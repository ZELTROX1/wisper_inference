#!/usr/bin/env python3
"""
Script de monitoring GPU/CPU/VRAM pendant les tests de requêtes simultanées.
Lance le monitoring en parallèle pendant que tu exécutes tes tests.
"""
import time
import subprocess
import json
import sys
from collections import deque
import psutil
import torch

class GPUMonitor:
    def __init__(self, interval=0.1, max_samples=1000):
        self.interval = interval
        self.max_samples = max_samples
        self.samples = {
            'timestamp': deque(maxlen=max_samples),
            'gpu_util': deque(maxlen=max_samples),
            'gpu_memory_used': deque(maxlen=max_samples),
            'gpu_memory_total': deque(maxlen=max_samples),
            'gpu_power': deque(maxlen=max_samples),
            'gpu_temp': deque(maxlen=max_samples),
            'cpu_percent': deque(maxlen=max_samples),
            'cpu_memory_percent': deque(maxlen=max_samples),
            'gpu_sm_util': deque(maxlen=max_samples),  # Streaming Multiprocessor utilization
        }
        self.running = False
        
    def get_nvidia_smi(self):
        """Récupère les stats GPU via nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 5:
                    return {
                        'gpu_util': float(parts[0]),
                        'memory_used': float(parts[1]),
                        'memory_total': float(parts[2]),
                        'power': float(parts[3]),
                        'temp': float(parts[4]),
                        'sm_util': float(parts[0])  # Utilise gpu_util comme approximation
                    }
        except Exception as e:
            print(f"[DEBUG] nvidia-smi error: {e}", file=sys.stderr)
        return None
    
    def get_pytorch_stats(self):
        """Récupère les stats via PyTorch"""
        stats = {}
        if torch.cuda.is_available():
            stats['torch_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['torch_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
            stats['torch_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return stats
    
    def collect_sample(self):
        """Collecte un échantillon de métriques"""
        timestamp = time.time()
        
        # GPU via nvidia-smi
        gpu_stats = self.get_nvidia_smi()
        if gpu_stats:
            self.samples['gpu_util'].append(gpu_stats['gpu_util'])
            self.samples['gpu_memory_used'].append(gpu_stats['memory_used'])
            self.samples['gpu_memory_total'].append(gpu_stats['memory_total'])
            self.samples['gpu_power'].append(gpu_stats['power'])
            self.samples['gpu_temp'].append(gpu_stats['temp'])
            self.samples['gpu_sm_util'].append(gpu_stats['sm_util'])
        else:
            # Si nvidia-smi échoue, essayer PyTorch
            if torch.cuda.is_available():
                torch_stats = self.get_pytorch_stats()
                self.samples['gpu_util'].append(0)  # Pas disponible via PyTorch
                self.samples['gpu_memory_used'].append(torch_stats.get('torch_allocated', 0) * 1024)  # Convert GB to MB
                self.samples['gpu_memory_total'].append(0)  # Pas disponible via PyTorch
                self.samples['gpu_power'].append(0)
                self.samples['gpu_temp'].append(0)
                self.samples['gpu_sm_util'].append(0)
            else:
                self.samples['gpu_util'].append(0)
                self.samples['gpu_memory_used'].append(0)
                self.samples['gpu_memory_total'].append(0)
                self.samples['gpu_power'].append(0)
                self.samples['gpu_temp'].append(0)
                self.samples['gpu_sm_util'].append(0)
        
        # CPU
        self.samples['cpu_percent'].append(psutil.cpu_percent(interval=None))
        self.samples['cpu_memory_percent'].append(psutil.virtual_memory().percent)
        
        self.samples['timestamp'].append(timestamp)
    
    def monitor_loop(self):
        """Boucle de monitoring"""
        self.running = True
        start_time = time.time()
        
        print("Monitoring démarré. Appuie sur Ctrl+C pour arrêter.\n")
        print(f"{'Time':<8} {'GPU%':<6} {'SM%':<6} {'VRAM':<12} {'Power':<8} {'Temp':<6} {'CPU%':<6} {'RAM%':<6}")
        print("-" * 70)
        
        try:
            while self.running:
                self.collect_sample()
                
                if len(self.samples['timestamp']) > 0:
                    elapsed = time.time() - start_time
                    gpu_util = self.samples['gpu_util'][-1]
                    sm_util = self.samples['gpu_sm_util'][-1]
                    vram_used = self.samples['gpu_memory_used'][-1]
                    vram_total = self.samples['gpu_memory_total'][-1]
                    power = self.samples['gpu_power'][-1]
                    temp = self.samples['gpu_temp'][-1]
                    cpu = self.samples['cpu_percent'][-1]
                    ram = self.samples['cpu_memory_percent'][-1]
                    
                    if vram_total > 0:
                        vram_str = f"{vram_used/1024:.1f}/{vram_total/1024:.1f}GB"
                    elif vram_used > 0:
                        vram_str = f"{vram_used/1024:.1f}GB"
                    else:
                        vram_str = "N/A"
                    
                    print(f"{elapsed:>6.1f}s {gpu_util:>5.1f}% {sm_util:>5.1f}% {vram_str:>12} {power:>6.1f}W {temp:>4.0f}°C {cpu:>5.1f}% {ram:>5.1f}%")
                
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring arrêté.")
            self.running = False
    
    def get_summary(self):
        """Retourne un résumé des stats"""
        if not self.samples['timestamp']:
            return None
        
        def avg(d):
            return sum(d) / len(d) if d else 0
        
        def max_val(d):
            return max(d) if d else 0
        
        def min_val(d):
            return min(d) if d else 0
        
        return {
            'duration': self.samples['timestamp'][-1] - self.samples['timestamp'][0] if len(self.samples['timestamp']) > 1 else 0,
            'gpu_util': {
                'avg': avg(self.samples['gpu_util']),
                'max': max_val(self.samples['gpu_util']),
                'min': min_val(self.samples['gpu_util'])
            },
            'gpu_sm_util': {
                'avg': avg(self.samples['gpu_sm_util']),
                'max': max_val(self.samples['gpu_sm_util']),
                'min': min_val(self.samples['gpu_sm_util'])
            },
            'gpu_memory': {
                'avg_used': avg(self.samples['gpu_memory_used']),
                'max_used': max_val(self.samples['gpu_memory_used']),
                'total': self.samples['gpu_memory_total'][0] if self.samples['gpu_memory_total'] else 0
            },
            'gpu_power': {
                'avg': avg(self.samples['gpu_power']),
                'max': max_val(self.samples['gpu_power'])
            },
            'gpu_temp': {
                'avg': avg(self.samples['gpu_temp']),
                'max': max_val(self.samples['gpu_temp'])
            },
            'cpu': {
                'avg': avg(self.samples['cpu_percent']),
                'max': max_val(self.samples['cpu_percent'])
            },
            'ram': {
                'avg': avg(self.samples['cpu_memory_percent']),
                'max': max_val(self.samples['cpu_memory_percent'])
            }
        }
    
    def print_summary(self):
        """Affiche un résumé"""
        summary = self.get_summary()
        if not summary:
            print("Aucune donnée collectée.")
            return
        
        print("\n" + "="*70)
        print("RÉSUMÉ DU MONITORING")
        print("="*70)
        print(f"Durée: {summary['duration']:.1f}s")
        print(f"\nGPU:")
        print(f"  Utilisation: {summary['gpu_util']['avg']:.1f}% (max: {summary['gpu_util']['max']:.1f}%)")
        print(f"  SM Utilisation: {summary['gpu_sm_util']['avg']:.1f}% (max: {summary['gpu_sm_util']['max']:.1f}%)")
        if summary['gpu_memory']['total'] > 0:
            print(f"  VRAM: {summary['gpu_memory']['avg_used']/1024:.1f}GB / {summary['gpu_memory']['total']/1024:.1f}GB (max: {summary['gpu_memory']['max_used']/1024:.1f}GB)")
        else:
            print(f"  VRAM: {summary['gpu_memory']['avg_used']/1024:.1f}GB (max: {summary['gpu_memory']['max_used']/1024:.1f}GB)")
        print(f"  Power: {summary['gpu_power']['avg']:.1f}W (max: {summary['gpu_power']['max']:.1f}W)")
        print(f"  Temp: {summary['gpu_temp']['avg']:.1f}°C (max: {summary['gpu_temp']['max']:.1f}°C)")
        print(f"\nCPU:")
        print(f"  Utilisation: {summary['cpu']['avg']:.1f}% (max: {summary['cpu']['max']:.1f}%)")
        print(f"  RAM: {summary['ram']['avg']:.1f}% (max: {summary['ram']['max']:.1f}%)")
        print("="*70)
        
        # Analyse
        print("\nANALYSE:")
        if summary['gpu_util']['avg'] < 50:
            print("⚠️  GPU sous-utilisé (<50%) - le bottleneck n'est probablement pas le GPU")
        elif summary['gpu_util']['avg'] > 90:
            print("✅ GPU bien utilisé (>90%) - le bottleneck est probablement le GPU")
        else:
            print("⚠️  GPU modérément utilisé (50-90%) - peut-être un problème de batching")
        
        if summary['gpu_sm_util']['avg'] < summary['gpu_util']['avg'] * 0.8:
            print("⚠️  SM utilization < GPU utilization - possible problème de parallélisme")
        
        if summary['cpu']['avg'] > 80:
            print("⚠️  CPU saturé - possible bottleneck CPU")
    
    def save_json(self, filename):
        """Sauvegarde les données en JSON"""
        data = {
            'samples': {k: list(v) for k, v in self.samples.items()},
            'summary': self.get_summary()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nDonnées sauvegardées dans {filename}")

def main():
    monitor = GPUMonitor(interval=0.1)
    
    try:
        monitor.monitor_loop()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.print_summary()
        if len(sys.argv) > 1:
            monitor.save_json(sys.argv[1])

if __name__ == "__main__":
    main()

