import asyncio
import aiohttp
import time
import os
import json
import uuid
import statistics
from pathlib import Path
from typing import List, Dict
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("load_test")

# API URL - can be overridden via environment variable
API_URL = os.getenv("API_URL", "http://localhost:8080/transcribe")
API_KEY = os.getenv("API_KEY", "your-api-key-here")
MODEL_ID = os.getenv("MODEL_ID", "your-model-id-here")

# Concurrency levels to test - can be overridden via environment variable
CONCURRENCY_LEVELS_STR = os.getenv("CONCURRENCY_LEVELS", "3,5,7,9,10,12,14")
CONCURRENCY_LEVELS = [int(x.strip()) for x in CONCURRENCY_LEVELS_STR.split(",")]


async def transcribe_audio_async(session: aiohttp.ClientSession, audio_bytes: bytes, audio_name: str, api_key: str = None, model_id: str = None) -> Dict:
    """
    Sends audio file to transcription API and measures latency
    """
    start_time = time.time()
    api_key = api_key or API_KEY
    model_id = model_id or MODEL_ID
    
    try:
        # Prepare multipart form data
        form_data = aiohttp.FormData()
        form_data.add_field(
            'audio_file',
            audio_bytes,
            filename=f"audio_{uuid.uuid4()}.wav",
            content_type='audio/wav'
        )
        
        # Send request
        headers = {
            "api-key": api_key,
            "Accept": "application/json",
            "model-id": model_id
        }

        async with session.post(f"{API_URL}", data=form_data, headers=headers) as response:
            end_time = time.time()
            latency = end_time - start_time
            
            if response.status == 200:
                result = await response.json()
                print(result)
                return {
                    'success': True,
                    'latency': latency,
                    'transcription': result.get('transcription', ''),
                    'audio_name': audio_name,
                    'status': response.status
                }
            else:
                return {
                    'success': False,
                    'latency': latency,
                    'error': f"HTTP {response.status} {await response.text()}",
                    'audio_name': audio_name,
                    'status': response.status
                }
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        return {
            'success': False,
            'latency': latency,
            'error': str(e),
            'audio_name': audio_name,
            'status': None
        }


async def run_concurrent_tests(audio_samples: List[Dict], num_concurrent: int) -> Dict:
    """
    Runs num_concurrent simultaneous requests and collects statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Test with {num_concurrent} concurrent requests")
    logger.info(f"{'='*80}")
    
    # Create shared HTTP session
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Select audios to use (cycle if necessary)
        tasks = []
        for i in range(num_concurrent):
            audio_sample = audio_samples[i % len(audio_samples)]
            task = transcribe_audio_async(
                session,
                audio_sample['audio_bytes'],
                audio_sample['name']
            )
            tasks.append(task)
        
        # Measure total time
        start_time = time.time()
        
        # Launch all requests in parallel
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    latencies = [r['latency'] for r in successful_results]
    
    stats = {
        'num_concurrent': num_concurrent,
        'total_requests': num_concurrent,
        'successful_requests': len(successful_results),
        'failed_requests': len(failed_results),
        'success_rate': (len(successful_results) / num_concurrent * 100) if num_concurrent > 0 else 0,
        'total_time': total_time,
        'throughput': num_concurrent / total_time if total_time > 0 else 0,  # requests/second
    }
    
    if latencies:
        stats.update({
            'latency_min': min(latencies),
            'latency_max': max(latencies),
            'latency_mean': statistics.mean(latencies),
            'latency_median': statistics.median(latencies),
            'latency_stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        })
    else:
        stats.update({
            'latency_min': None,
            'latency_max': None,
            'latency_mean': None,
            'latency_median': None,
            'latency_stdev': None,
        })
    
    # Display results
    logger.info(f"\nResults:")
    logger.info(f"  ✓ Successful requests: {stats['successful_requests']}/{stats['total_requests']} ({stats['success_rate']:.1f}%)")
    logger.info(f"  ✗ Failed requests: {stats['failed_requests']}")
    logger.info(f"  ⏱  Total time: {stats['total_time']:.2f}s")
    logger.info(f"  📊 Throughput: {stats['throughput']:.2f} requests/second")
    for i, result in enumerate(successful_results):
        transcription_preview = result['transcription'][:50] if result.get('transcription') else 'N/A'
        logger.info(f"  • Request {i+1} ({result['latency']:.3f}s): {transcription_preview}...")
    if latencies:
        logger.info(f"\nLatencies (seconds):")
        logger.info(f"  • Minimum:  {stats['latency_min']:.3f}s")
        logger.info(f"  • Maximum:  {stats['latency_max']:.3f}s")
        logger.info(f"  • Mean:     {stats['latency_mean']:.3f}s")
        logger.info(f"  • Median:   {stats['latency_median']:.3f}s")
        logger.info(f"  • Std Dev:  {stats['latency_stdev']:.3f}s")
    
    if failed_results:
        logger.warning(f"\nErrors encountered:")
        error_counts = {}
        for r in failed_results:
            error = r.get('error', 'Unknown')
            error_counts[error] = error_counts.get(error, 0) + 1
        for error, count in error_counts.items():
            logger.warning(f"  • {error}: {count} times")
    
    return stats


async def load_audio_samples(samples_dir: str = None, max_samples: int = 10) -> List[Dict]:
    """
    Loads audio samples from directory
    """
    samples_dir = samples_dir or os.getenv("AUDIO_SAMPLES_DIR", "./audio_samples")
    samples = []
    samples_path = Path(samples_dir)
    
    if not samples_path.exists():
        raise FileNotFoundError(f"Directory {samples_dir} does not exist")
    
    logger.info(f"Loading audio samples from {samples_dir}...")
    
    audio_files = list(samples_path.glob('*.wav'))
    if not audio_files:
        audio_files = list(samples_path.glob('*.mp3'))
    # Limit number of samples to avoid memory overload
    audio_files = audio_files[:max_samples]
    
    for audio_file in audio_files:
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        samples.append({
            'name': audio_file.name,
            'path': str(audio_file),
            'audio_bytes': audio_bytes,
            'size_kb': len(audio_bytes) / 1024
        })
    
    logger.info(f"✓ {len(samples)} audio samples loaded")
    for sample in samples[:5]:  # Display first 5
        logger.info(f"  • {sample['name']} ({sample['size_kb']:.1f} KB)")
    if len(samples) > 5:
        logger.info(f"  ... and {len(samples) - 5} more")
    
    return samples


async def check_api_health():
    """
    Checks if API is accessible
    """
    try:
        async with aiohttp.ClientSession() as session:
            # Try /health endpoint first
            health_url = API_URL.replace('/transcribe', '/health')
            try:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✓ API health check OK: {data}")
                        return True
            except:
                pass
            
            logger.warning(f"⚠️  Health check endpoint not available")
            logger.info(f"→ Proceeding with direct /transcribe endpoint test")
            return True  # Continue anyway to test /transcribe
            
    except Exception as e:
        logger.error(f"✗ Cannot contact API: {e}")
        return False


async def main():
    """
    Main load test function
    """
    logger.info("="*80)
    logger.info("LOAD TEST - Whisper API")
    logger.info(f"URL: {API_URL}")
    logger.info("="*80)
    
    # Check API
    # if not await check_api_health():
    #     logger.error("API is not accessible. Stopping test.")
    #     return
    
    # Load audio samples
    max_samples = int(os.getenv("MAX_SAMPLES", "10"))
    try:
        audio_samples = await load_audio_samples(max_samples=max_samples)
    except Exception as e:
        logger.error(f"Error loading samples: {e}")
        return
    
    if not audio_samples:
        logger.error("No audio samples found. Stopping test.")
        return
    
    # Run tests for each concurrency level
    all_stats = []
    
    for num_concurrent in CONCURRENCY_LEVELS:
        try:
            stats = await run_concurrent_tests(audio_samples, num_concurrent)
            all_stats.append(stats)
            
            # Pause between tests to let server recover
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error during test with {num_concurrent} requests: {e}")
            continue
    
    # Save results
    results_file = os.getenv("RESULTS_FILE", "load_test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS SAVED TO: {results_file}")
    logger.info(f"{'='*80}")
    
    # Display summary
    logger.info("\n📊 PERFORMANCE SUMMARY:")
    logger.info(f"\n{'Concurrent':<12} {'Max Latency':<15} {'Success Rate':<15}")
    logger.info("-" * 80)
    
    for stats in all_stats:
        concurrent = f"{stats['num_concurrent']}"
        latency = f"{stats['latency_max']:.3f}s" if stats['latency_max'] is not None else "N/A"
        success_rate = f"{stats['success_rate']:.1f}%"
        logger.info(f"{concurrent:<12} {latency:<15} {success_rate:<15}")
    
    # Find best configuration (lowest max latency with good success rate)
    valid_stats = [s for s in all_stats if s['success_rate'] >= 90 and s['latency_max'] is not None]
    if valid_stats:
        best_config = min(valid_stats, key=lambda x: x['latency_max'])
        logger.info(f"\n🎯 OPTIMAL CONFIGURATION:")
        logger.info(f"   Concurrency: {best_config['num_concurrent']} simultaneous requests")
        logger.info(f"   Max latency: {best_config['latency_max']:.3f}s")
        logger.info(f"   Success rate: {best_config['success_rate']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())