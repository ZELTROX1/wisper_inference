import os
import time
import requests
from typing import Dict, Any

# Configuration
REFRESH_INTERVAL_SECONDS = 5 * 60  # 5 minutes
REFRESH_INTERVAL_ALL_MODELS_SECONDS = 60 * 5  # 5 minutes
FLUSH_INTERVAL_SECONDS = 60 * 5  # 1 minute

class GPUQuotaManager:
    def __init__(self):
        self.user_caches = {}  # {api_key: cache_data}
        self.user_last_usage_record = {}  # {api_key: timestamp}
        self.user_pending_usage = {}  # {api_key: [usage_list]}
        self.all_existing_models = []
        self.all_existing_models_last_update = None
        self.users_data = {}  # {api_key: user_data}
        self.api_base_url = os.getenv("API_BASE_URL", "https://app.latice.ai")
        self.private_secured_key = os.getenv("PRIVATE_SECURED_KEY", "")
        self.batchers = {}
        # Standalone mode: no Latice SaaS backend to call (app.latice.ai down / not ours anymore).
        # Any api-key/model-id pair is accepted and mapped straight to LOCAL_MODEL_REPO_ID,
        # with unlimited quota. Set LOCAL_MODE=false to restore the original hosted-backend behavior.
        self.local_mode = os.getenv("LOCAL_MODE", "true").lower() == "true"
        self.local_model_repo_id = os.getenv("LOCAL_MODEL_REPO_ID", "")

    def can_transcribe(self, api_key: str, duration_ms: int) -> bool:
        """Checks if transcription is allowed for a given API key"""
        if self.local_mode:
            return True
        # Si pas dans le cache, charger depuis users_data (mis à jour par get_all_existing_models)
        if api_key not in self.user_caches:
            if api_key in self.users_data:
                self.user_caches[api_key] = self.users_data[api_key]
            else:
                # Premier appel: récupérer le PK puis chercher dans users_data
                try:
                    headers = {"X-API-Key": api_key}
                    if self.private_secured_key:
                        headers["private_secured_key"] = self.private_secured_key
                    response = requests.get(
                        f"{self.api_base_url}/api/get_user_pk",
                        headers=headers,
                        timeout=10
                    )
                    response.raise_for_status()
                    pk = response.json().get("pk")
                    if pk and pk in self.users_data:
                        self.user_caches[api_key] = self.users_data[pk]
                    else:
                        raise Exception(f"User not found for API key")
                except Exception as e:
                    print(f"Erreur lors de la récupération du PK pour {api_key}: {e}")
                    raise Exception(f"API unavailable: {e}")
        
        return self.user_caches[api_key]["available_milliseconds"] >= duration_ms
    
    def resolve_repo_id(self, api_key: str, model_id: str) -> str:
        """Maps (api_key, model_id) -> HuggingFace repo_id.

        In local_mode this ignores the Latice backend entirely and always returns
        LOCAL_MODEL_REPO_ID, so any api-key/model-id header works against a single
        model deployed on this GPU box.
        """
        if self.local_mode:
            if not self.local_model_repo_id:
                raise RuntimeError("LOCAL_MODEL_REPO_ID env var not set")
            return self.local_model_repo_id
        user_cache = self.user_caches.get(api_key, {})
        models = user_cache.get("models", [])
        for model in models:
            if model.get("model_id") == model_id:
                return model.get("hugging_face_repo_id")
        return None

    def consume_quota(self, api_key: str, duration_ms: int):
        """Consumes quota locally for a given API key"""
        if self.local_mode:
            return
        if api_key in self.user_caches:
            self.user_caches[api_key]["available_milliseconds"] -= duration_ms
    
    def record_usage(self, api_key: str, cost: float, duration_seconds: float, latency_ms: int, requested_at: float, fallbacked: bool = False, streaming: bool = False):
        """Adds usage to queue for batch sending every 5 minutes"""
        if self.local_mode:
            return
        if api_key not in self.user_pending_usage:
            self.user_pending_usage[api_key] = []
        
        self.user_pending_usage[api_key].append({
            "cost": cost,
            "duration_seconds": duration_seconds,
            "infer_latency": latency_ms,
            "requested_at": requested_at,
            "fallbacked": fallbacked,
            "streaming": streaming
        })
    
    def _flush_usage_records(self):
        """Sends usage records for all users if > 1 minute since last send"""
        current_time = time.time()
        
        for api_key in list(self.user_pending_usage.keys()):
            if not self.user_pending_usage[api_key]:
                continue
                
            if (api_key not in self.user_last_usage_record or 
                current_time - self.user_last_usage_record[api_key] > FLUSH_INTERVAL_SECONDS):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/api/record_usages",
                        headers={
                            "X-API-Key": api_key,
                            "Content-Type": "application/json",
                        },
                        json={"usages": self.user_pending_usage[api_key]},
                        timeout=10
                    )
                    response.raise_for_status()
                    self.user_pending_usage[api_key] = []
                    self.user_last_usage_record[api_key] = current_time
                except Exception as e:
                    print(f"Erreur lors de l'enregistrement batch de l'usage pour {api_key}: {e}")
    
    def get_available_milliseconds(self, api_key: str) -> int:
        """Returns the number of available milliseconds for a given API key"""
        if self.local_mode:
            return 2**31
        if api_key not in self.user_caches:
            # Essayer de charger depuis users_data
            if api_key in self.users_data:
                self.user_caches[api_key] = self.users_data[api_key]
            else:
                return 0
        return self.user_caches[api_key].get("available_milliseconds", 0)

    def get_all_existing_models(self) -> list[str]:
        """Returns the list of all existing models and updates user quotas"""
        if self.local_mode:
            return [self.local_model_repo_id] if self.local_model_repo_id else []
        current_time = time.time()
        if (not self.all_existing_models or 
            not self.all_existing_models_last_update or
            current_time - self.all_existing_models_last_update > REFRESH_INTERVAL_ALL_MODELS_SECONDS):
            headers = {}
            if self.private_secured_key:
                headers["private_secured_key"] = self.private_secured_key
            response = requests.get(
                f"{self.api_base_url}/api/get_all_infos",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Extraire les models
            self.all_existing_models = [m.get("hugging_face_repo_id") for m in data.get("models", [])]
            
            # Mettre à jour users_data depuis les users retournés
            # Si format: users = [{"api_key": "...", "pk": "...", "available_milliseconds": X, "models": [...]}]
            for user in data.get("users", []):
                api_key = user.get("api_key")
                pk = user.get("pk")
                if api_key:
                    self.users_data[api_key] = user
                if pk:
                    self.users_data[pk] = user
            
            self.all_existing_models_last_update = current_time
        return self.all_existing_models

    def get_batcher(self, repo_id: str):
        """Returns a shared batcher for this repo_id, creating it if needed."""
        if repo_id in self.batchers:
            return self.batchers[repo_id]
        # Import local pour éviter les dépendances circulaires au chargement
        from model_manager import get_model
        from batcher import CoalescingBatcher
        model = get_model(repo_id)
        self.batchers[repo_id] = CoalescingBatcher(model)
        return self.batchers[repo_id]