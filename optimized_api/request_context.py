import os


def resolve_request_context(api_key: str | None, model_id: str | None):
    """Return a usable api-key/model-id pair for local standalone mode.

    When headers are missing, fall back to environment defaults so the repo can
    be used without the original Latice backend or a dedicated API key.
    """
    resolved_api_key = api_key or os.getenv("DEFAULT_API_KEY", "local")
    resolved_model_id = model_id or os.getenv("DEFAULT_MODEL_ID", "local-model")
    return resolved_api_key, resolved_model_id
