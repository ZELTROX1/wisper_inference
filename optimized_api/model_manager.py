import os
from huggingface_hub import snapshot_download
import whisper_s2t
from whisper_s2t.backends.ctranslate2.model import BEST_ASR_CONFIG

CACHE = {}

BEST_ASR_CONFIG['word_timestamps'] = True

def get_model(repo_id: str):
    """Downloads the model if not already loaded, then returns it."""
    if repo_id in CACHE:
        return CACHE[repo_id]

    # Already a local CTranslate2 model directory (e.g. output of
    # ct2-transformers-converter, see scripts/convert_to_ct2.sh) -> load as-is,
    # no HuggingFace download/snapshot involved.
    if os.path.isdir(repo_id):
        local_dir = repo_id
    else:
        local_dir = f"./models/{repo_id.replace('/', '_')}"
        os.makedirs(local_dir, exist_ok=True)

        # Get HuggingFace token from environment variable
        hf_token = os.getenv("HUGGINGFACE_TOKEN")

        print(f"Downloading model {repo_id}...")
        if hf_token:
            snapshot_download(repo_id=repo_id, local_dir=local_dir, token=hf_token)
        else:
            snapshot_download(repo_id=repo_id, local_dir=local_dir)

    print("Loading CTranslate2 model...")
    model = whisper_s2t.load_model(local_dir, backend='CTranslate2', asr_options=BEST_ASR_CONFIG)

    CACHE[repo_id] = model
    print(f"Model {repo_id} ready.")
    return model
