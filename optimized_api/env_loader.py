import os
from pathlib import Path


def load_env_file(env_path: str | None = None) -> bool:
    """Load environment variables from a .env file into os.environ."""
    path = Path(env_path or Path(__file__).resolve().parents[1] / ".env")
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    return True
