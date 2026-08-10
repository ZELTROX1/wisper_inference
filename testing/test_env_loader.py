import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYS_PATH = str(ROOT / "optimized_api")
if SYS_PATH not in sys.path:
    sys.path.insert(0, SYS_PATH)

from env_loader import load_env_file


class EnvLoaderTests(unittest.TestCase):
    def test_load_env_file_populates_missing_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("LOCAL_MODE=true\nLOCAL_MODEL_REPO_ID=demo/model\n")
            with unittest.mock.patch.dict(os.environ, {}, clear=True):
                loaded = load_env_file(env_path)
                self.assertTrue(loaded)
                self.assertEqual(os.environ["LOCAL_MODE"], "true")
                self.assertEqual(os.environ["LOCAL_MODEL_REPO_ID"], "demo/model")


if __name__ == "__main__":
    unittest.main()
