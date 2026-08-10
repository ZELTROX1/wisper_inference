import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SYS_PATH = str(ROOT / "optimized_api")
if SYS_PATH not in sys.path:
    sys.path.insert(0, SYS_PATH)

from request_context import resolve_request_context


class ResolveRequestContextTests(unittest.TestCase):
    def test_uses_environment_defaults_when_headers_are_missing(self):
        with patch.dict(os.environ, {"DEFAULT_API_KEY": "local-token", "DEFAULT_MODEL_ID": "local-model"}, clear=False):
            self.assertEqual(resolve_request_context(None, None), ("local-token", "local-model"))

    def test_prefers_explicit_headers_when_present(self):
        self.assertEqual(resolve_request_context("provided-key", "provided-model"), ("provided-key", "provided-model"))


if __name__ == "__main__":
    unittest.main()
