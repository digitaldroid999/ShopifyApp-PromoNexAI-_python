"""
Vertex AI (Google GenAI) client for Imagen image generation.
Uses google.genai with vertexai=True. Credentials from promo-nex-ai-vertex-ai-key.json
in project root, or GOOGLE_APPLICATION_CREDENTIALS. Project from env or from that key file.
"""

import json
import os
from pathlib import Path
from typing import Optional, Any

# Key file in project root (used when GOOGLE_APPLICATION_CREDENTIALS is not set)
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_DEFAULT_CREDENTIALS_PATH = _PROJECT_ROOT / "promo-nex-ai-vertex-ai-key.json"

# Optional: only set if google-genai is installed and configured
vertex_manager: Optional[Any] = None


def _ensure_credentials_and_project() -> Optional[str]:
    """
    Set GOOGLE_APPLICATION_CREDENTIALS to promo-nex-ai-vertex-ai-key.json if the file
    exists and env is not set. Return project_id from env or from that key file.
    """
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path and _DEFAULT_CREDENTIALS_PATH.is_file():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_DEFAULT_CREDENTIALS_PATH.resolve())

    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT")
    if project:
        return project
    # Read project_id from key file (only project_id, no secrets)
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or str(_DEFAULT_CREDENTIALS_PATH)
    if not os.path.isfile(key_path):
        return None
    try:
        with open(key_path) as f:
            data = json.load(f)
        return data.get("project_id")
    except Exception:
        return None


def _create_vertex_manager() -> Optional[Any]:
    """Create Vertex AI client manager if dependencies and config are available."""
    project = _ensure_credentials_and_project()
    location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_LOCATION", "us-central1")
    if not project:
        return None
    try:
        from google import genai
    except ImportError:
        return None
    try:
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
    except Exception:
        return None

    class _VertexManager:
        def __init__(self, c):
            self.client = c

        def is_available(self) -> bool:
            try:
                return self.client is not None
            except Exception:
                return False

    return _VertexManager(client)


vertex_manager = _create_vertex_manager()
