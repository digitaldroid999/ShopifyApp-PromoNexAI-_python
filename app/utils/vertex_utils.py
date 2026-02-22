"""
Vertex AI (Google GenAI) client for Imagen image generation.
Uses google.genai with vertexai=True; requires GOOGLE_CLOUD_PROJECT and optionally
GOOGLE_CLOUD_LOCATION. Credentials via ADC or GOOGLE_APPLICATION_CREDENTIALS.
"""

import os
from typing import Optional, Any

# Optional: only set if google-genai is installed and configured
vertex_manager: Optional[Any] = None


def _create_vertex_manager() -> Optional[Any]:
    """Create Vertex AI client manager if dependencies and config are available."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT")
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
