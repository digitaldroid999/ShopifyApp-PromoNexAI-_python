"""Audio script generation and audio generation endpoints."""

import traceback

from fastapi import APIRouter, HTTPException

from app.models import (
    AudioScriptGenerationRequest,
    AudioScriptGenerationResponse,
    AudioGenerationRequest,
    AudioGenerationResponse,
)
from app.services.audio_generation_service import audio_generation_service
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/generate-script", response_model=AudioScriptGenerationResponse)
def generate_audio_script(request: AudioScriptGenerationRequest) -> AudioScriptGenerationResponse:
    """
    Generate an audio script for a short. User can review/edit the script
    before calling POST /audio/generate.
    """
    try:
        return audio_generation_service.generate_audio_script(request)
    except Exception as e:
        logger.error(f"Audio script generation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=AudioGenerationResponse)
def generate_audio(request: AudioGenerationRequest) -> AudioGenerationResponse:
    """
    Generate audio from the provided script. Script typically comes from
    /audio/generate-script (possibly edited by the user).
    Saves the file to public/generated_audio/{user_id}/{short_id}/{file_name}
    and returns audio_url as generated_audio/{user_id}/{short_id}/{file_name}.
    """
    try:
        return audio_generation_service.generate_audio(request)
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
