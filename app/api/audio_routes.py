"""Audio script generation and audio generation endpoints."""

import traceback

from fastapi import APIRouter, HTTPException

from app.models import (
    AudioScriptGenerationRequest,
    AudioScriptGenerationResponse,
    AudioGenerationRequest,
    AudioGenerationResponse,
    TestAudioRequest,
    TestAudioResponse,
)
from app.services.audio_generation_service import audio_generation_service
from app.services.test_audio_service import test_audio_service
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


@router.post("/test-audio", response_model=TestAudioResponse)
def get_test_audio(request: TestAudioRequest) -> TestAudioResponse:
    """
    Get or generate a short test audio sample for a voice and language.
    Uses a fixed sample sentence per language. Results are cached in MongoDB;
    if a sample already exists for (voice_id, language), the cached audio_url is returned.
    Audio is saved under public/generated_audio/test_audios/{user_id}/{voice_id}_{language}.mp3.
    """
    try:
        return test_audio_service.get_test_audio(request)
    except Exception as e:
        logger.error(f"Test audio failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
