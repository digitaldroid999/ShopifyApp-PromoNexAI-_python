"""Background generation endpoints: async task + extract prompt."""

import traceback

from fastapi import APIRouter, HTTPException

from app.models import (
    BackgroundGenerationRequest,
    BackgroundGenerationResponse,
    ExtractBackgroundPromptRequest,
    ExtractBackgroundPromptResponse,
)
from app.services.background_generation_service import background_generation_service
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/background", tags=["background"])


@router.post("/generate", response_model=BackgroundGenerationResponse)
def start_background_generation(request: BackgroundGenerationRequest) -> BackgroundGenerationResponse:
    """
    Start an async background generation task (OpenAI prompt + Vertex Imagen).
    Returns immediately with task_id; poll GET /background/status/{task_id} for result.
    """
    try:
        result = background_generation_service.start_background_generation_task(
            user_id=request.user_id,
            product_description=request.product_description,
            mood=request.mood,
            style=request.style,
            environment=request.environment,
            manual_prompt=request.manual_prompt,
            scene_id=request.scene_id,
            short_id=request.short_id,
        )
        return BackgroundGenerationResponse(
            success=True,
            task_id=result["task_id"],
            status=result["status"],
            message=result["message"],
            created_at=result["created_at"],
        )
    except Exception as e:
        logger.error(f"Start background generation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=BackgroundGenerationResponse)
def get_background_task_status(task_id: str) -> BackgroundGenerationResponse:
    """
    Get status of a background generation task. Use after POST /background/generate.
    When status is 'completed', image_url is set.
    """
    task = background_generation_service.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    status = task.get("status", "unknown")
    success = status == "completed"
    return BackgroundGenerationResponse(
        success=success,
        task_id=task.get("task_id"),
        status=status,
        image_url=task.get("result_image_url"),
        message=task.get("current_step") or status,
        error=task.get("error_message"),
        progress=task.get("progress"),
        current_step=task.get("current_step"),
        created_at=task.get("created_at"),
    )


@router.post("/extract-prompt", response_model=ExtractBackgroundPromptResponse)
def extract_background_prompt(request: ExtractBackgroundPromptRequest) -> ExtractBackgroundPromptResponse:
    """
    Extract a background-generation prompt from a product description (OpenAI).
    Returns the prompt suitable for Vertex Imagen; does not generate an image.
    """
    try:
        result = background_generation_service.extract_background_prompt(
            product_description=request.product_description,
            mood=request.mood,
            style=request.style,
            environment=request.environment,
        )
        success = result.get("error") is None
        return ExtractBackgroundPromptResponse(
            success=success,
            prompt=result.get("prompt", ""),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"Extract background prompt failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
