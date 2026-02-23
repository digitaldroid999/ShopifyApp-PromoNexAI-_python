"""
Merge (finalize short) API: start finalization task and poll status.

Request/response samples:

--- Start finalize (POST /merge/finalize) ---
Request:
  POST /merge/finalize
  Content-Type: application/json

  {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "short_id": "660e8400-e29b-41d4-a716-446655440001"
  }

Response (200):
  {
    "task_id": "770e8400-e29b-41d4-a716-446655440002",
    "status": "pending",
    "short_id": "660e8400-e29b-41d4-a716-446655440001",
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Finalization task started",
    "created_at": "2025-02-23T12:00:00.000000Z",
    "progress": null,
    "current_step": null,
    "error_message": null,
    "thumbnail_url": null,
    "final_video_url": null,
    "completed_at": null
  }

--- Poll status (GET /merge/status/{task_id}) ---
Request:
  GET /merge/status/770e8400-e29b-41d4-a716-446655440002

Response while running (200):
  {
    "task_id": "770e8400-e29b-41d4-a716-446655440002",
    "status": "running",
    "short_id": "660e8400-e29b-41d4-a716-446655440001",
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Merging videos and audio",
    "created_at": "2025-02-23T12:00:00.000000Z",
    "progress": 40,
    "current_step": "Merging videos and audio",
    "error_message": null,
    "thumbnail_url": null,
    "final_video_url": null,
    "completed_at": null
  }

Response when completed (200):
  {
    "task_id": "770e8400-e29b-41d4-a716-446655440002",
    "status": "completed",
    "short_id": "660e8400-e29b-41d4-a716-446655440001",
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Uploading final video",
    "created_at": "2025-02-23T12:00:00.000000Z",
    "progress": 100,
    "current_step": null,
    "error_message": null,
    "thumbnail_url": null,
    "final_video_url": "https://storage.example.com/final_videos/.../short.mp4",
    "completed_at": "2025-02-23T12:05:00.000000Z"
  }

Response when failed (200, status="failed"):
  {
    "task_id": "770e8400-e29b-41d4-a716-446655440002",
    "status": "failed",
    "short_id": "660e8400-e29b-41d4-a716-446655440001",
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "No video scenes found for this short",
    "created_at": "2025-02-23T12:00:00.000000Z",
    "progress": null,
    "current_step": null,
    "error_message": "No video scenes found for this short",
    "thumbnail_url": null,
    "final_video_url": null,
    "completed_at": null
  }
"""

import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.models import FinalizeShortRequest, FinalizeShortResponse, TaskStatus
from app.services.merging_service import merging_service
from app.utils.task_management import get_task_status
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/merge", tags=["merge"])


def _task_to_response(task: dict) -> FinalizeShortResponse:
    """Map task dict from task_management to FinalizeShortResponse."""
    status_str = task.get("status") or "pending"
    try:
        status = TaskStatus(status_str)
    except ValueError:
        status = TaskStatus.PENDING

    created_at = task.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    elif created_at is None:
        created_at = datetime.now(timezone.utc)

    updated_at = task.get("updated_at")
    completed_at = None
    if updated_at and status == TaskStatus.COMPLETED:
        completed_at = (
            datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if isinstance(updated_at, str)
            else updated_at
        )

    metadata = task.get("metadata") or {}
    short_id = task.get("short_id") or metadata.get("short_id") or ""
    user_id = task.get("user_id") or ""

    msg = task.get("message")
    message = msg if isinstance(msg, str) else (str(msg) if msg is not None else status_str)
    current_step = msg if isinstance(msg, str) else (str(msg) if msg is not None else None)
    progress_val = task.get("progress")
    progress = float(progress_val) if progress_val is not None else None

    return FinalizeShortResponse(
        task_id=task.get("task_id", ""),
        status=status,
        short_id=short_id,
        user_id=user_id,
        message=message,
        created_at=created_at,
        progress=progress,
        current_step=current_step,
        error_message=task.get("error_message"),
        thumbnail_url=task.get("thumbnail_url"),
        final_video_url=metadata.get("final_video_url"),
        completed_at=completed_at,
    )


@router.post("/finalize", response_model=FinalizeShortResponse)
def start_finalize_short(request: FinalizeShortRequest) -> FinalizeShortResponse:
    """
    Start the merge/finalize process for a short video.

    Runs in the background: downloads scene videos and audio, merges them,
    adds watermark/subtitles if needed, uploads the final video and updates the short.

    Returns immediately with `task_id`. Poll GET /merge/status/{task_id} for progress
    and `final_video_url` when status is `completed`.
    """
    try:
        result = merging_service.start_finalize_short_task(
            user_id=request.user_id,
            short_id=request.short_id,
        )
        # Build response from result + request (task store may not be updated yet)
        task_id = result["task_id"]
        task = get_task_status(task_id)
        if task:
            return _task_to_response(task)
        return FinalizeShortResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            short_id=request.short_id,
            user_id=request.user_id,
            message=result.get("message", "Finalization task started"),
            created_at=datetime.fromisoformat(result["created_at"].replace("Z", "+00:00"))
            if isinstance(result.get("created_at"), str)
            else datetime.now(timezone.utc),
            progress=None,
            current_step=None,
            error_message=None,
            thumbnail_url=None,
            final_video_url=None,
            completed_at=None,
        )
    except Exception as e:
        logger.error(f"Start finalize short failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=FinalizeShortResponse)
def get_finalize_task_status(task_id: str) -> FinalizeShortResponse:
    """
    Get status of a finalize-short (merge) task.

    When status is `completed`, `final_video_url` is set.
    When status is `failed`, `error_message` is set.
    """
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return _task_to_response(task)
