"""Image composite endpoint: composite two images and save to Shopify app public folder."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from app.models import ImageCompositeResponse
from app.services.image_processing_service import image_processing_service
from app.logging_config import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)

router = APIRouter(prefix="/image", tags=["image"])


class CompositeRequest(BaseModel):
    """Request body for compositing two images (overlay on background)."""
    overlay_url: str = Field(..., description="URL of the overlay image (e.g. product with transparent background)")
    background_url: str = Field(..., description="URL of the background image")
    scene_id: str = Field(..., description="Scene ID for organizing files")
    user_id: str = Field(..., description="User ID for organizing files")
    position_x: Optional[int] = Field(0, description="X position for overlay (0 = auto-center)")
    position_y: Optional[int] = Field(0, description="Y position for overlay (0 = auto-center)")
    resize_overlay: Optional[bool] = Field(True, description="Whether to resize overlay to fit background")


class MergeVideoRequest(BaseModel):
    """Request body for merging product image with background video (Scene2)."""
    product_image_url: str = Field(..., description="URL of product image (PNG with transparent background)")
    background_video_url: str = Field(..., description="URL of background video")
    scene_id: str = Field(..., description="Scene ID for organizing files")
    user_id: str = Field(..., description="User ID for organizing files")
    short_id: Optional[str] = Field(None, description="Short ID for path (defaults to scene_id)")
    scale: float = Field(0.4, description="Product scale relative to video width (0-1)")
    position: str = Field("center", description="One of: center, top, bottom, left, right")
    duration: Optional[int] = Field(None, description="Output duration in seconds; null = full video")
    add_animation: bool = Field(True, description="Apply zoom/floating animation")
    add_shadow: bool = Field(True, description="Add shadow to product")
    shadow_blur_radius: int = Field(25, description="Shadow blur radius")
    shadow_offset: List[int] = Field([15, 15], description="Shadow offset [x, y]")


class MergeVideoResponse(BaseModel):
    """Response for merge-image-with-video (sync)."""
    success: bool
    video_url: Optional[str] = None
    error: Optional[str] = None


@router.post("/merge-video", response_model=MergeVideoResponse)
def merge_video(request: MergeVideoRequest) -> MergeVideoResponse:
    """
    Merge product image with background video (Scene2). Synchronous: waits until done.
    Returns video_url under generated_videos/{user_id}/{short_id}/scene2/{file_name}.
    """
    result = image_processing_service.merge_image_with_video(
        product_image_url=request.product_image_url,
        background_video_url=request.background_video_url,
        scene_id=request.scene_id,
        user_id=request.user_id,
        short_id=request.short_id,
        scale=request.scale,
        position=request.position,
        duration=request.duration,
        add_animation=request.add_animation,
        add_shadow=request.add_shadow,
        shadow_blur_radius=request.shadow_blur_radius,
        shadow_offset=tuple(request.shadow_offset) if request.shadow_offset else (15, 15),
    )
    return MergeVideoResponse(
        success=result["success"],
        video_url=result.get("video_url"),
        error=result.get("error"),
    )


# ---- Async / polling (merge-video) ----

class MergeVideoStartResponse(BaseModel):
    """Response when starting an async merge-video task."""
    task_id: str
    status: str
    scene_id: str
    user_id: str
    message: str
    created_at: str


class MergeVideoTaskStatusResponse(BaseModel):
    """Response for polling merge-video task status."""
    task_id: str
    status: str
    scene_id: Optional[str] = None
    user_id: Optional[str] = None
    message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None
    video_url: Optional[str] = None


@router.post("/merge-video/start", response_model=MergeVideoStartResponse)
def merge_video_start(request: MergeVideoRequest) -> MergeVideoStartResponse:
    """
    Start an async image-video merge task (Scene2). Returns immediately with task_id.
    Poll GET /image/merge-video/tasks/{task_id} for status and video_url when completed.
    """
    short_id = request.short_id or request.scene_id
    result = image_processing_service.start_image_merge_task(
        product_image_url=request.product_image_url,
        background_video_url=request.background_video_url,
        scene_id=request.scene_id,
        user_id=request.user_id,
        short_id=short_id,
        scale=request.scale,
        position=request.position,
        duration=request.duration,
        add_animation=request.add_animation,
        add_shadow=request.add_shadow,
        shadow_blur_radius=request.shadow_blur_radius,
        shadow_offset=tuple(request.shadow_offset) if request.shadow_offset else (15, 15),
    )
    return MergeVideoStartResponse(
        task_id=result["task_id"],
        status=result["status"],
        scene_id=result["scene_id"],
        user_id=result["user_id"],
        message=result["message"],
        created_at=result["created_at"],
    )


@router.get("/merge-video/tasks/{task_id}", response_model=MergeVideoTaskStatusResponse)
def merge_video_task_status(task_id: str) -> MergeVideoTaskStatusResponse:
    """
    Get status of an async merge-video task. When status is completed, video_url is set.
    """
    task = image_processing_service.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return MergeVideoTaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        scene_id=task.get("scene_id"),
        user_id=task.get("user_id"),
        message=task.get("message"),
        created_at=task.get("created_at"),
        updated_at=task.get("updated_at"),
        error_message=task.get("error_message"),
        video_url=task.get("video_url"),
    )


@router.post("/composite", response_model=ImageCompositeResponse)
def composite(request: CompositeRequest) -> ImageCompositeResponse:
    """
    Composite two images and save to public folder under composited_images/{user_id}/{short_id}/{file_name}.
    Returns relative URL: composited_images/{user_id}/{short_id}/{file_name}.
    """
    result = image_processing_service.composite_images_to_public_folder(
        background_url=request.background_url,
        overlay_url=request.overlay_url,
        user_id=request.user_id,
        scene_id=request.scene_id,
        position=(request.position_x or 0, request.position_y or 0),
        resize_overlay=request.resize_overlay if request.resize_overlay is not None else True,
    )
    if result["success"]:
        return ImageCompositeResponse(
            success=True,
            image_url=result["image_url"],
            error=None,
            message="Images composited and saved to public folder",
            created_at=datetime.now(timezone.utc),
        )
    return ImageCompositeResponse(
        success=False,
        image_url=None,
        error=result.get("error"),
        message=result.get("error", "Composite failed"),
        created_at=datetime.now(timezone.utc),
    )
