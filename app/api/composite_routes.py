"""Image composite endpoint: composite two images and save to Shopify app public folder."""

from fastapi import APIRouter
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
