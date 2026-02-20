"""Image composite endpoint: composite two images and save to Shopify app public folder."""

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union

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
    user_id: int = Field(..., description="Shopify user ID (BigInt); may be sent as number or string")
    position_x: Optional[int] = Field(0, description="X position for overlay (0 = auto-center)")

    position_y: Optional[int] = Field(0, description="Y position for overlay (0 = auto-center)")
    resize_overlay: Optional[bool] = Field(True, description="Whether to resize overlay to fit background")

    @field_validator("user_id", mode="before")
    @classmethod
    def user_id_to_int(cls, v: Union[int, str]) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            return int(v)
        raise ValueError("user_id must be an int or string representation of an integer")


@router.post("/composite", response_model=ImageCompositeResponse)
def composite(request: CompositeRequest) -> ImageCompositeResponse:
    """
    Composite two images and save to public folder under composited_images/{user_id}/{scene_id}/{file_name}.
    Returns relative URL: composited_images/{user_id}/{scene_id}/{file_name}.
    """
    result = image_processing_service.composite_images_to_public_folder(
        background_url=request.background_url,
        overlay_url=request.overlay_url,
        user_id=str(request.user_id),
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
