"""Image composite endpoint: composite two images and save to Shopify app public folder."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from app.models import ImageCompositeResponse
from app.services.image_processing_service import image_processing_service
from app.logging_config import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)

router = APIRouter(prefix="/image", tags=["image"])


class CompositeToPublicRequest(BaseModel):
    background_url: str = Field(..., description="URL of the background image")
    overlay_url: str = Field(..., description="URL of the overlay image (e.g. product with transparent background)")
    position_x: Optional[int] = Field(0, description="X position for overlay (0 = auto-center)")
    position_y: Optional[int] = Field(0, description="Y position for overlay (0 = auto-center)")
    resize_overlay: Optional[bool] = Field(True, description="Whether to resize overlay to fit background")


@router.post("/composite-to-public", response_model=ImageCompositeResponse)
def composite_to_public(request: CompositeToPublicRequest) -> ImageCompositeResponse:
    """
    Composite two images and save the result to the Shopify app public folder.
    Returns URL path for the frontend: /composited_images/{image_name}.
    """
    result = image_processing_service.composite_images_to_public_folder(
        background_url=request.background_url,
        overlay_url=request.overlay_url,
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
