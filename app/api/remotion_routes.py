"""
Remotion API Routes

Middleware endpoints that bridge Next.js and Remotion server.
These routes receive requests from Next.js, forward to Remotion server,
and return responses back to Next.js.

Before forwarding, the product title is optionally rewritten with OpenAI
to a shorter, casual version for display in Remotion scenes.
"""

import asyncio
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal

import openai

from app.middleware.remotion_proxy import remotion_proxy
from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/remotion", tags=["remotion"])


# Request/Response Models
class ProductInfo(BaseModel):
    """Product information for video generation - accepts flexible product data."""
    # Fields can be anything - we pass them as-is to Remotion server
    class Config:
        extra = "allow"  # Allow any additional fields
    
    # Optional common fields that Remotion might expect
    title: Optional[str] = None
    name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[float] = None
    reviewCount: Optional[int] = None
    currency: Optional[str] = "USD"
    description: Optional[str] = None


class VideoMetadata(BaseModel):
    """Metadata for video generation."""
    short_id: str
    scene_id: str
    sceneNumber: int


class StartVideoRequest(BaseModel):
    """Request to start video generation."""
    template: Literal["product-modern-v1", "product-minimal-v1"]
    imageUrl: str
    product: ProductInfo
    metadata: VideoMetadata


class StartVideoResponse(BaseModel):
    """Response from starting video generation."""
    taskId: str
    status: str


class TaskStatusResponse(BaseModel):
    """Response from checking task status."""
    status: Literal["pending", "processing", "completed", "failed"]
    stage: Optional[Literal["downloading", "rendering", "uploading", "done"]] = None
    progress: Optional[int] = Field(None, ge=0, le=100)
    videoUrl: Optional[str] = None
    error: Optional[str] = None


async def _rewrite_title_for_remotion(original_title: str) -> str:
    """
    Rewrite product title with OpenAI to a short, casual, friendly version for video display.
    Returns original title if API key is missing or rewrite fails (no change to behavior).
    """
    if not original_title or not original_title.strip():
        return original_title
    if not getattr(settings, "OPENAI_API_KEY", None):
        logger.debug("[Remotion] OPENAI_API_KEY not set, skipping title rewrite")
        return original_title

    def _call_openai() -> str:
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite product titles to be short, casual, and friendly for video overlays. "
                        "Output ONLY the rewritten title, no quotes or explanation. "
                        "Keep it under 8 words when possible. Examples: "
                        "'Ski Goggles PRO' instead of long technical names, "
                        "'Newton Ridge Hiking Shoe' instead of full product lines."
                    ),
                },
                {"role": "user", "content": original_title},
            ],
            max_tokens=60,
            temperature=0.5,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return original_title

    try:
        rewritten = await asyncio.to_thread(_call_openai)
        if rewritten and rewritten != original_title:
            logger.info(f"[Remotion] Title rewritten: '{original_title[:50]}...' -> '{rewritten}'")
        return rewritten or original_title
    except Exception as e:
        logger.warning(f"[Remotion] Title rewrite failed, using original: {e}")
        return original_title


# API Endpoints

@router.post("/videos", response_model=StartVideoResponse)
async def start_video_generation(request: StartVideoRequest):
    """
    Start video generation on Remotion server.
    
    This endpoint receives requests from Next.js and forwards them to Remotion server.
    
    **Next.js calls**: `/api/remotion/videos`
    **Forwards to**: `POST {REMOTION_SERVER}/videos`
    
    Example with full product info:
    ```
    POST /api/remotion/videos
    {
        "template": "product-modern-v1",
        "imageUrl": "https://...",
        "product": {
            "title": "Product Name",
            "price": "$99.99",
            "rating": 4.5,
            "reviewCount": 123,
            "currency": "USD"
        },
        "metadata": {
            "short_id": "abc123",
            "scene_id": "uuid",
            "sceneNumber": 1
        }
    }
    ```
    
    Example with minimal product info:
    ```
    POST /api/remotion/videos
    {
        "template": "product-modern-v1",
        "imageUrl": "https://...",
        "product": {
            "name": "Columbia Men's Newton Ridge Plus Ii Waterproof Hiking Shoe",
            "price": "USD 50.49",
            "description": "High quality hiking shoe"
        },
        "metadata": {
            "short_id": "abc123",
            "scene_id": "uuid",
            "sceneNumber": 1
        }
    }
    ```
    
    Returns:
    ```
    {
        "taskId": "task-uuid",
        "status": "pending"
    }
    ```
    """
    try:
        logger.info(
            f"[API] Received video generation request: "
            f"template={request.template}, scene={request.metadata.sceneNumber}"
        )
        logger.info(f"[API] Product data received: {request.product.dict()}")
        
        # Convert product dict and exclude None values to keep payload clean
        product_dict = request.product.dict(exclude_none=True)
        
        # Rewrite title to short, casual version for Remotion (only update title; rest unchanged)
        original_title = product_dict.get("title") or product_dict.get("name")
        if original_title and isinstance(original_title, str):
            updated_title = await _rewrite_title_for_remotion(original_title)
            product_dict["title"] = updated_title
            product_dict["name"] = updated_title  # Remotion may use either field
        
        # Forward request to Remotion server with updated product (same form, title updated)
        result = await remotion_proxy.start_video_generation(
            template=request.template,
            image_url=request.imageUrl,
            product=product_dict,
            metadata=request.metadata.dict()
        )
        
        logger.info(f"[API] Video generation started successfully: taskId={result.get('taskId')}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions from proxy
        raise
    except Exception as e:
        logger.error(f"[API] Failed to start video generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start video generation: {str(e)}"
        )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def check_task_status(
    task_id: str,
    shortId: Optional[str] = Query(None, description="Short ID for logging"),
    sceneNumber: Optional[int] = Query(None, description="Scene number for logging")
):
    """
    Check task status on Remotion server.
    
    This endpoint receives status check requests from Next.js and forwards them to Remotion server.
    
    **Next.js calls**: `/api/remotion/tasks/{taskId}?shortId=xxx&sceneNumber=1`
    **Forwards to**: `GET {REMOTION_SERVER}/tasks/{taskId}`
    
    Example:
    ```
    GET /api/remotion/tasks/task-uuid?shortId=abc123&sceneNumber=1
    ```
    
    Returns:
    ```
    {
        "status": "completed",
        "stage": "uploading",
        "progress": 100,
        "videoUrl": "https://..."
    }
    ```
    """
    try:
        logger.info(f"[API] Checking task status: taskId={task_id}")
        
        # Forward request to Remotion server
        result = await remotion_proxy.check_task_status(
            task_id=task_id,
            short_id=shortId,
            scene_number=sceneNumber
        )
        
        logger.info(f"[API] Task status retrieved: status={result.get('status')}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions from proxy
        raise
    except Exception as e:
        logger.error(f"[API] Failed to check task status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check task status: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Check health of Remotion proxy and server connection.
    
    Returns:
    ```
    {
        "proxy": "healthy",
        "remotion_server": "connected",
        "base_url": "http://localhost:5050"
    }
    ```
    """
    try:
        remotion_status = await remotion_proxy.health_check()
        
        return {
            "proxy": "healthy",
            **remotion_status
        }
    except Exception as e:
        logger.error(f"[API] Health check failed: {e}", exc_info=True)
        return {
            "proxy": "unhealthy",
            "error": str(e)
        }
