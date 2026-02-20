from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, List
from datetime import datetime, timezone
import os
import httpx
import json

from app.models import (
    ScrapeRequest, TaskStatusResponse, HealthResponse,
    TaskStatus, TaskPriority, VideoGenerationRequest, VideoGenerationResponse,
    FinalizeShortRequest, FinalizeShortResponse, ImageAnalysisRequest, ImageAnalysisResponse,
    ScenarioGenerationRequest, ScenarioGenerationResponse, SaveScenarioRequest, SaveScenarioResponse,
    TestAudioRequest, TestAudioResponse,
    AudioScriptGenerationRequest, AudioScriptGenerationResponse,
    AudioGenerationRequest, AudioGenerationResponse,
    ImageCompositeRequest, ImageCompositeResponse, GenerateScene1RequestFromNextJS, 
    GenerateScene1Request, GenerateScene1Response, Scene1Metadata, Scene1ProductInfo,
    MergeImageWithVideoRequest, MergeImageWithVideoResponse, ShadowGenerationRequest, ShadowGenerationResponse,
    BackgroundGenerationRequest, BackgroundGenerationResponse,
    ExtractBackgroundPromptRequest, ExtractBackgroundPromptResponse
)
from app.services.scraping_service import scraping_service
from app.services.video_generation_service import video_generation_service
from app.services.merging_service import merging_service
from app.services.image_analysis_service import image_analysis_service
from app.services.scenario_generation_service import scenario_generation_service
from app.services.save_scenario_service import save_scenario_service
from app.services.test_audio_service import test_audio_service
from app.services.audio_generation_service import audio_generation_service
from app.services.image_processing_service import image_processing_service
from app.services.scheduler_service import get_scheduler_status, run_cleanup_now
from app.services.session_service import session_service
from app.services.shadow_generation_service import shadow_generation_service
from app.services.background_generation_service import background_generation_service
from app.config import settings
from app.security import (
    get_api_key, validate_request_security, validate_scrape_request,
    get_security_stats, security_manager, API_KEYS
)
from app.logging_config import get_logger
from app.utils.credit_utils import can_perform_action
from app.api.remotion_routes import _rewrite_title_for_remotion

logger = get_logger(__name__)

router = APIRouter()

@router.post("/scrape", response_model=TaskStatusResponse)
def scrape_product(
    request: ScrapeRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> TaskStatusResponse:
    """
    Scrape product information from a URL
    
    This endpoint accepts a URL and starts scraping asynchronously using threads.
    Returns immediately with a task ID for polling.
    
    Authentication: Optional API key via Bearer token
    Rate Limits: 
    - With API key: Based on key configuration
    - Without API key: 10 requests per minute
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        # validate_scrape_request(str(request.url), api_key)
        
        # Check user's credit before starting scraping
        credit_check = can_perform_action(request.user_id, "scraping")
        if credit_check.get("error"):
            logger.error(f"Credit check failed for user {request.user_id}: {credit_check['error']}")
            raise HTTPException(status_code=400, detail=f"Credit check failed: {credit_check['error']}")
        
        if not credit_check.get("can_perform", False):
            reason = credit_check.get("reason", "Insufficient credits")
            current_credits = credit_check.get("current_credits", 0)
            required_credits = credit_check.get("required_credits", 1)
            logger.warning(f"Credit check failed for user {request.user_id}: {reason}. Current: {current_credits}, Required: {required_credits}")
            raise HTTPException(
                status_code=402, 
                detail={
                    "error": "Insufficient credits",
                    "reason": reason,
                    "current_credits": current_credits,
                    "required_credits": required_credits,
                    "message": f"You need {required_credits} credit(s) to perform this action. You currently have {current_credits} credit(s)."
                }
            )
        
        logger.info(f"Credit check passed for user {request.user_id}. Can perform scraping action.")
        
        # Start scraping using threads (no need for background tasks with threading)
        response = scraping_service.start_scraping_task(
            url=str(request.url),
            user_id=request.user_id,
            proxy=request.proxy,
            user_agent=request.user_agent,
            target_language=request.target_language
        )
        
        # Convert response to TaskStatusResponse format
        detail = {}
        # Note: target_language is not included in detail for scraping tasks
        if hasattr(response, 'supabase_product_id') and response.supabase_product_id:
            detail['supabase_product_id'] = response.supabase_product_id
        if hasattr(response, 'short_id') and response.short_id:
            detail['short_id'] = response.short_id
        
        task_response = TaskStatusResponse(
            task_id=response.task_id,
            status=response.status,
            url=response.url,
            task_type="scraping",
            progress=getattr(response, 'progress', None),
            message=getattr(response, 'message', None),
            created_at=response.created_at,
            updated_at=getattr(response, 'completed_at', response.created_at),
            priority=getattr(response, 'priority', TaskPriority.NORMAL),
            user_id=getattr(response, 'user_id', None),
            session_id=getattr(response, 'session_id', None),
            detail=detail
        )
        
        logger.info(f"Started scraping task {task_response.task_id} for {request.url} by user {request.user_id}")
        
        return task_response
        
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get the status of a scraping task. Returns simplified response with essential fields only.
    """
    task_info = scraping_service.get_task_status(task_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Build the response using the task info dictionary
    detail = {}
    
    # Add product_id to detail if it exists
    if task_info.get('product_id'):
        detail['product_id'] = task_info['product_id']
    
    # Add short_id to detail if it exists
    if task_info.get('short_id'):
        detail['short_id'] = task_info['short_id']
    
    # Add platform information to detail if available
    if task_info.get('platform'):
        detail['platform'] = task_info['platform']
    if task_info.get('platform_confidence'):
        detail['platform_confidence'] = task_info['platform_confidence']
    if task_info.get('platform_indicators'):
        detail['platform_indicators'] = task_info['platform_indicators']
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info.get('status', 'unknown'),
        url=task_info.get('url') or "",  # Ensure url is never None
        task_type='scraping',
        progress=task_info.get('progress'),
        message=task_info.get('message'),
        created_at=task_info.get('created_at'),
        updated_at=task_info.get('updated_at'),
        priority=TaskPriority.NORMAL,
        user_id=task_info.get('user_id'),
        session_id=None,
        detail=detail
    )

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint
    """
    try:
        # Get security stats
        security_stats = get_security_stats()
        
        # Get task stats
        all_tasks = scraping_service.get_all_tasks()
        task_stats = {
            'total_tasks': len(all_tasks),
            'pending_tasks': len([t for t in all_tasks.values() if t['status'] == TaskStatus.PENDING]),
            'running_tasks': len([t for t in all_tasks.values() if t['status'] == TaskStatus.RUNNING]),
            'completed_tasks': len([t for t in all_tasks.values() if t['status'] == TaskStatus.COMPLETED]),
            'failed_tasks': len([t for t in all_tasks.values() if t['status'] == TaskStatus.FAILED])
        }
        
        # Get cache stats
        
        health_response = HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version=settings.VERSION,
            uptime=0,  # TODO: Implement uptime tracking
            memory_usage=0,  # TODO: Implement memory usage tracking
            security_stats=security_stats,
            task_stats=task_stats,
        )
        
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/stats")
def get_stats():
    """
    Get detailed statistics about the scraper
    """
    try:
        # Get all available stats
        all_tasks = scraping_service.get_all_tasks()
        security_stats = get_security_stats()
        
        stats = {
            'tasks': {
                'total': len(all_tasks),
                'by_status': {}
            },
            'security': security_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Count tasks by status
        for task in all_tasks.values():
            status = task['status'].value if hasattr(task['status'], 'value') else str(task['status'])
            if status not in stats['tasks']['by_status']:
                stats['tasks']['by_status'][status] = 0
            stats['tasks']['by_status'][status] += 1
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.delete("/tasks/{task_id}")
def cancel_task(task_id: str):
    """
    Cancel a running task
    """
    try:
        task_info = scraping_service.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task_info['status'] not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed or failed task")
        
        # Update task status to cancelled
        scraping_service._update_task_status(task_id, TaskStatus.FAILED, "Task cancelled by user")
        
        return {"message": f"Task {task_id} cancelled successfully"}
        
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.delete("/tasks")
def cleanup_tasks():
    """
    Clean up old completed/failed tasks
    """
    try:
        scraping_service.cleanup_completed_tasks()
        return {"message": "Task cleanup completed"}
        
    except Exception as e:
        logger.error(f"Error cleaning up tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cleanup tasks: {str(e)}")





# ============================================================================
# Video Generation Endpoints
# ============================================================================

@router.post("/video/generate", response_model=VideoGenerationResponse)
def generate_video_from_scene(
    request: VideoGenerationRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> VideoGenerationResponse:
    """
    Generate video from a scene using AI generation.
    
    This endpoint accepts a scene_id and starts video generation asynchronously using threads.
    Returns immediately with a task ID for polling.
    
    The process includes:
    1. Image generation (if no image exists) using Vertex AI or Flux API
    2. Video generation from the image using Vertex AI
    3. Storage of both image and video in Supabase
    4. Credit deduction for each generation step
    
    Authentication: Optional API key via Bearer token
    Rate Limits: Based on API key configuration
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        
        logger.info(f"Starting video generation for scene {request.scene_id} by user {request.user_id}")
        
        # Start video generation using threads
        response = video_generation_service.start_video_generation_task(
            scene_id=request.scene_id,
            user_id=request.user_id,
            force_regenerate_first_frame=request.force_regenerate_first_frame
        )
        
        # Convert response to VideoGenerationResponse format
        video_generation_response = VideoGenerationResponse(
            task_id=response['task_id'],
            status=response['status'],
            scene_id=response['scene_id'],
            user_id=response['user_id'],
            message=response['message'],
            created_at=response['created_at'],
            progress=response.get('progress'),
            current_step=response.get('current_step'),
            error_message=response.get('error_message'),
            video_url=response.get('video_url'),
            image_url=response.get('image_url')
        )
        
        logger.info(f"Started video generation task {video_generation_response.task_id} for scene {request.scene_id}")
        
        return video_generation_response
        
    except Exception as e:
        logger.error(f"Error in video generation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")


@router.get("/video/generate/tasks/{task_id}", response_model=VideoGenerationResponse)
def get_video_generation_task_status(task_id: str) -> VideoGenerationResponse:
    """
    Get the status of a video generation task.
    Returns VideoGenerationResponse with current status and progress.
    """
    try:
        task_info = video_generation_service.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Video generation task not found")
        
        # Convert task info to VideoGenerationResponse
        return VideoGenerationResponse(
            task_id=task_id,
            status=task_info['status'],
            scene_id=task_info['scene_id'],
            user_id=task_info['user_id'],
            message=task_info.get('message', ''),
            created_at=task_info['created_at'],
            progress=task_info.get('progress'),
            current_step=task_info.get('current_step'),
            error_message=task_info.get('error_message'),
            video_url=task_info.get('video_url'),
            image_url=task_info.get('image_url'),
            completed_at=task_info.get('updated_at') if task_info.get('status') == TaskStatus.COMPLETED else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video generation task status {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get video generation task status: {str(e)}")



@router.delete("/video/generate/tasks/{task_id}")
def cancel_video_generation_task(task_id: str):
    """
    Cancel a running video generation task
    """
    try:
        task_info = video_generation_service.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Video generation task not found")
        
        if task_info['status'] not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed or failed video generation task")
        
        # Update task status to cancelled
        video_generation_service._update_task(task_id, status=TaskStatus.FAILED, message="Video generation task cancelled by user")
        
        return {"message": f"Video generation task {task_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling video generation task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel video generation task: {str(e)}")


@router.delete("/video/generate/tasks")
def cleanup_video_generation_tasks():
    """
    Clean up old completed/failed video generation tasks
    """
    try:
        video_generation_service.cleanup()
        return {"message": "Video generation task cleanup completed"}
        
    except Exception as e:
        logger.error(f"Error cleaning up video generation tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cleanup video generation tasks: {str(e)}")


# ============================================================================
# Short Finalization Endpoints
# ============================================================================

@router.post("/shorts/finalize", response_model=FinalizeShortResponse)
def finalize_short(
    request: FinalizeShortRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> FinalizeShortResponse:
    """
    Finalize a short video by merging scenes, generating thumbnail, and optionally upscaling.
    
    This endpoint starts the finalization process asynchronously using threads.
    Returns immediately with a task ID for polling.
    
    Authentication: Optional API key via Bearer token
    Rate Limits: 
    - With API key: Based on key configuration
    - Without API key: 10 requests per minute
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        # validate_scrape_request(str(request.url), api_key)
        
        logger.info(f"Starting short finalization for user {request.user_id}, short {request.short_id}")
        
        # Start finalization using merging service
        response = merging_service.start_finalize_short_task(
            user_id=request.user_id,
            short_id=request.short_id
        )
        
        # Convert response to FinalizeShortResponse format
        finalize_response = FinalizeShortResponse(
            task_id=response['task_id'],
            status=TaskStatus.PENDING,
            short_id=request.short_id,
            user_id=request.user_id,
            message=response['message'],
            created_at=datetime.fromisoformat(response['created_at']),
            progress=0.0,
            current_step="Initializing"
        )
        
        logger.info(f"Started short finalization task {finalize_response.task_id} for short {request.short_id}")
        
        return finalize_response
        
    except Exception as e:
        logger.error(f"Error in short finalization endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Short finalization failed: {str(e)}")


@router.get("/shorts/finalize/tasks/{task_id}", response_model=FinalizeShortResponse)
def get_short_finalization_task_status(task_id: str) -> FinalizeShortResponse:
    """
    Get the status of a short finalization task.
    Returns FinalizeShortResponse with current status and progress.
    """
    try:
        task_info = merging_service.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Short finalization task not found")
        
        # Convert task info to FinalizeShortResponse
        # Handle both Task object and dict formats
        if hasattr(task_info, 'task_status'):
            # Task object
            status = task_info.task_status
            created_at = task_info.created_at
            progress = task_info.progress
            current_step = task_info.current_step_name
            error_message = task_info.error_message
            short_id = task_info.task_metadata.get('short_id', '') if task_info.task_metadata else ''
            user_id = task_info.user_id or ''
            message = task_info.task_status_message or ''
            thumbnail_url = task_info.task_metadata.get('thumbnail_url', '') if task_info.task_metadata else ''
            video_url = task_info.task_metadata.get('video_url', '') if task_info.task_metadata else ''
            final_video_url = task_info.task_metadata.get('final_video_url', '') if task_info.task_metadata else ''
            updated_at = task_info.updated_at
        else:
            # Dict format
            status = task_info['status']
            created_at = task_info['created_at']
            progress = task_info.get('progress')
            current_step = task_info.get('current_step')
            error_message = task_info.get('error_message')
            short_id = task_info.get('short_id', '')
            user_id = task_info.get('user_id', '')
            message = task_info.get('message', '')
            thumbnail_url = task_info.get('thumbnail_url', '')
            video_url = task_info.get('video_url', '')
            final_video_url = task_info.get('final_video_url', '')
            updated_at = task_info.get('updated_at')
        
        return FinalizeShortResponse(
            task_id=task_id,
            status=status,
            short_id=short_id,
            user_id=user_id,
            message=message,
            created_at=created_at,
            progress=progress,
            current_step=current_step,
            error_message=error_message,
            thumbnail_url=thumbnail_url,
            final_video_url=final_video_url,
            completed_at=updated_at if status == TaskStatus.COMPLETED else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting short finalization task status {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get short finalization task status: {str(e)}")


@router.delete("/shorts/finalize/tasks/{task_id}")
def cancel_short_finalization_task(task_id: str):
    """
    Cancel a running short finalization task
    """
    try:
        task_info = merging_service.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Short finalization task not found")
        
        # Handle both Task object and dict formats
        if hasattr(task_info, 'task_status'):
            status = task_info.task_status
        else:
            status = task_info['status']
            
        if status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed or failed short finalization task")
        
        # Update task status to failed (cancelled)
        # Note: We'll need to add a cancel method to the merging service
        # For now, we'll mark it as failed
        logger.warning(f"Short finalization task {task_id} cancelled by user")
        
        return {"message": f"Short finalization task {task_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling short finalization task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel short finalization task: {str(e)}")


@router.delete("/shorts/finalize/tasks")
def cleanup_short_finalization_tasks():
    """
    Clean up old completed/failed short finalization tasks
    """
    try:
        merging_service.cleanup()
        return {"message": "Short finalization task cleanup completed"}
        
    except Exception as e:
        logger.error(f"Error cleaning up short finalization tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cleanup short finalization tasks: {str(e)}")


# ============================================================================
# Image Analysis Endpoints
# ============================================================================

@router.post("/image/analyze", response_model=ImageAnalysisResponse)
def analyze_image(
    request: ImageAnalysisRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> ImageAnalysisResponse:
    """
    Analyze images for a product based on product_id.
    
    This endpoint accepts a product_id and starts image analysis asynchronously.
    Returns immediately with a task ID for polling.
    
    The analysis includes:
    1. Finding the product by product_id
    2. Identifying unanalyzed images
    3. Analyzing up to 4 images simultaneously using OpenAI Vision API
    4. Storing results in the product's images field
    
    Authentication: Optional API key via Bearer token
    Rate Limits: Based on API key configuration
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        
        logger.info(f"Starting image analysis for product_id {request.product_id} by user {request.user_id}")
        
        # Check if all images are already analyzed before creating a task
        product_data = image_analysis_service._get_product_by_id(request.product_id)
        if not product_data:
            raise HTTPException(status_code=404, detail=f"Product not found for product_id: {request.product_id}")
        
        images = product_data.get('images', {})
        if not images:
            raise HTTPException(status_code=400, detail="No images found for this product")
        
        # Check if all images already have analysis data
        unanalyzed_images = image_analysis_service._get_unanalyzed_images(images)
        if not unanalyzed_images:
            # All images already analyzed - return immediate response
            return ImageAnalysisResponse(
                task_id="",  # No task created
                status=TaskStatus.COMPLETED,
                product_id=request.product_id,
                user_id=request.user_id,
                message="All images already analyzed",
                created_at=datetime.now(),
                progress=100.0,
                current_step="Already completed",
                error_message=None,
                total_images=len(images),
                analyzed_images=len(images),
                failed_images=0,
                analyzedData=None,
                completed_at=datetime.now()
            )
        
        # Start image analysis using the service
        response = image_analysis_service.start_image_analysis_task(
            product_id=request.product_id,
            user_id=request.user_id
        )
        
        # Convert response to ImageAnalysisResponse format
        image_analysis_response = ImageAnalysisResponse(
            task_id=response['task_id'],
            status=TaskStatus.PENDING,
            product_id=request.product_id,
            user_id=request.user_id,
            message=response['message'],
            created_at=datetime.now(),
            progress=0.0,
            current_step="Task started",
            error_message=None,
            total_images=len(unanalyzed_images),
            analyzed_images=0,
            failed_images=0,
            analyzedData=None,
            completed_at=None
        )
        
        logger.info(f"Started image analysis task {image_analysis_response.task_id} for product_id {request.product_id}")
        
        return image_analysis_response
        
    except Exception as e:
        logger.error(f"Error in image analysis endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@router.post("/image/add-shadow", response_model=ShadowGenerationResponse)
def add_shadow_to_image(
    request: ShadowGenerationRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> ShadowGenerationResponse:
    """
    Add a realistic shadow effect to a product image (Async/Polling Pattern).
    
    Returns immediately with a task_id for polling.
    Poll GET `/image/add-shadow/tasks/{task_id}` to check status.
    
    This endpoint accepts a product image URL and description, then:
    1. Extracts a shadow generation prompt from the product description using OpenAI
    2. Generates a new image with shadow effect using DALL-E
    3. Uploads the result to storage
    
    The shadow effect enhances depth and makes the product stand out with natural lighting.
    
    Authentication: Optional API key via Bearer token
    Rate Limits: Based on API key configuration
    """
    logger.info("\n" + "=" * 80)
    logger.info("API ENDPOINT: /image/add-shadow - REQUEST RECEIVED (ASYNC)")
    logger.info("=" * 80)
    
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        
        logger.info("Request Details:")
        logger.info(f"  → User ID: {request.user_id}")
        logger.info(f"  → Image URL: {request.image_url[:80]}...")
        logger.info(f"  → Product Description: {request.product_description[:100]}...")
        logger.info(f"  → Scene ID: {request.scene_id or 'N/A'}")
        logger.info(f"  → API Key: {'Provided' if api_key else 'Not provided'}")
        logger.info("-" * 80)
        
        logger.info("Starting async shadow generation task...")
        
        # Start async shadow generation task
        result = shadow_generation_service.start_shadow_generation_task(request)
        
        logger.info("-" * 80)
        logger.info("Task Created:")
        logger.info(f"  → Task ID: {result['task_id']}")
        logger.info(f"  → Status: {result['status']}")
        logger.info(f"  → Message: {result['message']}")
        logger.info("=" * 80)
        logger.info("API ENDPOINT: /image/add-shadow - TASK STARTED SUCCESSFULLY")
        logger.info("=" * 80 + "\n")
        
        return ShadowGenerationResponse(
            success=True,
            task_id=result['task_id'],
            status=result['status'],
            message=result['message'],
            progress=0,
            current_step='Task started',
            created_at=datetime.fromisoformat(result['created_at'])
        )
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("API ENDPOINT: /image/add-shadow - UNEXPECTED ERROR")
        logger.error("=" * 80)
        logger.error(f"Exception Type: {type(e).__name__}")
        logger.error(f"Exception Message: {str(e)}")
        logger.error("=" * 80, exc_info=True)
        logger.error("=" * 80 + "\n")
        raise HTTPException(status_code=500, detail=f"Failed to start shadow generation: {str(e)}")


@router.get("/image/add-shadow/tasks/{task_id}", response_model=ShadowGenerationResponse)
def get_shadow_task_status(
    task_id: str,
    api_key: Optional[str] = Depends(get_api_key)
) -> ShadowGenerationResponse:
    """
    Poll the status of a shadow generation task.
    
    Returns the current status of the task:
    - pending: Task is queued
    - processing: Task is being processed
    - completed: Task finished successfully (image_url available)
    - failed: Task failed (error message available)
    
    Authentication: Optional API key via Bearer token
    """
    try:
        task_info = shadow_generation_service.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Shadow generation task not found")
        
        # Check if completed and has image URL
        if task_info['status'] == TaskStatus.COMPLETED and task_info.get('result_image_url'):
            return ShadowGenerationResponse(
                success=True,
                task_id=task_id,
                status=task_info['status'],
                image_url=task_info['result_image_url'],
                message='Shadow generation completed successfully',
                progress=task_info.get('progress', 100),
                current_step=task_info.get('current_step', 'Completed'),
                created_at=datetime.fromisoformat(task_info['created_at'])
            )
        elif task_info['status'] == TaskStatus.FAILED:
            return ShadowGenerationResponse(
                success=False,
                task_id=task_id,
                status=task_info['status'],
                image_url=None,
                message='Shadow generation failed',
                error=task_info.get('error_message'),
                progress=task_info.get('progress', 0),
                current_step=task_info.get('current_step', 'Failed'),
                created_at=datetime.fromisoformat(task_info['created_at'])
            )
        else:
            # Still pending or processing
            return ShadowGenerationResponse(
                success=True,
                task_id=task_id,
                status=task_info['status'],
                image_url=None,
                message=f"Task is {task_info['status']}",
                progress=task_info.get('progress', 0),
                current_step=task_info.get('current_step', 'Processing'),
                created_at=datetime.fromisoformat(task_info['created_at'])
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting shadow task status {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.post("/image/generate-background", response_model=BackgroundGenerationResponse)
def generate_background_image(
    request: BackgroundGenerationRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> BackgroundGenerationResponse:
    """
    Generate a product background image using AI (Async Pattern)
    
    Returns immediately with a task_id for polling.
    Poll GET `/image/generate-background/tasks/{task_id}` to check status.
    
    This endpoint accepts a product description and environment variables, then:
    1. Extracts a background generation prompt from the product description and environment using OpenAI
    2. Generates a background image using Vertex AI Imagen 4.0
    3. Uploads the result to storage
    
    Environment variables:
    - mood: Mood/feeling for the background (e.g., "Energetic", "Calm", "Professional")
    - style: Visual style for the background (e.g., "Film Grain", "Minimalist", "Vibrant")
    - environment: Environment setting (e.g., "Indoor Studio", "Outdoor Nature", "Urban") - optional
    
    Authentication: Optional API key via Bearer token
    Rate Limits: Based on API key configuration
    """
    logger.info("\n" + "=" * 80)
    logger.info("API ENDPOINT: /image/generate-background - REQUEST RECEIVED (ASYNC)")
    logger.info("=" * 80)
    
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        
        logger.info("Request Details:")
        logger.info(f"  → User ID: {request.user_id}")
        logger.info(f"  → Product Description: {request.product_description[:100]}...")
        logger.info(f"  → Mood: {request.mood or 'N/A'}")
        logger.info(f"  → Style: {request.style or 'N/A'}")
        logger.info(f"  → Environment: {request.environment or 'N/A'}")
        logger.info(f"  → Manual Prompt: {request.manual_prompt[:100] + '...' if request.manual_prompt and len(request.manual_prompt) > 100 else request.manual_prompt or 'N/A'}")
        logger.info(f"  → Scene ID: {request.scene_id or 'N/A'}")
        logger.info(f"  → API Key: {'Provided' if api_key else 'Not provided'}")
        logger.info("-" * 80)
        
        logger.info("Starting async background generation task...")
        
        # Start async background generation task
        result = background_generation_service.start_background_generation_task(
            user_id=request.user_id,
            product_description=request.product_description,
            mood=request.mood,
            style=request.style,
            environment=request.environment,
            manual_prompt=request.manual_prompt,
            scene_id=request.scene_id,
            short_id=request.short_id
        )
        
        logger.info("-" * 80)
        logger.info("Task Created:")
        logger.info(f"  → Task ID: {result['task_id']}")
        logger.info(f"  → Status: {result['status']}")
        logger.info(f"  → Message: {result['message']}")
        logger.info("=" * 80)
        logger.info("API ENDPOINT: /image/generate-background - TASK STARTED SUCCESSFULLY")
        logger.info("=" * 80 + "\n")
        
        return BackgroundGenerationResponse(
            success=True,
            task_id=result['task_id'],
            status=result['status'],
            message=result['message'],
            progress=0,
            current_step='Task started',
            created_at=datetime.fromisoformat(result['created_at'])
        )
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("API ENDPOINT: /image/generate-background - UNEXPECTED ERROR")
        logger.error("=" * 80)
        logger.error(f"Exception Type: {type(e).__name__}")
        logger.error(f"Exception Message: {str(e)}")
        logger.error("=" * 80, exc_info=True)
        logger.error("=" * 80 + "\n")
        raise HTTPException(status_code=500, detail=f"Failed to start background generation: {str(e)}")


@router.get("/image/generate-background/tasks/{task_id}", response_model=BackgroundGenerationResponse)
def get_background_task_status(
    task_id: str,
    api_key: Optional[str] = Depends(get_api_key)
) -> BackgroundGenerationResponse:
    """
    Poll the status of a background generation task.
    
    Returns the current status of the task:
    - pending: Task is queued
    - processing: Task is being processed
    - completed: Task finished successfully (image_url available)
    - failed: Task failed (error message available)
    
    Authentication: Optional API key via Bearer token
    """
    try:
        task_info = background_generation_service.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Background generation task not found")
        
        # Check if completed and has image URL
        if task_info['status'] == TaskStatus.COMPLETED and task_info.get('result_image_url'):
            return BackgroundGenerationResponse(
                success=True,
                task_id=task_id,
                status=task_info['status'],
                image_url=task_info['result_image_url'],
                message='Background generation completed successfully',
                progress=task_info.get('progress', 100),
                current_step=task_info.get('current_step', 'Completed'),
                created_at=datetime.fromisoformat(task_info['created_at'])
            )
        elif task_info['status'] == TaskStatus.FAILED:
            return BackgroundGenerationResponse(
                success=False,
                task_id=task_id,
                status=task_info['status'],
                image_url=None,
                message='Background generation failed',
                error=task_info.get('error_message'),
                progress=task_info.get('progress', 0),
                current_step=task_info.get('current_step', 'Failed'),
                created_at=datetime.fromisoformat(task_info['created_at'])
            )
        else:
            # Still pending or processing
            return BackgroundGenerationResponse(
                success=True,
                task_id=task_id,
                status=task_info['status'],
                image_url=None,
                message=f"Task is {task_info['status']}",
                progress=task_info.get('progress', 0),
                current_step=task_info.get('current_step', 'Processing'),
                created_at=datetime.fromisoformat(task_info['created_at'])
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting background task status {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.post("/image/extract-background-prompt", response_model=ExtractBackgroundPromptResponse)
def extract_background_prompt(
    request: ExtractBackgroundPromptRequest,
    api_key: Optional[str] = Depends(get_api_key)
) -> ExtractBackgroundPromptResponse:
    """
    Extract the background-generation prompt from a product description using OpenAI.
    Returns the prompt string that would be sent to Vertex for background generation.
    Use when the next server sends only product description and you need the prompt.
    """
    result = background_generation_service.extract_background_prompt(
        product_description=request.product_description,
        mood=request.mood,
        style=request.style,
        environment=request.environment,
    )
    return ExtractBackgroundPromptResponse(
        success=result["error"] is None,
        prompt=result["prompt"],
        error=result["error"],
    )


@router.get("/image/analyze/tasks/{task_id}", response_model=ImageAnalysisResponse)
def get_image_analysis_task_status(task_id: str) -> ImageAnalysisResponse:
    """
    Get the status of an image analysis task.
    Returns ImageAnalysisResponse with current status and progress.
    """
    try:
        # Get task status from task management system
        from app.utils.task_management import get_task_status
        task_info = get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Image analysis task not found")
        
        # Convert task info to ImageAnalysisResponse
        return ImageAnalysisResponse(
            task_id=task_id,
            status=task_info.task_status,
            product_id=task_info.task_metadata.get('product_id', ''),
            user_id=task_info.user_id or '',
            message=task_info.task_status_message,
            created_at=task_info.created_at,
            progress=task_info.progress,
            current_step=task_info.current_step_name,
            error_message=task_info.error_message,
            total_images=task_info.task_metadata.get('total_images'),
            analyzed_images=task_info.task_metadata.get('analyzed_images'),
            failed_images=task_info.task_metadata.get('failed_images'),
            analyzedData=None,  # Don't include analyzed data in task response
            completed_at=task_info.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image analysis task status {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get image analysis task status: {str(e)}")


@router.delete("/image/analyze/tasks/{task_id}")
def cancel_image_analysis_task(task_id: str):
    """
    Cancel a running image analysis task
    """
    try:
        # Get task status from task management system
        from app.utils.task_management import get_task_status, fail_task
        task_info = get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Image analysis task not found")
        
        if task_info.task_status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed or failed image analysis task")
        
        # Update task status to failed with cancellation message
        fail_task(task_id, "Image analysis task cancelled by user")
        
        return {"message": f"Image analysis task {task_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling image analysis task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel image analysis task: {str(e)}")


@router.delete("/image/analyze/tasks")
def cleanup_image_analysis_tasks():
    """
    Clean up old completed/failed image analysis tasks
    """
    try:
        # Use task management system cleanup
        from app.utils.task_management import task_manager
        deleted_count = task_manager.cleanup_old_tasks(days_old=30)
        return {"message": "Image analysis task cleanup completed", "deleted_tasks_count": deleted_count}
        
    except Exception as e:
        logger.error(f"Error cleaning up image analysis tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cleanup image analysis tasks: {str(e)}")


# ============================================================================
# Image Compositing Endpoints
# ============================================================================

@router.post("/image/composite", response_model=ImageCompositeResponse)
def composite_images(
    request: ImageCompositeRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> ImageCompositeResponse:
    """
    Composite two images together (overlay on top of background).
    
    This endpoint accepts URLs for a background image and an overlay image (typically 
    a product with transparent background), composites them together, uploads the result 
    to Supabase storage, and updates the scene's image_url in the database.
    
    The overlay can be positioned and resized to fit the background. By default, it's 
    centered and resized to 80% of the background dimensions while maintaining aspect ratio.
    
    Authentication: Optional API key via Bearer token
    Rate Limits: Based on API key configuration
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        
        logger.info(f"Starting image compositing for scene {request.scene_id} by user {request.user_id}")
        logger.info(f"Background URL: {request.background_url[:80]}...")
        logger.info(f"Overlay URL: {request.overlay_url[:80]}...")
        
        # Composite the images using the service
        result = image_processing_service.composite_images(
            background_url=request.background_url,
            overlay_url=request.overlay_url,
            scene_id=request.scene_id,
            user_id=request.user_id,
            position=(request.position_x, request.position_y),
            resize_overlay=request.resize_overlay
        )
        
        # Build response
        if result['success']:
            logger.info(f"Image compositing completed successfully for scene {request.scene_id}")
            return ImageCompositeResponse(
                success=True,
                image_url=result['image_url'],
                error=None,
                message="Images composited successfully"
            )
        else:
            logger.error(f"Image compositing failed for scene {request.scene_id}: {result['error']}")
            return ImageCompositeResponse(
                success=False,
                image_url=None,
                error=result['error'],
                message="Image compositing failed"
            )
        
    except Exception as e:
        logger.error(f"Error in image composite endpoint: {e}", exc_info=True)
        # Return error response instead of raising exception for better client handling
        return ImageCompositeResponse(
            success=False,
            image_url=None,
            error=str(e),
            message="Image compositing failed with exception"
        )


# Scenario Generation Endpoints
@router.post("/scenario/generate", response_model=ScenarioGenerationResponse)
def generate_scenario(
    request: ScenarioGenerationRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> ScenarioGenerationResponse:
    """
    Generate AI-powered video scenario for a product
    
    This endpoint accepts product information and generates a complete video scenario
    including scenes, audio script, and preview image using OpenAI and Vertex AI.
    Returns immediately with a task ID for polling.
    
    Authentication: Optional API key via Bearer token
    Rate Limits: 
    - With API key: Based on key configuration
    - Without API key: 10 requests per minute
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        # validate_scrape_request(str(request.product_id), api_key)
        
        # Check user's credit before starting scenario generation
        credit_check = can_perform_action(request.user_id, "generate_scenario")
        if credit_check.get("error"):
            logger.error(f"Credit check failed for user {request.user_id}: {credit_check['error']}")
            raise HTTPException(status_code=400, detail=f"Credit check failed: {credit_check['error']}")
        
        if not credit_check.get("can_perform", False):
            reason = credit_check.get("reason", "Insufficient credits")
            current_credits = credit_check.get("current_credits", 0)
            required_credits = credit_check.get("required_credits", 1)
            logger.warning(f"Credit check failed for user {request.user_id}: {reason}. Current: {current_credits}, Required: {required_credits}")
            raise HTTPException(
                status_code=402, 
                detail={
                    "error": "Insufficient credits",
                    "reason": reason,
                    "current_credits": current_credits,
                    "required_credits": required_credits,
                    "message": f"You need {required_credits} credit(s) to perform this action. You currently have {current_credits} credit(s)."
                }
            )
        
        logger.info(f"Credit check passed for user {request.user_id}. Can perform scenario generation action.")
        
        # Start scenario generation using threads
        response = scenario_generation_service.start_scenario_generation_task(request)
        
        # Convert response to ScenarioGenerationResponse format
        return ScenarioGenerationResponse(
            task_id=response["task_id"],
            status=TaskStatus.PENDING,
            short_id="",  # Will be populated when task completes
            user_id=request.user_id,
            message=response["message"],
            created_at=datetime.now(timezone.utc),
            progress=0.0,
            current_step="Starting scenario generation",
            error_message=None,
            scenario=None,
            completed_at=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting scenario generation for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start scenario generation: {str(e)}")


@router.get("/scenario/generate/tasks/{task_id}", response_model=ScenarioGenerationResponse)
def get_scenario_generation_task_status(task_id: str):
    """
    Get the status of a scenario generation task
    """
    try:
        # Get task status from task management system
        from app.utils.task_management import get_task_status
        task_info = get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Scenario generation task not found")
        
        # Convert task info to ScenarioGenerationResponse
        return ScenarioGenerationResponse(
            task_id=task_id,
            status=task_info.task_status,
            short_id=task_info.task_metadata.get('short_id', ''),
            user_id=task_info.user_id or '',
            message=task_info.task_status_message,
            created_at=task_info.created_at,
            progress=task_info.progress,
            current_step=task_info.current_step_name,
            error_message=task_info.error_message,
            scenario=task_info.task_metadata.get('scenario') if task_info.task_metadata else None,
            completed_at=task_info.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scenario generation task status {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get scenario generation task status: {str(e)}")


@router.delete("/scenario/generate/tasks/{task_id}")
def cancel_scenario_generation_task(task_id: str):
    """
    Cancel a running scenario generation task
    """
    try:
        # Get task status from task management system
        from app.utils.task_management import get_task_status, fail_task
        task_info = get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Scenario generation task not found")
        
        if task_info.task_status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed or failed scenario generation task")
        
        # Update task status to failed with cancellation message
        fail_task(task_id, "Scenario generation task cancelled by user")
        
        return {"message": f"Scenario generation task {task_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling scenario generation task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel scenario generation task: {str(e)}")


# Save Scenario Endpoints
@router.post("/scenario/save", response_model=SaveScenarioResponse)
def save_scenario(
    request: SaveScenarioRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> SaveScenarioResponse:
    """
    Save AI-generated scenario and generate images for scenes
    
    This endpoint accepts a scenario JSON string and saves it to the database,
    then generates images for all scenes (except the first one which is already generated).
    Returns immediately with a task ID for polling.
    
    Authentication: Optional API key via Bearer token
    Rate Limits: 
    - With API key: Based on key configuration
    - Without API key: 10 requests per minute
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        # validate_scrape_request(str(request.short_id), api_key)
        
        # No credit check needed for save_scenario itself
        # Credits will be deducted for each image generation (scene_count * generate_image)
        logger.info(f"Starting save scenario for user {request.user_id}")
        
        # Start save scenario using threads
        response = save_scenario_service.start_save_scenario_task(request)
        
        # Convert response to SaveScenarioResponse format
        return SaveScenarioResponse(
            task_id=response["task_id"],
            status=TaskStatus.PENDING,
            short_id=request.short_id,
            user_id=request.user_id,
            message=response["message"],
            created_at=datetime.now(timezone.utc),
            progress=0.0,
            current_step="Starting scenario save process",
            error_message=None,
            completed_at=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting save scenario for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start save scenario: {str(e)}")


@router.get("/scenario/save/tasks/{task_id}", response_model=SaveScenarioResponse)
def get_save_scenario_task_status(task_id: str):
    """
    Get the status of a save scenario task
    """
    try:
        # Get task status from task management system
        from app.utils.task_management import get_task_status
        task_info = get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Save scenario task not found")
        
        # Convert task info to SaveScenarioResponse
        return SaveScenarioResponse(
            task_id=task_id,
            status=task_info.task_status,
            short_id=task_info.task_metadata.get('short_id', ''),
            user_id=task_info.user_id or '',
            message=task_info.task_status_message,
            created_at=task_info.created_at,
            progress=task_info.progress,
            current_step=task_info.current_step_name,
            error_message=task_info.error_message,
            completed_at=task_info.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting save scenario task status {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get save scenario task status: {str(e)}")


@router.delete("/scenario/save/tasks/{task_id}")
def cancel_save_scenario_task(task_id: str):
    """
    Cancel a running save scenario task
    """
    try:
        # Get task status from task management system
        from app.utils.task_management import get_task_status, fail_task
        task_info = get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Save scenario task not found")
        
        if task_info.task_status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed or failed save scenario task")
        
        # Update task status to failed with cancellation message
        fail_task(task_id, "Save scenario task cancelled by user")
        
        return {"message": f"Save scenario task {task_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling save scenario task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel save scenario task: {str(e)}")


# Scheduler Management Endpoints
@router.get("/scheduler/status")
def get_scheduler_status_endpoint():
    """
    Get the current status of the scheduler service
    """
    try:
        status = get_scheduler_status()
        return status
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler status: {str(e)}")


@router.post("/scheduler/cleanup/now")
def trigger_cleanup_now():
    """
    Manually trigger cleanup of old tasks now
    """
    try:
        deleted_count = run_cleanup_now()
        return {
            "message": "Manual cleanup completed successfully",
            "deleted_tasks_count": deleted_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering manual cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to trigger cleanup: {str(e)}")


# ============================================================================
# Session Management Endpoints
# ============================================================================

@router.get("/sessions/{short_id}")
def get_sessions_by_short_id(short_id: str):
    """
    Get all sessions for a specific short_id
    
    This endpoint returns all active and completed sessions associated with a short_id.
    Useful for tracking task progress and session management.
    
    Authentication: Optional API key via Bearer token
    """
    try:
        sessions = session_service.get_sessions_by_short_id(short_id)
        
        # Convert sessions to response format
        session_data = []
        for session in sessions:
            session_data.append({
                "short_id": session.short_id,
                "task_type": session.task_type,
                "task_id": session.task_id,
                "user_id": session.user_id,
                "status": session.status,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            })
        
        return {
            "short_id": short_id,
            "sessions": session_data,
            "total_sessions": len(session_data),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting sessions for short_id {short_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


@router.get("/sessions/task/{task_id}")
def get_session_by_task_id(task_id: str):
    """
    Get session information for a specific task_id
    
    This endpoint returns session information for a specific task.
    Useful for checking if a task has an associated session.
    
    Authentication: Optional API key via Bearer token
    """
    try:
        session = session_service.get_session(task_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "short_id": session.short_id,
            "task_type": session.task_type,
            "task_id": session.task_id,
            "user_id": session.user_id,
            "status": session.status,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session for task_id {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.delete("/sessions/task/{task_id}")
def remove_session(task_id: str):
    """
    Remove a session for a specific task_id
    
    This endpoint manually removes a session. This is typically done automatically
    when tasks complete, but can be used for manual cleanup if needed.
    
    Authentication: Optional API key via Bearer token
    """
    try:
        success = session_service.remove_session(task_id)
        
        if success:
            return {
                "message": f"Session for task {task_id} removed successfully",
                "task_id": task_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found or already removed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing session for task_id {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to remove session: {str(e)}")

@router.get("/sessions/user/{user_id}")
def get_sessions_by_user_id(user_id: str):
    """
    Get all sessions for a specific user_id
    
    This endpoint returns all active and completed sessions associated with a user_id.
    Useful for tracking all tasks for a specific user.
    """
    try:
        sessions = session_service.get_sessions_by_user_id(user_id)
        
        # Convert sessions to response format
        session_data = []
        for session in sessions:
            session_data.append({
                "short_id": session.short_id,
                "task_type": session.task_type,
                "task_id": session.task_id,
                "user_id": session.user_id,
                "status": session.status,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            })
        
        return {
            "sessions": session_data,
            "total_sessions": len(session_data),
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error getting sessions for user_id {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


# ============================================================================
# Test Audio Endpoints
# ============================================================================

@router.post("/test-audio", response_model=TestAudioResponse)
def get_test_audio(
    request: TestAudioRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> TestAudioResponse:
    """
    Get or generate test audio for a specific voice and language.
    
    This endpoint checks if test audio already exists in MongoDB for the given
    voice_id and language combination. If it exists, it returns the cached URL.
    If not, it generates new test audio using ElevenLabs and stores it for future use.
    
    Supported languages:
    - en-US, en-CA, en-GB (English variants)
    - es, es-MX (Spanish variants)
    - pt-BR (Portuguese - Brazil)
    - fr (French)
    - de (German)
    - nl (Dutch)
    
    Authentication: Optional API key via Bearer token
    Rate Limits: Based on API key configuration
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        
        logger.info(f"Getting test audio for voice {request.voice_id} in language {request.language} by user {request.user_id}")
        
        # Get test audio using the service
        response = test_audio_service.get_test_audio(request)
        
        logger.info(f"Test audio request completed for voice {request.voice_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in test audio endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Test audio request failed: {str(e)}")


@router.post("/generate-audio-script", response_model=AudioScriptGenerationResponse)
def generate_audio_script(
    request: AudioScriptGenerationRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> AudioScriptGenerationResponse:
    """
    Generate an audio script only (no audio). User can review/edit the script, then call POST /generate-audio with the script.
    
    This endpoint:
    1. Gets or generates test audio and analyzes words-per-minute for the voice
    2. Fetches short title/description from the database
    3. Generates a promotional script using OpenAI (hook, main, CTA)
    
    Returns the script so the Next server can show it to the user for editing. After user confirms, call POST /generate-audio with the (possibly edited) script.
    
    No credits are deducted for script generation; credits are deducted when generating audio.
    """
    try:
        logger.info(f"Generating audio script for short {request.short_id} with voice {request.voice_id} by user {request.user_id}")
        result = audio_generation_service.generate_audio_script(request)
        logger.info(f"Successfully generated script for short {request.short_id}")
        return result
    except Exception as e:
        logger.error(f"Error in audio script generation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio script generation failed: {str(e)}")


@router.post("/generate-audio", response_model=AudioGenerationResponse)
def generate_audio(
    request: AudioGenerationRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
) -> AudioGenerationResponse:
    """
    Generate audio from a provided script (script is required).
    
    Call this after the user has confirmed the script (from POST /generate-audio-script, possibly edited).
    This endpoint:
    1. Gets test audio and WPM for metadata
    2. Generates final audio using ElevenLabs from the script sent by the Next server
    3. Uploads to storage, deducts credits, saves to audio_info
    
    Request must include the script field (the confirmed/edited script from the client).
    
    Authentication: Optional API key via Bearer token
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        
        logger.info(f"Starting audio generation for short {request.short_id} with voice {request.voice_id} by user {request.user_id}")

        # Check if user has enough credits
        credit_check = can_perform_action(request.user_id, "generate_audio")
        if credit_check.get("error"):
            raise HTTPException(status_code=400, detail=f"Credit check failed: {credit_check['error']}")
        
        if not credit_check.get("can_perform", False):
            reason = credit_check.get("reason", "Insufficient credits")
            current_credits = credit_check.get("current_credits", 0)
            required_credits = credit_check.get("required_credits", 1)
            raise HTTPException(
                status_code=402, 
                detail={
                    "error": "Insufficient credits",
                    "reason": reason,
                    "current_credits": current_credits,
                    "required_credits": required_credits,
                    "message": f"You need {required_credits} credit(s) to perform this action. You currently have {current_credits} credit(s)."
                }
            )

        # Generate audio directly and return result
        result = audio_generation_service.generate_audio(request)
        
        logger.info(f"Successfully generated audio for short {request.short_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in audio generation endpoint: {e}", exc_info=True)
        msg = str(e)
        if "payment" in msg.lower() and ("ElevenLabs" in msg or "invoice" in msg.lower()):
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "elevenlabs_payment_required",
                    "message": "ElevenLabs subscription has a failed or incomplete payment. Complete the latest invoice at ElevenLabs to continue.",
                },
            )
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {msg}")


# ============================================================================
# Remotion Bridge Endpoints (Pass-through to Node.js)
# ============================================================================

@router.post("/remotion/generate-scene1")
async def generate_scene1_bridge(
    request: GenerateScene1Request,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
):
    """
    Bridge endpoint: Forward scene1 generation request from Next.js to Node.js Remotion server.
    
    This endpoint acts as a simple pass-through:
    1. Receives request from Next.js (with template and metadata)
    2. Forwards to Node.js Remotion server
    3. Returns Node.js response back to Next.js
    
    Authentication: Optional API key via Bearer token
    """
    try:
        # Get Node.js server URL from environment
        remotion_url = os.getenv("REMOTION_SERVER_URL", "http://localhost:5050")
        endpoint = f"{remotion_url}/videos"
        
        # Log incoming request from Next.js
        logger.info("=" * 80)
        logger.info("📥 RECEIVED REQUEST FROM NEXT.JS - POST /remotion/generate-scene1")
        logger.info("=" * 80)
        
        # Convert request to dict for logging and forwarding
        request_data = request.model_dump(exclude_none=True)
        
        # Rewrite product title to short, casual version before sending to Remotion
        if request_data.get("product"):
            product = request_data["product"]
            original_title = product.get("title") or product.get("name")
            if original_title and isinstance(original_title, str):
                updated_title = await _rewrite_title_for_remotion(original_title)
                request_data["product"] = {**product, "title": updated_title, "name": updated_title}
        
        # Log request details
        logger.info(f"📋 Request from Next.js:")
        logger.info(f"   - template: {request_data.get('template', 'N/A')}")
        logger.info(f"   - imageUrl: {request_data.get('imageUrl', '')[:80]}..." if request_data.get('imageUrl') else "   - imageUrl: None")
        
        if 'product' in request_data:
            product = request_data['product']
            logger.info(f"   - product:")
            logger.info(f"     * title: {product.get('title', 'N/A')}")
            logger.info(f"     * price: {product.get('price', 'N/A')}")
            logger.info(f"     * rating: {product.get('rating', 'N/A')}")
            logger.info(f"     * reviewCount: {product.get('reviewCount', 'N/A')}")
        
        if 'metadata' in request_data:
            metadata = request_data['metadata']
            logger.info(f"   - metadata:")
            logger.info(f"     * short_id: {metadata.get('short_id', 'N/A')}")
            logger.info(f"     * scene_id: {metadata.get('scene_id', 'N/A')}")
            logger.info(f"     * sceneNumber: {metadata.get('sceneNumber', 'N/A')}")
        
        # Log forwarding to Node.js
        logger.info("-" * 80)
        logger.info(f"🚀 FORWARDING TO NODE.JS SERVER")
        logger.info(f"   - Target URL: {endpoint}")
        logger.info(f"   - Method: POST")
        logger.info(f"   - Timeout: 60 seconds")
        logger.info(f"   - Payload being sent:")
        logger.info(json.dumps(request_data, indent=2))
        
        # Forward request to Node.js server
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                endpoint,
                json=request_data
            )
            response.raise_for_status()
            
            # Get response data from Node.js
            node_response = response.json()
            
            # Log response from Node.js
            logger.info("-" * 80)
            logger.info("✅ RECEIVED RESPONSE FROM NODE.JS")
            logger.info(f"   - Status Code: {response.status_code}")
            logger.info(f"   - Response Body:")
            logger.info(f"     * success: {node_response.get('success')}")
            logger.info(f"     * taskId: {node_response.get('taskId')}")
            logger.info(f"     * status: {node_response.get('status')}")
            logger.info(f"     * message: {node_response.get('message')}")
            if node_response.get('videoUrl'):
                logger.info(f"     * videoUrl: {node_response.get('videoUrl')}")
            if node_response.get('error'):
                logger.info(f"     * error: {node_response.get('error')}")
            
            # Log returning to Next.js
            logger.info("-" * 80)
            logger.info("📤 RETURNING RESPONSE TO NEXT.JS")
            logger.info(f"   - Forwarding Node.js response back to Next.js")
            logger.info(f"   - Full response being sent back:")
            logger.info(json.dumps(node_response, indent=2))
            logger.info("=" * 80)
            
            # Return Node.js response to Next.js
            return GenerateScene1Response(**node_response)
        
    except httpx.HTTPStatusError as e:
        logger.error("=" * 80)
        logger.error("❌ NODE.JS SERVER ERROR")
        logger.error(f"   - Status Code: {e.response.status_code}")
        logger.error(f"   - Response Text: {e.response.text}")
        logger.error("=" * 80)
        try:
            error_data = e.response.json()
            return GenerateScene1Response(
                success=False,
                error=error_data.get('error', str(e))
            )
        except:
            return GenerateScene1Response(
                success=False,
                error=f"Node.js server error: {e.response.status_code}"
            )
    except httpx.RequestError as e:
        logger.error("=" * 80)
        logger.error("❌ CONNECTION ERROR TO NODE.JS")
        logger.error(f"   - Error: {str(e)}")
        logger.error(f"   - Target: {os.getenv('REMOTION_SERVER_URL', 'http://localhost:5050')}/videos")
        logger.error(f"   - Make sure Node.js server is running on port 5050")
        logger.error("=" * 80)
        return JSONResponse(
            content={"error": f"Failed to connect to Remotion server: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ UNEXPECTED ERROR IN BRIDGE")
        logger.error(f"   - Error: {str(e)}")
        logger.error(f"   - Type: {type(e).__name__}")
        logger.error("=" * 80)
        logger.error(f"Full traceback:", exc_info=True)
        return JSONResponse(
            content={"error": f"Bridge error: {str(e)}"},
            status_code=500
        )


@router.get("/remotion/tasks/{task_id}")
async def check_scene1_status_bridge(
    task_id: str,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
):
    """
    Bridge endpoint: Forward scene1 status check from Next.js to Node.js Remotion server.
    
    This endpoint acts as a simple pass-through:
    1. Receives status check from Next.js
    2. Forwards to Node.js Remotion server
    3. Returns Node.js response back to Next.js
    
    Authentication: Optional API key via Bearer token
    """
    try:
        # Get Node.js server URL from environment
        remotion_url = os.getenv("REMOTION_SERVER_URL", "http://localhost:5050")
        endpoint = f"{remotion_url}/tasks/{task_id}"
        
        # Log incoming request from Next.js
        logger.info("=" * 80)
        logger.info("📥 RECEIVED STATUS CHECK FROM NEXT.JS - GET /remotion/tasks/{task_id}")
        logger.info("=" * 80)
        logger.info(f"📋 Path Parameters:")
        logger.info(f"   - task_id: {task_id}")
        
        # Log forwarding to Node.js
        logger.info("-" * 80)
        logger.info(f"🚀 FORWARDING TO NODE.JS SERVER")
        logger.info(f"   - Target URL: {endpoint}")
        logger.info(f"   - Method: GET")
        logger.info(f"   - Timeout: 60 seconds")
        
        # Forward request to Node.js server
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            
            # Get response data from Node.js
            node_response = response.json()
            
            # Log response from Node.js
            logger.info("-" * 80)
            logger.info("✅ RECEIVED RESPONSE FROM NODE.JS")
            logger.info(f"   - Status Code: {response.status_code}")
            logger.info(f"   - Response Body:")
            logger.info(f"     * success: {node_response.get('success')}")
            logger.info(f"     * status: {node_response.get('status')}")
            logger.info(f"     * progress: {node_response.get('progress', 'N/A')}")
            logger.info(f"     * stage: {node_response.get('stage', 'N/A')}")
            if node_response.get('videoUrl'):
                logger.info(f"     * videoUrl: {node_response.get('videoUrl')}")
            if node_response.get('error'):
                logger.info(f"     * error: {node_response.get('error')}")
            if node_response.get('message'):
                logger.info(f"     * message: {node_response.get('message')}")
            
            # Log returning to Next.js
            logger.info("-" * 80)
            logger.info("📤 RETURNING RESPONSE TO NEXT.JS")
            logger.info(f"   - Forwarding Node.js response back to Next.js")
            logger.info(f"   - Full response being sent back:")
            logger.info(json.dumps(node_response, indent=2))
            logger.info("=" * 80)
            
            # Return Node.js response to Next.js EXACTLY as-is (no Pydantic transformation)
            return JSONResponse(content=node_response)
        
    except httpx.HTTPStatusError as e:
        logger.error("=" * 80)
        logger.error("❌ NODE.JS SERVER ERROR (STATUS CHECK)")
        logger.error(f"   - Status Code: {e.response.status_code}")
        logger.error(f"   - Response Text: {e.response.text}")
        logger.error(f"   - Task ID: {task_id}")
        logger.error("=" * 80)
        try:
            error_data = e.response.json()
            return JSONResponse(content=error_data, status_code=e.response.status_code)
        except:
            return JSONResponse(
                content={"error": f"Node.js server error: {e.response.status_code}"},
                status_code=e.response.status_code
            )
    except httpx.RequestError as e:
        logger.error("=" * 80)
        logger.error("❌ CONNECTION ERROR TO NODE.JS (STATUS CHECK)")
        logger.error(f"   - Error: {str(e)}")
        logger.error(f"   - Target: {os.getenv('REMOTION_SERVER_URL', 'http://localhost:5050')}/tasks/{task_id}")
        logger.error(f"   - Make sure Node.js server is running on port 5050")
        logger.error("=" * 80)
        return JSONResponse(
            content={"error": f"Failed to connect to Remotion server: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ UNEXPECTED ERROR IN BRIDGE (STATUS CHECK)")
        logger.error(f"   - Error: {str(e)}")
        logger.error(f"   - Type: {type(e).__name__}")
        logger.error(f"   - Task ID: {task_id}")
        logger.error("=" * 80)
        logger.error(f"Full traceback:", exc_info=True)
        return JSONResponse(
            content={"error": f"Bridge error: {str(e)}"},
            status_code=500
        )


@router.post("/image/merge-with-video")
def merge_image_with_video(
    request: MergeImageWithVideoRequest,
    http_request: Request = None,
    api_key: Optional[str] = Depends(get_api_key)
):
    """
    Start async image-video merge task (Scene2 generation).
    Returns immediately with a task_id for polling.
    
    This endpoint is designed for scene2 generation where the user selects:
    - A product image (without background/transparent PNG)
    - A background video from Storyblocks
    
    The service will (in background):
    1. Download both the product image and background video
    2. Composite the product onto the video with optional animations (zoom, floating)
    3. Upload the merged video to Supabase storage
    4. Update the scene's generated_video_url in the database
    
    Animation features:
    - Zoom animation: Product zooms from 5% to target scale over 3 seconds
    - Floating animation: Gentle up-down movement after zoom completes
    - Smooth position transitions
    
    **NEW**: Now returns task_id immediately and processes in background.
    Poll GET `/image/merge-with-video/tasks/{task_id}` to check status.
    
    Authentication: Optional API key via Bearer token
    Rate Limits: Based on API key configuration
    """
    try:
        # Security validation - DISABLED FOR DEVELOPMENT
        # TODO: Re-enable security checks for production by uncommenting the lines below
        # validate_request_security(http_request, api_key)
        
        print("\n" + "🔔 "*40)
        print("📨 NEW SCENE2 VIDEO MERGE REQUEST RECEIVED (ASYNC)")
        print("🔔 "*40)
        logger.info(
            "[Scene2] REQUEST | scene_id=%s user_id=%s | product_image=%s | background_video=%s | scale=%s position=%s duration=%s animation=%s",
            request.scene_id,
            request.user_id,
            request.product_image_url[:60] + "..." if len(request.product_image_url) > 60 else request.product_image_url,
            request.background_video_url[:60] + "..." if len(request.background_video_url) > 60 else request.background_video_url,
            request.scale,
            request.position,
            request.duration,
            request.add_animation,
        )
        
        # Validate required fields
        if not request.product_image_url:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "product_image_url is required"
                }
            )
        
        if not request.background_video_url:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "background_video_url is required"
                }
            )
        
        # Validate URL formats
        if not request.product_image_url.startswith(('http://', 'https://')):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid image URL format"
                }
            )
        
        if not request.background_video_url.startswith(('http://', 'https://')):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid video URL format"
                }
            )
        
        # Get short_id from request (try multiple field names for compatibility)
        short_id = request.short_id or request.shortId or request.scene_id
        
        # Start async image-video merge task
        result = image_processing_service.start_image_merge_task(
            product_image_url=request.product_image_url,
            background_video_url=request.background_video_url,
            scene_id=request.scene_id,
            user_id=request.user_id,
            short_id=short_id,
            scale=request.scale,
            position=request.position,
            duration=request.duration,
            add_animation=request.add_animation
        )
        
        logger.info(f"✅ Started async merge task {result['task_id']} for scene {request.scene_id}")
        print("🔔 "*40)
        print()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "task_id": result['task_id'],
                "status": "pending",
                "message": "Video merge task started successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting image-video merge task: {e}", exc_info=True)
        print("🔔 "*40)
        print()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to start video merge task"
            }
        )


@router.get("/image/merge-with-video/tasks/{task_id}")
def get_image_merge_task_status(
    task_id: str,
    api_key: Optional[str] = Depends(get_api_key)
):
    """
    Poll the status of an image-video merge task (Scene2 generation).
    
    Returns the current status of the task:
    - pending: Task is queued
    - processing: Task is being processed
    - completed: Task finished successfully (video_url available)
    - failed: Task failed (error message available)
    
    Authentication: Optional API key via Bearer token
    """
    try:
        task_info = image_processing_service.get_task_status(task_id)
        
        if not task_info:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": "Task not found"
                }
            )
        
        # Map internal status to expected format
        status_map = {
            'pending': 'pending',
            'running': 'processing',
            'processing': 'processing',
            'completed': 'completed',
            'failed': 'failed',
            TaskStatus.PENDING: 'pending',
            TaskStatus.RUNNING: 'processing',
            TaskStatus.COMPLETED: 'completed',
            TaskStatus.FAILED: 'failed'
        }
        
        status = status_map.get(task_info.get('status'), 'pending')
        
        # Check if completed and has video URL
        if status == 'completed' and task_info.get('video_url'):
            return JSONResponse(
                content={
                    "success": True,
                    "task_id": task_id,
                    "status": status,
                    "video_url": task_info['video_url'],
                    "message": "Video merge completed successfully",
                    "progress": 100
                }
            )
        elif status == 'failed':
            return JSONResponse(
                content={
                    "success": False,
                    "task_id": task_id,
                    "status": status,
                    "error": task_info.get('error_message', 'Failed to merge video: Unknown error'),
                    "message": "Video merge failed"
                }
            )
        elif status == 'processing':
            return JSONResponse(
                content={
                    "success": True,
                    "task_id": task_id,
                    "status": status,
                    "video_url": None,
                    "message": "Merging product image with background video",
                    "progress": 50
                }
            )
        else:
            # pending
            return JSONResponse(
                content={
                    "success": True,
                    "task_id": task_id,
                    "status": status,
                    "video_url": None,
                    "message": "Video merge task is queued",
                    "progress": 0
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image merge task status {task_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Failed to get task status: {str(e)}"
            }
        )

