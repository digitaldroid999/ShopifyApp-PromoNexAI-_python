from pydantic import BaseModel, HttpUrl, Field, ConfigDict
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime, timezone


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ScrapeRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL of the product to scrape")
    user_id: str = Field(..., description="User ID associated with the task (required)")
    proxy: Optional[str] = Field(None, description="Custom proxy to use")
    user_agent: Optional[str] = Field(None, description="Custom user agent to use")
    target_language: Optional[str] = Field(None, description="Target language for content extraction (e.g., 'en', 'es', 'fr')")
    priority: TaskPriority = Field(TaskPriority.NORMAL, description="Task priority level")
    session_id: Optional[str] = Field(None, description="Session ID for the task")


class ProductInfo(BaseModel):
    title: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    description: Optional[str] = None
    images: List[str] = []
    rating: Optional[float] = None
    review_count: Optional[int] = None
    specifications: Dict[str, Any] = {}


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    url: str
    task_type: Optional[str] = Field(None, description="Type of task (e.g., 'scraping', 'media_processing')")
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    priority: TaskPriority = TaskPriority.NORMAL
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    detail: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional task details and metadata including supabase_product_id")


class TaskListResponse(BaseModel):
    tasks: List[TaskStatusResponse]
    total: int
    page: int
    page_size: int


class TaskStatisticsResponse(BaseModel):
    total_tasks: int
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    timeout_tasks: int
    retrying_tasks: int
    avg_progress: float
    avg_duration_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]

class VideoGenerationRequest(BaseModel):
    scene_id: str = Field(..., description="UUID of the scene to generate video for")
    user_id: str = Field(..., description="User ID who owns the scene")
    force_regenerate_first_frame: bool = Field(False, description="Force regenerate the first frame image even if it already exists")
class VideoGenerationResponse(BaseModel):
    task_id: str = Field(..., description="Unique task ID for tracking")
    status: TaskStatus = Field(..., description="Current status of the task")
    scene_id: str = Field(..., description="Scene ID being processed")
    user_id: str = Field(..., description="User ID who owns the scene")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="When the task was created")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    video_url: Optional[str] = Field(None, description="Generated video URL (signed URL) when task is completed")
    image_url: Optional[str] = Field(None, description="Generated image URL when task is completed")
    error: Optional[str] = None
    completed_at: Optional[datetime] = None


class FinalizeShortRequest(BaseModel):
    user_id: str = Field(..., description="User ID associated with the task")
    short_id: str = Field(..., description="Short ID to finalize")

class FinalizeShortResponse(BaseModel):
    task_id: str
    status: TaskStatus
    short_id: str
    user_id: str
    message: str
    created_at: datetime
    progress: Optional[float] = None
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    thumbnail_url: Optional[str] = None
    final_video_url: Optional[str] = None
    completed_at: Optional[datetime] = None 


class ImageAnalysisRequest(BaseModel):
    product_id: str = Field(..., description="Product ID to analyze images for")
    user_id: str = Field(..., description="User ID associated with the task")

class ImageAnalysisResult(BaseModel):
    image_url: str
    description: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    analyzed_at: Optional[datetime] = None
    objects: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    style: Optional[str] = None
    mood: Optional[str] = None
    text: Optional[List[str]] = None
    productFeatures: Optional[List[str]] = None
    videoScenarios: Optional[List[str]] = None
    targetAudience: Optional[str] = None
    useCases: Optional[List[str]] = None 
class ImageAnalysisResponse(BaseModel):
    task_id: str
    status: TaskStatus
    product_id: str
    user_id: str
    message: str
    created_at: datetime
    progress: Optional[float] = None
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    total_images: Optional[int] = None
    analyzed_images: Optional[int] = None
    failed_images: Optional[int] = None
    analyzedData: Optional[Dict[str, ImageAnalysisResult]] = None
    completed_at: Optional[datetime] = None


# Scenario Generation Models
class ScenarioGenerationRequest(BaseModel):
    short_id: Optional[str] = Field(None, description="Short ID if updating existing short")
    product_id: str = Field(..., description="Product ID to generate scenario for")
    user_id: str = Field(..., description="User ID associated with the task")
    style: str = Field(..., description="Video style (e.g., 'trendy-influencer-vlog', 'cinematic-storytelling')")
    mood: str = Field(..., description="Video mood (e.g., 'energetic', 'calm', 'professional')")
    video_length: int = Field(..., description="Video length in seconds (16, 24, 32, 40, 48, 56 or 64)")
    resolution: str = Field(..., description="Video resolution (e.g., '720:1280', '1280:720')")
    target_language: str = Field(..., description="Target language for content (e.g., 'en-US', 'es-ES')")
    environment: Optional[str] = Field(None, description="Environment context for the video (e.g., 'indoor', 'outdoor', 'studio', 'home', 'office')")


class DetectedDemographics(BaseModel):
    target_gender: str = Field(..., description="Target gender (male, female, unisex, children, adults, seniors, all-ages)")
    age_group: str = Field(..., description="Age group (children, teens, young-adults, adults, seniors, all-ages)")
    product_type: str = Field(..., description="Product category/type")
    demographic_context: str = Field(..., description="Description of target audience for character consistency")


class Scene(BaseModel):
    scene_id: str = Field(..., description="Unique identifier for this scene")
    scene_number: int = Field(..., description="Scene number")
    description: str = Field(..., description="Human-readable scene description")
    duration: int = Field(default=8, description="Duration in seconds (must be exactly 8)")
    image_prompt: str = Field(..., description="Detailed prompt for first frame image generation")
    visual_prompt: str = Field(..., description="Safe video prompt for video generation")
    image_reasoning: str = Field(..., description="Why this image was chosen for this scene")
    generated_image_url: Optional[str] = Field(None, description="Generated image URL from Vertex AI or Flux API")
    text_overlay_prompt: Optional[str] = Field(None, description="Text overlay prompt containing text, position, color and style information. NULL or empty if no text overlay needed.")


class GeneratedScenario(BaseModel):
    title: str = Field(..., description="Scenario title")
    description: str = Field(..., description="Brief description of the scenario approach")
    detected_demographics: Optional[DetectedDemographics] = Field(None, description="AI-detected demographic information")
    scenes: List[Scene] = Field(..., description="List of scenes for the video")
    total_duration: int = Field(..., description="Total duration of the video")
    style: str = Field(..., description="Style of the video")
    mood: str = Field(..., description="Mood of the video")
    resolution: str = Field(..., description="Resolution of the video")
    environment: Optional[str] = Field(None, description="Environment context for the video")
    thumbnail_prompt: Optional[str] = Field(None, description="AI-generated prompt for thumbnail image generation")
    thumbnail_url: Optional[str] = Field(None, description="Generated thumbnail image URL from Vertex AI")
    thumbnail_text_overlay_prompt: Optional[str] = Field(None, description="Thumbnail text overlay prompt containing text, position, color and style information. NULL or empty if no text overlay needed.")


class ScenarioGenerationResponse(BaseModel):
    task_id: str = Field(..., description="Unique task ID for tracking")
    status: TaskStatus = Field(..., description="Current status of the task")
    short_id: str = Field(..., description="Short ID associated with the scenario")
    user_id: str = Field(..., description="User ID who owns the scenario")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="When the task was created")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    scenario: Optional[GeneratedScenario] = Field(None, description="Generated scenario when task is completed")
    completed_at: Optional[datetime] = Field(None, description="When the task was completed")

# Save Scenario Models
class SaveScenarioRequest(BaseModel):
    short_id: str = Field(..., description="Short ID to save scenario for")
    user_id: str = Field(..., description="User ID who owns the scenario")
    scenario: str = Field(..., description="Scenario JSON string to save")

class SaveScenarioResponse(BaseModel):
    task_id: str = Field(..., description="Unique task ID for tracking")
    status: TaskStatus = Field(..., description="Current status of the task")
    short_id: str = Field(..., description="Short ID associated with the scenario")
    user_id: str = Field(..., description="User ID who owns the scenario")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="When the task was created")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    scenario_id: Optional[str] = Field(None, description="Generated scenario ID when task is completed")
    completed_at: Optional[datetime] = Field(None, description="When the task was completed")


# Test Audio Models
class TestAudioRequest(BaseModel):
    voice_id: str = Field(..., description="ElevenLabs voice ID for test audio")
    language: str = Field(..., description="Language code for the test audio (e.g., 'en-US', 'es', 'fr')")
    user_id: str = Field(..., description="User ID associated with the request")

class TestAudioResponse(BaseModel):
    voice_id: str = Field(..., description="Voice ID used for test audio")
    language: str = Field(..., description="Language code used")
    audio_url: str = Field(..., description="URL of the test audio")
    user_id: str = Field(..., description="User ID who requested the audio")
    created_at: datetime = Field(..., description="When the test audio was generated")
    is_cached: bool = Field(False, description="Whether this was a cached result")
    message: str = Field(..., description="Status message")
    test_text: str = Field(..., description="The text that was used to generate the test audio")


# Audio Generation Models
class AudioScriptGenerationRequest(BaseModel):
    """Request for generating an audio script only (user can edit before generating audio)."""
    voice_id: str = Field(..., description="ElevenLabs voice ID (used for WPM calibration)")
    user_id: str = Field(..., description="User ID associated with the request")
    short_id: str = Field(..., description="Short ID to generate script for")


class AudioScriptGenerationResponse(BaseModel):
    """Response containing the generated script for user review/edit before audio generation."""
    short_id: str = Field(..., description="Short ID the script was generated for")
    script: str = Field(..., description="Generated audio script (user can edit before calling generate-audio)")
    words_per_minute: float = Field(..., description="Detected words per minute for this voice")
    target_duration_seconds: int = Field(..., description="Target duration in seconds used for script length")
    message: str = Field("Script generated successfully", description="Status message")


class AudioGenerationRequest(BaseModel):
    """Request for generating audio from a confirmed script (script is required; get it from /generate-audio-script)."""
    voice_id: str = Field(..., description="ElevenLabs voice ID for audio generation")
    user_id: str = Field(..., description="User ID associated with the request")
    short_id: str = Field(..., description="Short ID to generate audio for")
    script: str = Field(..., description="Audio script to speak (from generate-audio-script, possibly edited by user)")


class AudioGenerationResponse(BaseModel):
    voice_id: str = Field(..., description="Voice ID used for audio generation")
    user_id: str = Field(..., description="User ID who requested the audio")
    short_id: str = Field(..., description="Short ID the audio was generated for")
    audio_url: str = Field(..., description="URL of the generated audio")
    script: str = Field(..., description="Generated audio script")
    words_per_minute: float = Field(..., description="Detected words per minute from test audio")
    duration: float = Field(..., description="Duration of the generated audio in seconds")
    created_at: datetime = Field(..., description="When the audio was generated")
    is_cached: bool = Field(False, description="Whether this was a cached result")
    message: str = Field(..., description="Status message")
    subtitle_timing: Optional[List[Dict[str, Any]]] = Field(None, description="Subtitle timing information")


# Image Compositing Models
class ImageCompositeRequest(BaseModel):
    background_url: str = Field(..., description="URL of the background image")
    overlay_url: str = Field(..., description="URL of the overlay image (product with transparent background)")
    scene_id: str = Field(..., description="Scene ID for organizing files and updating database")
    user_id: str = Field(..., description="User ID for organizing files")
    position_x: Optional[int] = Field(0, description="X position to place overlay on background (0 = auto-center)")
    position_y: Optional[int] = Field(0, description="Y position to place overlay on background (0 = auto-center)")
    resize_overlay: Optional[bool] = Field(True, description="Whether to resize overlay to fit background")


class ImageCompositeResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation succeeded")
    image_url: Optional[str] = Field(None, description="URL of the composited image in Supabase (if successful)")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the request was processed")


# Session Management Models
class SessionInfo(BaseModel):
    short_id: str = Field(..., description="Short ID associated with the session")
    task_type: str = Field(..., description="Type of task (e.g., 'scraping', 'scenario_generation')")
    task_id: str = Field(..., description="Task ID associated with the session")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the session was created")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the session was last updated")
    user_id: Optional[str] = Field(None, description="User ID associated with the session")
    status: str = Field("active", description="Session status (active, completed, failed)")


# ============================================================================
# Remotion Scene1 Models (Bridge to Node.js)
# ============================================================================

class Scene1ProductInfo(BaseModel):
    title: Optional[str] = Field(None, description="Product name")
    price: Optional[str] = Field(None, description="Product price")
    rating: Optional[float] = Field(None, description="Product rating 0-5")
    reviewCount: Optional[int] = Field(None, description="Number of reviews")


class Scene1Metadata(BaseModel):
    short_id: Optional[str] = Field(None, description="Short ID")
    scene_id: Optional[str] = Field(None, description="Scene ID from database")
    sceneNumber: Optional[int] = Field(None, description="Scene number")


class GenerateScene1RequestFromNextJS(BaseModel):
    """Request format coming from Next.js"""
    imageUrl: Optional[str] = Field(None, description="Image URL directly")
    shortId: Optional[str] = Field(None, description="Short ID")
    sceneNumber: Optional[int] = Field(None, description="Scene number")
    product: Scene1ProductInfo = Field(..., description="Product information")


class GenerateScene1Request(BaseModel):
    """Request format to send to Node.js"""
    template: str = Field(default="product-modern-v1", description="Remotion template to use")
    imageUrl: str = Field(..., description="Image URL for video generation")
    product: Scene1ProductInfo = Field(..., description="Product information")
    metadata: Optional[Scene1Metadata] = Field(None, description="Additional metadata")


class GenerateScene1Response(BaseModel):
    """
    Flexible response model that accepts any fields from Node.js server.
    This allows pass-through of all Node.js response data without modification.
    """
    model_config = ConfigDict(extra="allow")  # Allow any extra fields from Node.js
    
    # Common fields that might be present
    taskId: Optional[str] = Field(None, description="Task ID for polling status")
    id: Optional[str] = Field(None, description="Task ID (alternative field name)")
    status: Optional[str] = Field(None, description="Task status")
    message: Optional[str] = Field(None, description="Response message")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Progress fields
    stage: Optional[str] = Field(None, description="Current processing stage")
    progress: Optional[float] = Field(None, description="Progress percentage")
    
    # Result fields
    videoUrl: Optional[str] = Field(None, description="Video URL when completed")
    
    # Original request data (echoed back in GET response)
    template: Optional[str] = Field(None, description="Template used")
    imageUrl: Optional[str] = Field(None, description="Image URL used")
    product: Optional[Dict] = Field(None, description="Product data used")


# ============================================================================
# Scene2 (Image-Video Merge) Models (Bridge to Node.js)
# ============================================================================

class MergeImageWithVideoRequest(BaseModel):
    """Request format for Scene2 image-video merge"""
    product_image_url: str = Field(..., description="URL of product image (transparent PNG)")
    background_video_url: str = Field(..., description="URL of background video")
    scene_id: str = Field(..., description="Scene ID in database")
    user_id: str = Field(..., description="User ID")
    short_id: Optional[str] = Field(None, description="Short ID")
    shortId: Optional[str] = Field(None, description="Short ID (alternative field name for compatibility)")
    scale: float = Field(default=0.4, description="Product image scale (0.0-1.0)")
    position: str = Field(default="center", description="Position (center, top, bottom, left, right)")
    duration: int = Field(default=8, description="Video duration in seconds")
    add_animation: bool = Field(default=True, description="Add zoom/floating animation")


class MergeImageWithVideoResponse(BaseModel):
    """Response format for Scene2 image-video merge (Python format with snake_case)"""
    success: bool = Field(..., description="Whether the operation succeeded")
    task_id: str = Field(..., description="Task ID for polling status")
    status: str = Field(..., description="Task status (pending, processing, completed, failed)")
    video_url: Optional[str] = Field(None, description="Merged video URL when completed")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[int] = Field(None, description="Progress percentage (0-100)")
    created_at: Optional[str] = Field(None, description="When the task was created (ISO format)")


class ShadowGenerationRequest(BaseModel):
    """Request format for adding shadow effect to product images"""
    image_url: str = Field(..., description="URL of the product image")
    product_description: str = Field(..., description="Product description to extract shadow prompt context")
    user_id: str = Field(..., description="User ID for credit tracking")
    scene_id: Optional[str] = Field(None, description="Scene ID to update after completion")
    short_id: Optional[str] = Field(None, description="Short ID")


class ShadowGenerationResponse(BaseModel):
    """Response format for shadow generation (async pattern)"""
    success: bool = Field(..., description="Whether the operation succeeded")
    task_id: Optional[str] = Field(None, description="Task ID for polling status")
    status: Optional[str] = Field(None, description="Task status (pending, processing, completed, failed)")
    image_url: Optional[str] = Field(None, description="URL of the image with shadow effect (when completed)")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    created_at: Optional[datetime] = Field(None, description="When the task was created")


class BackgroundGenerationRequest(BaseModel):
    """Request format for generating product backgrounds using AI"""
    product_description: str = Field(..., description="Product description to extract background context")
    user_id: str = Field(..., description="User ID for credit tracking")
    mood: Optional[str] = Field(None, description="Mood/feeling for the background (e.g., 'Energetic', 'Calm')")
    style: Optional[str] = Field(None, description="Visual style for the background (e.g., 'Film Grain', 'Minimalist')")
    environment: Optional[str] = Field(None, description="Environment setting (e.g., 'Indoor Studio', 'Outdoor Nature')")
    manual_prompt: Optional[str] = Field(None, description="Manual prompt for background generation (skips OpenAI if provided)")
    scene_id: Optional[str] = Field(None, description="Scene ID to update after completion")
    short_id: Optional[str] = Field(None, description="Short ID")


class ExtractBackgroundPromptRequest(BaseModel):
    """Request to extract the background prompt from a product description (OpenAI → prompt for Vertex)"""
    product_description: str = Field(..., description="Product description from next server")
    mood: Optional[str] = Field(None, description="Optional mood for the background")
    style: Optional[str] = Field(None, description="Optional visual style")
    environment: Optional[str] = Field(None, description="Optional environment setting")


class ExtractBackgroundPromptResponse(BaseModel):
    """Response with the extracted prompt usable for Vertex background generation"""
    success: bool = Field(..., description="Whether extraction succeeded")
    prompt: str = Field("", description="The background prompt for Vertex")
    error: Optional[str] = Field(None, description="Error message if failed")


class BackgroundGenerationResponse(BaseModel):
    """Response format for background generation (async pattern)"""
    success: bool = Field(..., description="Whether the operation succeeded")
    task_id: Optional[str] = Field(None, description="Task ID for polling status")
    status: Optional[str] = Field(None, description="Task status (pending, processing, completed, failed)")
    image_url: Optional[str] = Field(None, description="URL of the generated background image (when completed)")
    message: str = Field(..., description="Status message")
    error: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    created_at: Optional[datetime] = Field(None, description="When the task was created")