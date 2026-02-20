"""
Remotion Server Proxy Middleware

Acts as a bridge between Next.js frontend and Remotion server.
Forwards requests from Next.js to Remotion server and returns responses.
"""

import httpx
from typing import Dict, Any, Optional
from fastapi import HTTPException

from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)

# Remotion server configuration
REMOTION_SERVER_URL = getattr(settings, 'REMOTION_SERVER_URL', 'http://localhost:5050')
REQUEST_TIMEOUT = 300  # 5 minutes timeout for video generation requests
STATUS_CHECK_TIMEOUT = 30  # 30 seconds for status checks


class RemotionProxy:
    """Proxy service for forwarding requests to Remotion server."""

    def __init__(self, base_url: str = REMOTION_SERVER_URL):
        """
        Initialize Remotion proxy.
        
        Args:
            base_url: Base URL of Remotion server (default: http://localhost:5050)
        """
        self.base_url = base_url.rstrip('/')
        logger.info(f"Remotion proxy initialized with base URL: {self.base_url}")

    async def start_video_generation(
        self, 
        template: str,
        image_url: str,
        product: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Start video generation on Remotion server.
        
        POST {REMOTION_SERVER}/videos
        
        Args:
            template: Template name (e.g., "product-modern-v1", "product-minimal-v1")
            image_url: URL to product image
            product: Product information (title, price, rating, etc.)
            metadata: Metadata (short_id, scene_id, sceneNumber)
            
        Returns:
            {
                "taskId": "task-uuid",
                "status": "pending"
            }
            
        Raises:
            HTTPException: If request fails
        """
        try:
            url = f"{self.base_url}/videos"
            
            payload = {
                "template": template,
                "imageUrl": image_url,
                "product": product,
                "metadata": metadata
            }
            
            logger.info(f"[REMOTION PROXY] Starting video generation: {url}")
            logger.info(f"[REMOTION PROXY] Template: {template}, Scene: {metadata.get('sceneNumber')}")
            logger.info(f"[REMOTION PROXY] Product data being sent: {product}")
            logger.info(f"[REMOTION PROXY] Full payload: {payload}")
            
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"[REMOTION PROXY] ✅ Response from Remotion server:")
                logger.info(f"[REMOTION PROXY] Status Code: {response.status_code}")
                logger.info(f"[REMOTION PROXY] Response Body: {result}")
                logger.info(f"[REMOTION PROXY] Video generation started: taskId={result.get('taskId')}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"[REMOTION PROXY] HTTP error from Remotion server: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Remotion server error: {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(f"[REMOTION PROXY] Failed to connect to Remotion server: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Remotion server at {self.base_url}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"[REMOTION PROXY] Unexpected error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start video generation: {str(e)}"
            )

    async def check_task_status(
        self,
        task_id: str,
        short_id: Optional[str] = None,
        scene_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check task status on Remotion server.
        
        GET {REMOTION_SERVER}/tasks/{taskId}
        
        Args:
            task_id: Task ID from Remotion server
            short_id: Optional short ID for logging
            scene_number: Optional scene number for logging
            
        Returns:
            {
                "status": "pending" | "processing" | "completed" | "failed",
                "stage": "downloading" | "rendering" | "uploading",
                "progress": 0-100,
                "videoUrl": "https://..." // When completed
                "error": "error message" // When failed
            }
            
        Raises:
            HTTPException: If request fails
        """
        try:
            url = f"{self.base_url}/tasks/{task_id}"
            
            logger.info(f"[REMOTION PROXY] Checking task status: {url}")
            if short_id:
                logger.info(f"[REMOTION PROXY] Short ID: {short_id}, Scene: {scene_number}")
            
            async with httpx.AsyncClient(timeout=STATUS_CHECK_TIMEOUT) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"[REMOTION PROXY] ✅ Response from Remotion server:")
                logger.info(f"[REMOTION PROXY] Status Code: {response.status_code}")
                logger.info(f"[REMOTION PROXY] Response Body: {result}")
                
                status = result.get('status')
                stage = result.get('stage')
                progress = result.get('progress')
                
                logger.info(f"[REMOTION PROXY] Task {task_id}: status={status}, stage={stage}, progress={progress}%")
                
                if status == 'completed':
                    logger.info(f"[REMOTION PROXY] Task completed: videoUrl={result.get('videoUrl')}")
                elif status == 'failed':
                    logger.error(f"[REMOTION PROXY] Task failed: {result.get('error')}")
                
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"[REMOTION PROXY] HTTP error from Remotion server: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Remotion server error: {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(f"[REMOTION PROXY] Failed to connect to Remotion server: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Remotion server at {self.base_url}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"[REMOTION PROXY] Unexpected error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to check task status: {str(e)}"
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Remotion server is reachable.
        
        Returns:
            {
                "remotion_server": "connected" | "unreachable",
                "base_url": "http://localhost:5050"
            }
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Try to connect to Remotion server
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                
                logger.info(f"[REMOTION PROXY] Health check passed: Remotion server is reachable")
                return {
                    "remotion_server": "connected",
                    "base_url": self.base_url
                }
        except Exception as e:
            logger.warning(f"[REMOTION PROXY] Health check failed: {e}")
            return {
                "remotion_server": "unreachable",
                "base_url": self.base_url,
                "error": str(e)
            }


# Global instance
remotion_proxy = RemotionProxy()
