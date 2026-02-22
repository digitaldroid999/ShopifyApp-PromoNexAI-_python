"""
Merging Service for Finalizing Shorts

This module provides functionality for:
- Merging generated video scenes into final videos
- Merging audio with videos
- Embedding subtitles into videos
- Adding watermarks for free plan users
- Managing the finalization process with task management
"""

import os
import re
import uuid
import threading
import tempfile
import httpx
import subprocess
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.logging_config import get_logger
from app.utils.credit_utils import credit_manager
from app.utils.task_management import (
    create_task, start_task, update_task_progress,
    complete_task, fail_task, get_task_status,
    TaskType
)
from app.models import TaskStatus
from app.config import settings

logger = get_logger(__name__)

# Timeout configurations
HTTP_TIMEOUT = 300  # 5 minutes for HTTP operations
DOWNLOAD_TIMEOUT = 600  # 10 minutes for file downloads
UPLOAD_TIMEOUT = 900  # 15 minutes for file uploads
MAX_RETRIES = 3  # Maximum retry attempts for failed operations
RETRY_DELAY = 5  # Seconds to wait between retries


class MergingService:
    """Service for finalizing shorts by merging videos, audio, and processing final videos."""

    def __init__(self):
        self._active_threads: Dict[str, threading.Thread] = {}

    def start_finalize_short_task(
        self,
        user_id: str,
        short_id: str
    ) -> Dict[str, Any]:
        """
        Start the finalization process for a short video.

        Args:
            user_id: The user's UUID
            short_id: The short's UUID

        Returns:
            Dict containing task information
        """
        try:
            # Create task
            task_metadata = {
                "user_id": user_id,
                "short_id": short_id,
                "task_type": "finalize_short"
            }

            task_id = create_task(
                task_type=TaskType.FINALIZE_SHORT,
                task_metadata=task_metadata,
                user_id=user_id
            )

            # Start task in background thread
            thread = threading.Thread(
                target=self._finalize_short_worker,
                args=(task_id, user_id, short_id),
                daemon=True
            )

            self._active_threads[task_id] = thread
            thread.start()

            logger.info(
                f"Started finalize short task {task_id} for short {short_id}")

            return {
                "task_id": task_id,
                "status": "pending",
                "message": "Finalization task started",
                "created_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(
                f"Failed to start finalize short task: {e}", exc_info=True)
            raise

    def _finalize_short_worker(
        self,
        task_id: str,
        user_id: str,
        short_id: str
    ):
        """Background worker for finalizing shorts."""
        try:
            logger.info("=" * 80)
            logger.info(f"[MERGE FLOW] Starting merge process for short_id: {short_id}, task_id: {task_id}")
            logger.info("=" * 80)
            
            # Start task
            logger.info(f"[STEP 0] Initializing merge task...")
            start_task(task_id)
            update_task_progress(
                task_id, 0.1, "Fetching video scenes and audio data")
            logger.info(f"[STEP 0] ✓ Task initialized successfully")

            # Step 1: Fetch video scenes and audio data
            logger.info(f"[STEP 1] Fetching video scenes and audio data...")
            logger.info(f"[STEP 1.1] Fetching video scenes for short_id: {short_id}")
            scenes_data = self._fetch_video_scenes(short_id)
            if not scenes_data:
                raise Exception("No video scenes found for this short")
            logger.info(f"[STEP 1.1] ✓ Found {len(scenes_data)} video scenes")
            for i, scene in enumerate(scenes_data, 1):
                logger.info(f"[STEP 1.1]   Scene {i}: id={scene.get('id')}, number={scene.get('scene_number')}, "
                          f"duration={scene.get('duration')}s, url={scene.get('generated_video_url', 'N/A')[:80]}...")

            # Product info no longer needed since thumbnail generation is removed

            logger.info(f"[STEP 1.2] Fetching audio data for short_id: {short_id}")
            audio_data = self._fetch_audio_data(short_id)
            if audio_data:
                logger.info(f"[STEP 1.2] ✓ Audio data found: url={audio_data.get('generated_audio_url', 'N/A')[:80]}..., "
                          f"has_subtitles={bool(audio_data.get('subtitles'))}, status={audio_data.get('status')}")
            else:
                logger.info(f"[STEP 1.2] ⚠ No audio data found for this short")
            logger.info(f"[STEP 1] ✓ Completed fetching data")

            update_task_progress(task_id, 0.2, "Downloading videos and audio")

            # Step 2: Download videos and audio
            logger.info(f"[STEP 2] Downloading videos and audio...")
            logger.info(f"[STEP 2.1] Downloading {len(scenes_data)} video files...")
            video_files = self._download_videos(scenes_data, user_id)
            if not video_files:
                raise Exception("Failed to download videos")
            logger.info(f"[STEP 2.1] ✓ Successfully downloaded {len(video_files)} video files")
            for i, video_file in enumerate(video_files, 1):
                file_size = os.path.getsize(video_file) if os.path.exists(video_file) else 0
                logger.info(f"[STEP 2.1]   Video {i}: {video_file} ({file_size / 1024 / 1024:.2f} MB)")

            audio_file = None
            if audio_data and audio_data.get('generated_audio_url'):
                logger.info(f"[STEP 2.2] Downloading audio file...")
                audio_file = self._download_audio(
                    audio_data['generated_audio_url'], task_id)
                if audio_file:
                    file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
                    logger.info(f"[STEP 2.2] ✓ Audio downloaded: {audio_file} ({file_size / 1024 / 1024:.2f} MB)")
                else:
                    logger.warning(f"[STEP 2.2] ⚠ Audio download returned None")
            else:
                logger.info(f"[STEP 2.2] ⚠ Skipping audio download (no audio URL available)")
            logger.info(f"[STEP 2] ✓ Completed downloading files")

            update_task_progress(task_id, 0.4, "Merging videos and audio")

            # Step 3: Merge videos
            logger.info(f"[STEP 3] Merging videos...")
            logger.info(f"[STEP 3] Merging {len(video_files)} video files into single video...")
            merged_video_path = self._merge_videos(video_files, task_id)
            if not merged_video_path:
                raise Exception("Failed to merge videos")
            merged_size = os.path.getsize(merged_video_path) if os.path.exists(merged_video_path) else 0
            logger.info(f"[STEP 3] ✓ Videos merged successfully: {merged_video_path} ({merged_size / 1024 / 1024:.2f} MB)")

            # Step 3.5: Fetch and download background music if available
            music_files = []
            logger.info(f"[STEP 3.5] Checking for background music...")
            music_metadata = self._get_music_metadata(short_id)
            if music_metadata:
                temp_music_dir = tempfile.mkdtemp()
                
                # Download track1 if available (from downloadUrl or path in Supabase)
                if music_metadata.get('track1'):
                    track1 = music_metadata['track1']
                    logger.info(f"[STEP 3.5] Downloading track1: {track1.get('name')} ({track1.get('genre')})")
                    track1_path = self._download_music_track(track1, temp_music_dir)
                    if track1_path:
                        music_files.append(track1_path)
                        logger.info(f"[STEP 3.5] ✓ Track1 downloaded successfully")
                    else:
                        logger.warning(f"[STEP 3.5] ⚠ Failed to download track1")
                
                # Download track2 if available (for 48s videos)
                if music_metadata.get('track2'):
                    track2 = music_metadata['track2']
                    logger.info(f"[STEP 3.5] Downloading track2: {track2.get('name')} ({track2.get('genre')})")
                    track2_path = self._download_music_track(track2, temp_music_dir)
                    if track2_path:
                        music_files.append(track2_path)
                        logger.info(f"[STEP 3.5] ✓ Track2 downloaded successfully")
                    else:
                        logger.warning(f"[STEP 3.5] ⚠ Failed to download track2")
                
                if music_files:
                    logger.info(f"[STEP 3.5] ✓ Downloaded {len(music_files)} background music track(s)")
                else:
                    logger.warning(f"[STEP 3.5] ⚠ No background music tracks downloaded")
            else:
                logger.info(f"[STEP 3.5] ⚠ No background music metadata found")

            # Step 4: Merge audio and/or background music if available
            if audio_file:
                logger.info(f"[STEP 4] Merging audio with video...")
                logger.info(f"[STEP 4] Combining audio file with merged video...")
                merged_video_path = self._merge_audio_with_video(
                    merged_video_path, audio_file, task_id, music_files
                )
                if merged_video_path:
                    merged_size = os.path.getsize(merged_video_path) if os.path.exists(merged_video_path) else 0
                    logger.info(f"[STEP 4] ✓ Audio merged successfully: {merged_video_path} ({merged_size / 1024 / 1024:.2f} MB)")
                else:
                    logger.warning(f"[STEP 4] ⚠ Audio merge returned None")
            elif music_files:
                # No voice audio but background music exists - merge music only
                logger.info(f"[STEP 4] No voice audio, but merging background music with video...")
                logger.info(f"[STEP 4] Adding {len(music_files)} background music track(s) to video...")
                merged_video_path = self._merge_music_with_video(
                    merged_video_path, task_id, music_files
                )
                if merged_video_path:
                    merged_size = os.path.getsize(merged_video_path) if os.path.exists(merged_video_path) else 0
                    logger.info(f"[STEP 4] ✓ Background music merged successfully: {merged_video_path} ({merged_size / 1024 / 1024:.2f} MB)")
                else:
                    logger.warning(f"[STEP 4] ⚠ Music merge returned None")
            else:
                logger.info(f"[STEP 4] ⚠ Skipping audio merge (no audio or music available)")

            update_task_progress(task_id, 0.5, "Processing final video")

            # Step 5: Add watermark if free plan
            logger.info(f"[STEP 5] Processing final video (watermark, subtitles)...")
            logger.info(f"[STEP 5.1] Checking user plan and adding watermark if needed...")
            final_video_path = self._add_watermark_if_needed(
                merged_video_path, user_id, task_id
            )
            if final_video_path != merged_video_path:
                final_size = os.path.getsize(final_video_path) if os.path.exists(final_video_path) else 0
                logger.info(f"[STEP 5.1] ✓ Watermark added: {final_video_path} ({final_size / 1024 / 1024:.2f} MB)")
            else:
                logger.info(f"[STEP 5.1] ✓ No watermark needed (user is not on free plan)")

            # Step 6: Add subtitles if available
            if audio_data and audio_data.get('subtitles'):
                logger.info(f"[STEP 5.2] Embedding subtitles into video...")
                try:
                    subtitle_count = len(audio_data.get('subtitles', []))
                    logger.info(f"[STEP 5.2] Found {subtitle_count} subtitle entries to embed")
                    final_video_path = self._embed_subtitles(
                        final_video_path, audio_data['subtitles'], task_id
                    )
                    if final_video_path:
                        final_size = os.path.getsize(final_video_path) if os.path.exists(final_video_path) else 0
                        logger.info(f"[STEP 5.2] ✓ Subtitles embedded successfully: {final_video_path} ({final_size / 1024 / 1024:.2f} MB)")
                    else:
                        logger.warning(f"[STEP 5.2] ⚠ Subtitle embedding returned None")
                except Exception as subtitle_error:
                    logger.warning(
                        f"[STEP 5.2] ⚠ Failed to embed subtitles, continuing without them: {subtitle_error}")
                    # Continue with the video without subtitles
            else:
                logger.info(f"[STEP 5.2] ⚠ Skipping subtitles (no subtitle data available)")
            logger.info(f"[STEP 5] ✓ Completed processing final video")

            update_task_progress(task_id, 0.7, "Uploading final video")

            # Step 7: Upload final video
            logger.info(f"[STEP 6] Uploading final video to Supabase...")
            final_size = os.path.getsize(final_video_path) if os.path.exists(final_video_path) else 0
            logger.info(f"[STEP 6] Uploading video: {final_video_path} ({final_size / 1024 / 1024:.2f} MB)")
            final_video_url = self._upload_final_video(
                final_video_path, short_id, task_id)
            if not final_video_url:
                raise Exception("Failed to upload final video")
            logger.info(f"[STEP 6] ✓ Final video uploaded successfully: {final_video_url}")

            # Update shorts table with final video URL
            logger.info(f"[STEP 7] Updating database with final video URL...")
            self._update_shorts_final_video(short_id, final_video_url)
            logger.info(f"[STEP 7] ✓ Database updated successfully")

            # Clean up temporary files
            logger.info(f"[STEP 8] Cleaning up temporary files...")
            temp_files = video_files + [merged_video_path, final_video_path]
            if audio_file:
                temp_files.append(audio_file)
            
            # Also clean up music files
            if music_files:
                temp_files.extend(music_files)
                logger.info(f"[STEP 8] Including {len(music_files)} music file(s) in cleanup")
            
            # Also clean up any subtitle temporary directories
            if audio_data and audio_data.get('subtitles'):
                try:
                    # Find and clean up subtitle temp directories
                    self._cleanup_subtitle_temp_dirs()
                    logger.info(f"[STEP 8] ✓ Subtitle temp directories cleaned")
                except Exception as cleanup_error:
                    logger.warning(f"[STEP 8] ⚠ Failed to cleanup subtitle temp directories: {cleanup_error}")
            
            self._cleanup_temp_files(temp_files)
            logger.info(f"[STEP 8] ✓ Cleaned up {len(temp_files)} temporary files")

            # Complete task
            logger.info(f"[STEP 9] Completing merge task...")
            complete_task(task_id, {
                "final_video_url": final_video_url,
                "short_id": short_id
            })
            logger.info(f"[STEP 9] ✓ Task marked as completed")

            logger.info("=" * 80)
            logger.info(f"[MERGE FLOW] ✓ Successfully completed merge process for short_id: {short_id}")
            logger.info(f"[MERGE FLOW] Final video URL: {final_video_url}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"[MERGE FLOW] ✗ Failed to finalize short {short_id}: {e}", exc_info=True)
            logger.error("=" * 80)
            fail_task(task_id, str(e))
        finally:
            # Remove thread reference
            if task_id in self._active_threads:
                del self._active_threads[task_id]
                logger.info(f"[MERGE FLOW] Thread reference cleaned up for task_id: {task_id}")

    def _fetch_video_scenes(self, short_id: str) -> List[Dict[str, Any]]:
        """Fetch video scenes for a short. Supabase removed - return empty; use alternative data source (e.g. Prisma/API)."""
        logger.warning("Supabase removed: _fetch_video_scenes returns empty. Provide scenes via alternative source.")
        return []

    # def _fetch_product_info(self, short_id: str) -> Dict[str, Any]:
    #     """Fetch product information for thumbnail generation - REMOVED: No longer generating thumbnails."""
    #     # This method has been removed as thumbnail generation is no longer needed
    #     pass

    def _fetch_audio_data(self, short_id: str) -> Optional[Dict[str, Any]]:
        """Fetch audio data. Supabase removed - return None; use alternative source (e.g. local path from audio generation)."""
        logger.warning("Supabase removed: _fetch_audio_data returns None.")
        return None

    def _get_music_metadata(self, short_id: str) -> Optional[Dict[str, Any]]:
        """Get background music metadata. Supabase removed - return None."""
        logger.warning("Supabase removed: _get_music_metadata returns None.")
        return None

    # def _generate_thumbnail(self, user_id: str, product_info: Dict[str, Any], task_id: str, short_id: str) -> str:
    #     """Generate thumbnail using Vertex AI - REMOVED: No longer generating thumbnails."""
    #     # This method has been removed as thumbnail generation is no longer needed
    #     pass

    # def _download_image(self, image_url: str, task_id: str) -> str:
    #     """Download generated image to temporary file - REMOVED: No longer generating thumbnails."""
    #     # This method has been removed as thumbnail generation is no longer needed
    #     pass

    # def _upload_thumbnail_from_url(self, image_url: str, user_id: str, task_id: str) -> str:
    #     """Upload thumbnail directly from URL to Supabase storage without saving to filesystem - REMOVED: No longer generating thumbnails."""
    #     # This method has been removed as thumbnail generation is no longer needed
    #     pass

    # def _upload_thumbnail(self, thumbnail_path: str, user_id: str, task_id: str) -> str:
    #     """Upload thumbnail from local file to Supabase storage - REMOVED: No longer generating thumbnails."""
    #     # This method has been removed as thumbnail generation is no longer needed
    #     pass

    # def _upload_thumbnail_fallback(self, image_url: str, user_id: str, task_id: str) -> str:
    #     """Fallback method: download image to temp file first, then upload - REMOVED: No longer generating thumbnails."""
    #     # This method has been removed as thumbnail generation is no longer needed
    #     pass

    # def _update_shorts_thumbnail(self, short_id: str, thumbnail_url: str):
    #     """Update shorts table with thumbnail URL - REMOVED: No longer generating thumbnails."""
    #     # This method has been removed as thumbnail generation is no longer needed
    #     pass

    def _download_music_file(self, music_path: str, temp_dir: str) -> Optional[str]:
        """
        Download background music file from Supabase storage.
        
        Args:
            music_path: Path to music file in Supabase (e.g., 'Music/Classical/Eternal.mp3')
            temp_dir: Temporary directory to save the downloaded file
            
        Returns:
            Path to temporary music file, or None if download fails
        """
        try:
            # Construct full Supabase URL for background music
            base_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/background-music"
            music_url = f"{base_url}/{music_path}"
            
            logger.info(f"Downloading background music from: {music_url}")

            # Download music file
            with httpx.Client(timeout=DOWNLOAD_TIMEOUT) as client:
                response = client.get(music_url)
                response.raise_for_status()

                # Save to temp file
                music_filename = os.path.basename(music_path)
                temp_music_path = os.path.join(temp_dir, music_filename)
                
                with open(temp_music_path, 'wb') as f:
                    f.write(response.content)

            file_size = os.path.getsize(temp_music_path)
            logger.info(f"Successfully downloaded background music: {temp_music_path} ({file_size / 1024:.2f} KB)")
            return temp_music_path

        except Exception as e:
            logger.error(f"Failed to download background music from {music_path}: {e}")
            return None

    def _download_music_from_url(self, download_url: str, temp_dir: str, suggested_filename: Optional[str] = None) -> Optional[str]:
        """
        Download background music file from an external URL (e.g. musicInfo.track1.downloadUrl).
        
        Args:
            download_url: Full URL to the music file (e.g. AudioBlocks CDN)
            temp_dir: Temporary directory to save the downloaded file
            suggested_filename: Optional filename (e.g. from track name); extension will be preserved or default to .mp3
            
        Returns:
            Path to temporary music file, or None if download fails
        """
        try:
            logger.info(f"Downloading background music from external URL (length={len(download_url)})")
            # Many CDNs (e.g. AudioBlocks/CloudFront) return 403 for non-browser User-Agents
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "audio/mpeg,audio/*,*/*",
            }
            with httpx.Client(timeout=DOWNLOAD_TIMEOUT) as client:
                response = client.get(download_url, headers=headers)
                response.raise_for_status()
                content = response.content
                content_disposition = response.headers.get("content-disposition")
            if not content:
                logger.error("Download returned empty content")
                return None
            # Determine filename: use suggested or derive from URL (path segment) or default
            if suggested_filename:
                base = suggested_filename.strip()
                if not base.endswith(('.mp3', '.m4a', '.wav')):  
                    base = f"{base}.mp3"
                music_filename = "".join(c for c in base if c.isalnum() or c in "._- ") or "track.mp3"
            else:
                # Try Content-Disposition header
                cd = content_disposition
                if cd and "filename=" in cd:
                    m = re.search(r'filename[*]?=(?:UTF-8\'\')?["\']?([^"\';]+)', cd, re.I)
                    if m:
                        music_filename = m.group(1).strip().strip('"')
                    else:
                        music_filename = "track.mp3"
                else:
                    path_part = download_url.split("?")[0].rstrip("/")
                    music_filename = os.path.basename(path_part) or "track.mp3"
            music_filename = music_filename[:200]
            temp_music_path = os.path.join(temp_dir, music_filename)
            with open(temp_music_path, 'wb') as f:
                f.write(content)
            file_size = os.path.getsize(temp_music_path)
            logger.info(f"Successfully downloaded background music from URL: {temp_music_path} ({file_size / 1024:.2f} KB)")
            return temp_music_path
        except Exception as e:
            logger.error(f"Failed to download background music from URL: {e}")
            return None

    def _download_music_track(self, track: Dict[str, Any], temp_dir: str) -> Optional[str]:
        """
        Download one background music track. Prefers downloadUrl (external) if present, else path (Supabase).
        
        Args:
            track: Track dict with either downloadUrl (full URL) or path (Supabase path)
            temp_dir: Temporary directory to save the file
            
        Returns:
            Path to temporary music file, or None if download fails
        """
        if not track:
            return None
        download_url = track.get("downloadUrl") or track.get("download_url")
        path = track.get("path")
        name = track.get("name") or "track"
        if download_url:
            return self._download_music_from_url(download_url, temp_dir, suggested_filename=name)
        if path:
            return self._download_music_file(path, temp_dir)
        logger.warning(f"Track has neither downloadUrl nor path: {list(track.keys())}")
        return None

    def _download_audio(self, audio_url: str, task_id: str) -> str:
        """
        Download audio file from Supabase storage.
        
        Converts public URL to signed URL and downloads the file.
        
        Args:
            audio_url: Public URL from database
            task_id: Task ID for logging
            
        Returns:
            Path to temporary audio file
            
        Raises:
            Exception: If URL conversion or download fails
        """
        try:
            # Local path: /generated_audio/... or generated_audio/... (Supabase removed)
            path_stripped = (audio_url or "").strip().lstrip("/")
            if path_stripped.startswith("generated_audio/"):
                local_path = Path(settings.PUBLIC_OUTPUT_BASE) / path_stripped
                if local_path.exists():
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                        shutil.copy2(local_path, temp_file.name)
                        logger.info(f"Copied local audio to {temp_file.name}")
                        return temp_file.name
                raise Exception(f"Local audio file not found: {local_path}")

            raise Exception(
                "Supabase removed. Audio URL must be a local path (e.g. /generated_audio/user_id/short_id/file.mp3)."
            )

        except Exception as e:
            logger.error(f"Failed to download audio: {e}")
            raise

    def _get_signed_audio_url(self, audio_url: str) -> str:
        """Supabase removed. Use local path /generated_audio/... for audio."""
        raise NotImplementedError("Supabase removed. Audio must use local path /generated_audio/...")

    def _get_signed_video_url(self, video_url: str) -> str:
        """Supabase removed. Use local paths for video."""
        raise NotImplementedError("Supabase removed. Video URLs must use local paths.")

    def _handle_expired_url(self, url: str, user_id: str) -> str:
        """
        Handle expired signed URLs by converting them to public URLs or regenerating signed URLs.
        
        Args:
            url: The potentially expired URL
            user_id: The user ID for regenerating URLs if needed
            
        Returns:
            A working URL for the video
        """
        try:
            # Supabase removed - return URL as-is (no regeneration)
            if 'supabase.co' in url and 'token=' in url:
                logger.warning("Supabase removed: cannot regenerate signed URL, returning URL as-is")
            return url
                
        except Exception as e:
            logger.error(f"Error handling expired URL {url}: {e}")
            # Return original URL if we can't handle it
            return url

    def _download_videos(self, scenes_data: List[Dict[str, Any]], user_id: str) -> List[str]:
        """Download all video files from scenes."""
        try:
            video_files = []

            for scene in scenes_data:
                video_url = scene.get('generated_video_url')
                if not video_url:
                    logger.warning(f"Scene {scene.get('id')} has no generated_video_url, skipping")
                    continue

                # Convert Supabase public URL to signed URL for download
                # This will raise an exception if conversion fails (no fallback)
                try:
                    working_url = self._get_signed_video_url(video_url)
                except Exception as url_error:
                    raise Exception(
                        f"Failed to convert public URL to signed URL for scene {scene.get('id')}: {url_error}"
                    )
                
                # Validate that we got a signed URL (not the original public URL)
                if working_url == video_url:
                    raise Exception(
                        f"Signed URL conversion returned original public URL. This should not happen. "
                        f"Scene: {scene.get('id')}, URL: {video_url}"
                    )
                
                if 'token=' not in working_url:
                    raise Exception(
                        f"Invalid signed URL received (missing token). Scene: {scene.get('id')}, "
                        f"URL: {working_url}"
                    )
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name

                # Download video with retry logic
                download_success = False
                for attempt in range(MAX_RETRIES):
                    try:
                        with httpx.Client(timeout=DOWNLOAD_TIMEOUT) as client:
                            response = client.get(working_url)
                            response.raise_for_status()

                            with open(temp_path, 'wb') as f:
                                f.write(response.content)

                        download_success = True
                        logger.info(
                            f"Successfully downloaded video for scene {scene.get('id')} on attempt {attempt + 1}"
                        )
                        break
                        
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in [400, 403, 404] and 'supabase.co' in working_url:
                            logger.warning(
                                f"Attempt {attempt + 1}: Got {e.response.status_code} error for Supabase URL, "
                                f"trying to refresh signed URL: {e}"
                            )
                            # Try to refresh the signed URL (it may have expired)
                            try:
                                working_url = self._get_signed_video_url(video_url)
                                # Continue to next attempt with new signed URL
                            except Exception as refresh_error:
                                logger.error(
                                    f"Failed to refresh signed URL on attempt {attempt + 1}: {refresh_error}"
                                )
                                if attempt == MAX_RETRIES - 1:
                                    raise Exception(
                                        f"Failed to refresh signed URL after {MAX_RETRIES} attempts: {refresh_error}"
                                    )
                        else:
                            logger.error(f"HTTP error downloading video (attempt {attempt + 1}): {e}")
                            if attempt == MAX_RETRIES - 1:
                                raise Exception(
                                    f"HTTP error downloading video after {MAX_RETRIES} attempts: {e}"
                                )
                            break
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1} failed: {e}")
                        if attempt < MAX_RETRIES - 1:
                            import time
                            time.sleep(RETRY_DELAY)
                        else:
                            raise Exception(
                                f"Failed to download video after {MAX_RETRIES} attempts: {e}"
                            )

                if not download_success:
                    raise Exception(
                        f"Failed to download video for scene {scene.get('id')} after {MAX_RETRIES} attempts"
                    )

                # Verify downloaded video duration
                downloaded_duration = self._get_video_duration(temp_path)
                expected_duration = scene.get('duration', 8)
                logger.info(
                    f"✅ Downloaded scene {scene.get('scene_number')}: "
                    f"{downloaded_duration:.2f}s (expected: {expected_duration}s)"
                )
                
                # Warn if duration mismatch
                if abs(downloaded_duration - expected_duration) > 1.0:
                    logger.warning(
                        f"⚠️  Scene {scene.get('scene_number')} duration mismatch: "
                        f"got {downloaded_duration:.2f}s, expected {expected_duration}s"
                    )

                video_files.append(temp_path)

            if not video_files:
                raise Exception("No videos were downloaded successfully")

            logger.info(f"✅ Successfully downloaded {len(video_files)} videos")
            return video_files

        except Exception as e:
            logger.error(f"Failed to download videos: {e}")
            raise

    def _merge_videos(self, video_files: List[str], task_id: str) -> str:
        """Merge videos using FFmpeg concat filter to ensure proper scene timing."""
        try:
            if len(video_files) == 1:
                return video_files[0]

            # Log each video's duration and audio status before merging
            logger.info(f"📊 Analyzing {len(video_files)} video scenes before merge:")
            total_expected_duration = 0.0
            has_audio_streams = []
            
            for i, video_file in enumerate(video_files, 1):
                duration = self._get_video_duration(video_file)
                has_audio = self._check_video_has_audio(video_file)
                has_audio_streams.append(has_audio)
                total_expected_duration += duration
                audio_status = "✅ with audio" if has_audio else "⚠️  no audio"
                logger.info(f"   Scene {i}: {duration:.2f}s {audio_status} - {os.path.basename(video_file)}")
            logger.info(f"   Total expected duration: {total_expected_duration:.2f}s")

            # Create temporary directory for merged video
            temp_dir = tempfile.mkdtemp()
            merged_video_path = os.path.join(temp_dir, "merged.mp4")

            # Check if all videos have audio
            all_have_audio = all(has_audio_streams)
            none_have_audio = not any(has_audio_streams)
            
            if none_have_audio:
                # No videos have audio - use simple concat demuxer
                logger.info("🎬 All videos are silent - using simple concat demuxer")
                file_list_path = os.path.join(temp_dir, "file_list.txt")
                with open(file_list_path, 'w') as f:
                    for video_file in video_files:
                        f.write(f"file '{video_file}'\n")
                
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', file_list_path,
                    '-c', 'copy',
                    merged_video_path
                ]
            elif not all_have_audio:
                # Mixed: some have audio, some don't - need to add silent audio to silent videos first
                logger.info("🎬 Mixed audio streams detected - normalizing videos first")
                normalized_videos = []
                for i, (video_file, has_audio) in enumerate(zip(video_files, has_audio_streams), 1):
                    if not has_audio:
                        logger.info(f"   Adding silent audio to scene {i}...")
                        normalized_path = os.path.join(temp_dir, f"normalized_{i}.mp4")
                        norm_cmd = [
                            'ffmpeg', '-y',
                            '-i', video_file,
                            '-f', 'lavfi',
                            '-i', 'anullsrc=r=48000:cl=stereo',
                            '-shortest',
                            '-c:v', 'copy',
                            '-c:a', 'aac',
                            normalized_path
                        ]
                        norm_result = subprocess.run(norm_cmd, capture_output=True, text=True, timeout=120)
                        if norm_result.returncode != 0:
                            logger.error(f"Failed to add silent audio: {norm_result.stderr}")
                            raise Exception(f"Failed to normalize video {i}")
                        normalized_videos.append(normalized_path)
                        logger.info(f"   ✅ Scene {i} normalized")
                    else:
                        normalized_videos.append(video_file)
                
                # Now use concat FILTER (not demuxer) to properly handle mixed sources
                logger.info("🎬 Using concat filter for mixed sources (ensures proper timing)")
                filter_parts = []
                for i in range(len(normalized_videos)):
                    filter_parts.append(f"[{i}:v][{i}:a]")
                
                filter_complex = f"{''.join(filter_parts)}concat=n={len(normalized_videos)}:v=1:a=1[outv][outa]"
                
                cmd = ['ffmpeg', '-y']
                
                # Add all normalized input files
                for video_file in normalized_videos:
                    cmd.extend(['-i', video_file])
                
                # Add filter complex and output mapping
                cmd.extend([
                    '-filter_complex', filter_complex,
                    '-map', '[outv]',
                    '-map', '[outa]',
                    '-c:v', 'libx264',      # Re-encode video for consistency
                    '-preset', 'medium',     # Balanced encoding speed
                    '-crf', '23',            # Good quality
                    '-c:a', 'aac',           # Re-encode audio for consistency
                    '-b:a', '192k',          # Good audio quality
                    merged_video_path
                ])
            else:
                # All videos have audio - use concat filter for best quality
                logger.info("🎬 All videos have audio - using concat filter")
                filter_parts = []
                for i in range(len(video_files)):
                    filter_parts.append(f"[{i}:v][{i}:a]")
                
                filter_complex = f"{''.join(filter_parts)}concat=n={len(video_files)}:v=1:a=1[outv][outa]"
                
                cmd = ['ffmpeg', '-y']
                
                # Add all input files
                for video_file in video_files:
                    cmd.extend(['-i', video_file])
                
                # Add filter complex and output mapping
                cmd.extend([
                    '-filter_complex', filter_complex,
                    '-map', '[outv]',
                    '-map', '[outa]',
                    '-c:v', 'libx264',      # Re-encode video for consistency
                    '-preset', 'medium',     # Balanced encoding speed
                    '-crf', '23',            # Good quality
                    '-c:a', 'aac',           # Re-encode audio for consistency
                    '-b:a', '192k',          # Good audio quality
                    merged_video_path
                ])

            logger.info("🎬 Executing FFmpeg merge command...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for re-encoding
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg concat filter failed, stderr: {result.stderr}")
                raise Exception(f"FFmpeg failed: {result.stderr}")

            if not os.path.exists(merged_video_path):
                raise Exception("Merged video file was not created")

            # Verify merged video duration
            merged_duration = self._get_video_duration(merged_video_path)
            logger.info(f"✅ Successfully merged {len(video_files)} videos")
            logger.info(f"   Expected duration: {total_expected_duration:.2f}s")
            logger.info(f"   Actual duration: {merged_duration:.2f}s")
            
            if abs(merged_duration - total_expected_duration) > 0.5:
                logger.warning(f"   ⚠️  Duration mismatch: {merged_duration - total_expected_duration:.2f}s difference")
            else:
                logger.info(f"   ✅ Duration verified: All scenes included properly")
            
            return merged_video_path

        except Exception as e:
            logger.error(f"Failed to merge videos: {e}")
            raise

    def _add_watermark_if_needed(self, video_path: str, user_id: str, task_id: str) -> str:
        """Add watermark based on user's plan watermark_enabled setting."""
        try:
            # Step 1: Get user's plan name
            user_plan = self._get_user_plan(user_id)
            
            # Step 2: Get watermark_enabled setting for this plan
            watermark_enabled = self._get_plan_watermark_setting(user_plan)

            if watermark_enabled:
                logger.info(f"Adding watermark for user {user_id} (plan: {user_plan}, watermark_enabled: true)")
                return self._add_watermark_to_video(video_path, task_id)
            else:
                logger.info(f"User {user_id} is on {user_plan} plan, watermark disabled (watermark_enabled: false)")
                return video_path

        except Exception as e:
            logger.error(f"Failed to check user plan or add watermark: {e}")
            # Continue without watermark if there's an error
            return video_path

    def _get_user_plan(self, user_id: str) -> str:
        """Get user's subscription plan. Supabase removed - default to free."""
        return "free"

    def _get_plan_watermark_setting(self, plan_name: str) -> bool:
        """Get watermark_enabled for plan. Supabase removed - default to True."""
        return True

    def _add_watermark_to_video(self, video_path: str, task_id: str) -> str:
        """Add watermark to video using FFmpeg."""
        try:
            # Create temporary directory for watermarked video
            temp_dir = tempfile.mkdtemp()
            watermarked_video_path = os.path.join(temp_dir, "watermarked.mp4")

            # Add watermark using FFmpeg with bigger, transparent gray text and subtle shadow
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', 'drawtext=text=\'PromoNexAI\':fontcolor=gray@0.6:fontsize=120:x=(w-text_w)/2:y=(h-text_h)/2:shadowcolor=black@0.2:shadowx=1:shadowy=1',
                '-c:a', 'copy',
                watermarked_video_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg watermark failed: {result.stderr}")

            if not os.path.exists(watermarked_video_path):
                raise Exception("Watermarked video file was not created")

            logger.info(f"Successfully added watermark to video")
            return watermarked_video_path

        except Exception as e:
            logger.error(f"Failed to add watermark: {e}")
            raise


    def _check_video_has_audio(self, video_path: str) -> bool:
        """Check if video file has an audio stream using FFprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-select_streams', 'a',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                video_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # If there's an audio stream, ffprobe will output "audio"
                has_audio = "audio" in result.stdout.strip()
                logger.info(f"Video {video_path} has audio: {has_audio}")
                return has_audio
            else:
                logger.warning(f"Could not check for audio stream, assuming no audio: {result.stderr}")
                return False

        except Exception as e:
            logger.warning(f"Failed to check for audio stream: {e}, assuming no audio")
            return False

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using FFmpeg (returns float for precise timing)."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                video_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                duration = float(result.stdout.strip())
                return duration
            else:
                logger.warning(
                    f"Could not get video duration, defaulting to 30 seconds: {result.stderr}")
                return 30.0

        except Exception as e:
            logger.warning(
                f"Failed to get video duration: {e}, defaulting to 30 seconds")
            return 30.0


    def _merge_audio_with_video(self, video_path: str, audio_path: str, task_id: str, music_files: List[str] = None) -> str:
        """
        Merge audio with video using FFmpeg, mixing video sound effects, voice script, and optional background music.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the voice audio file
            task_id: Task ID for logging
            music_files: Optional list of background music file paths (1 or 2 tracks)
            
        Returns:
            Path to the merged video file
        """
        try:
            # First check if the video has an audio stream
            has_audio = self._check_video_has_audio(video_path)
            
            if not has_audio:
                logger.info("Video has no audio stream, using voice-only approach")
                return self._merge_audio_with_video_basic(video_path, audio_path, task_id, music_files)

            # Create temporary directory for merged video
            temp_dir = tempfile.mkdtemp()
            merged_path = os.path.join(temp_dir, "video_with_audio.mp4")

            # Get video duration for proper audio mixing
            video_duration = self._get_video_duration(video_path)
            audio_duration = self._get_video_duration(audio_path) if audio_path else 0
            
            logger.info(f"📏 Duration Analysis:")
            logger.info(f"   - Video duration: {video_duration:.2f}s (ALL scenes merged)")
            logger.info(f"   - Audio duration: {audio_duration:.2f}s")
            
            if audio_duration > video_duration:
                trim_amount = audio_duration - video_duration
                logger.warning(f"   ⚠️  Audio is {trim_amount:.2f}s LONGER than video - will be trimmed to {video_duration:.2f}s")
                logger.warning(f"   ✂️  Audio will be CUT at {video_duration:.2f}s to match video length")
            elif audio_duration < video_duration:
                silence_amount = video_duration - audio_duration
                logger.info(f"   ✅ Audio is {silence_amount:.2f}s shorter than video - will have silence at end")
            else:
                logger.info(f"   ✅ Audio and video durations match perfectly")

            # Build FFmpeg command: trim/pad all audio to exactly video duration so output = video length
            vd = round(video_duration, 2)
            logger.info(f"   - Output duration will be exactly {vd}s (video length)")
            if music_files and len(music_files) > 0:
                logger.info(f"Merging with background music: {len(music_files)} track(s)")
                
                if len(music_files) == 1:
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', video_path,
                        '-i', audio_path,
                        '-i', music_files[0],
                        '-c:v', 'copy',
                        '-filter_complex',
                        f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.2[vid_audio];'
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice_audio];'
                        f'[2:a]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[bg_music];'
                        '[vid_audio][voice_audio][bg_music]amix=inputs=3:duration=longest:dropout_transition=2[out]',
                        '-map', '0:v', '-map', '[out]', '-c:a', 'aac',
                        merged_path
                    ]
                else:
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', video_path,
                        '-i', audio_path,
                        '-i', music_files[0], '-i', music_files[1],
                        '-c:v', 'copy',
                        '-filter_complex',
                        '[2:a][3:a]concat=n=2:v=0:a=1[bg_raw];'
                        f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.2[vid_audio];'
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice_audio];'
                        f'[bg_raw]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[bg_audio];'
                        '[vid_audio][voice_audio][bg_audio]amix=inputs=3:duration=longest:dropout_transition=2[out]',
                        '-map', '0:v', '-map', '[out]', '-c:a', 'aac',
                        merged_path
                    ]
            else:
                logger.info("Merging without background music")
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path, '-i', audio_path,
                    '-c:v', 'copy',
                    '-filter_complex',
                    f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.3[vid_audio];'
                    f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice_audio];'
                    '[vid_audio][voice_audio]amix=inputs=2:duration=longest:dropout_transition=2[out]',
                    '-map', '0:v', '-map', '[out]', '-c:a', 'aac',
                    merged_path
                ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                # If the complex filter fails, try a simpler approach
                logger.warning(f"Complex audio mixing failed, trying simpler approach: {result.stderr}")
                return self._merge_audio_with_video_simple(video_path, audio_path, task_id, music_files)

            if not os.path.exists(merged_path):
                raise Exception("Audio-merged video file was not created")

            logger.info(f"Successfully merged audio with video using complex mixing")
            return merged_path

        except Exception as e:
            logger.error(f"Failed to merge audio with video: {e}")
            raise

    def _merge_audio_with_video_simple(self, video_path: str, audio_path: str, task_id: str, music_files: List[str] = None) -> str:
        """Fallback method for merging audio with video using a simpler approach."""
        try:
            # First check if the video has an audio stream
            has_audio = self._check_video_has_audio(video_path)
            
            if not has_audio:
                logger.info("Video has no audio stream, using voice-only approach")
                return self._merge_audio_with_video_basic(video_path, audio_path, task_id, music_files)

            temp_dir = tempfile.mkdtemp()
            merged_path = os.path.join(temp_dir, "video_with_audio_simple.mp4")
            video_duration = self._get_video_duration(video_path)
            vd = round(video_duration, 2)

            if music_files and len(music_files) > 0:
                logger.info(f"Simple merge with background music: {len(music_files)} track(s)")
                if len(music_files) == 1:
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-i', music_files[0],
                        '-c:v', 'copy', '-filter_complex',
                        f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.3[vid_audio];'
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice_audio];'
                        f'[2:a]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[bg_music];'
                        '[vid_audio][voice_audio][bg_music]amix=inputs=3:duration=longest[out]',
                        '-map', '0:v', '-map', '[out]', '-c:a', 'aac', merged_path
                    ]
                else:
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
                        '-i', music_files[0], '-i', music_files[1],
                        '-c:v', 'copy', '-filter_complex',
                        '[2:a][3:a]concat=n=2:v=0:a=1[bg_raw];'
                        f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.3[vid_audio];'
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice_audio];'
                        f'[bg_raw]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[bg_audio];'
                        '[vid_audio][voice_audio][bg_audio]amix=inputs=3:duration=longest[out]',
                        '-map', '0:v', '-map', '[out]', '-c:a', 'aac', merged_path
                    ]
            else:
                cmd = [
                    'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
                    '-c:v', 'copy', '-filter_complex',
                    f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.5[vid_audio];'
                    f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice_audio];'
                    '[vid_audio][voice_audio]amix=inputs=2:duration=longest[out]',
                    '-map', '0:v', '-map', '[out]', '-c:a', 'aac', merged_path
                ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                # If even the simple approach fails, try the voice-priority method
                logger.warning(f"Simple audio mixing failed, trying voice-priority approach: {result.stderr}")
                return self._merge_audio_with_video_voice_priority(video_path, audio_path, task_id, music_files)

            if not os.path.exists(merged_path):
                raise Exception("Audio-merged video file was not created")

            logger.info(f"Successfully merged audio with video using simple mixing")
            return merged_path

        except Exception as e:
            logger.error(f"Failed to merge audio with video (simple): {e}")
            raise

    def _merge_audio_with_video_voice_priority(self, video_path: str, audio_path: str, task_id: str, music_files: List[str] = None) -> str:
        """Voice-priority fallback method for merging audio with video."""
        try:
            # First check if the video has an audio stream
            has_audio = self._check_video_has_audio(video_path)
            
            if not has_audio:
                logger.info("Video has no audio stream, using basic approach")
                return self._merge_audio_with_video_basic(video_path, audio_path, task_id, music_files)

            temp_dir = tempfile.mkdtemp()
            merged_path = os.path.join(temp_dir, "video_with_audio_voice_priority.mp4")
            video_duration = self._get_video_duration(video_path)
            vd = round(video_duration, 2)

            if music_files and len(music_files) > 0:
                logger.info(f"Voice-priority merge with background music: {len(music_files)} track(s)")
                if len(music_files) == 1:
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-i', music_files[0],
                        '-filter_complex',
                        f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.15[vid_audio];'
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice];'
                        f'[2:a]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[music];'
                        '[vid_audio][voice][music]amix=inputs=3:duration=longest:dropout_transition=2[out]',
                        '-map', '0:v', '-map', '[out]', '-c:v', 'copy', '-c:a', 'aac', merged_path
                    ]
                else:
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
                        '-i', music_files[0], '-i', music_files[1],
                        '-filter_complex',
                        '[2:a][3:a]concat=n=2:v=0:a=1[bg_raw];'
                        f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.15[vid_audio];'
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice];'
                        f'[bg_raw]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[music];'
                        '[vid_audio][voice][music]amix=inputs=3:duration=longest:dropout_transition=2[out]',
                        '-map', '0:v', '-map', '[out]', '-c:v', 'copy', '-c:a', 'aac', merged_path
                    ]
            else:
                logger.info("Voice-priority merge: video sound + voice (no background music)")
                cmd = [
                    'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
                    '-c:v', 'copy', '-filter_complex',
                    f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.2[vid_audio];'
                    f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice_audio];'
                    '[vid_audio][voice_audio]amix=inputs=2:duration=longest:dropout_transition=2[out]',
                    '-map', '0:v', '-map', '[out]', '-c:a', 'aac', merged_path
                ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                # If this fails too, try the basic approach
                logger.warning(f"Voice-priority mixing failed, trying basic approach: {result.stderr}")
                return self._merge_audio_with_video_basic(video_path, audio_path, task_id, music_files)

            if not os.path.exists(merged_path):
                raise Exception("Audio-merged video file was not created")

            if music_files and len(music_files) > 0:
                logger.info(f"Successfully merged audio with video using voice-priority (with {len(music_files)} music track(s))")
            else:
                logger.info(f"Successfully merged audio with video using voice-priority")
            return merged_path

        except Exception as e:
            logger.error(f"Failed to merge audio with video (voice-priority): {e}")
            raise

    def _merge_audio_with_video_basic(self, video_path: str, audio_path: str, task_id: str, music_files: List[str] = None) -> str:
        """Most basic fallback method for merging audio with video."""
        try:
            # Create temporary directory for merged video
            temp_dir = tempfile.mkdtemp()
            merged_path = os.path.join(temp_dir, "video_with_audio_basic.mp4")

            # Get durations for logging
            video_duration = self._get_video_duration(video_path)
            audio_duration = self._get_video_duration(audio_path) if audio_path else 0
            
            logger.info(f"📏 Duration Analysis (Basic):")
            logger.info(f"   - Video: {video_duration:.2f}s | Audio: {audio_duration:.2f}s")
            if audio_duration > video_duration:
                logger.warning(f"   ⚠️  Audio will be trimmed by {audio_duration - video_duration:.2f}s")

            vd = round(video_duration, 2)
            if music_files and len(music_files) > 0:
                logger.info(f"Basic merge with background music: {len(music_files)} track(s)")
                if len(music_files) == 1:
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-i', music_files[0],
                        '-filter_complex',
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice];'
                        f'[2:a]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[music];'
                        '[voice][music]amix=inputs=2:duration=longest[out]',
                        '-map', '0:v', '-map', '[out]', '-c:v', 'copy', '-c:a', 'aac',
                        merged_path
                    ]
                else:
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
                        '-i', music_files[0], '-i', music_files[1],
                        '-filter_complex',
                        '[2:a][3:a]concat=n=2:v=0:a=1[bg_raw];'
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=1.0[voice];'
                        f'[bg_raw]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[music];'
                        '[voice][music]amix=inputs=2:duration=longest[out]',
                        '-map', '0:v', '-map', '[out]', '-c:v', 'copy', '-c:a', 'aac',
                        merged_path
                    ]
            else:
                logger.info("Basic merge: voice only (no background music)")
                cmd = [
                    'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
                    '-filter_complex',
                    f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd}[out]',
                    '-map', '0:v', '-map', '[out]', '-c:v', 'copy', '-c:a', 'aac',
                    merged_path
                ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg audio merge failed (basic): {result.stderr}")

            if not os.path.exists(merged_path):
                raise Exception("Audio-merged video file was not created")

            if music_files and len(music_files) > 0:
                logger.info(f"Successfully merged audio with video using basic approach (voice + {len(music_files)} music track(s))")
            else:
                logger.info(f"Successfully merged audio with video using basic approach (voice only)")
            return merged_path

        except Exception as e:
            logger.error(f"Failed to merge audio with video (basic): {e}")
            raise

    def _merge_music_with_video(self, video_path: str, task_id: str, music_files: List[str]) -> str:
        """
        Merge only background music with video (no voice audio).
        Mixes video sound effects with background music.
        
        Args:
            video_path: Path to the video file
            task_id: Task ID for logging
            music_files: List of background music file paths (1 or 2 tracks)
            
        Returns:
            Path to the merged video file
        """
        try:
            # Create temporary directory for merged video
            temp_dir = tempfile.mkdtemp()
            merged_path = os.path.join(temp_dir, "video_with_music.mp4")

            # Check if video has audio stream
            has_audio = self._check_video_has_audio(video_path)
            
            video_duration = self._get_video_duration(video_path)
            vd = round(video_duration, 2)
            logger.info(f"📏 Video duration: {video_duration:.2f}s - Output will be exactly {vd}s")
            
            if has_audio:
                if len(music_files) == 1:
                    logger.info(f"Merging video audio + 1 background music track")
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', music_files[0],
                        '-filter_complex',
                        f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.2[vid_audio];'
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[bg_music];'
                        '[vid_audio][bg_music]amix=inputs=2:duration=longest:dropout_transition=2[out]',
                        '-map', '0:v', '-map', '[out]', '-c:v', 'copy', '-c:a', 'aac',
                        merged_path
                    ]
                else:
                    logger.info(f"Merging video audio + 2 background music tracks")
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path,
                        '-i', music_files[0], '-i', music_files[1],
                        '-filter_complex',
                        '[1:a][2:a]concat=n=2:v=0:a=1[bg_raw];'
                        f'[0:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.2[vid_audio];'
                        f'[bg_raw]atrim=0:{vd},asetpts=PTS-STARTPTS,volume=0.4[bg_music];'
                        '[vid_audio][bg_music]amix=inputs=2:duration=longest:dropout_transition=2[out]',
                        '-map', '0:v', '-map', '[out]', '-c:v', 'copy', '-c:a', 'aac',
                        merged_path
                    ]
            else:
                if len(music_files) == 1:
                    logger.info(f"Adding 1 background music track to silent video")
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', music_files[0],
                        '-filter_complex',
                        f'[1:a]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.4[out]',
                        '-map', '0:v', '-map', '[out]', '-c:v', 'copy', '-c:a', 'aac',
                        merged_path
                    ]
                else:
                    logger.info(f"Adding 2 background music tracks to silent video")
                    cmd = [
                        'ffmpeg', '-y', '-i', video_path,
                        '-i', music_files[0], '-i', music_files[1],
                        '-filter_complex',
                        '[1:a][2:a]concat=n=2:v=0:a=1[bg_raw];'
                        f'[bg_raw]atrim=0:{vd},asetpts=PTS-STARTPTS,apad=whole_dur={vd},volume=0.4[out]',
                        '-map', '0:v', '-map', '[out]', '-c:v', 'copy', '-c:a', 'aac',
                        merged_path
                    ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg music merge failed: {result.stderr}")

            if not os.path.exists(merged_path):
                raise Exception("Music-merged video file was not created")

            logger.info(f"Successfully merged background music with video")
            return merged_path

        except Exception as e:
            logger.error(f"Failed to merge background music with video: {e}")
            raise

    def _embed_subtitles(self, video_path: str, subtitles: List[Dict[str, Any]], task_id: str) -> str:
        """Embed subtitles into video using FFmpeg with SRT format."""
        try:
            # Create temporary directory at the same level as the app directory
            app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            temp_dir = os.path.join(app_dir, "temp_subtitles")
            
            # Create the temp directory if it doesn't exist
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            subtitled_path = os.path.join(temp_dir, "video_with_subtitles.mp4")

            # Create SRT file from subtitles data
            srt_path = os.path.join(temp_dir, "subtitles.srt")
            self._create_srt_file(subtitles, srt_path)

            # Convert Windows path to proper format for FFmpeg
            # Use forward slashes for FFmpeg on Windows - it handles them better
            escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
            
            # Get appropriate font for the language
            font_name = self._get_subtitle_font()
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f"subtitles='{escaped_srt_path}':force_style='FontSize=16,Bold=1,FontName={font_name},PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,Outline=1,Shadow=1,BackColour=&H000000&'",
                '-c:a', 'copy',  # Copy audio codec
                subtitled_path
            ]

            # Log the command for debugging
            logger.info(f"FFmpeg command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg subtitle embedding failed: {result.stderr}")

            if not os.path.exists(subtitled_path):
                raise Exception("Subtitled video file was not created")
            
            logger.info(f"Successfully embedded subtitles into video")

            # Don't clean up temporary directory here - let the caller handle cleanup
            # This ensures the file exists when the upload method tries to access it

            return subtitled_path

        except Exception as e:
            logger.error(f"Failed to embed subtitles: {e}")
            # Clean up temporary files
            try:
                if 'temp_dir' in locals() and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up temporary directory: {cleanup_error}")
            raise

    def _get_subtitle_font(self) -> str:
        """Get appropriate font for subtitles based on system."""
        try:
            # Common fonts that work well across different systems
            common_fonts = [
                'Arial',
                'Helvetica',
                'DejaVu Sans',
                'Liberation Sans',
                'FreeSans',
                'Verdana'
            ]
            
            # For Windows, Arial is usually available
            if os.name == 'nt':  # Windows
                return 'Arial'
            else:
                # For Unix-like systems, try to find a common font
                return 'DejaVu Sans'
                
        except Exception as e:
            logger.warning(f"Could not determine subtitle font, using default: {e}")
            return 'Arial'

    def _create_srt_file(self, subtitles: List[Dict[str, Any]], srt_path: str):
        """Create SRT file from subtitles data."""
        try:
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, subtitle in enumerate(subtitles, 1):
                    start_time = subtitle.get('start_time', 0)
                    end_time = subtitle.get('end_time', 0)
                    text = subtitle.get('text', '')

                    # Convert seconds to SRT time format (HH:MM:SS,mmm)
                    start_srt = self._seconds_to_srt_time(start_time)
                    end_srt = self._seconds_to_srt_time(end_time)

                    f.write(f"{i}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(f"{text}\n\n")

            logger.info(f"Created SRT file at {srt_path}")

        except Exception as e:
            logger.error(f"Failed to create SRT file: {e}")
            raise
        

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds % 1) * 1000)

            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

        except Exception as e:
            logger.error(f"Failed to convert seconds to SRT time: {e}")
            return "00:00:00,000"

    def _upload_final_video(self, video_path: str, short_id: str, task_id: str) -> str:
        """Upload final video. Supabase removed - raise; use local save or alternative storage."""
        raise NotImplementedError("Supabase removed. Save final video to local public folder or alternative storage.")

    def _update_shorts_final_video(self, short_id: str, final_video_url: str):
        """Update shorts with final video URL. Supabase removed - no-op."""
        logger.warning("Supabase removed: _update_shorts_final_video is a no-op.")

    def _cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files."""
        try:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)

                    # Also remove parent directory if empty
                    parent_dir = os.path.dirname(file_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                        logger.info(f"Cleaned up empty directory: {parent_dir}")

        except Exception as e:
            logger.warning(f"Failed to cleanup some temporary files: {e}")

    def _cleanup_subtitle_temp_dirs(self):
        """Clean up subtitle temporary directories that may have been created."""
        try:
            # Clean up the subtitle temp directory at the same level as the app directory
            app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            temp_dir = os.path.join(app_dir, "temp_subtitles")
            
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup subtitle temp directory: {e}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a finalization task."""
        try:
            return get_task_status(task_id)
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None

    def cleanup(self):
        """Clean up completed tasks and threads."""
        try:
            # Clean up completed threads
            completed_tasks = []
            for task_id, thread in self._active_threads.items():
                if not thread.is_alive():
                    completed_tasks.append(task_id)

            for task_id in completed_tasks:
                del self._active_threads[task_id]

            logger.info(
                f"Cleaned up {len(completed_tasks)} completed finalization tasks")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def test_supabase_connection(self) -> Dict[str, Any]:
        """Supabase removed."""
        return {"success": False, "error": "Supabase has been removed from this project."}


# Global instance
merging_service = MergingService()
