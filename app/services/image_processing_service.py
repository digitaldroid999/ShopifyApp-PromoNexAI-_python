"""
Image Processing Service

This module provides image processing capabilities including:
- Background removal using Remove.bg API
- Image compositing (merging two images)
- Saving composited images to local/Shopify public folder
"""

import os
import shutil
import uuid
import tempfile
import requests
import httpx
import subprocess
import math
import threading
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
import numpy as np


from app.logging_config import get_logger
from app.config import settings
from app.utils.task_management import create_task, get_task_status, complete_task, fail_task, start_task, TaskType, TaskStatus, task_manager

logger = get_logger(__name__)

# Timeout configurations
HTTP_TIMEOUT = 300  # 5 minutes for HTTP operations
DOWNLOAD_TIMEOUT = 600  # 10 minutes for file downloads
UPLOAD_TIMEOUT = 900  # 15 minutes for file uploads
MAX_RETRIES = 3  # Maximum retry attempts for failed operations
RETRY_DELAY = 5  # Seconds to wait between retries

# Remove.bg API configuration
REMOVEBG_API_KEY = os.getenv("REMOVEBG_API_KEY", "")
REMOVEBG_API_URL = "https://api.remove.bg/v1.0/removebg"


class ImageProcessingService:
    """Service for processing images including background removal and compositing."""

    def __init__(self):
        self._temp_dir = self._get_temp_dir()
        self._check_ffmpeg()
        self._active_threads = {}  # Track background threads for image merge tasks

    def _get_temp_dir(self) -> Path:
        """Get or create the temp directory for temporary files."""
        project_root = Path(__file__).parent.parent.parent
        temp_dir = project_root / "temp"
        temp_dir.mkdir(exist_ok=True)
        return temp_dir
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("✅ FFmpeg is available for video processing")
            else:
                logger.warning("⚠️  FFmpeg check returned non-zero exit code")
        except FileNotFoundError:
            logger.error("❌ FFmpeg not found! Video merge will fail. Install from: https://ffmpeg.org/download.html")
        except Exception as e:
            logger.warning(f"⚠️  Could not check FFmpeg availability: {e}")

    def remove_background(self, image_url: str, scene_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Remove background from an image using Remove.bg API.

        Args:
            image_url: URL of the image to process
            scene_id: Optional scene ID for organizing files

        Returns:
            Dict containing:
                - success (bool): Whether the operation succeeded
                - image_url (str): None (Supabase removed); use local_path for the file
                - local_path (str): Local path of the processed image (if successful)
                - error (str): Error message (if failed)
        """
        temp_input_path = None
        temp_output_path = None

        try:
            logger.info(f"Starting background removal for image: {image_url}")

            # Validate API key
            if not REMOVEBG_API_KEY:
                raise Exception("Remove.bg API key not configured. Please set REMOVEBG_API_KEY in .env file")

            # Download the image from URL
            logger.info("Downloading image from URL...")
            temp_input_path = self._download_image_from_url(image_url)

            # Call Remove.bg API
            logger.info("Calling Remove.bg API...")
            temp_output_path = self._call_removebg_api(temp_input_path)

            logger.info("Background removal completed successfully.")

            return {
                'success': True,
                'image_url': None,
                'local_path': temp_output_path,
                'error': None
            }

        except Exception as e:
            error_msg = f"Background removal failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'image_url': None,
                'local_path': None,
                'error': error_msg
            }

        finally:
            # Clean up temporary input file
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    os.unlink(temp_input_path)
                    logger.debug(f"Cleaned up temporary input file: {temp_input_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary input file {temp_input_path}: {cleanup_error}")

    def _call_removebg_api(self, image_path: str) -> str:
        """
        Call Remove.bg API to remove background from image.

        Args:
            image_path: Path to the local image file

        Returns:
            Path to the output image file with background removed
        """
        try:
            # Prepare output path
            output_path = str(self._temp_dir / f"no-bg-{uuid.uuid4()}.png")

            # Call Remove.bg API
            with open(image_path, 'rb') as image_file:
                response = requests.post(
                    REMOVEBG_API_URL,
                    files={'image_file': image_file},
                    data={'size': 'auto'},
                    headers={'X-Api-Key': REMOVEBG_API_KEY},
                    timeout=HTTP_TIMEOUT
                )

            # Check response
            if response.status_code == requests.codes.ok:
                # Save the result
                with open(output_path, 'wb') as out_file:
                    out_file.write(response.content)
                logger.info(f"Background removed successfully. Output saved to: {output_path}")
                return output_path
            else:
                error_msg = f"Remove.bg API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except requests.exceptions.Timeout:
            raise Exception("Remove.bg API request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Remove.bg API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to call Remove.bg API: {str(e)}")

    def _match_color_temperature(self, overlay, background, strength=0.4):
        """
        🎯 NEW FEATURE: Color Temperature Matching
        Adjusts overlay to match background lighting (warm/cool).
        strength: 0 = no change, 1 = full effect (default 0.4 = subtle, user-identifiable).
        """
        # Convert to numpy for analysis
        bg_array = np.array(background.convert("RGB"))
        overlay_array = np.array(overlay.convert("RGBA"))
        
        # Calculate average color of background
        avg_color = np.mean(bg_array.reshape(-1, 3), axis=0)
        
        # Determine temperature (B/R ratio)
        blue_red_ratio = avg_color[2] / (avg_color[0] + 1)  # R/B in PIL
        
        # Extract RGB and alpha
        r, g, b, a = overlay.split()
        r_orig = np.array(r, dtype=float)
        g_orig = np.array(g, dtype=float)
        b_orig = np.array(b, dtype=float)
        r_arr, g_arr, b_arr = r_orig.copy(), g_orig.copy(), b_orig.copy()
        
        if blue_red_ratio < 0.9:  # Warm (more red)
            r_arr *= 1.12
            g_arr *= 1.05
            b_arr *= 0.92
        elif blue_red_ratio > 1.1:  # Cool (more blue)
            r_arr *= 0.92
            g_arr *= 1.05
            b_arr *= 1.12
        # else: neutral, no adjustment
        
        # Blend with original so effect is subtle but identifiable
        r_arr = strength * np.clip(r_arr, 0, 255) + (1 - strength) * r_orig
        g_arr = strength * np.clip(g_arr, 0, 255) + (1 - strength) * g_orig
        b_arr = strength * np.clip(b_arr, 0, 255) + (1 - strength) * b_orig
        
        r = Image.fromarray(np.clip(r_arr, 0, 255).astype(np.uint8))
        g = Image.fromarray(np.clip(g_arr, 0, 255).astype(np.uint8))
        b = Image.fromarray(np.clip(b_arr, 0, 255).astype(np.uint8))
        
        return Image.merge("RGBA", (r, g, b, a))
    
    
    def _match_histogram(self, overlay, background, strength=0.5):
        """
        🎯 NEW FEATURE: Histogram Matching
        Makes overlay colors blend naturally with background
        """
        bg_array = np.array(background.convert("RGB"))
        overlay_array = np.array(overlay.convert("RGBA"))
        
        # Extract alpha
        alpha = overlay_array[:, :, 3]
        
        # Match each RGB channel
        for channel in range(3):
            # Get histograms
            bg_hist, _ = np.histogram(bg_array[:, :, channel].flatten(), 256, [0, 256])
            overlay_hist, _ = np.histogram(overlay_array[:, :, channel].flatten(), 256, [0, 256])
            
            # Calculate CDFs
            bg_cdf = bg_hist.cumsum()
            bg_cdf = bg_cdf / bg_cdf[-1]
            
            overlay_cdf = overlay_hist.cumsum()
            overlay_cdf = overlay_cdf / (overlay_cdf[-1] + 1e-7)
            
            # Create lookup table
            lookup = np.interp(overlay_cdf, bg_cdf, np.arange(256))
            
            # Apply lookup
            matched = np.interp(
                overlay_array[:, :, channel].flatten(),
                np.arange(256),
                lookup
            ).reshape(overlay_array[:, :, channel].shape)
            
            # Blend with original based on strength
            original = overlay_array[:, :, channel].astype(float)
            overlay_array[:, :, channel] = (
                strength * matched + (1 - strength) * original
            ).astype(np.uint8)
        
        # Restore alpha
        overlay_array[:, :, 3] = alpha
        
        return Image.fromarray(overlay_array, "RGBA")
    
    
    def _adjust_brightness_contrast(self, overlay, brightness=1.0, contrast=1.0, strength=0.4):
        """
        🎯 NEW FEATURE: Brightness/Contrast Matching
        Adjusts overlay exposure to match background.
        strength: 0 = no change, 1 = full effect (default 0.4 = subtle, user-identifiable).
        """
        # Damp adjustments toward 1.0 so effect is subtle but identifiable
        effective_brightness = 1.0 + (brightness - 1.0) * strength
        effective_contrast = 1.0 + (contrast - 1.0) * strength

        r, g, b, a = overlay.split()
        
        # Apply brightness
        if effective_brightness != 1.0:
            r = ImageEnhance.Brightness(r).enhance(effective_brightness)
            g = ImageEnhance.Brightness(g).enhance(effective_brightness)
            b = ImageEnhance.Brightness(b).enhance(effective_brightness)
        
        # Apply contrast
        if effective_contrast != 1.0:
            r = ImageEnhance.Contrast(r).enhance(effective_contrast)
            g = ImageEnhance.Contrast(g).enhance(effective_contrast)
            b = ImageEnhance.Contrast(b).enhance(effective_contrast)
        
        return Image.merge("RGBA", (r, g, b, a))
    
    
    def _create_cast_shadow(self, overlay, angle=135, distance=30, opacity=0.35, blur=40):
        """
        🎯 NEW FEATURE: Directional Cast Shadow with Perspective
        Creates realistic angled shadow matching light direction
        
        Args:
            angle: Shadow direction in degrees (0=right, 90=down, 180=left, 270=up)
            distance: Shadow distance in pixels
            opacity: Shadow darkness (0-1)
            blur: Shadow blur radius
        """
        # Extract alpha channel as shadow base
        alpha = overlay.split()[3]
        
        # Calculate shadow offset
        angle_rad = math.radians(angle)
        offset_x = int(distance * math.cos(angle_rad))
        offset_y = int(distance * math.sin(angle_rad))
        
        # Create larger canvas for shadow
        canvas_w = overlay.width + abs(offset_x) + blur * 2
        canvas_h = overlay.height + abs(offset_y) + blur * 2
        
        # Create shadow canvas
        shadow = Image.new("L", (canvas_w, canvas_h), 0)
        
        # Paste alpha as shadow
        paste_x = blur + max(0, offset_x)
        paste_y = blur + max(0, offset_y)
        shadow.paste(alpha, (paste_x, paste_y))
        
        # Apply perspective distortion (shadow stretches away from light)
        shadow_array = np.array(shadow, dtype=np.uint8)
        h, w = shadow_array.shape
        
        # Simple perspective: stretch bottom if light from top
        if 45 < angle < 135:  # Light from top-right to top-left
            # Stretch bottom horizontally
            perspective_factor = 0.2
            for y in range(h):
                progress = y / h
                stretch = int(w * perspective_factor * progress)
                if stretch > 0:
                    row = shadow_array[y, :]
                    stretched = np.interp(
                        np.linspace(0, len(row), len(row) + stretch),
                        np.arange(len(row)),
                        row
                    )
                    # Center the stretched row
                    start = stretch // 2
                    shadow_array[y, :] = stretched[start:start + w]
        
        shadow = Image.fromarray(shadow_array, "L")
        
        # Apply blur for soft edges
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
        
        # Apply opacity
        shadow_array = np.array(shadow, dtype=float)
        shadow_array = (shadow_array * opacity).astype(np.uint8)
        shadow = Image.fromarray(shadow_array, "L")
        
        # Convert to RGBA
        shadow_rgba = Image.new("RGBA", shadow.size, (0, 0, 0, 255))
        shadow_rgba.putalpha(shadow)
        
        return shadow_rgba, (offset_x, offset_y)
    
    def _add_rim_light(self, overlay, light_angle=135, intensity=0.25, thickness=4):
        """
        🎯 Rim Lighting (Edge Highlights) - NO SCIPY REQUIRED
        Adds subtle bright edge on lit side of product
        Uses PIL's built-in edge detection
        
        Args:
            light_angle: Direction light comes from (degrees)
            intensity: Edge brightness (0-1)
            thickness: Edge thickness in pixels
        """
        try:
            # Extract alpha for edge detection
            alpha = overlay.split()[3]
            
            # Use PIL's FIND_EDGES filter (fast and built-in!)
            edges = alpha.filter(ImageFilter.FIND_EDGES)
            
            # Dilate edges using blur instead of MaxFilter (more compatible)
            if thickness > 1:
                # Multiple blur passes to expand edges
                for _ in range(thickness):
                    edges = edges.filter(ImageFilter.BLUR)
            
            edges_array = np.array(edges, dtype=float) / 255.0
            
            # Calculate which edges should be lit based on angle
            h, w = edges_array.shape
            y_grad, x_grad = np.mgrid[0:h, 0:w]
            
            # Normalize to -1 to 1
            x_grad = (x_grad / w) * 2 - 1
            y_grad = (y_grad / h) * 2 - 1
            
            # Calculate edge brightness based on light angle
            angle_rad = math.radians(light_angle)
            edge_brightness = (
                x_grad * math.cos(angle_rad) + 
                y_grad * math.sin(angle_rad)
            )
            edge_brightness = np.clip(edge_brightness, 0, 1)
            
            # Combine edge detection with directional brightness
            rim_mask = edges_array * edge_brightness * intensity
            
            # Apply rim light to RGB channels
            r, g, b, a = overlay.split()
            r_arr = np.array(r, dtype=float)
            g_arr = np.array(g, dtype=float)
            b_arr = np.array(b, dtype=float)
            
            # Brighten edges
            r_arr = np.clip(r_arr + rim_mask * 255, 0, 255)
            g_arr = np.clip(g_arr + rim_mask * 255, 0, 255)
            b_arr = np.clip(b_arr + rim_mask * 255, 0, 255)
            
            r = Image.fromarray(r_arr.astype(np.uint8))
            g = Image.fromarray(g_arr.astype(np.uint8))
            b = Image.fromarray(b_arr.astype(np.uint8))
            
            return Image.merge("RGBA", (r, g, b, a))
            
        except Exception as e:
            # If rim lighting fails, just return original overlay
            logger.warning(f"Rim lighting skipped: {e}")
            return overlay

    def _create_improved_ao(self, overlay, intensity=0.4):
        """
        🎯 IMPROVED: Better Ambient Occlusion with Gradient
        Creates darker shadow at base, fading upward
        """
        alpha = overlay.split()[3]
        
        # Create vertical gradient (darker at bottom)
        h, w = overlay.size[1], overlay.size[0]
        gradient = np.linspace(1, 0, h) ** 2  # Quadratic falloff
        gradient = np.tile(gradient.reshape(-1, 1), (1, w))
        
        # Apply to alpha channel
        alpha_array = np.array(alpha, dtype=float) / 255.0
        ao_mask = alpha_array * gradient * intensity
        
        # Create AO layer
        ao = Image.new("RGBA", overlay.size, (0, 0, 0, 255))
        ao_alpha = (ao_mask * 255).astype(np.uint8)
        ao.putalpha(Image.fromarray(ao_alpha))
        
        return ao

    def composite_images_to_public_folder(
        self,
        background_url: str,
        overlay_url: str,
        user_id: str,
        scene_id: str,
        position: Tuple[int, int] = (0, 0),
        resize_overlay: bool = True,
    ) -> Dict[str, Any]:
        """
        Composite two images and save to public folder under composited_images/{user_id}/{scene_id}/{file_name}.
        Returns relative URL: composited_images/{user_id}/{scene_id}/{file_name}.

        Args:
            background_url: URL of the background image
            overlay_url: URL of the overlay image (e.g. product with transparent background)
            user_id: Shopify user ID (string form for path)
            scene_id: Scene ID for organizing files
            position: (x, y) position to place overlay (0,0 = auto-center)
            resize_overlay: Whether to resize overlay to fit background

        Returns:
            Dict with success, image_url (e.g. "composited_images/{user_id}/{scene_id}/{file_name}"), error
        """
        temp_bg_path = None
        temp_overlay_path = None
        temp_output_path = None
        base_dir = getattr(settings, "PUBLIC_OUTPUT_BASE", None) or getattr(settings, "COMPOSITED_IMAGES_OUTPUT_DIR", None)
        if not base_dir:
            return {"success": False, "image_url": None, "error": "PUBLIC_OUTPUT_BASE not configured"}
        base_dir = Path(base_dir)
        # Save under composited_images/{user_id}/{scene_id}/{file_name}
        output_dir = base_dir / "composited_images" / user_id / scene_id
        try:
            logger.info("Compositing images to Shopify public folder: %s/%s/...", user_id, scene_id)
            temp_bg_path = self._download_image_from_url(background_url)
            temp_overlay_path = self._download_image_from_url(overlay_url)
            temp_output_path = self._composite_images_locally(
                background_path=temp_bg_path,
                overlay_path=temp_overlay_path,
                position=position,
                resize_overlay=resize_overlay,
                shadow_angle=135,
                shadow_distance=30,
                enable_color_matching=True,
                enable_histogram_matching=True,
                enable_rim_light=True,
                brightness_adjust=1.0,
                contrast_adjust=1.05,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"composited-{uuid.uuid4().hex[:12]}.png"
            dest_path = output_dir / file_name
            shutil.copy2(temp_output_path, str(dest_path))
            # Return relative URL without leading slash: composited_images/{user_id}/{scene_id}/{file_name}
            image_url = f"composited_images/{user_id}/{scene_id}/{file_name}"
            logger.info("Saved composited image to %s, URL: %s", dest_path, image_url)
            return {"success": True, "image_url": image_url, "error": None}
        except Exception as e:
            error_msg = str(e)
            logger.error("Composite to public folder failed: %s", error_msg, exc_info=True)
            return {"success": False, "image_url": None, "error": error_msg}
        finally:
            for temp_path in [temp_bg_path, temp_overlay_path, temp_output_path]:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

    def _composite_images_locally(
        self,
        background_path: str,
        overlay_path: str,
        position=(0, 0),
        resize_overlay=True,
        landscape_width=1920,
        landscape_height=1080,
        add_reflection=True,
        # 🆕 NEW PARAMETERS
        shadow_angle=135,  # Light direction (135 = top-left, 45 = top-right)
        shadow_distance=30,  # How far shadow extends
        enable_color_matching=True,  # Match colors to background
        enable_histogram_matching=True,  # Blend color distributions
        enable_rim_light=True,  # Add edge highlights
        brightness_adjust=1.0,  # Brightness multiplier
        contrast_adjust=1.05,  # Contrast multiplier
    ) -> str:
        """
        ENHANCED compositing with all professional features
        """
        
        # -----------------------------
        # 1️⃣ Load images
        # -----------------------------
        background = Image.open(background_path).convert("RGB")
        overlay = Image.open(overlay_path).convert("RGBA")

        # -----------------------------
        # 2️⃣ Resize background without distortion
        # -----------------------------
        bg_ratio = background.width / background.height
        target_ratio = landscape_width / landscape_height

        if bg_ratio > target_ratio:
            new_height = landscape_height
            new_width = int(bg_ratio * new_height)
        else:
            new_width = landscape_width
            new_height = int(new_width / bg_ratio)

        background = background.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_width - landscape_width) // 2
        top = (new_height - landscape_height) // 2
        background = background.crop((
            left, top, left + landscape_width, top + landscape_height
        ))
        background = background.convert("RGBA")

        # -----------------------------
        # 3️⃣ Resize overlay proportionally
        # -----------------------------
        if resize_overlay:
            max_width = int(landscape_width * 0.6)
            max_height = int(landscape_height * 0.6)
            scale = min(max_width / overlay.width, max_height / overlay.height)
            overlay = overlay.resize(
                (int(overlay.width * scale), int(overlay.height * scale)),
                Image.Resampling.LANCZOS
            )

        # -----------------------------
        # 🆕 3.5️⃣ COLOR & LIGHTING MATCHING
        # -----------------------------
        if enable_color_matching:
            print("🎨 Matching color temperature...")
            overlay = self._match_color_temperature(overlay, background, strength=0)
        
        if enable_histogram_matching:
            print("🎨 Matching histogram...")
            overlay = self._match_histogram(overlay, background, strength=0.1)
        
        if brightness_adjust != 1.0 or contrast_adjust != 1.0:
            print(f"💡 Adjusting brightness ({brightness_adjust}) & contrast ({contrast_adjust})...")
            overlay = self._adjust_brightness_contrast(
                overlay, 
                brightness=brightness_adjust, 
                contrast=contrast_adjust,
                strength=0
            )

        # -----------------------------
        # 4️⃣ Smooth overlay edges
        # -----------------------------
        r, g, b, a = overlay.split()
        a = a.filter(ImageFilter.GaussianBlur(1))
        overlay = Image.merge("RGBA", (r, g, b, a))

        # -----------------------------
        # 🆕 4.5️⃣ ADD RIM LIGHTING
        # -----------------------------
        if enable_rim_light:
            print("✨ Adding rim lighting...")
            overlay = self._add_rim_light(
                overlay, 
                light_angle=shadow_angle - 180,  # Opposite of shadow
                intensity=0.25,
                thickness=4
            )

        # -----------------------------
        # 5️⃣ Determine shadow opacity based on background brightness
        # -----------------------------
        bg_gray = np.mean(np.array(background.convert("L")))
        if bg_gray > 180:
            shadow_opacity = 0.25
        elif bg_gray > 120:
            shadow_opacity = 0.35
        else:
            shadow_opacity = 0.45

        # -----------------------------
        # 6️⃣ Center overlay if no position provided
        # -----------------------------
        if position == (0, 0):
            position = (
                (landscape_width - overlay.width) // 2,
                (landscape_height - overlay.height) // 2
            )

        composited = Image.new("RGBA", background.size)
        composited.paste(background, (0, 0))

        alpha = overlay.split()[3]

        # -----------------------------
        # 🆕 6.5️⃣ DIRECTIONAL CAST SHADOW (canvas = background size so shadow isn't cropped by overlay)
        # -----------------------------
        print(f"🌑 Creating cast shadow (angle: {shadow_angle}°, distance: {shadow_distance}px)...")
        cast_shadow, shadow_offset = self._create_cast_shadow(
            overlay,
            angle=shadow_angle,
            distance=shadow_distance,
            opacity=shadow_opacity * 0.9,
            blur=45
        )
        cast_shadow_pos = (
            position[0] + shadow_offset[0] - 45,  # Account for blur padding
            position[1] + shadow_offset[1] - 45
        )
        if cast_shadow_pos[0] + cast_shadow.width > 0 and cast_shadow_pos[1] + cast_shadow.height > 0:
            cast_canvas = Image.new("RGBA", background.size, (0, 0, 0, 0))
            cast_canvas.paste(cast_shadow, cast_shadow_pos, cast_shadow)
            composited.paste(cast_canvas, (0, 0), cast_canvas)

        # -----------------------------
        # 7️⃣ Contact shadow (canvas = background size so shadow isn't cropped by overlay)
        # -----------------------------
        print("🌑 Creating contact shadow...")
        contact_shadow = Image.new("RGBA", overlay.size, (0, 0, 0, 255))
        contact_shadow.putalpha(alpha)
        new_h = int(overlay.height * 0.15)
        contact_shadow = contact_shadow.resize((overlay.width, new_h), Image.Resampling.LANCZOS)
        contact_shadow = contact_shadow.filter(ImageFilter.GaussianBlur(8))
        shadow_alpha = contact_shadow.split()[3]
        shadow_alpha = ImageEnhance.Brightness(shadow_alpha).enhance(shadow_opacity * 1.4)
        contact_shadow.putalpha(shadow_alpha)
        contact_position = (position[0], position[1] + overlay.height - contact_shadow.height)
        contact_canvas = Image.new("RGBA", background.size, (0, 0, 0, 0))
        contact_canvas.paste(contact_shadow, contact_position, contact_shadow)
        composited.paste(contact_canvas, (0, 0), contact_canvas)

        # -----------------------------
        # 🆕 7.5️⃣ IMPROVED AMBIENT OCCLUSION (canvas = background size so shadow isn't cropped by overlay)
        # -----------------------------
        print("🌑 Adding ambient occlusion...")
        ao = self._create_improved_ao(overlay, intensity=0.4)
        ao_position = (position[0], position[1] + int(overlay.height * 0.4))
        ao_resized = ao.resize((overlay.width, int(overlay.height * 0.6)), Image.Resampling.LANCZOS)
        ao_resized = ao_resized.filter(ImageFilter.GaussianBlur(30))
        ao_canvas = Image.new("RGBA", background.size, (0, 0, 0, 0))
        ao_canvas.paste(ao_resized, ao_position, ao_resized)
        composited.paste(ao_canvas, (0, 0), ao_canvas)

        # -----------------------------
        # 9️⃣ Optional reflection
        # -----------------------------
        if add_reflection:
            print("✨ Adding reflection...")
            reflection = overlay.copy().transpose(Image.FLIP_TOP_BOTTOM)
            r, g, b, a = reflection.split()
            a = ImageEnhance.Brightness(a).enhance(0.15)
            a = a.filter(ImageFilter.GaussianBlur(5))
            reflection.putalpha(a)
            reflection_position = (position[0], position[1] + overlay.height)
            composited.paste(reflection, reflection_position, reflection)

        # -----------------------------
        # 🔟 Paste overlay
        # -----------------------------
        composited.paste(overlay, position, overlay)

        # -----------------------------
        # 1️⃣1️⃣ Adaptive brightness / color blending
        # -----------------------------
        avg_color = np.mean(np.array(background.convert("RGB")).reshape(-1, 3), axis=0)
        tint_layer = Image.new("RGBA", composited.size,
                            (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]), 10))
        composited = Image.alpha_composite(composited, tint_layer)

        # -----------------------------
        # 1️⃣2️⃣ Add subtle grain / noise
        # -----------------------------
        final = composited.convert("RGB")
        np_img = np.array(final)
        noise = np.random.randint(-1, 2, np_img.shape, dtype='int16')
        np_img = np.clip(np_img + noise, 0, 255).astype('uint8')
        final = Image.fromarray(np_img)

        # -----------------------------
        # 1️⃣3️⃣ Save
        # -----------------------------
        output_path = str(self._temp_dir / f"composited-{uuid.uuid4()}.png")
        final.save(output_path, "PNG", optimize=True)
        
        print(f"✅ Composite saved: {output_path}")

        return output_path

    def _download_image_from_url(self, image_url: str) -> str:
        """
        Download image from URL to local file.

        Args:
            image_url: URL of the image to download

        Returns:
            Path to the downloaded image file
        """
        try:
            logger.info(f"Downloading image from URL: {image_url}")

            # Generate temp file path
            temp_path = str(self._temp_dir / f"download-{uuid.uuid4()}.png")

            with httpx.Client(timeout=DOWNLOAD_TIMEOUT) as client:
                response = client.get(image_url)
                response.raise_for_status()

                # Save to local file
                with open(temp_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Successfully downloaded image to: {temp_path}")
                return temp_path

        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
            raise Exception(f"Failed to download image: {e}")

    def _upload_to_supabase(
        self,
        file_path: str,
        folder: str = "processed-images",
        scene_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """No-op: Supabase removed. Returns None."""
        return None

    def _update_scene_image_url(self, scene_id: str, image_url: str) -> None:
        """No-op: Supabase removed."""
        pass


    def replace_background(
        self,
        product_image_url: str,
        background_image_url: str,
        scene_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Complete workflow: Remove background from product image and composite with new background.

        Args:
            product_image_url: URL of the product image
            background_image_url: URL of the new background image
            scene_id: Scene ID for organizing files and updating database
            user_id: User ID for organizing files

        Returns:
            Dict containing:
                - success (bool): Whether the operation succeeded
                - image_url (str): None
                - error (str): Error message (if failed)
        """
        try:
            logger.info(f"Starting background replacement for scene {scene_id}")

            # Step 1: Remove background from product image
            logger.info("Step 1: Removing background from product image...")
            remove_result = self.remove_background(product_image_url, scene_id)

            if not remove_result['success']:
                raise Exception(f"Background removal failed: {remove_result['error']}")

            no_bg_local_path = remove_result['local_path']

            # Step 2: Download background image
            logger.info("Step 2: Downloading background image...")
            temp_bg_path = self._download_image_from_url(background_image_url)

            # Step 3: Composite the images
            logger.info("Step 3: Compositing images...")
            temp_output_path = self._composite_images_locally(
                background_path=temp_bg_path,
                overlay_path=no_bg_local_path,
                position=(0, 0),  # Center the product
                resize_overlay=True,
                shadow_angle=135,
                shadow_distance=30,
                enable_color_matching=True,
                enable_histogram_matching=True,
                enable_rim_light=True,
                brightness_adjust=1.0,
                contrast_adjust=1.05,
            )

            logger.info("Background replacement completed successfully.")

            return {
                'success': True,
                'image_url': None,
                'error': None
            }

        except Exception as e:
            error_msg = f"Background replacement failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'image_url': None,
                'error': error_msg
            }

        finally:
            # Clean up temporary files
            for temp_path in [no_bg_local_path, temp_bg_path, temp_output_path]:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                        logger.debug(f"Cleaned up temporary file: {temp_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary file {temp_path}: {cleanup_error}")

    def start_image_merge_task(
        self,
        product_image_url: str,
        background_video_url: str,
        scene_id: str,
        user_id: str,
        short_id: str,
        scale: float = 0.4,
        position: str = "center",
        duration: Optional[int] = None,
        add_animation: bool = True
    ) -> Dict[str, Any]:
        """
        Start an async image-video merge task (Scene2 generation).
        Returns immediately with a task_id for polling.
        
        Args:
            product_image_url: URL of the product image
            background_video_url: URL of the background video
            scene_id: Scene ID
            user_id: User ID
            short_id: Short ID (for task creation)
            scale: Product scale relative to video width
            position: Product position on video
            duration: Optional duration in seconds
            add_animation: Whether to add animations
            
        Returns:
            Dict with task_id, status, and message
        """
        try:
            logger.info(f"🎬 Creating async image merge task for scene {scene_id}")
            
            # Create task using task management system
            task_id = create_task(
                task_type=TaskType.VIDEO_GENERATION,  # Reuse VIDEO_GENERATION type
                user_id=user_id,
                scene_id=scene_id,
                short_id=short_id,
                task_name="Scene2 Image-Video Merge",
                description=f"Merge product image with background video for scene {scene_id}"
            )
            
            if not task_id:
                raise Exception("Failed to create image merge task")
            
            logger.info(f"✅ Created task {task_id} for scene {scene_id}")
            
            # Start background thread
            thread = threading.Thread(
                target=self._process_image_merge_task,
                args=(
                    task_id,
                    product_image_url,
                    background_video_url,
                    scene_id,
                    user_id,
                    short_id,
                    scale,
                    position,
                    duration,
                    add_animation
                ),
                daemon=True,
                name=f"image_merge_{task_id}"
            )
            thread.start()
            
            # Store thread reference
            self._active_threads[task_id] = thread
            
            logger.info(f"🚀 Started background thread for task {task_id}")
            
            return {
                "task_id": task_id,
                "status": "pending",
                "scene_id": scene_id,
                "user_id": user_id,
                "message": "Image-video merge task started",
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to start image merge task: {e}", exc_info=True)
            raise
    
    def _process_image_merge_task(
        self,
        task_id: str,
        product_image_url: str,
        background_video_url: str,
        scene_id: str,
        user_id: str,
        short_id: str,
        scale: float,
        position: str,
        duration: Optional[int],
        add_animation: bool
    ):
        """
        Background worker that processes the image-video merge.
        Updates task status as it progresses.
        """
        try:
            logger.info("[Scene2] TASK picked up | task_id=%s scene_id=%s short_id=%s", task_id, scene_id, short_id)
            logger.info("🔄 Processing image merge task %s", task_id)

            # Update task to processing
            start_task(task_id)
            
            # Perform the actual merge (this is the synchronous method)
            result = self.merge_image_with_video(
                product_image_url=product_image_url,
                background_video_url=background_video_url,
                scene_id=scene_id,
                user_id=user_id,
                short_id=short_id or scene_id,
                scale=scale,
                position=position,
                duration=duration,
                add_animation=add_animation
            )
            
            # Update task based on result
            if result['success']:
                complete_task(task_id, metadata={"video_url": result['video_url']})
                logger.info(f"✅ Task {task_id} completed successfully")
            else:
                fail_task(task_id, error_message=result.get('error', 'Unknown error'))
                logger.error(f"❌ Task {task_id} failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error processing image merge task {task_id}: {e}", exc_info=True)
            fail_task(task_id, error_message=str(e))
        finally:
            # Clean up thread reference
            if task_id in self._active_threads:
                del self._active_threads[task_id]
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an image merge task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Dict with task information or None if not found
        """
        try:
            task_info = get_task_status(task_id)
            if not task_info:
                return None
            
            # Convert Task object to dict if needed
            if not isinstance(task_info, dict):
                # Convert TaskStatus enum to string
                status = task_info.task_status
                if hasattr(status, 'value'):
                    status = status.value
                
                # Extract scene_id from task_metadata
                metadata = task_info.task_metadata or {}
                scene_id = metadata.get('scene_id') if isinstance(metadata, dict) else None
                    
                task_info_dict = {
                    "task_id": task_info.task_id,
                    "status": status,
                    "scene_id": scene_id,
                    "user_id": task_info.user_id,
                    "message": task_info.task_status_message or "",
                    "created_at": task_info.created_at.isoformat() if task_info.created_at else None,
                    "updated_at": task_info.updated_at.isoformat() if task_info.updated_at else None,
                    "error_message": task_info.error_message,
                    "metadata": metadata
                }
            else:
                task_info_dict = task_info
                # Convert status enum to string if needed
                status = task_info_dict.get('status')
                if hasattr(status, 'value'):
                    task_info_dict['status'] = status.value
            
            # Format response similar to video generation tasks
            response = {
                "task_id": task_id,
                "status": task_info_dict.get('status'),
                "scene_id": task_info_dict.get('scene_id'),
                "user_id": task_info_dict.get('user_id'),
                "message": task_info_dict.get('message', ''),
                "created_at": task_info_dict.get('created_at'),
                "updated_at": task_info_dict.get('updated_at'),
                "error_message": task_info_dict.get('error_message')
            }
            
            # Add video_url if completed
            status_str = str(response.get('status', '')).lower()
            if status_str == 'completed' or status_str == TaskStatus.COMPLETED.value:
                metadata = task_info_dict.get('metadata', {})
                if isinstance(metadata, dict):
                    response['video_url'] = metadata.get('video_url')
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error getting task status for {task_id}: {e}", exc_info=True)
            return None
   
    def merge_image_with_video(
        self,
        product_image_url: str,
        background_video_url: str,
        scene_id: str,
        user_id: str,
        short_id: Optional[str] = None,
        scale: float = 0.4,
        position: str = "center",
        duration: Optional[int] = None,
        add_animation: bool = True,
        add_shadow: bool = True,
        shadow_blur_radius: int = 25,
        shadow_offset: Tuple[int, int] = (15, 15)
    ) -> Dict[str, Any]:
        """
        Merge a product image (without background) with a background video using OpenCV.
        Saves the result to public folder: generated_videos/{user_id}/{short_id}/scene2/{file_name}.
        
        Args:
            product_image_url: URL of the product image (PNG with transparent background)
            background_video_url: URL of the background video
            scene_id: Scene ID for organizing files
            user_id: User ID for organizing files
            short_id: Short ID for path (defaults to scene_id if not provided)
            scale: Scale of product relative to video width (default: 0.4)
            position: Position of product ("center", "top", "bottom", "left", "right")
            duration: Optional duration in seconds (if None, uses full video duration)
            add_animation: Whether to add zoom and floating animations
            add_shadow: Whether to add shadow effect to the product
            shadow_blur_radius: Blur radius for shadow (default: 25)
            shadow_offset: Shadow offset as (x, y) tuple (default: (15, 15))
        
        Returns:
            Dict with success, video_url (e.g. "generated_videos/{user_id}/{short_id}/scene2/{file_name}"), error
        """
        short_id = short_id or scene_id
        temp_product_path = None
        temp_video_path = None
        temp_output_path = None
        temp_shadow_product_path = None
        
        try:
            print("\n" + "="*80)
            print("🎬 SCENE 2 GENERATION - VIDEO MERGE STARTED")
            print("="*80)
            logger.info("[Scene2] START | scene_id=%s user_id=%s", scene_id, user_id)
            logger.info("[Scene2] Parameters: scale=%s position=%s duration=%s animation=%s shadow=%s blur=%s offset=%s",
                       scale, position, duration, add_animation, add_shadow, shadow_blur_radius if add_shadow else None, shadow_offset if add_shadow else None)
            logger.info("🚀 Starting image-video merge for scene %s", scene_id)
            logger.info("📦 Parameters: Scene ID=%s, User ID=%s, Scale=%s%%, Position=%s, Duration=%s, Animation=%s, Shadow=%s",
                       scene_id, user_id, round(scale * 100), position, duration, add_animation, add_shadow)

            # Step 1: Download product image
            logger.info("[Scene2] Step 1/6: Downloading product image - started")
            print("\n📥 STEP 1/6: Downloading Product Image")
            print("-" * 80)
            logger.info("🖼️  Product image URL: %s...", product_image_url[:80])
            temp_product_path = self._download_image_from_url(product_image_url)
            logger.info("[Scene2] Step 1/6: Downloading product image - done | path=%s", temp_product_path)
            logger.info("✅ Product image downloaded: %s", temp_product_path)

            # Step 2: Add shadow effect to product image using PIL
            if add_shadow:
                logger.info("[Scene2] Step 2/6: Adding shadow effect (PIL) - started")
                print("\n✨ STEP 2/6: Adding Shadow Effect to Product Image (PIL)")
                print("-" * 80)
                logger.info("🔄 Adding shadow effect to product image using PIL...")

                temp_shadow_product_path = self._add_pil_shadow_to_image(
                    temp_product_path,
                    blur_radius=shadow_blur_radius,
                    offset=shadow_offset
                )
                logger.info("[Scene2] Step 2/6: Adding shadow effect (PIL) - done | path=%s", temp_shadow_product_path)
                logger.info("✅ Shadow effect added: %s", temp_shadow_product_path)

                # Use the shadow-enhanced image for further processing
                product_path_for_merge = temp_shadow_product_path
            else:
                logger.info("[Scene2] Step 2/6: Skipped (shadow disabled)")
                product_path_for_merge = temp_product_path

            # Step 3: Download background video
            logger.info("[Scene2] Step 3/6: Downloading background video - started")
            print("\n📥 STEP 3/6: Downloading Background Video")
            print("-" * 80)
            logger.info("🎥 Background video URL: %s...", background_video_url[:80])
            temp_video_path = self._download_video_from_url(background_video_url)
            logger.info("[Scene2] Step 3/6: Downloading background video - done | path=%s", temp_video_path)
            logger.info("✅ Background video downloaded: %s", temp_video_path)
            
            # Step 4: Merge using OpenCV
            logger.info("[Scene2] Step 4/6: Merging product with video (OpenCV) - started")
            print("\n🎨 STEP 4/6: Merging Product with Video (OpenCV Processing)")
            print("-" * 80)
            logger.info("🔄 Starting OpenCV video processing...")
            temp_output_path = self._merge_with_opencv(
                product_path_for_merge,  # Use shadow-enhanced image if shadow was added
                temp_video_path,
                scale=scale,
                position=position,
                duration=duration,
                add_animation=add_animation
            )
            logger.info("[Scene2] Step 4/6: Merging product with video (OpenCV) - done | path=%s", temp_output_path)
            logger.info("✅ Video merge completed: %s", temp_output_path)

            # Step 5: Save to public folder and build response URL
            public_base = getattr(settings, "PUBLIC_OUTPUT_BASE", None)
            if not public_base:
                return {'success': False, 'video_url': None, 'error': 'PUBLIC_OUTPUT_BASE not configured'}
            out_dir = Path(public_base) / "generated_videos" / user_id / short_id / "scene2"
            out_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"scene2-{uuid.uuid4().hex[:12]}.mp4"
            dest_path = out_dir / file_name
            shutil.copy2(temp_output_path, str(dest_path))
            video_url = f"generated_videos/{user_id}/{short_id}/scene2/{file_name}"
            logger.info("[Scene2] Step 5/6: Saved to public folder | url=%s", video_url)
            logger.info("[Scene2] Step 6/6: Done")

            print("\n" + "="*80)
            print("✅ SCENE 2 GENERATION COMPLETED SUCCESSFULLY")
            print("="*80)
            logger.info("[Scene2] COMPLETED | scene_id=%s | video_url=%s", scene_id, video_url)
            logger.info("🎉 Image-video merge completed successfully for scene %s", scene_id)
            print()
            
            return {
                'success': True,
                'video_url': video_url,
                'error': None
            }
            
        except Exception as e:
            error_msg = f"Image-video merge failed: {str(e)}"
            print("\n" + "="*80)
            print("❌ SCENE 2 GENERATION FAILED")
            print("="*80)
            logger.error("[Scene2] FAILED | scene_id=%s | error=%s", scene_id, error_msg, exc_info=True)
            print()
            return {
                'success': False,
                'video_url': None,
                'error': error_msg
            }
        
        finally:
            # Clean up temporary files
            for temp_path in [temp_product_path, temp_video_path, temp_output_path, temp_shadow_product_path]:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                        logger.debug(f"Cleaned up temporary file: {temp_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary file {temp_path}: {cleanup_error}")

    def _add_pil_shadow_to_image(
        self,
        image_path: str,
        blur_radius: int = 25,
        offset: Tuple[int, int] = (15, 15),
        shadow_color: Tuple[int, int, int] = (0, 0, 0),
        shadow_angle: float = 135,
        shadow_distance: float = 30,
        shadow_opacity: float = 0.35,
        contact_height_ratio: float = 0.15,
        contact_blur: int = 8,
        ao_intensity: float = 0.4,
        ao_blur: int = 30,
    ) -> str:
        """
        Add cast shadow, contact shadow, and ambient occlusion to a transparent PNG (same pipeline as compositing).
        
        Args:
            image_path: Path to the input PNG image with transparency
            blur_radius: Blur for directional cast shadow (clamped; used as cast shadow blur)
            offset: Optional (x, y) offset; if non-zero, used to derive shadow_angle and shadow_distance
            shadow_color: Shadow color as RGB tuple (default: black); reserved for future use
            shadow_angle: Cast shadow direction in degrees (0=right, 90=down, 135=down-left)
            shadow_distance: Cast shadow distance in pixels
            shadow_opacity: Shadow darkness (0–1); cast uses opacity*0.9, contact uses opacity*1.4
            contact_height_ratio: Height of contact shadow as fraction of overlay height (default 0.15)
            contact_blur: Gaussian blur radius for contact shadow (default 8)
            ao_intensity: Ambient occlusion intensity (default 0.4)
            ao_blur: Gaussian blur radius for AO layer (default 30)
        
        Returns:
            Path to the new image with shadow effect
        """
        try:
            logger.info(f"🖼️  Adding PIL shadow to image (cast+contact+AO): {image_path}")
            
            # Load and ensure RGBA without altering product colors (avoid convert("RGBA") color shift)
            img = Image.open(image_path)
            if img.mode == "RGBA":
                overlay = img.copy()
            elif img.mode == "RGB":
                r, g, b = img.split()
                overlay = Image.merge("RGBA", (r, g, b, Image.new("L", img.size, 255)))
            else:
                overlay = img.convert("RGBA")
            # Untouched copy for final paste so only shadow is added, product colors preserved
            product_only = overlay.copy()
            alpha = overlay.split()[3]
            
            # If offset is provided and non-zero, derive angle and distance for cast shadow
            use_angle = shadow_angle
            use_distance = shadow_distance
            if offset and (offset[0] != 0 or offset[1] != 0):
                use_angle = math.degrees(math.atan2(offset[1], offset[0]))
                use_distance = math.hypot(offset[0], offset[1])
            
            cast_blur = min(45, max(blur_radius, 20))
            pad = int(cast_blur + abs(use_distance) + 20)
            canvas_w = overlay.width + 2 * pad
            canvas_h = overlay.height + 2 * pad
            result = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            
            pos_x, pos_y = pad, pad

            # 1. Directional cast shadow (same as compositing step 6.5)
            cast_shadow, shadow_offset = self._create_cast_shadow(
                overlay,
                angle=use_angle,
                distance=use_distance,
                opacity=shadow_opacity * 0.9,
                blur=cast_blur,
            )
            cast_shadow_pos = (pos_x + shadow_offset[0] - cast_blur, pos_y + shadow_offset[1] - cast_blur)
            if cast_shadow_pos[0] + cast_shadow.width > 0 and cast_shadow_pos[1] + cast_shadow.height > 0:
                result.paste(cast_shadow, cast_shadow_pos, cast_shadow)

            # 2. Contact shadow under the object (same as compositing step 7)
            contact_shadow = Image.new("RGBA", overlay.size, (0, 0, 0, 255))
            contact_shadow.putalpha(alpha)
            new_h = int(overlay.height * contact_height_ratio)
            contact_shadow = contact_shadow.resize((overlay.width, new_h), Image.Resampling.LANCZOS)
            contact_shadow = contact_shadow.filter(ImageFilter.GaussianBlur(contact_blur))
            shadow_alpha = contact_shadow.split()[3]
            shadow_alpha = ImageEnhance.Brightness(shadow_alpha).enhance(shadow_opacity * 1.4)
            contact_shadow.putalpha(shadow_alpha)
            contact_position = (pos_x, pos_y + overlay.height - contact_shadow.height)
            result.paste(contact_shadow, contact_position, contact_shadow)

            # 3. Improved ambient occlusion (same as compositing step 7.5)
            ao = self._create_improved_ao(overlay, intensity=ao_intensity)
            ao_position = (pos_x, pos_y + int(overlay.height * 0.4))
            ao_resized = ao.resize((overlay.width, int(overlay.height * 0.6)), Image.Resampling.LANCZOS)
            ao_resized = ao_resized.filter(ImageFilter.GaussianBlur(ao_blur))
            result.paste(ao_resized, ao_position, ao_resized)

            # 4. Product on top — paste untouched copy with alpha mask (no color change)
            product_alpha = product_only.split()[3]
            result.paste(product_only, (pos_x, pos_y), product_alpha)

            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_with_shadow{ext}"
            result.save(output_path, "PNG")
            logger.info(f"✅ PIL shadow image saved (cast+contact+AO): {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error adding PIL shadow to image: {str(e)}", exc_info=True)
            # If shadow addition fails, return the original image path
            return image_path

    def _extract_alpha_pil(self, image):
        """Extract the alpha channel from a PIL image."""
        return image.split()[-1]

    def _create_shadow_from_alpha_pil(self, alpha, blur_radius, shadow_color=(0, 0, 0)):
        """Create a shadow based on a blurred version of the alpha channel."""
        alpha_blur = alpha.filter(ImageFilter.BoxBlur(blur_radius))
        shadow = Image.new("RGBA", alpha_blur.size, (*shadow_color, 0))
        shadow.putalpha(alpha_blur)
        return shadow

    def _download_video_from_url(self, video_url: str) -> str:
        """
        Download video from URL to local file.
        
        Args:
            video_url: URL of the video to download
        
        Returns:
            Path to the downloaded video file
        """
        try:
            logger.info(f"Downloading video from URL: {video_url}")
            
            # Generate temp file path
            temp_path = str(self._temp_dir / f"download-video-{uuid.uuid4()}.mp4")
            
            with httpx.Client(timeout=DOWNLOAD_TIMEOUT) as client:
                response = client.get(video_url)
                response.raise_for_status()
                
                # Save to local file
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Successfully downloaded video to: {temp_path}")
                return temp_path
                
        except Exception as e:
            logger.error(f"Failed to download video from {video_url}: {e}")
            raise Exception(f"Failed to download video: {e}")

    def _merge_with_opencv(
        self,
        product_path: str,
        video_path: str,
        scale: float = 0.4,
        position: str = "center",
        duration: Optional[int] = None,
        add_animation: bool = True
    ) -> str:
        """
        Merge product image with background video using OpenCV.
        
        Args:
            product_path: Path to product image (PNG with transparency)
            video_path: Path to background video
            scale: Scale of product relative to video width
            position: Position of product on video
            duration: Optional duration limit in seconds
            add_animation: Whether to add zoom and floating animations
        
        Returns:
            Path to the output video file
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Force output to 1920x1080 for high quality
            width = 1920
            height = 1080
            
            # Limit duration if specified
            if duration:
                max_frames = int(fps * duration)
                total_frames = min(total_frames, max_frames)
                logger.info(f"⏱️  Duration limited to {duration}s ({max_frames} frames)")
            
            video_duration = total_frames / fps
            logger.info(f"📹 Input Video Info:")
            logger.info(f"   - Original Resolution: {original_width}x{original_height}")
            logger.info(f"   - Output Resolution: {width}x{height} (1080p HD)")
            logger.info(f"   - FPS: {fps}")
            logger.info(f"   - Total Frames: {total_frames}")
            logger.info(f"   - Duration: {video_duration:.2f}s")
            
            # Load product image (PIL gives RGB; OpenCV uses BGR — convert so product colors match video)
            product_img = Image.open(product_path).convert("RGBA")
            product_np = np.array(product_img)
            # PIL RGBA = R,G,B,A; OpenCV frame = B,G,R. Convert product to BGRA so compositing doesn't swap R/B.
            product_np = product_np[:, :, [2, 1, 0, 3]]

            logger.info(f"🖼️  Product Image: {product_np.shape[1]}x{product_np.shape[0]} pixels")
            
            # Create output video - use temp AVI first for better compatibility
            temp_output_path = str(self._temp_dir / f"merged-temp-{uuid.uuid4()}.avi")
            output_path = str(self._temp_dir / f"merged-video-{uuid.uuid4()}.mp4")
            
            # Use MJPEG codec for AVI (more reliable than mp4v)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            
            logger.info(f"💾 Temp output file: {temp_output_path}")
            logger.info(f"💾 Final output file: {output_path}")
            
            # Animation parameters
            zoom_duration = 3.0 if add_animation else 0.0
            min_scale = 0.05 if add_animation else scale
            
            if add_animation:
                logger.info(f"✨ Animation enabled:")
                logger.info(f"   - Zoom: {min_scale*100}% → {scale*100}% over {zoom_duration}s")
                logger.info(f"   - Floating: ±30px sine wave after zoom")
            
            # Process frames
            frame_idx = 0
            smooth_x, smooth_y = None, None
            smooth_factor = 0.08
            last_log_time = 0
            
            if original_width != width or original_height != height:
                logger.info(f"🔄 Each frame will be upscaled: {original_width}x{original_height} → {width}x{height}")
            
            logger.info(f"🎬 Starting frame processing...")
            print(f"   Progress: [", end="", flush=True)
            
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame to 1920x1080 if needed
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
                
                t = frame_idx / fps
                
                # Calculate scale with zoom animation
                if add_animation and t < zoom_duration:
                    # Ease-out zoom animation
                    progress = t / zoom_duration
                    eased = 1 - (1 - progress) ** 3
                    current_scale = min_scale + eased * (scale - min_scale)
                else:
                    current_scale = scale
                
                # Calculate product size
                product_w = max(1, int(width * current_scale))
                ratio = product_w / product_np.shape[1]
                product_h = max(1, int(product_np.shape[0] * ratio))
                
                # Resize product
                product_resized = cv2.resize(product_np, (product_w, product_h), 
                                            interpolation=cv2.INTER_AREA)
                
                # Calculate position with optional floating animation
                dy = 0
                if add_animation and t >= zoom_duration:
                    # Floating animation
                    phase = t - zoom_duration
                    dy = int(30 * math.sin(phase * 0.8))
                
                # Position based on parameter
                if position == "center":
                    target_x = width // 2 - product_w // 2
                    target_y = int(height * 0.52 - product_h // 2 + dy)
                elif position == "top":
                    target_x = width // 2 - product_w // 2
                    target_y = int(height * 0.2 + dy)
                elif position == "bottom":
                    target_x = width // 2 - product_w // 2
                    target_y = int(height * 0.8 - product_h + dy)
                elif position == "left":
                    target_x = int(width * 0.2)
                    target_y = int(height * 0.52 - product_h // 2 + dy)
                elif position == "right":
                    target_x = int(width * 0.8 - product_w)
                    target_y = int(height * 0.52 - product_h // 2 + dy)
                else:
                    target_x = width // 2 - product_w // 2
                    target_y = int(height * 0.52 - product_h // 2 + dy)
                
                # Smooth position transition
                if smooth_x is None:
                    smooth_x, smooth_y = target_x, target_y
                smooth_x += (target_x - smooth_x) * smooth_factor
                smooth_y += (target_y - smooth_y) * smooth_factor
                
                x = int(smooth_x)
                y = int(smooth_y)
                
                # Ensure product stays within frame
                x = max(0, min(width - product_w, x))
                y = max(0, min(height - product_h, y))
                
                # Composite product onto frame using alpha channel
                alpha = product_resized[:, :, 3] / 255.0
                
                for c in range(3):
                    frame[y:y+product_h, x:x+product_w, c] = (
                        frame[y:y+product_h, x:x+product_w, c] * (1 - alpha)
                        + product_resized[:, :, c] * alpha
                    )
                
                # Write frame
                out.write(frame)
                frame_idx += 1
                
                # Log progress - show progress bar
                progress_pct = (frame_idx / total_frames) * 100
                
                # Update progress bar every 5%
                if progress_pct >= last_log_time + 5 or frame_idx == total_frames:
                    last_log_time = progress_pct
                    bar_length = int(progress_pct / 5)
                    print("=" * bar_length, end="", flush=True)
                
                # Detailed logging every 2 seconds
                if frame_idx % int(fps * 2) == 0 or frame_idx == total_frames:
                    elapsed_time = frame_idx / fps
                    remaining_frames = total_frames - frame_idx
                    estimated_remaining = remaining_frames / fps if fps > 0 else 0
                    
                    phase = "ZOOM" if (add_animation and elapsed_time < zoom_duration) else "FLOAT"
                    
                    logger.info(
                        f"   Frame {frame_idx}/{total_frames} | "
                        f"{progress_pct:.1f}% | "
                        f"Time: {elapsed_time:.1f}s/{video_duration:.1f}s | "
                        f"Phase: {phase} | "
                        f"ETA: {estimated_remaining:.1f}s"
                    )
            
            print("] 100%")
            
            cap.release()
            out.release()
            
            logger.info(f"✅ OpenCV processing complete")
            logger.info(f"💾 Temp AVI saved to: {temp_output_path}")
            
            # Convert to H.264 MP4 using FFmpeg for better compatibility
            logger.info(f"🎞️  Converting to H.264 MP4 with FFmpeg...")
            self._convert_to_h264(temp_output_path, output_path)
            
            # Clean up temp AVI file
            try:
                os.unlink(temp_output_path)
                logger.info(f"🗑️  Cleaned up temp AVI file")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")
            
            logger.info(f"💾 Final MP4 saved to: {output_path}")
            return output_path
            
        except Exception as e:
            raise Exception(f"Failed to merge video with OpenCV: {e}")

    def _convert_to_h264(self, input_path: str, output_path: str):
        """
        Convert video to H.264 MP4 format using FFmpeg for maximum compatibility.
        
        Args:
            input_path: Path to input video (AVI)
            output_path: Path to output video (MP4)
        """
        try:
            # FFmpeg command for high-quality H.264 encoding at 1920x1080
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',  # Ensure 1920x1080
                '-c:v', 'libx264',          # H.264 video codec
                '-preset', 'slow',           # Better quality (slow = higher quality)
                '-crf', '18',                # High quality (18 = visually lossless)
                '-profile:v', 'high',        # H.264 high profile for better compression
                '-level', '4.2',             # H.264 level 4.2
                '-pix_fmt', 'yuv420p',      # Pixel format for compatibility
                '-movflags', '+faststart',   # Enable fast start for web streaming
                '-b:v', '8M',                # 8 Mbps bitrate for high quality 1080p
                '-maxrate', '10M',           # Max bitrate
                '-bufsize', '16M',           # Buffer size
                '-y',                        # Overwrite output file
                output_path
            ]
            
            logger.info(f"🎞️  FFmpeg Quality Settings:")
            logger.info(f"   - Resolution: 1920x1080 (forced)")
            logger.info(f"   - Codec: H.264 High Profile")
            logger.info(f"   - Bitrate: 8Mbps (high quality)")
            logger.info(f"   - CRF: 18 (visually lossless)")
            logger.info(f"   - Preset: slow (higher quality)")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg stderr: {result.stderr}")
                raise Exception(f"FFmpeg conversion failed with code {result.returncode}")
            
            logger.info(f"✅ FFmpeg conversion complete")
            
        except FileNotFoundError:
            raise Exception("FFmpeg not found! Please install FFmpeg: https://ffmpeg.org/download.html")
        except subprocess.TimeoutExpired:
            raise Exception("FFmpeg conversion timed out after 5 minutes")
        except Exception as e:
            raise Exception(f"FFmpeg conversion failed: {e}")

    def _upload_video_to_supabase(
        self,
        file_path: str,
        scene_id: str,
        user_id: str
    ) -> Optional[str]:
        """No-op: Supabase removed. Returns None."""
        return None

    def _update_scene_video_url(self, scene_id: str, video_url: str) -> None:
        """No-op: Supabase removed."""
        pass


# Global instance
image_processing_service = ImageProcessingService()
