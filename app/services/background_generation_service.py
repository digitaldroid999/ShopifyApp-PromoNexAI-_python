"""
Background Generation Service for creating product backgrounds using AI.
Uses OpenAI GPT-4 for prompt extraction and Vertex AI Imagen 4.0 for background generation.
Supports async/polling pattern with threading.

Process:
1. Extract background requirements from product description and environment variables using OpenAI GPT-4o-mini
2. Generate background image using Vertex AI Imagen 4.0
3. Upload result to Supabase storage
"""

import logging
import os
import uuid
import openai
import threading
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timezone
from app.models import TaskStatus
from app.config import settings
from app.utils.supabase_utils import supabase_manager
from app.utils.vertex_utils import vertex_manager
from app.logging_config import get_logger

logger = get_logger(__name__)


class BackgroundGenerationService:
    """Service for generating product backgrounds with async task management"""

    def __init__(self):
        self.openai_client = None
        self.vertex_manager = vertex_manager  # Use global Vertex AI instance
        self.tasks = {}  # In-memory task storage {task_id: task_info}
        self.tasks_lock = threading.Lock()  # Thread-safe access to tasks
        self._initialize_openai()
        self._check_vertex_availability()

    def _initialize_openai(self):
        """Initialize OpenAI client for prompt extraction"""
        logger.info("Initializing Background Generation Service (Async + Vertex AI)...")
        try:
            logger.info("→ Checking for OpenAI API key...")
            if not settings.OPENAI_API_KEY:
                logger.warning("✗ OpenAI API key not configured in settings!")
                logger.warning("→ Background generation service will not be available")
                return

            logger.info("✓ OpenAI API key found")
            logger.info("→ Creating OpenAI client instance...")
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)                                                               
            logger.info("✓ OpenAI client initialized for prompt extraction")

        except Exception as e:
            logger.error(f"✗ Failed to initialize OpenAI client!")
            logger.error(f"→ Error: {str(e)}")
            self.openai_client = None
            logger.warning("→ Background generation service will not be available")
    
    def _check_vertex_availability(self):
        """Check if Vertex AI is available for image generation"""
        try:
            # Basic presence check
            if not self.vertex_manager:
                logger.warning("✗ Vertex manager instance is None")
                logger.warning("→ Background generation will not work without Vertex AI manager")
                return

            # Try to call is_available and surface any exception details
            try:
                available = self.vertex_manager.is_available()
            except Exception as inner_exc:
                logger.error("✗ Exception while calling vertex_manager.is_available()")
                logger.exception(inner_exc)
                logger.warning("→ Background generation may not work properly due to the above error")
                return

            if available:
                logger.info("✓ Vertex AI Imagen initialized for image generation")
                logger.info("✓ Background Generation Service ready (Async mode: OpenAI + Vertex AI)")
            else:
                logger.warning("✗ Vertex AI is not available (is_available() returned False)!")
                # Log common environment/setting clues to help debug credential/config issues
                try:
                    ga_creds = getattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS', None)
                    vertex_proj = getattr(settings, 'VERTEX_PROJECT', None)
                    logger.warning(f"→ GOOGLE_APPLICATION_CREDENTIALS={ga_creds}")
                    logger.warning(f"→ VERTEX_PROJECT={vertex_proj}")
                except Exception:
                    logger.debug("Could not read some settings for additional diagnostics")
                logger.warning("→ Background generation will not work without Vertex AI")
        except Exception as e:
            # Catch-all to ensure service initialization continues but logs useful info
            logger.error(f"✗ Failed to check Vertex AI availability: {e}")
            logger.exception(e)
            logger.warning("→ Background generation may not work properly")
    
    def start_background_generation_task(
        self, 
        user_id: str,
        product_description: str,
        mood: Optional[str] = None,
        style: Optional[str] = None,
        environment: Optional[str] = None,
        manual_prompt: Optional[str] = None,
        scene_id: Optional[str] = None,
        short_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start an async background generation task.
        Returns immediately with task_id for polling.
        
        Args:
            user_id: User ID for storage organization
            product_description: Description of the product
            mood: Mood/feeling for the background (e.g., "Energetic", "Calm", "Professional")
            style: Visual style for the background (e.g., "Film Grain", "Minimalist", "Vibrant")
            environment: Environment setting (e.g., "Indoor Studio", "Outdoor Nature", "Urban")
            manual_prompt: Manual prompt for background generation (skips OpenAI if provided)
            scene_id: Optional scene ID for database updates
            short_id: Optional short ID for tracking
        """
        task_id = f"bg_task_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        logger.info("=" * 80)
        logger.info(f"🚀 STARTING ASYNC BACKGROUND GENERATION TASK: {task_id}")
        logger.info("=" * 80)
        logger.info(f"User ID: {user_id}")
        logger.info(f"Product Description: {product_description}")
        logger.info(f"Mood: {mood or 'N/A'}")
        logger.info(f"Style: {style or 'N/A'}")
        logger.info(f"Environment: {environment or 'N/A'}")
        logger.info(f"Manual Prompt: {manual_prompt[:100] + '...' if manual_prompt and len(manual_prompt) > 100 else manual_prompt or 'N/A'}")
        logger.info(f"Scene ID: {scene_id or 'N/A'}")
        
        # Create task record
        task_info = {
            'task_id': task_id,
            'status': TaskStatus.PENDING,
            'user_id': user_id,
            'scene_id': scene_id,
            'short_id': short_id,
            'product_description': product_description,
            'mood': mood,
            'style': style,
            'environment': environment,
            'manual_prompt': manual_prompt,
            'result_image_url': None,
            'error_message': None,
            'progress': 0,
            'current_step': 'Initializing',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        with self.tasks_lock:
            self.tasks[task_id] = task_info
        
        # Start processing in a background thread
        thread = threading.Thread(
            target=self._process_background_generation,
            args=(task_id, user_id, product_description, mood, style, environment, manual_prompt, scene_id),
            daemon=True
        )
        thread.start()
        
        logger.info(f"✅ Task created and started in background thread")
        logger.info("=" * 80)
        
        return {
            'task_id': task_id,
            'status': TaskStatus.PENDING,
            'message': 'Background generation task started successfully',
            'created_at': task_info['created_at']
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a background generation task"""
        with self.tasks_lock:
            return self.tasks.get(task_id)
    
    def _update_task(self, task_id: str, **kwargs):
        """Update task information thread-safely"""
        with self.tasks_lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
                self.tasks[task_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    def _process_background_generation(
        self, 
        task_id: str, 
        user_id: str,
        product_description: str,
        mood: Optional[str],
        style: Optional[str],
        environment: Optional[str],
        manual_prompt: Optional[str],
        scene_id: Optional[str]
    ):
        """
        Background worker function that processes background generation.
        This runs in a separate thread.
        """
        try:
            logger.info(f"🔄 [Task {task_id}] Starting background generation process...")
            
            self._update_task(task_id, status=TaskStatus.RUNNING, current_step='Starting', progress=10)
            
            if not self.vertex_manager or not self.vertex_manager.is_available():
                raise Exception("Vertex AI is not available")
            
            # Step 1: Get background prompt (either from manual_prompt or extract with OpenAI)
            if manual_prompt:
                # Use manual prompt directly
                logger.info(f"[Task {task_id}] Step 1/3: Using manual prompt (skipping OpenAI)...")
                self._update_task(task_id, current_step='Using manual prompt', progress=40)
                background_prompt = manual_prompt
                logger.info(f"[Task {task_id}] ✓ Manual prompt loaded")
                logger.info(f"[Task {task_id}]   → Background prompt: {background_prompt[:100]}...")
            else:
                # Extract background prompt with OpenAI (40% progress)
                logger.info(f"[Task {task_id}] Step 1/3: Extracting background prompt with OpenAI GPT-4o-mini...")
                self._update_task(task_id, current_step='Generating background prompt with OpenAI', progress=40)
                
                if not self.openai_client:
                    raise Exception("OpenAI client not initialized and no manual prompt provided")
                
                background_prompt = self._extract_background_prompt(
                    product_description, mood, style, environment
                )
                logger.info(f"[Task {task_id}] ✓ Background prompt extraction completed")
                logger.info(f"[Task {task_id}]   → Background prompt: {background_prompt[:100]}...")
            
            # Step 2: Generate background with Vertex AI Imagen 4.0 (70% progress)
            logger.info(f"[Task {task_id}] Step 2/3: Generating background with Vertex AI Imagen 4.0...")
            self._update_task(task_id, current_step='Generating background with Vertex AI', progress=70)
            generated_image_path = self._generate_background_with_vertex(background_prompt)
            
            if not generated_image_path:
                raise Exception("Failed to generate background using Vertex AI Imagen 4.0")
            
            logger.info(f"[Task {task_id}] ✓ Background generated successfully (Vertex AI Imagen 4.0)")
            logger.info(f"[Task {task_id}]   → Local file path: {generated_image_path}")
            
            # Step 3: Upload to Supabase (90% progress)
            logger.info(f"[Task {task_id}] Step 3/3: Uploading to Supabase...")
            self._update_task(task_id, current_step='Uploading to storage', progress=90)
            final_image_url = self._upload_to_supabase(generated_image_path, user_id)
            
            if not final_image_url:
                raise Exception("Failed to upload image to storage")
            
            logger.info(f"[Task {task_id}] ✓ Image upload completed")
            
            # Cleanup temp files
            self._cleanup_temp_files([generated_image_path])
            
            # Update scene in database if scene_id provided
            if scene_id:
                logger.info(f"[Task {task_id}] Updating scene {scene_id} in database...")
                try:
                    import asyncio
                    asyncio.run(supabase_manager.update_record(
                        table='video_scenes',
                        filters={'id': scene_id},
                        updates={'image_url': final_image_url, 'updated_at': datetime.now(timezone.utc).isoformat()}
                    ))
                    logger.info(f"[Task {task_id}] ✓ Scene updated in database")
                except Exception as e:
                    logger.warning(f"[Task {task_id}] Failed to update scene in database: {e}")
            
            # Mark task as completed (100% progress)
            self._update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                result_image_url=final_image_url,
                current_step='Completed',
                progress=100
            )
            
            logger.info("=" * 80)
            logger.info(f"✅ [Task {task_id}] BACKGROUND GENERATION COMPLETED SUCCESSFULLY!")
            logger.info(f"Final Image URL: {final_image_url}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ [Task {task_id}] BACKGROUND GENERATION FAILED!")
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 80)
            
            self._update_task(
                task_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                current_step='Failed',
                progress=0
            )

    def _extract_background_prompt(
        self, 
        product_description: str,
        mood: Optional[str],
        style: Optional[str],
        environment: Optional[str]
    ) -> str:
        """
        Build a background generation prompt directly from the product description
        and environment variables (NO OpenAI call).
        
        Args:
            product_description: The product description text
            mood: Mood/feeling for the background (optional)
            style: Visual style for the background (optional)
            environment: Environment setting (optional)
            
        Returns:
            A detailed prompt for background generation
        """
        try:
            logger.info("  → Building background prompt from product description and environment (no OpenAI)...")

            # Build environment context
            env_context = []
            if mood:
                env_context.append(f"Mood/Feeling: {mood}")
            if style:
                env_context.append(f"Visual Style: {style}")
            if environment:
                env_context.append(f"Environment: {environment}")
            
            env_text = "\n".join(env_context) if env_context else "No specific environment variables provided"

            # Do NOT embed raw product description — image models can latch onto product nouns and draw them.
            # Use only env vars and category/mood guidelines so the scene stays product-free.
            # This prompt is used directly with Vertex Imagen as the generation prompt
            background_prompt = f"""Generate a professional BACKGROUND-ONLY empty scene suitable for compositing e-commerce product photography later. Do not depict any product, object, or main subject.

Environment Variables:
{env_text}

CRITICAL REQUIREMENTS:

A) Product-Absent Rules (Strict):
1. DO NOT generate or include the actual product in the image - NO exceptions.
2. DO NOT generate any object, shape, silhouette, shadow, reflection, or abstract form that represents, implies, or substitutes the product.
3. DO NOT generate any product-specific accessories, packaging, stands, holders, or display fixtures.
4. Generate ONLY the environment, surface, backdrop, and atmospheric elements that would surround the product.
5. Keep the primary product placement zone (center 1/3 of frame) visually clean, empty, and completely unobstructed.
6. Treat the scene as an empty retail stage or photography set, NOT a product composition.

B) Compositing-Ready Technical Rules:
7. Design the background so ANY product can be naturally composited later without visual conflict.
8. Lighting must be realistic, neutral, and oriented toward the central placement area (45-degree key light recommended).
9. Shadows and reflections must be subtle, physically plausible, and originate ONLY from environmental structures/surfaces - NOT from the product itself.
10. Perspective and camera angle must support standard e-commerce photography (eye-level, 45-degree, or flat lay overhead).
11. Color palette must be neutral enough to complement diverse product colors, not compete with them.
12. Surface texture must provide subtle visual interest without overpowering the composited product.
13. If any ambiguity occurs, ALWAYS prioritize emptiness, neutrality, and simplicity in the center area.

C) Mood & Context Rules (Use Product Description ONLY for these):
14. Extract ONLY the following from product description and ignore all physical form attributes:
    - Target audience (men/women/unisex, age, lifestyle)
    - Price tier (budget/mid-range/luxury)
    - Category (apparel/electronics/home/beauty/sports/food/pets/office)
    - Seasonality (spring/summer/fall/winter/all-season)
    - Style aesthetic (minimal/rustic/modern/bohemian/industrial/classic)
15. Map these attributes to appropriate:
    - Color psychology and palette temperature
    - Lighting mood (bright and airy / dramatic / soft and diffused / warm and cozy / cool and clean)
    - Environmental context (studio / indoor lifestyle / outdoor urban / outdoor nature / abstract)
    - Surface materials and textures
    - Supporting prop categories (ONLY if non-competitive and placed OUTSIDE center zone)

D) Universal Background Requirements:
- Suitable for professional e-commerce, Amazon/Target/Wayfair/Sephora style product photography.
- Photorealistic quality, 8K resolution capable, commercial grade lighting.
- Clean, modern aesthetic with strong negative space - center zone must be IMMACULATE.
- Consistent visual style throughout the entire frame (no style mixing).
- If environment specified, integrate it appropriately; if not specified, default to professional studio seamless.
- Use professional photography terminology for lighting, composition, depth of field, and surface treatment.
- Props are OPTIONAL and STRICTLY LIMITED to:
    * Non-competitive contextual objects (e.g., foliage, fabric, books - NOT product-related)
    * Must be placed in background/edges ONLY
    * Must be heavily diffused or shallow depth of field
    * Must NOT resemble the product or its accessories
- Maintain strong compositional balance with clear visual path to empty center.

E) UNIVERSAL MOOD-TO-VISUAL MAPPING TABLE (Follow strictly):

| Category              | Budget                                     | Mid-Range                                   | Luxury                                      |
|-----------------------|--------------------------------------------|---------------------------------------------|---------------------------------------------|
| **Apparel**           | Bright white seamless, clean, high key     | Textured plaster, soft grays, diffused      | Marble, sculptural forms, dramatic shadow   |
| **Electronics**       | Clean white, blue accent, tech minimal     | Charcoal gradient, rim light, precision     | Dark studio, amber accent, museum quality   |
| **Home & Living**     | Bright airy, light wood, casual            | Warm neutrals, layered textures, curated    | Rich depths, art objects, gallery lighting  |
| **Beauty**            | Pink/white, luminous, dewy soft focus      | Rose quartz, satin surfaces, soft glam      | Gold veining, black marble, spotlight       |
| **Sports/Fitness**    | Bright primary, energetic, clean lines     | Concrete gray, amber light, athletic        | Polished charcoal, dramatic rim, premium    |
| **Food & Beverage**   | Bright tabletop, casual, friendly          | Rustic wood, natural light, artisanal       | Dark stone, directional beam, jewel tones   |
| **Pet Products**      | Warm home setting, soft, approachable      | Lifestyle interior, natural fibers          | Curated modern, architectural, quiet        |
| **Office**            | Clean white, functional, productive        | Warm wood, soft focus, professional         | Leather textures, library tones, heritage   |

| Season     | Palette                              | Lighting                              | Context                                |
|------------|--------------------------------------|---------------------------------------|----------------------------------------|
| Spring     | Soft pastels, blush, sage, cream     | Bright diffused, morning freshness    | Fresh botanicals, open air             |
| Summer     | Warm whites, sand, sky blue          | Golden hour, high sun, vibrant        | Beach, pool, outdoor lifestyle         |
| Fall       | Terracotta, olive, warm taupe        | Golden sidelight, cozy shadows        | Forest edge, harvest warmth            |
| Winter     | Cool whites, charcoal, ice blue      | Soft overcast, crisp clean            | Snowy minimal, cozy interior           |
| All-Season | True neutrals, balanced warmth       | Studio controlled, timeless           | Versatile, classic, permanent          |

F) PRODUCT CATEGORY - SPECIFIC GUIDELINES:

**APPAREL / ACCESSORIES:**
- Surface: Seamless paper, textured concrete, aged wood, polished floor
- Format: Full-body, three-quarter, or flat lay perspective
- Key requirement: Clean vertical or horizontal plane, no clothing forms

**ELECTRONICS / TECH:**
- Surface: Matte black, brushed aluminum effect, clean acrylic
- Format: 45-degree hero angle, overhead for smaller items
- Key requirement: Minimal, precision-focused, reflection-ready

**HOME / KITCHEN:**
- Surface: Marble, butcher block, linen, ceramic tile
- Format: Lifestyle context or clean tabletop
- Key requirement: Warm but neutral, residential scale

**BEAUTY / SKINCARE:**
- Surface: Satin acrylic, soft stone, frosted glass
- Format: Beauty flat lay, hero angled shot
- Key requirement: Luminous, clean, spa-like atmosphere

**SPORTS / FITNESS:**
- Surface: Rubber gym flooring, track surface, grass texture
- Format: Dynamic angles, ground-level possible
- Key requirement: Energetic but empty, motion-ready

**FOOD / BEVERAGE:**
- Surface: Wood, stone, concrete, linen
- Format: Overhead flat lay or 45-degree tabletop
- Key requirement: Surface storytelling, NO food/drink props

**PETS:**
- Surface: Hardwood, indoor-outdoor fabric, grass
- Format: Low angle, ground perspective
- Key requirement: Warm, approachable, empty floor space

**OFFICE / STATIONERY:**
- Surface: Deskscape, leather mat, clean white
- Format: Desk-level, overhead organization
- Key requirement: Productive calm, clean surfaces

EXAMPLES OF WHAT TO GENERATE (Pattern only - ADAPT, NEVER COPY):

**Product: Wireless earbuds (Tech/Budget)**
→ Scene: Bright white studio cove with subtle blue LED rim light from rear. Clean seamless floor with soft gradient shadow fade. No charging cases, cables, or earbud silhouettes.

**Product: Cashmere sweater (Apparel/Luxury)**
→ Scene: Sculpted plaster wall in warm gray, museum-quality track lighting creating soft geometric shadows. Polished concrete floor with subtle reflection. No hangers, mannequins, or garment forms.

**Product: Ceramic dinnerware (Home/Mid-Range)**
→ Scene: Honed marble countertop in warm white, morning window light from left, eucalyptus sprig in distant soft focus background corner. No plates, bowls, or utensils.

**Product: Face serum (Beauty/Luxury)**
→ Scene: Black marble with subtle gold veining, single directional spotlight from above-right creating dramatic falloff. Soft atmospheric haze. No droppers, bottles, or cosmetic tools.

**Product: Running shoes (Sports/Budget)**
→ Scene: Weathered blue track surface with faint white lane lines receding into soft focus. Golden hour sidelight raking across texture. Empty foreground lane. No shoe forms or footprints.

**Product: Pet bed (Pet/Mid-Range)**
→ Scene: Light hardwood floor in Scandinavian interior, soft diffused window light, woven basket in distant corner. Empty floor plane. No pet toys or bedding.

**Product: Desk lamp (Office/Mid-Range)**
→ Scene: Warm oak desk surface, soft window light, leather journal stack in far background edge. Clean central desk area. No lamps, cords, or office supplies.

UNIVERSAL OUTPUT FORMAT:

Return a SINGLE, COHESIVE PARAGRAPH of 150-200 words. Structure:

1. Establishing shot + environment type + surface material
2. Lighting setup (key light position, quality, modifiers, temperature)
3. Color palette (dominant hues, accent colors, psychology)
4. Spatial layout + depth layers + prop placement (if any - MUST be non-competitive and peripheral)
5. Surface texture + material qualities + reflection behavior
6. Atmosphere + negative space emphasis + compositing notes

Use cinematic, professional photography terminology ONLY.
NO lists, bullet points, or markdown.
NO product names, product shapes, or product-like objects.
NO human figures, mannequins, products, or animal subjects.
NO food, beverages, or edible props.
NO electronic screens displaying content.
NO text, logos, or branding elements.
NO shoes, books, or other product-like objects.

The background must:
- Be detailed and specific about environmental elements ONLY.
- Be suitable for high-end commercial e-commerce use across any product category.
- Explicitly emphasize that NO product or product-like object appears in the scene.
- Describe the scene using professional, cinematic, commercial photography language.
- Maintain immaculate negative space in the central product placement zone.

GENERATE UNIVERSAL PRODUCT BACKDROP DESCRIPTION NOW:"""



            logger.info("  → Background prompt built successfully (no OpenAI)")
            logger.info(f"  → Prompt length: {len(background_prompt)} characters")
            logger.info(f"  → Prompt preview: {background_prompt[:100]}...")
            
            return background_prompt

        except Exception as e:
            logger.error("  → Failed to build background prompt from description!")
            logger.error(f"  → Error: {str(e)}")
            logger.warning("  → Using fallback default background prompt")
            
            # Return a default background prompt if something unexpected happens
            mood_text = f" with a {mood.lower()} mood" if mood else ""
            style_text = f" in {style.lower()} style" if style else ""
            env_text = f" set in {environment.lower()}" if environment else ""
            
            return (
                f"A professional product photography background ONLY{mood_text}{style_text}{env_text}. "
                "Clean, commercial-grade backdrop with studio lighting, suitable for showcasing products. "
                "DO NOT include any products, objects, or items in the image. Generate ONLY the empty background "
                "scene with appropriate lighting and atmosphere. High-quality, professional composition with "
                "appropriate color palette and atmospheric elements. The background should have a clean center area "
                "for placing a product later while providing visual interest and context through lighting, textures, "
                "and environmental elements."
            )

    def extract_background_prompt(
        self,
        product_description: str,
        mood: Optional[str] = None,
        style: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call OpenAI to extract the background-generation prompt from a product description.
        Returns the prompt string that would be sent to Vertex for background generation.
        Use this when the next server sends only the product description and you need the prompt.

        Args:
            product_description: Description of the product (from next server request).
            mood: Optional mood/feeling for the background.
            style: Optional visual style.
            environment: Optional environment setting.

        Returns:
            {"prompt": str, "error": str | None}
        """
        if not self.openai_client:
            return {"prompt": "", "error": "OpenAI client not initialized"}
        try:
            env_parts = []
            if mood:
                env_parts.append(f"Mood: {mood}")   
            if style:
                env_parts.append(f"Style: {style}")
            if environment:
                env_parts.append(f"Environment: {environment}")
            context_block = "\n".join(env_parts) if env_parts else ""

            user_content = (
                "You are a professional commercial product scene designer.\n\n"
                "Your task is to generate a high-quality background image generation prompt for a product.\n\n"
                "The image will be used as a background only.\n"
                "The product itself will be added later in post-processing.\n"
                "Do NOT include the product in the scene.\n\n"
                "CRITICAL - NO PRODUCT IN OUTPUT:\n"
                "Your output prompt will be sent to an image model. You must NEVER mention, name, or describe the product itself in your output.\n"
                "Do NOT describe any object, item, or subject that could be drawn (no bottles, shoes, clothes, devices, etc.).\n"
                "Use the product description ONLY to infer: mood, category, price tier, and style — then describe ONLY the empty environment: surfaces, lighting, colors, atmosphere, depth.\n"
                "The output must read like a description of an empty set or stage — no main subject, no product, no props that look like the product.\n\n"
                "OBJECTIVE:\n"
                "Create a visually appealing, realistic, high-end background that enhances the product's perceived value.\n\n"
                "REQUIREMENTS:\n"
                "- The center of the image must remain clean and not cluttered (reserved space for product placement).\n"
                "- No text, logos, watermarks, or typography.\n"
                "- No main subject in the center. No product, no objects that could be mistaken for the product.\n"
                "- Background must match the product's mood, category, and positioning (infer from description; do not name the product).\n"
                "- Use cinematic lighting description.\n"
                "- Describe environment, materials, atmosphere, depth, and lens style only.\n"
                "- Must feel professional, commercial-grade, photorealistic.\n\n"
                "STYLE:\n"
                "- High resolution\n"
                "- Depth of field\n"
                "- Professional product photography lighting\n"
                "- Balanced composition\n"
                "- Natural shadows\n"
                "- Realistic textures\n\n"
                "OUTPUT FORMAT:\n"
                "Return ONLY the final image generation prompt. The prompt must describe an EMPTY scene (surfaces, lighting, atmosphere) with NO subject or product.\n"
                "Do not explain anything.\n"
                "Do not add extra commentary.\n\n"
                "PRODUCT DESCRIPTION (use only to infer mood/category/style — do not mention or describe the product in your output):\n"
                f"{product_description}\n"
            )
            if context_block:
                user_content += f"\nADDITIONAL CONTEXT:\n{context_block}\n"

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_content}],
                max_tokens=500,
            )
            prompt = (response.choices[0].message.content or "").strip()
            return {"prompt": prompt, "error": None}
        except Exception as e:
            logger.exception("extract_background_prompt failed")
            return {"prompt": "", "error": str(e)}

    def _generate_background_with_vertex(self, background_prompt: str) -> Optional[str]:
        """
        Generate a background image using Vertex AI Imagen 4.0.
        
        Args:
            background_prompt: The detailed prompt for background generation
            
        Returns:
            Path to the generated image file or None if failed
        """
        try:
            logger.info("  → Preparing Vertex AI Imagen 4.0 for background generation...")
            
            if not self.vertex_manager or not self.vertex_manager.is_available():
                raise RuntimeError("Vertex AI is not available")
            
            # Import required types
            try:
                from google.genai.types import GenerateImagesConfig
            except ImportError:
                raise RuntimeError("Google Vertex AI types not available")
            
            # Use Imagen 4.0 model
            model = "imagen-4.0-generate-001"
            
            logger.info(f"  → Model: {model}")
            logger.info(f"  → Aspect Ratio: 16:9 (1920x1080)")
            logger.info("  → Background Prompt:")
            logger.info("  " + "-" * 76)
            for line in background_prompt.split('\n'):
                logger.info(f"  {line}")
            logger.info("  " + "-" * 76)
            
            logger.info("  → Calling Vertex AI Imagen API...")

            # Mandatory prefix/suffix so the image model never draws the product (critical for compositing)
            vertex_prompt = (
                "BACKGROUND ONLY. Empty scene. Do not draw or depict any product, object, or main subject. "
                "No items in the center. This is an empty set for compositing. "
                f"{background_prompt} "
                "Remember: background only, no product, no main subject, empty center."
            )

            # Generate the background image
            # Note: Imagen 4.0 generates images in 16:9 aspect ratio which produces approximately 1920x1080
            result = self.vertex_manager.client.models.generate_images(
                model=model,
                prompt=vertex_prompt,
                config=GenerateImagesConfig(
                    aspect_ratio="16:9",  # 1920x1080 resolution (Full HD)
                    number_of_images=1,
                    safety_filter_level="BLOCK_ONLY_HIGH",
                    person_generation="ALLOW_ALL"
                )
            )
            
            if not result.generated_images:
                raise RuntimeError("No images were generated")
            
            generated_image = result.generated_images[0].image
            logger.info("  → Background generated successfully with Imagen 4.0!")
            logger.info(f"  → Image bytes: {len(generated_image.image_bytes)} bytes")
            
            # Save the generated image
            temp_dir = self._get_temp_dir()
            output_path = str(temp_dir / f"background_{uuid.uuid4()}.png")
            generated_image.save(output_path)
            
            logger.info(f"  → Generated image saved: {output_path}")
            
            # Return the local file path
            return output_path

        except Exception as e:
            logger.error(f"  → Vertex AI Imagen 4.0 generation failed!")
            logger.error(f"  → Error: {str(e)}")
            logger.error(f"  → Error type: {type(e).__name__}")
            return None

    def _upload_to_supabase(self, image_path: str, user_id: str) -> Optional[str]:
        """
        Upload the generated image to Supabase storage
        
        Args:
            image_path: Path to the image file to upload
            user_id: User ID for organizing uploads
            
        Returns:
            Public URL of the uploaded image or None if failed
        """
        try:
            logger.info("  → Checking Supabase connection...")
            if not supabase_manager.is_connected():
                logger.info("  → Supabase not connected, establishing connection...")
                supabase_manager.ensure_connection()
                logger.info("  → Supabase connection established")
            else:
                logger.info("  → Supabase already connected")
            
            # Read the image file
            logger.info(f"  → Reading image file from: {image_path}")
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            file_size = len(image_data)
            logger.info(f"  → File size to upload: {file_size} bytes")
            
            # Generate unique filename
            image_uuid = uuid.uuid4()
            filename = f"background-images/{user_id}/{image_uuid}.png"
            logger.info(f"  → Target storage path: {filename}")
            logger.info(f"  → Storage bucket: generated-content")
            
            # Upload to Supabase storage
            logger.info("  → Uploading to Supabase storage...")
            supabase_manager.client.storage.from_('generated-content').upload(
                path=filename,
                file=image_data,
                file_options={'content-type': 'image/png'}
            )
            logger.info("  → Upload successful")
            
            # Get public URL
            logger.info("  → Retrieving public URL...")
            public_url = supabase_manager.client.storage.from_('generated-content').get_public_url(filename)
            
            logger.info("  → Public URL generated successfully")
            logger.info(f"  → Public URL: {public_url}")
            
            return public_url

        except Exception as e:
            logger.error(f"  → Supabase upload failed!")
            logger.error(f"  → Error type: {type(e).__name__}")
            logger.error(f"  → Error message: {str(e)}")
            return None

    def _get_temp_dir(self) -> Path:
        """Get or create the temp directory for temporary files"""
        project_root = Path(__file__).parent.parent.parent
        temp_dir = project_root / "temp"
        temp_dir.mkdir(exist_ok=True)
        return temp_dir

    def _cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files"""
        logger.info(f"  → Cleaning up {len(file_paths)} temporary file(s)...")
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                if file_path and os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.info(f"  → [{i}/{len(file_paths)}] Deleting: {file_path} ({file_size} bytes)")
                    os.unlink(file_path)
                    logger.info(f"  → [{i}/{len(file_paths)}] Deleted successfully")
                else:
                    logger.warning(f"  → [{i}/{len(file_paths)}] File not found or invalid: {file_path}")
            except Exception as e:
                logger.warning(f"  → [{i}/{len(file_paths)}] Failed to delete: {file_path}")
                logger.warning(f"  → Error: {str(e)}")
        
        logger.info(f"  → Cleanup process completed")


# Global service instance
background_generation_service = BackgroundGenerationService()
