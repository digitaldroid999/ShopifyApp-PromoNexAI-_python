"""
Audio Generation Service for creating AI-powered audio content.
Handles test audio generation, speed analysis, script generation, and final audio creation.
"""

import uuid
import asyncio
import base64
from elevenlabs.types import CharacterAlignmentResponseModel
import openai
import requests
import struct
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from elevenlabs import ElevenLabs

from app.models import (
    AudioGenerationRequest,
    AudioGenerationResponse,
    AudioScriptGenerationRequest,
    AudioScriptGenerationResponse,
)
from app.utils.mongodb_manager import mongodb_manager
from app.utils.supabase_utils import supabase_manager
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class AudioGenerationService:
    """Service for generating AI-powered audio content"""

    def __init__(self):
        self.openai_client = None
        self.elevenlabs_client = None
        self.mongodb = mongodb_manager
        self._initialize_clients()
        
        # Language-specific test texts (same as test_audio_service)
        self.test_texts = {
            'en-US': "The first move is what sets everything in motion.",
            'en-CA': "The first move is what sets everything in motion.",
            'en-GB': "The first move is what sets everything in motion.",
            'es': "El primer movimiento es lo que pone todo en marcha.",
            'es-MX': "El primer movimiento es lo que pone todo en marcha.",
            'pt-BR': "O primeiro movimento é o que coloca tudo em movimento.",
            'fr': "Le premier mouvement est ce qui met tout en mouvement.",
            'de': "Der erste Zug ist es, der alles in Bewegung setzt.",
            'nl': "De eerste zet is wat alles in beweging zet.",
            'zh': "第一步是让一切开始运转的关键。",
            'ja': "最初の一歩がすべてを動かすきっかけとなる。",
            'ar': "الخطوة الأولى هي ما يضع كل شيء في حركة."
        }

    def _initialize_clients(self):
        """Initialize OpenAI and ElevenLabs clients"""
        try:
            # Initialize OpenAI
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not configured")

            # Initialize ElevenLabs
            if settings.ELEVENLABS_API_KEY:
                self.elevenlabs_client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
                logger.info("ElevenLabs client initialized successfully")
            else:
                logger.warning("ElevenLabs API key not configured")

        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            self.openai_client = None
            self.elevenlabs_client = None

    def generate_audio_script(self, request: AudioScriptGenerationRequest) -> AudioScriptGenerationResponse:
        """Generate audio script only using OpenAI. User can review/edit before calling generate_audio."""
        try:
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")

            logger.info(f"Generating audio script for short {request.short_id} with voice {request.voice_id} by user {request.user_id}")

            # Step 1: Get target language from shorts table
            target_language = asyncio.run(self._get_target_language(request.short_id))
            if not target_language:
                logger.warning(f"Could not fetch target_language for short_id {request.short_id}, defaulting to en-US")
                target_language = "en-US"

            # Step 2: Get or generate test audio and calculate speed
            test_audio_result = asyncio.run(self._get_or_generate_test_audio(request.voice_id, request.user_id, target_language))
            if not test_audio_result:
                raise Exception("Failed to get or generate test audio")

            words_per_minute = asyncio.run(self._analyze_audio_speed(test_audio_result["url"]))
            if not words_per_minute:
                words_per_minute = 150.0

            test_audio_duration = asyncio.run(self._calculate_audio_duration(test_audio_result["url"]))
            test_text = test_audio_result.get("text", "")

            # Step 3: Get short description
            short_info = asyncio.run(self._get_short_description(request.short_id))
            if not short_info:
                raise Exception(f"No short found for short_id: {request.short_id}")

            total_duration = 24
            style = "trendy-influencer-vlog"
            mood = "energetic"

            audio_script = asyncio.run(self._generate_audio_script(
                total_duration=total_duration,
                style=style,
                mood=mood,
                words_per_minute=words_per_minute,
                test_text=test_text,
                test_audio_duration=test_audio_duration,
                short_info=short_info,
            ))
            if not audio_script:
                raise Exception("Failed to generate audio script")

            return AudioScriptGenerationResponse(
                short_id=request.short_id,
                script=audio_script,
                words_per_minute=words_per_minute,
                target_duration_seconds=total_duration,
                message="Script generated successfully. Edit if needed, then call POST /generate-audio with this script.",
            )
        except Exception as e:
            logger.error(f"Audio script generation failed: {e}")
            raise Exception(f"Audio script generation failed: {str(e)}")

    def generate_audio(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        """Generate audio from the provided script (script comes from Next server; user may have edited it)."""
        try:
            if not self.elevenlabs_client:
                raise Exception("ElevenLabs client not initialized")

            logger.info(f"Starting audio generation for short {request.short_id} with voice {request.voice_id} by user {request.user_id}")
            audio_script = request.script

            # Step 1: Get target language (for test audio if needed)
            target_language = asyncio.run(self._get_target_language(request.short_id))
            if not target_language:
                target_language = "en-US"

            # Step 2: Get or generate test audio and calculate speed (for metadata and duration validation)
            test_audio_result = asyncio.run(self._get_or_generate_test_audio(request.voice_id, request.user_id, target_language))
            if not test_audio_result:
                raise Exception("Failed to get or generate test audio")

            words_per_minute = asyncio.run(self._analyze_audio_speed(test_audio_result["url"]))
            if not words_per_minute:
                words_per_minute = 150.0

            total_duration = 24  # Used for validation logging

            # Step 3: Generate final audio using ElevenLabs from the provided script
            logger.info("Generating final audio from provided script...")
            final_audio_result = asyncio.run(self._generate_final_audio_with_info(request.voice_id, audio_script, request.user_id))
            if not final_audio_result:
                raise Exception("Failed to generate final audio")

            # final_audio_url is the public URL from Supabase storage
            final_audio_url = final_audio_result["url"]
            upload_info = final_audio_result["upload_info"]
            subtitle_timing = final_audio_result.get("subtitle_timing", [])
            
            logger.info(f"Audio uploaded successfully. Public URL: {final_audio_url}")
            
            # Step 5: Deduct credits for successful audio generation
            credit_result = asyncio.run(self._deduct_credits_for_audio_generation(request, audio_script, request.short_id))
            
            # Step 6: Generate SRT content from subtitle timing
            srt_content = self._generate_srt_content(subtitle_timing)
            
            # Step 7: Get signed URL for duration calculation (signed URLs work for both public and private buckets)
            signed_audio_url = asyncio.run(self._get_signed_url_from_supabase_url(final_audio_url))
            if not signed_audio_url:
                logger.warning("Failed to get signed URL, using public URL for duration calculation")
                signed_audio_url = final_audio_url

            # Step 8: Calculate duration using signed URL
            duration = asyncio.run(self._calculate_audio_duration(signed_audio_url))
            logger.info(f"Actual audio duration: {duration:.2f} seconds (target: {total_duration} seconds, difference: {duration - total_duration:.2f}s)")
            
            # Validate duration doesn't exceed target
            if duration > total_duration:
                logger.error(f"⚠️ AUDIO TOO LONG! Generated audio is {duration:.2f}s, exceeding target of {total_duration}s by {duration - total_duration:.2f}s")
                logger.error(f"This audio may not fit properly in the video. Consider regenerating with stricter word count.")
            elif duration > total_duration * 0.95:
                logger.warning(f"⚠️ Audio duration ({duration:.2f}s) is very close to target ({total_duration}s). Only {total_duration - duration:.2f}s buffer remaining.")
            else:
                logger.info(f"✅ Audio duration ({duration:.2f}s) is within target ({total_duration}s). Buffer: {total_duration - duration:.2f}s")
            
            # Step 9: Save to Supabase audio_info table with PUBLIC URL (not signed URL)
            # The public URL is stable and doesn't expire, so it's stored in the database
            # Signed URLs are generated on-demand during merge/download phase
            logger.info(f"Saving public URL to database: {final_audio_url}")
            asyncio.run(self._save_audio_to_supabase_audio_info(
                request, final_audio_url, audio_script, words_per_minute, 
                duration, upload_info, credit_result, srt_content, subtitle_timing
            ))

            logger.info(f"Successfully generated audio for short {request.short_id}")
            
            # Return the result directly with signed URL
            return AudioGenerationResponse(
                voice_id=request.voice_id,
                user_id=request.user_id,
                short_id=request.short_id,
                audio_url=signed_audio_url,
                script=audio_script,
                words_per_minute=words_per_minute,
                duration=duration,
                created_at=datetime.now(timezone.utc),
                is_cached=False,
                message="Audio generated successfully",
                subtitle_timing=subtitle_timing
            )

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            err_str = str(e).lower()
            if "payment_issue" in err_str or "incomplete payment" in err_str or ("401" in err_str and "invoice" in err_str):
                raise Exception(
                    "ElevenLabs subscription has a failed or incomplete payment. "
                    "Complete the latest invoice at ElevenLabs to continue usage."
                )
            raise Exception(f"Audio generation failed: {str(e)}")

    async def _get_target_language(self, short_id: str) -> Optional[str]:
        """Get target_language from shorts table based on short_id"""
        try:
            if not supabase_manager.is_connected():
                supabase_manager.ensure_connection()
            
            # Get target_language from shorts table
            result = supabase_manager.client.table('shorts').select('target_language').eq('id', short_id).execute()
            
            if result.data and len(result.data) > 0:
                target_language = result.data[0].get('target_language', 'en-US')
                logger.info(f"Found target_language '{target_language}' for short_id {short_id}")
                return target_language
            else:
                logger.warning(f"No shorts record found for short_id: {short_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching target_language for short_id {short_id}: {e}")
            return None

    async def _get_short_description(self, short_id: str) -> Optional[Dict[str, Any]]:
        """Get short title and description from shorts table"""
        try:
            if not supabase_manager.is_connected():
                supabase_manager.ensure_connection()
            
            # Get title and description from shorts table
            result = supabase_manager.client.table('shorts').select('title, description').eq('id', short_id).execute()
            
            if result.data and len(result.data) > 0:
                short_data = result.data[0]
                logger.info(f"Found short info for short_id {short_id}: title='{short_data.get('title', 'N/A')}'")
                return short_data
            else:
                logger.warning(f"No shorts record found for short_id: {short_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching short description for short_id {short_id}: {e}")
            return None

    async def _get_or_generate_test_audio(self, voice_id: str, user_id: str, language: str = "en-US") -> Optional[Dict[str, Any]]:
        """Check MongoDB for existing test audio or generate it using test_audio_service logic"""
        try:
            # Check MongoDB for existing test audio with matching language
            if not self.mongodb.ensure_connection():
                logger.error("Failed to connect to MongoDB")
                return None

            # Query for existing test audio with voice_id and language
            query = {
                "voice_id": voice_id,
                "language": language,
                "type": "test_audio"
            }
            
            result = self.mongodb.database.test_audio.find_one(query)
            if result:
                logger.info(f"Found cached test audio for voice {voice_id} with language {language}")
                # Convert Supabase URL to signed URL for access
                signed_url = await self._get_signed_url_from_supabase_url(result['audio_url'])
                if signed_url:
                    return {
                        "url": signed_url,
                        "text": result.get('text', self.test_texts.get(language, self.test_texts['en-US']))
                    }
                else:
                    logger.warning("Failed to get signed URL for cached test audio, will regenerate")
                    # Fall through to generate new test audio

            # Generate new test audio if not found or signed URL failed
            logger.info(f"Generating new test audio for voice {voice_id} with language {language}")
            return await self._generate_test_audio(voice_id, user_id, language)

        except Exception as e:
            logger.error(f"Error getting or generating test audio: {e}")
            return None

    async def _get_signed_url_from_supabase_url(self, supabase_url: str) -> Optional[str]:
        """Extract path from Supabase URL and get signed URL"""
        try:
            # Extract bucket and path from Supabase URL
            # URL format: https://project.supabase.co/storage/v1/object/public/bucket/path
            if '/storage/v1/object/public/' in supabase_url:
                # Extract bucket and path from public URL
                parts = supabase_url.split('/storage/v1/object/public/')
                if len(parts) == 2:
                    bucket_and_path = parts[1]
                    bucket_path_parts = bucket_and_path.split('/', 1)
                    if len(bucket_path_parts) == 2:
                        bucket = bucket_path_parts[0]
                        path = bucket_path_parts[1]
                        
                        # Get signed URL using Supabase helper
                        signed_url_result = await supabase_manager.get_file_url(bucket, path, expires_in=3600)
                        if signed_url_result:
                            # Extract the actual URL from the result (could be dict or string)
                            if isinstance(signed_url_result, dict):
                                # Try different possible keys for the URL
                                signed_url = (signed_url_result.get('signedURL') or 
                                            signed_url_result.get('signedUrl') or 
                                            signed_url_result.get('url'))
                                if signed_url:
                                    logger.info(f"Successfully converted Supabase URL to signed URL")
                                    return signed_url
                                else:
                                    logger.error(f"Could not extract URL from signed URL result: {signed_url_result}")
                                    return None
                            elif isinstance(signed_url_result, str):
                                logger.info(f"Successfully converted Supabase URL to signed URL")
                                return signed_url_result
                            else:
                                logger.error(f"Unexpected signed URL result type: {type(signed_url_result)}")
                                return None
                        else:
                            logger.error("Failed to get signed URL from Supabase helper")
                            return None
            
            logger.error(f"Could not parse Supabase URL: {supabase_url}")
            return None

        except Exception as e:
            logger.error(f"Error converting Supabase URL to signed URL: {e}")
            return None

    async def _generate_test_audio(self, voice_id: str, user_id: str, language: str = "en-US") -> Optional[Dict[str, Any]]:
        """Generate test audio using ElevenLabs (similar to test_audio_service)"""
        try:
            if not self.elevenlabs_client:
                logger.error("ElevenLabs client not initialized")
                return None

            if not supabase_manager.is_connected():
                logger.error("Supabase client not connected")
                return None

            # Use language-specific test text
            test_text = self.test_texts.get(language, self.test_texts['en-US'])
            logger.info(f"Generating test audio for voice {voice_id} with language {language}")
            logger.info(f"Test text: {test_text}")

            # Generate audio using ElevenLabs
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                voice_id=voice_id,
                output_format=settings.ELEVENLABS_DEFAULT_OUTPUT_FORMAT,
                text=test_text,
                model_id=settings.ELEVENLABS_DEFAULT_MODEL
            )

            # Convert generator to bytes
            audio_data = b''.join(audio_generator)

            # Upload to Supabase storage (test-audios bucket)
            upload_info = await self._upload_test_audio_to_supabase(audio_data, voice_id, language)
            if not upload_info:
                logger.error("Failed to upload test audio to Supabase storage")
                return None

            # Save to MongoDB with language
            await self._save_test_audio_to_mongodb(voice_id, upload_info["url"], user_id, language, test_text)

            logger.info(f"Successfully generated and uploaded test audio for voice {voice_id}")
            return {
                "url": upload_info["url"],
                "text": test_text
            }

        except Exception as e:
            logger.error(f"Error generating test audio: {e}")
            return None

    async def _analyze_audio_speed(self, audio_url: str) -> Optional[float]:
        """Analyze audio speed to determine words per minute"""
        try:
            # Download audio file
            response = requests.get(audio_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to download audio: {response.status_code}")
                logger.warning("Using default WPM due to audio download failure")
                return 150.0  # Default WPM

            audio_data = response.content
            
            # For now, we'll use a simple estimation based on the test text
            # In a real implementation, you might use speech recognition to get exact timing
            test_text = self.test_texts['en-US']
            word_count = len(test_text.split())
            
            # Estimate duration based on typical speech patterns
            # This is a simplified approach - in production you'd analyze the actual audio
            estimated_duration = len(audio_data) / 16000  # Rough estimation
            words_per_minute = (word_count / estimated_duration) * 60 if estimated_duration > 0 else 150.0
            
            logger.info(f"Analyzed audio speed: {words_per_minute:.2f} WPM")
            return words_per_minute

        except Exception as e:
            logger.error(f"Error analyzing audio speed: {e}")
            logger.warning("Using default WPM due to analysis error")
            return 150.0  # Default WPM

    async def _generate_audio_script(
        self, 
        total_duration: int,
        style: str,
        mood: str,
        words_per_minute: float, 
        test_text: str = "", 
        test_audio_duration: float = 0,
        short_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Generate audio script using OpenAI based on video settings and speed"""
        try:
            if not self.openai_client:
                logger.error("OpenAI client not initialized")
                return None
            
            # Calculate target word count with 25% safety buffer to ensure audio stays under duration
            # Using larger buffer because GPT may not strictly follow word count and actual speech rate can vary
            target_word_count = (total_duration / 60) * words_per_minute
            conservative_word_count = int(target_word_count * 0.75)  # 25% buffer to ensure audio ≤ target duration
            estimated_characters = conservative_word_count * 5  # Rough estimate: ~5 characters per word
            
            logger.info(f"Target duration: {total_duration}s, WPM: {words_per_minute:.1f}, Target words: {conservative_word_count} (with 25% buffer, original: {int(target_word_count)} words)")
            
            system_message = f"""You are an expert copywriter for social media product promotion videos.
Create a clean, spoken audio script with NO stage directions, music cues, emojis, or brackets.

SCRIPT STRUCTURE (3 parts):
1. HOOK (2-3 seconds): Start with a fun, intriguing, or relatable question or statement that grabs attention immediately.
2. MAIN (80% of script): Present product features and benefits, speaking in a friendly, conversational tone. Use natural speech patterns, like laughter, pauses, or excitement when appropriate. Focus on how the product makes life easier or more enjoyable.
3. CTA (2-3 seconds): Clear call-to-action that's casual but persuasive. Encourage the listener to take action with a friendly prompt or playful invitation.

REQUIREMENTS:
- Duration: {total_duration} seconds (MUST NOT EXCEED - audio must be shorter than or equal to this)
- Word count: EXACTLY {conservative_word_count} words (strict requirement - count carefully)
- Character count: Approximately {estimated_characters} characters
- Style: {style}
- Mood: {mood}
- Write ONLY what will be spoken - no [music cues], no emojis, no stage directions
- Keep it natural, conversational, and engaging
- Focus on product benefits and creating desire

VOICE CALIBRATION:
This voice speaks at {words_per_minute:.1f} words per minute.
Example: "{test_text}" took {test_audio_duration:.2f} seconds to speak.

WORD COUNT VALIDATION (CRITICAL - MUST FOLLOW):
The script MUST contain MAXIMUM {conservative_word_count} words - absolutely no more!
Before returning the script, count every single word. The audio MUST be under {total_duration} seconds.
If you exceed {conservative_word_count} words, the audio will be TOO LONG and rejected.
It's better to have slightly fewer words than too many. Aim for {conservative_word_count} words or less.

OUTPUT FORMAT: Return only the clean spoken script without any formatting, labels, or non-spoken elements."""

            # Get short title and description
            short_title = short_info.get('title', 'Product') if short_info else 'Product'
            short_description = short_info.get('description', 'Amazing product') if short_info else 'Amazing product'

            user_message = f"""Create a promotional audio script for this product:

Title: {short_title}
Description: {short_description}"""

            logger.info("Sending request to OpenAI for script generation...")
            # Limit max_tokens based on conservative word count to prevent overly long scripts
            # Each word is roughly 1.3 tokens, so add some buffer for response formatting
            max_tokens_limit = int(conservative_word_count * 1.5)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=min(max_tokens_limit, 300),  # Cap at 300 tokens max for 24s videos
                temperature=0.7
            )
  
            if response.choices and response.choices[0].message.content:
                script = response.choices[0].message.content.strip()
                actual_word_count = len(script.split())
                logger.info(f"Generated audio script: {len(script)} characters, {actual_word_count} words (target: {conservative_word_count} words)")
                
                # Truncate script if it exceeds target word count to ensure audio stays under duration
                if actual_word_count > conservative_word_count:
                    logger.warning(f"⚠️ Generated script has {actual_word_count} words, EXCEEDING target of {conservative_word_count} words. Truncating to fit duration...")
                    words = script.split()
                    truncated_words = words[:conservative_word_count]
                    script = ' '.join(truncated_words)
                    
                    # Ensure script ends properly (try to end at sentence boundary if possible)
                    if script and not script[-1] in '.!?':
                        # Try to find last sentence boundary
                        last_period = max(script.rfind('.'), script.rfind('!'), script.rfind('?'))
                        if last_period > len(script) * 0.7:  # If we're at least 70% through
                            script = script[:last_period + 1]
                        else:
                            script = script + '.'
                    
                    actual_word_count = len(script.split())
                    logger.info(f"✂️ Script truncated to {actual_word_count} words to ensure audio ≤ {total_duration}s")
                elif actual_word_count < conservative_word_count * 0.9:
                    logger.warning(f"Generated script has {actual_word_count} words, which is less than target of {conservative_word_count} words by more than 10%")
                
                logger.info(f"Final script: {len(script)} characters, {actual_word_count} words")
                return script

            return None

        except Exception as e:
            logger.error(f"Failed to generate audio script: {e}")
            return None


    async def _generate_final_audio_with_info(self, voice_id: str, script: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Generate final audio using ElevenLabs with timestamps and return URL, upload info, and timing data"""
        try:
            if not self.elevenlabs_client:
                logger.error("ElevenLabs client not initialized")
                return None

            if not supabase_manager.is_connected():
                logger.error("Supabase client not connected")
                return None

            logger.info(f"Generating final audio with timestamps for voice {voice_id}")

            # Generate audio using ElevenLabs with timestamps
            response = self.elevenlabs_client.text_to_speech.convert_with_timestamps(
                voice_id=voice_id,
                output_format=settings.ELEVENLABS_DEFAULT_OUTPUT_FORMAT,
                text=script,
                model_id=settings.ELEVENLABS_DEFAULT_MODEL
            )

            # Convert audio data to bytes
            # response.audio_base64 is a single base64-encoded string
            logger.info(f"Received audio response, decoding base64 audio data")
            audio_data = base64.b64decode(response.audio_base_64)
            logger.info(f"Total audio data size: {len(audio_data)} bytes")

            # Upload to Supabase storage
            upload_info = await self._upload_audio_to_supabase(audio_data, voice_id, user_id, "generated-audio")
            if not upload_info:
                logger.error("Failed to upload final audio to Supabase storage")
                return None

            # Process timing data for subtitles
            subtitle_timing = self._process_subtitle_timing(response.alignment, script)

            logger.info(f"Successfully generated and uploaded final audio for voice {voice_id}")
            return {
                "url": upload_info["url"],
                "upload_info": upload_info,
                "subtitle_timing": subtitle_timing
            }

        except Exception as e:
            logger.error(f"Error generating final audio: {e}")
            raise  # Propagate so caller can return a clear message (e.g. payment_issue)

    def _generate_srt_content(self, subtitle_timing: List[Dict[str, Any]]) -> str:
        """Generate SRT format content from subtitle timing segments"""
        try:
            if not subtitle_timing:
                logger.warning("No subtitle timing data available for SRT generation")
                return ""
            
            srt_lines = []
            for i, segment in enumerate(subtitle_timing, start=1):
                # Convert times to SRT format (HH:MM:SS,mmm)
                start_time = self._format_srt_time(segment['start_time'])
                end_time = self._format_srt_time(segment['end_time'])
                text = segment['text']
                
                # Add SRT entry
                srt_lines.append(f"{i}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(text)
                srt_lines.append("")  # Empty line between entries
            
            srt_content = "\n".join(srt_lines)
            logger.info(f"Generated SRT content with {len(subtitle_timing)} segments")
            return srt_content
            
        except Exception as e:
            logger.error(f"Error generating SRT content: {e}")
            return ""
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time in seconds to SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _process_subtitle_timing(self, alignment_data: CharacterAlignmentResponseModel , script: str) -> List[Dict[str, Any]]:
        """Process ElevenLabs alignment data into subtitle timing segments"""
        try:
            if not alignment_data or not hasattr(alignment_data, 'characters'):
                logger.warning("No alignment data available for subtitle timing")
                return []

            # Extract character-level timing data
            characters = alignment_data.characters
            start_times = alignment_data.character_start_times_seconds
            end_times = alignment_data.character_end_times_seconds

            if not all([characters, start_times, end_times]):
                logger.warning("Incomplete alignment data for subtitle timing")
                return []

            # Group characters into words for subtitle segments
            word_segments = []
            current_word = ""
            current_start = None
            current_end = None
            
            for i, char in enumerate(characters):
                if char in [' ', '\n', ',', '.', '!', '?', ':', ';']:
                    # End of word
                    if current_word.strip() and current_start is not None:
                        # Include punctuation with the word if present
                        word_text = current_word.strip()
                        if char in [',', '.', '!', '?', ':', ';']:
                            word_text += char
                        
                        word_segments.append({
                            "text": word_text,
                            "start_time": current_start,
                            "end_time": end_times[i] if char in [',', '.', '!', '?', ':', ';'] else (current_end or end_times[i]),
                            "duration": (end_times[i] if char in [',', '.', '!', '?', ':', ';'] else (current_end or end_times[i])) - current_start
                        })
                    
                    # Reset for next word
                    current_word = ""
                    current_start = None
                    current_end = None
                else:
                    # Add character to current word
                    current_word += char
                    if current_start is None:
                        current_start = start_times[i]
                    current_end = end_times[i]

            # Add the last word if exists
            if current_word.strip() and current_start is not None:
                word_segments.append({
                    "text": current_word.strip(),
                    "start_time": current_start,
                    "end_time": current_end,
                    "duration": current_end - current_start
                })

            # Group words into small phrases (3-5 words per segment) for better subtitle readability
            phrase_segments = []
            words_per_segment = 4  # Optimal words per subtitle segment
            current_phrase = []
            
            for i, word_segment in enumerate(word_segments):
                current_phrase.append(word_segment)
                
                # Create a new segment if:
                # 1. We have enough words
                # 2. Current word ends with sentence-ending punctuation (not commas, semicolons, etc.)
                # 3. It's the last word
                is_last_word = (i == len(word_segments) - 1)
                ends_with_sentence_punctuation = word_segment["text"].rstrip().endswith(('.', '!', '?'))
                has_enough_words = len(current_phrase) >= words_per_segment
                
                if has_enough_words or ends_with_sentence_punctuation or is_last_word:
                    if current_phrase:
                        # Combine words into a phrase
                        phrase_text = " ".join([w["text"] for w in current_phrase])
                        phrase_start = current_phrase[0]["start_time"]
                        phrase_end = current_phrase[-1]["end_time"]
                        
                        phrase_segments.append({
                            "text": phrase_text,
                            "start_time": phrase_start,
                            "end_time": phrase_end,
                            "duration": phrase_end - phrase_start
                        })
                        
                        # Reset for next phrase
                        current_phrase = []

            logger.info(f"Processed {len(phrase_segments)} subtitle segments from {len(word_segments)} words")
            return phrase_segments

        except Exception as e:
            logger.error(f"Error processing subtitle timing: {e}")
            return []

    async def _calculate_audio_duration(self, audio_url: str) -> float:
        """Calculate audio duration in seconds using MP3 metadata parsing"""
        try:
            # Download audio file
            response = requests.get(audio_url)
            if response.status_code != 200:
                logger.error(f"Failed to download audio for duration calculation: {response.status_code}")
                return 0.0

            audio_data = response.content
            
            # Try to parse MP3 metadata to get exact duration
            try:
                duration = self._parse_mp3_duration(audio_data)
                if duration > 0:
                    logger.info(f"Audio duration from metadata: {duration:.2f}s")
                    return duration
            except Exception as metadata_error:
                logger.warning(f"Failed to parse MP3 metadata: {metadata_error}")
            
            # Fallback to estimation based on file size and bitrate
            file_size_bytes = len(audio_data)
            estimated_bitrate_kbps = 128  # MP3 at 128 kbps
            bytes_per_second = (estimated_bitrate_kbps * 1000) / 8
            estimated_duration = file_size_bytes / bytes_per_second
            
            logger.info(f"Audio file size: {file_size_bytes} bytes, estimated duration: {estimated_duration:.2f}s")
            return estimated_duration

        except Exception as e:
            logger.error(f"Error calculating audio duration: {e}")
            return 0.0

    def _parse_mp3_duration(self, audio_data: bytes) -> float:
        """Parse MP3 duration from metadata (simplified version)"""
        try:
            # Look for ID3v2 tag
            if audio_data[:3] == b'ID3':
                # Skip ID3v2 tag
                tag_size = struct.unpack('>I', b'\x00' + audio_data[6:9])[0]
                audio_data = audio_data[10 + tag_size:]
            
            # Look for MP3 frame header
            for i in range(len(audio_data) - 4):
                if audio_data[i:i+2] == b'\xff\xfb' or audio_data[i:i+2] == b'\xff\xfa':
                    # Found MP3 frame, estimate duration based on bitrate
                    # This is a simplified approach - in production you'd use a proper MP3 parser
                    bitrate = self._get_mp3_bitrate(audio_data[i:i+4])
                    if bitrate > 0:
                        # Estimate duration based on file size and bitrate
                        file_size_bytes = len(audio_data)
                        bytes_per_second = (bitrate * 1000) / 8
                        duration = file_size_bytes / bytes_per_second
                        return duration
                    break
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error parsing MP3 duration: {e}")
            return 0.0

    def _get_mp3_bitrate(self, frame_header: bytes) -> int:
        """Extract bitrate from MP3 frame header"""
        try:
            if len(frame_header) < 4:
                return 0
            
            # MP3 bitrate lookup table (simplified)
            bitrate_table = {
                0b0001: 32, 0b0010: 40, 0b0011: 48, 0b0100: 56,
                0b0101: 64, 0b0110: 80, 0b0111: 96, 0b1000: 112,
                0b1001: 128, 0b1010: 160, 0b1011: 192, 0b1100: 224,
                0b1101: 256, 0b1110: 320
            }
            
            # Extract bitrate bits (bits 12-15 of second byte)
            bitrate_bits = (frame_header[2] >> 4) & 0x0F
            return bitrate_table.get(bitrate_bits, 128)  # Default to 128 kbps
            
        except Exception as e:
            logger.error(f"Error extracting MP3 bitrate: {e}")
            return 128  # Default bitrate

    async def _upload_audio_to_supabase(self, audio_data: bytes, voice_id: str, user_id: str, audio_type: str) -> Optional[Dict[str, Any]]:
        """Upload audio data to Supabase storage and return detailed upload info"""
        try:
            # Ensure bucket exists
            if not await self._ensure_audio_bucket_exists():
                logger.error("Failed to ensure audio bucket exists")
                return None

            # Create unique filename with UUID (following JavaScript pattern)
            audio_uuid = uuid.uuid4()
            filename = f"{user_id}/{audio_uuid}.mp3"
            
            logger.info(f"Uploading audio to Supabase storage: {filename}")

            # Upload to Supabase storage
            result = supabase_manager.client.storage.from_('audio-files').upload(
                path=filename,
                file=audio_data,
                file_options={"content-type": "audio/mpeg"}
            )

            # Check for upload errors
            if hasattr(result, 'error') and result.error:
                logger.error(f"Failed to upload audio: {result.error}")
                return None

            # Get public URL
            public_url = supabase_manager.client.storage.from_('audio-files').get_public_url(filename)

            # Calculate file size
            file_size = len(audio_data)

            upload_info = {
                "url": public_url,
                "path": filename,
                "size": file_size,
                "mimeType": "audio/mpeg",
                "uuid": str(audio_uuid)
            }

            logger.info(f"Successfully uploaded audio to: {public_url}")
            return upload_info

        except Exception as e:
            logger.error(f"Error uploading audio to Supabase: {e}")
            return None

    async def _ensure_audio_bucket_exists(self) -> bool:
        """Ensure the audio-files bucket exists in Supabase storage"""
        try:
            if not supabase_manager.is_connected():
                logger.error("Supabase client not connected")
                return False

            # Try to list all buckets to see if audio-files exists
            try:
                buckets = supabase_manager.client.storage.list_buckets()
                bucket_names = [bucket.name for bucket in buckets]
                
                if 'audio-files' in bucket_names:
                    logger.info("audio-files bucket already exists")
                    return True
                else:
                    logger.info("audio-files bucket not found, creating it...")
                    # Create the bucket with public access
                    result = supabase_manager.client.storage.create_bucket(
                        'audio-files', 
                        options={"public": True}
                    )
                    
                    if result:
                        logger.info("Successfully created audio-files bucket")
                        return True
                    else:
                        logger.error("Failed to create audio-files bucket")
                        return False
                        
            except Exception as e:
                logger.error(f"Error checking/creating audio-files bucket: {e}")
                return False
                    
        except Exception as e:
            logger.error(f"Error ensuring audio-files bucket: {e}")
            return False

    async def _upload_test_audio_to_supabase(self, audio_data: bytes, voice_id: str, language: str) -> Optional[Dict[str, Any]]:
        """Upload test audio data to Supabase test-audios bucket"""
        try:
            # Ensure bucket exists
            if not await self._ensure_test_audio_bucket_exists():
                logger.error("Failed to ensure test-audios bucket exists")
                return None

            # Create unique filename with voice_id, language, and UUID
            audio_uuid = uuid.uuid4().hex[:8]
            filename = f"test-audios/{voice_id}_{language}_{audio_uuid}.mp3"
            
            logger.info(f"Uploading test audio to Supabase storage: {filename}")

            # Upload to Supabase storage
            result = supabase_manager.client.storage.from_('test-audios').upload(
                path=filename,
                file=audio_data,
                file_options={"content-type": "audio/mpeg"}
            )

            # Check for upload errors
            if hasattr(result, 'error') and result.error:
                logger.error(f"Failed to upload test audio: {result.error}")
                return None

            # Get public URL
            public_url = supabase_manager.client.storage.from_('test-audios').get_public_url(filename)

            # Calculate file size
            file_size = len(audio_data)

            upload_info = {
                "url": public_url,
                "path": filename,
                "size": file_size,
                "mimeType": "audio/mpeg",
                "uuid": audio_uuid
            }

            logger.info(f"Successfully uploaded test audio to: {public_url}")
            return upload_info

        except Exception as e:
            logger.error(f"Error uploading test audio to Supabase: {e}")
            return None

    async def _ensure_test_audio_bucket_exists(self) -> bool:
        """Ensure the test-audios bucket exists in Supabase storage"""
        try:
            if not supabase_manager.is_connected():
                logger.error("Supabase client not connected")
                return False

            # Try to list all buckets to see if test-audios exists
            try:
                buckets = supabase_manager.client.storage.list_buckets()
                bucket_names = [bucket.name for bucket in buckets]
                
                if 'test-audios' in bucket_names:
                    logger.info("test-audios bucket already exists")
                    return True
                else:
                    logger.info("test-audios bucket not found, creating it...")
                    # Create the bucket with public access
                    result = supabase_manager.client.storage.create_bucket(
                        'test-audios', 
                        options={"public": True}
                    )
                    
                    if result:
                        logger.info("Successfully created test-audios bucket")
                        return True
                    else:
                        logger.error("Failed to create test-audios bucket")
                        return False
                        
            except Exception as e:
                logger.error(f"Error checking/creating test-audios bucket: {e}")
                return False
                    
        except Exception as e:
            logger.error(f"Error ensuring test-audios bucket: {e}")
            return False

    async def _save_test_audio_to_mongodb(self, voice_id: str, audio_url: str, user_id: str, language: str = "en-US", test_text: str = "") -> bool:
        """Save test audio info to MongoDB"""
        try:
            if not self.mongodb.ensure_connection():
                logger.error("Failed to connect to MongoDB")
                return False

            audio_doc = {
                "voice_id": voice_id,
                "audio_url": audio_url,
                "user_id": user_id,
                "language": language,
                "text": test_text,
                "type": "test_audio",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }

            result = self.mongodb.database.test_audio.insert_one(audio_doc)
            if result.inserted_id:
                logger.info(f"Saved test audio to MongoDB with ID: {result.inserted_id} for language: {language}")
                return True
            else:
                logger.error("Failed to save test audio to MongoDB")
                return False

        except Exception as e:
            logger.error(f"Error saving test audio to MongoDB: {e}")
            return False

    async def _deduct_credits_for_audio_generation(self, request: AudioGenerationRequest, script: str, task_id: str) -> Dict[str, Any]:
        """Deduct credits for successful audio generation"""
        try:
            # Import here to avoid circular imports
            from app.utils.credit_utils import deduct_credits, check_user_credits
            
            # Create description for credit deduction
            description = f"Generated audio for short {request.short_id}: {script[:50]}..."
            
            # Deduct credits for audio generation
            success = deduct_credits(
                user_id=request.user_id,
                action_name="generate_audio",
                reference_id=request.short_id,  # Use short_id instead of task_id
                reference_type="audio_generation",
                description=description
            )
            
            if success:
                # Get new balance after deduction
                credit_info = check_user_credits(request.user_id)
                new_balance = credit_info.get("available_credits")
                logger.info(f"Successfully deducted credits for audio generation for user {request.user_id}, new balance: {new_balance}")
                return {
                    "success": True,
                    "credits_used": 1,  # Assuming 1 credit per audio generation
                    "new_balance": new_balance,
                    "description": description
                }
            else:
                logger.warning(f"Failed to deduct credits for audio generation for user {request.user_id}")
                return {
                    "success": False,
                    "credits_used": 0,
                    "new_balance": None,
                    "description": description,
                    "error": "Credit deduction failed"
                }
                
        except Exception as e:
            logger.error(f"Error deducting credits for audio generation: {e}")
            return {
                "success": False,
                "credits_used": 0,
                "new_balance": None,
                "error": str(e)
            }

    async def _save_failed_audio_to_supabase(self, request: AudioGenerationRequest, audio_url: str, script: str, words_per_minute: float, duration: float, upload_info: Optional[Dict[str, Any]] = None, credit_result: Optional[Dict[str, Any]] = None, error_message: str = "") -> bool:
        """Save failed audio info to Supabase audio_info table"""
        try:
            if not supabase_manager.is_connected():
                logger.error("Supabase client not connected")
                return False

            # Check if audio_info record already exists for this short
            existing_audio = await self._get_existing_audio_info(request.short_id)
            
            audio_data = {
                "user_id": request.user_id,
                "short_id": request.short_id,
                "voice_id": request.voice_id,
                "generated_audio_url": audio_url if audio_url else None,
                "status": "failed",
                "metadata": {
                    "script": script if script else None,
                    "words_per_minute": words_per_minute if words_per_minute else None,
                    "duration": duration if duration else None,
                    "storage_path": upload_info.get("path") if upload_info else None,
                    "file_size": upload_info.get("size") if upload_info else None,
                    "mime_type": upload_info.get("mimeType") if upload_info else "audio/mpeg",
                    "uuid": upload_info.get("uuid") if upload_info else None,
                    "credit_info": credit_result if credit_result else None,
                    "error": error_message,
                    "failed_at": datetime.now(timezone.utc).isoformat()
                }
            }

            if existing_audio:
                # Update existing record with failed status
                logger.info(f"Updating existing audio_info record with failed status for short {request.short_id}")
                result = supabase_manager.client.table('audio_info').update(audio_data).eq('short_id', request.short_id).execute()
                
                if result.data:
                    logger.info(f"Successfully updated audio_info record with failed status for short {request.short_id}")
                    return True
                else:
                    logger.error(f"Failed to update audio_info record with failed status for short {request.short_id}")
                    return False
            else:
                # Create new record with failed status
                logger.info(f"Creating new audio_info record with failed status for short {request.short_id}")
                result = supabase_manager.client.table('audio_info').insert(audio_data).execute()
                
                if result.data:
                    logger.info(f"Successfully created audio_info record with failed status for short {request.short_id}")
                    return True
                else:
                    logger.error(f"Failed to create audio_info record with failed status for short {request.short_id}")
                    return False

        except Exception as e:
            logger.error(f"Error saving failed audio info to Supabase: {e}")
            return False

    async def _save_audio_to_supabase_audio_info(self, request: AudioGenerationRequest, audio_url: str, script: str, words_per_minute: float, duration: float, upload_info: Optional[Dict[str, Any]] = None, credit_result: Optional[Dict[str, Any]] = None, srt_content: str = "", subtitles: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Save generated audio info to Supabase audio_info table"""
        try:
            if not supabase_manager.is_connected():
                logger.error("Supabase client not connected")
                return False

            # Check if audio_info record already exists for this short
            existing_audio = await self._get_existing_audio_info(request.short_id)
            
            # Build metadata according to the new format
            metadata = {
                "modelId": settings.ELEVENLABS_DEFAULT_MODEL,
                "file_size": upload_info.get("size") if upload_info else 0,
                "mime_type": "audio/mp3",
                "textLength": len(script),
                "srt_content": srt_content,
                "outputFormat": settings.ELEVENLABS_DEFAULT_OUTPUT_FORMAT,
                "storage_path": upload_info.get("path") if upload_info else None,
                "creditDeducted": credit_result.get("success", False) if credit_result else False,
                "originalTextLength": len(script)
            }
            
            # Add newBalance if available from credit result
            if credit_result and credit_result.get("new_balance") is not None:
                metadata["newBalance"] = credit_result.get("new_balance")
            
            # Validate that audio_url is a public URL (should contain /storage/v1/object/public/)
            if '/storage/v1/object/public/' not in audio_url:
                logger.warning(
                    f"Audio URL does not appear to be a public URL format. "
                    f"Expected format: .../storage/v1/object/public/bucket/path, got: {audio_url[:100]}..."
                )
            else:
                logger.info(f"Saving public URL to PostgreSQL audio_info table: {audio_url[:100]}...")
            
            audio_data = {
                "user_id": request.user_id,
                "short_id": request.short_id,
                "voice_id": request.voice_id,
                "generated_audio_url": audio_url,  # Public URL stored in database
                "status": "completed",
                "metadata": metadata,
                "subtitles": subtitles if subtitles else []
            }

            if existing_audio:
                # Update existing record
                logger.info(f"Updating existing audio_info record for short {request.short_id}")
                result = supabase_manager.client.table('audio_info').update(audio_data).eq('short_id', request.short_id).execute()
                
                if result.data:
                    logger.info(f"Successfully updated audio_info record for short {request.short_id}")
                    return True
                else:
                    logger.error(f"Failed to update audio_info record for short {request.short_id}")
                    return False
            else:
                # Create new record
                logger.info(f"Creating new audio_info record for short {request.short_id}")
                result = supabase_manager.client.table('audio_info').insert(audio_data).execute()
                
                if result.data:
                    logger.info(f"Successfully created audio_info record for short {request.short_id}")
                    return True
                else:
                    logger.error(f"Failed to create audio_info record for short {request.short_id}")
                    return False

        except Exception as e:
            logger.error(f"Error saving audio info to Supabase: {e}")
            return False

    async def _get_existing_audio_info(self, short_id: str) -> Optional[Dict[str, Any]]:
        """Get existing audio_info record for a short"""
        try:
            if not supabase_manager.is_connected():
                logger.error("Supabase client not connected")
                return None

            result = supabase_manager.client.table('audio_info').select('*').eq('short_id', short_id).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            
            return None

        except Exception as e:
            logger.error(f"Error getting existing audio_info: {e}")
            return None




# Global service instance
audio_generation_service = AudioGenerationService()
