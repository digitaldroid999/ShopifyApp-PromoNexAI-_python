"""
Audio Generation Service for creating AI-powered audio content.
Handles test audio generation, speed analysis, script generation, and final audio creation.
"""

import os
import uuid
import asyncio
import base64
from pathlib import Path
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

            target_language = request.target_language or "en-US"

            # Step 1: Get or generate test audio and calculate speed
            test_audio_result = asyncio.run(self._get_or_generate_test_audio(request.voice_id, request.user_id, target_language))
            if not test_audio_result:
                raise Exception("Failed to get or generate test audio")

            if "audio_bytes" in test_audio_result:
                words_per_minute = self._analyze_audio_speed_from_bytes(
                    test_audio_result["audio_bytes"], test_audio_result.get("text", "")
                )
                test_audio_duration = self._calculate_audio_duration_from_bytes(test_audio_result["audio_bytes"])
            else:
                words_per_minute = asyncio.run(self._analyze_audio_speed(test_audio_result["url"]))
                test_audio_duration = asyncio.run(self._calculate_audio_duration(test_audio_result["url"]))
            if not words_per_minute:
                words_per_minute = 150.0
            test_text = test_audio_result.get("text", "")

            # Step 2: Build product info from request (product_description sent by client)
            short_info = {"title": "Product", "description": request.product_description or "Amazing product for your audience."}

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

            target_language = "en-US"

            # Step 1: Get or generate test audio and calculate speed (for metadata and duration validation)
            test_audio_result = asyncio.run(self._get_or_generate_test_audio(request.voice_id, request.user_id, target_language))
            if not test_audio_result:
                raise Exception("Failed to get or generate test audio")

            if "audio_bytes" in test_audio_result:
                words_per_minute = self._analyze_audio_speed_from_bytes(
                    test_audio_result["audio_bytes"], test_audio_result.get("text", "")
                )
            else:
                words_per_minute = asyncio.run(self._analyze_audio_speed(test_audio_result["url"]))
            if not words_per_minute:
                words_per_minute = 150.0

            total_duration = 24  # Used for validation logging

            # Step 3: Generate final audio using ElevenLabs (returns bytes + subtitle timing, no Supabase upload)
            logger.info("Generating final audio from provided script...")
            final_audio_result = asyncio.run(self._generate_final_audio_with_info(request.voice_id, audio_script, request.user_id))
            if not final_audio_result:
                raise Exception("Failed to generate final audio")

            audio_data = final_audio_result["audio_data"]
            subtitle_timing = final_audio_result.get("subtitle_timing", [])

            # Step 4: Save to public folder and get relative URL
            final_audio_url = self._save_audio_to_public(audio_data, request.user_id, request.short_id)
            if not final_audio_url:
                raise Exception("Failed to save audio to public folder")

            logger.info(f"Audio saved to public folder. URL: {final_audio_url}")

            upload_info = {"path": final_audio_url, "size": len(audio_data), "mimeType": "audio/mpeg"}

            # Step 5: Deduct credits for successful audio generation
            credit_result = asyncio.run(self._deduct_credits_for_audio_generation(request, audio_script, request.short_id))

            # Step 6: Generate SRT content from subtitle timing
            srt_content = self._generate_srt_content(subtitle_timing)

            # Step 7: Calculate duration from bytes (no URL fetch)
            duration = self._calculate_audio_duration_from_bytes(audio_data)
            logger.info(f"Actual audio duration: {duration:.2f} seconds (target: {total_duration} seconds, difference: {duration - total_duration:.2f}s)")

            if duration > total_duration:
                logger.error(f"⚠️ AUDIO TOO LONG! Generated audio is {duration:.2f}s, exceeding target of {total_duration}s by {duration - total_duration:.2f}s")
            elif duration > total_duration * 0.95:
                logger.warning(f"⚠️ Audio duration ({duration:.2f}s) is very close to target ({total_duration}s).")
            else:
                logger.info(f"✅ Audio duration ({duration:.2f}s) is within target ({total_duration}s).")

            logger.info(f"Successfully generated audio for short {request.short_id}")

            return AudioGenerationResponse(
                voice_id=request.voice_id,
                user_id=request.user_id,
                short_id=request.short_id,
                audio_url=final_audio_url,
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

    async def _get_or_generate_test_audio(self, voice_id: str, user_id: str, language: str = "en-US") -> Optional[Dict[str, Any]]:
        """Check MongoDB for existing test audio or generate it. Uses local public folder (no Supabase upload)."""
        try:
            # Check MongoDB for existing test audio with matching language
            if not self.mongodb.ensure_connection():
                logger.error("Failed to connect to MongoDB")
                return None

            query = {
                "voice_id": voice_id,
                "language": language,
                "type": "test_audio"
            }
            result = self.mongodb.database.test_audio.find_one(query)
            if result:
                logger.info(f"Found cached test audio for voice {voice_id} with language {language}")
                audio_url = result.get("audio_url", "")
                text = result.get("text", self.test_texts.get(language, self.test_texts["en-US"]))
                # Local path (saved under public/generated_audio/test_audios/...)
                if audio_url.startswith("generated_audio/"):
                    local_path = Path(settings.PUBLIC_OUTPUT_BASE) / audio_url
                    if local_path.exists():
                        audio_bytes = local_path.read_bytes()
                        return {"audio_bytes": audio_bytes, "text": text}
                    logger.warning("Cached test audio file not found on disk, will regenerate")
                else:
                    logger.warning("Cached test audio has non-local URL, will regenerate")

            logger.info(f"Generating new test audio for voice {voice_id} with language {language}")
            return await self._generate_test_audio(voice_id, user_id, language)

        except Exception as e:
            logger.error(f"Error getting or generating test audio: {e}")
            return None

    def _save_test_audio_to_public(self, audio_data: bytes, user_id: str, voice_id: str, language: str) -> Optional[str]:
        """Save test audio to public folder. Returns relative URL: generated_audio/test_audios/{user_id}/{voice_id}_{language}.mp3"""
        try:
            public_base = getattr(settings, "PUBLIC_OUTPUT_BASE", None)
            if not public_base:
                logger.error("PUBLIC_OUTPUT_BASE not configured")
                return None
            safe_lang = language.replace("-", "_")
            out_dir = Path(public_base) / "generated_audio" / "test_audios" / user_id
            out_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{voice_id}_{safe_lang}.mp3"
            dest_path = out_dir / file_name
            dest_path.write_bytes(audio_data)
            relative_url = f"generated_audio/test_audios/{user_id}/{file_name}"
            logger.info(f"Saved test audio to public folder: {relative_url}")
            return relative_url
        except Exception as e:
            logger.error(f"Error saving test audio to public folder: {e}")
            return None

    async def _generate_test_audio(self, voice_id: str, user_id: str, language: str = "en-US") -> Optional[Dict[str, Any]]:
        """Generate test audio using ElevenLabs and save to local public folder (no Supabase upload)."""
        try:
            if not self.elevenlabs_client:
                logger.error("ElevenLabs client not initialized")
                return None

            test_text = self.test_texts.get(language, self.test_texts["en-US"])
            logger.info(f"Generating test audio for voice {voice_id} with language {language}")

            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                voice_id=voice_id,
                output_format=settings.ELEVENLABS_DEFAULT_OUTPUT_FORMAT,
                text=test_text,
                model_id=settings.ELEVENLABS_DEFAULT_MODEL
            )
            audio_data = b"".join(audio_generator)

            relative_path = self._save_test_audio_to_public(audio_data, user_id, voice_id, language)
            if not relative_path:
                logger.error("Failed to save test audio to public folder")
                return None

            await self._save_test_audio_to_mongodb(voice_id, relative_path, user_id, language, test_text)
            logger.info(f"Successfully generated and saved test audio for voice {voice_id}")
            return {"audio_bytes": audio_data, "text": test_text}

        except Exception as e:
            logger.error(f"Error generating test audio: {e}")
            return None

    def _analyze_audio_speed_from_bytes(self, audio_data: bytes, test_text: str) -> float:
        """Analyze audio speed from bytes to determine words per minute."""
        try:
            word_count = len(test_text.split())
            duration = self._calculate_audio_duration_from_bytes(audio_data)
            if duration <= 0:
                return 150.0
            words_per_minute = (word_count / duration) * 60
            logger.info(f"Analyzed audio speed from bytes: {words_per_minute:.2f} WPM")
            return words_per_minute
        except Exception as e:
            logger.error(f"Error analyzing audio speed from bytes: {e}")
            return 150.0

    async def _analyze_audio_speed(self, audio_url: str) -> Optional[float]:
        """Analyze audio speed to determine words per minute (from URL)."""
        try:
            response = requests.get(audio_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to download audio: {response.status_code}")
                return 150.0
            audio_data = response.content
            test_text = self.test_texts["en-US"]
            return self._analyze_audio_speed_from_bytes(audio_data, test_text)
        except Exception as e:
            logger.error(f"Error analyzing audio speed: {e}")
            return 150.0

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
        """Generate final audio using ElevenLabs with timestamps. Returns audio bytes and subtitle timing (no Supabase upload)."""
        try:
            if not self.elevenlabs_client:
                logger.error("ElevenLabs client not initialized")
                return None

            logger.info(f"Generating final audio with timestamps for voice {voice_id}")

            response = self.elevenlabs_client.text_to_speech.convert_with_timestamps(
                voice_id=voice_id,
                output_format=settings.ELEVENLABS_DEFAULT_OUTPUT_FORMAT,
                text=script,
                model_id=settings.ELEVENLABS_DEFAULT_MODEL
            )

            logger.info("Received audio response, decoding base64 audio data")
            audio_data = base64.b64decode(response.audio_base_64)
            logger.info(f"Total audio data size: {len(audio_data)} bytes")

            subtitle_timing = self._process_subtitle_timing(response.alignment, script)

            logger.info(f"Successfully generated final audio for voice {voice_id}")
            return {
                "audio_data": audio_data,
                "subtitle_timing": subtitle_timing
            }

        except Exception as e:
            logger.error(f"Error generating final audio: {e}")
            raise

    def _save_audio_to_public(self, audio_data: bytes, user_id: str, short_id: str) -> Optional[str]:
        """Save audio to public folder. Returns URL with leading slash: /generated_audio/{user_id}/{short_id}/{file_name}"""
        try:
            public_base = getattr(settings, "PUBLIC_OUTPUT_BASE", None)
            if not public_base:
                logger.error("PUBLIC_OUTPUT_BASE not configured")
                return None
            out_dir = Path(public_base) / "generated_audio" / user_id / short_id
            out_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{uuid.uuid4().hex[:12]}.mp3"
            dest_path = out_dir / file_name
            dest_path.write_bytes(audio_data)
            relative_url = f"/generated_audio/{user_id}/{short_id}/{file_name}"
            logger.info(f"Saved audio to public folder: {relative_url}")
            return relative_url
        except Exception as e:
            logger.error(f"Error saving audio to public folder: {e}")
            return None

    def _calculate_audio_duration_from_bytes(self, audio_data: bytes) -> float:
        """Calculate audio duration in seconds from MP3 bytes."""
        try:
            duration = self._parse_mp3_duration(audio_data)
            if duration > 0:
                return duration
            file_size_bytes = len(audio_data)
            bytes_per_second = (128 * 1000) / 8
            return file_size_bytes / bytes_per_second
        except Exception as e:
            logger.error(f"Error calculating audio duration from bytes: {e}")
            return 0.0

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

# Global service instance
audio_generation_service = AudioGenerationService()
