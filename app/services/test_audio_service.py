import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from elevenlabs import ElevenLabs
from app.config import settings
from app.logging_config import get_logger
from app.utils.task_management import MongoDBManager
from app.utils.supabase_utils import supabase_manager
from app.models import TestAudioRequest, TestAudioResponse

logger = get_logger(__name__)

class TestAudioService:
    """Service for generating and retrieving test audio samples from ElevenLabs"""
    
    def __init__(self):
        self.mongodb = MongoDBManager()
        self.elevenlabs_client = None
        self._initialize_elevenlabs()
        
        # Language-specific test texts
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
        
        # Supported languages
        self.supported_languages = [
            'en-US', 'en-CA', 'en-GB', 'es', 'es-MX', 
            'pt-BR', 'fr', 'de', 'nl', 'zh', 
            'ja', 'ar'
        ]
    
    def _initialize_elevenlabs(self):
        """Initialize ElevenLabs client"""
        try:
            if not settings.ELEVENLABS_API_KEY:
                logger.warning("ElevenLabs API key not configured")
                return
                
            self.elevenlabs_client = ElevenLabs(
                api_key=settings.ELEVENLABS_API_KEY
            )
            logger.info("ElevenLabs client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs client: {e}")
            self.elevenlabs_client = None
    
    def _get_test_text(self, language: str) -> str:
        """Get appropriate test text for the given language"""
        return self.test_texts.get(language, self.test_texts['en-US'])
    
    def _validate_language(self, language: str) -> bool:
        """Validate if the language is supported"""
        return language in self.supported_languages
    
    def _get_cached_audio(self, voice_id: str, language: str) -> Optional[Dict[str, Any]]:
        """Check if test audio already exists in MongoDB"""
        try:
            if not self.mongodb.ensure_connection():
                logger.error("Failed to connect to MongoDB")
                return None
                
            # Query for existing test audio
            query = {
                "voice_id": voice_id,
                "language": language,
                "type": "test_audio"
            }
            
            result = self.mongodb.database.test_audio.find_one(query)
            if result:
                logger.info(f"Found cached test audio for voice {voice_id} in language {language}")
                return result
                
            return None
            
        except Exception as e:
            logger.error(f"Error checking cached audio: {e}")
            return None
    
    def _save_audio_to_mongodb(self, voice_id: str, language: str, audio_url: str, user_id: str) -> bool:
        """Save generated test audio info to MongoDB"""
        try:
            if not self.mongodb.ensure_connection():
                logger.error("Failed to connect to MongoDB")
                return False
                
            audio_doc = {
                "voice_id": voice_id,
                "language": language,
                "audio_url": audio_url,
                "user_id": user_id,
                "type": "test_audio",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            result = self.mongodb.database.test_audio.insert_one(audio_doc)
            if result.inserted_id:
                logger.info(f"Saved test audio to MongoDB with ID: {result.inserted_id}")
                return True
            else:
                logger.error("Failed to save test audio to MongoDB")
                return False
                
        except Exception as e:
            logger.error(f"Error saving audio to MongoDB: {e}")
            return False
    
    def _generate_test_audio(self, voice_id: str, language: str, user_id: str) -> Optional[str]:
        """Generate test audio using ElevenLabs and upload to Supabase storage"""
        try:
            if not self.elevenlabs_client:
                logger.error("ElevenLabs client not initialized")
                return None
                
            if not supabase_manager.is_connected():
                logger.error("Supabase client not connected")
                return None
                
            test_text = self._get_test_text(language)
            logger.info(f"Generating test audio for voice {voice_id} in language {language}")
            
            # Generate audio using ElevenLabs
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                voice_id=voice_id,
                output_format=settings.ELEVENLABS_DEFAULT_OUTPUT_FORMAT,
                text=test_text,
                model_id=settings.ELEVENLABS_DEFAULT_MODEL
            )
            
            # Convert generator to bytes
            audio_data = b''.join(audio_generator)
            
            # Upload to Supabase storage
            audio_url = self._upload_audio_to_supabase(audio_data, voice_id, language, user_id)
            if not audio_url:
                logger.error("Failed to upload audio to Supabase storage")
                return None
            
            logger.info(f"Successfully generated and uploaded test audio for voice {voice_id}")
            return audio_url
            
        except Exception as e:
            logger.error(f"Error generating test audio: {e}")
            return None
    
    def _ensure_test_audios_bucket(self) -> bool:
        """Ensure the test-audios bucket exists in Supabase storage"""
        try:
            if not supabase_manager.is_connected():
                logger.error("Supabase client not connected")
                return False
                
            # First, try to list all buckets to see if test-audios exists
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
                        logger.error("Failed to create test-audios bucket - no result returned")
                        return False
                        
            except Exception as e:
                logger.error(f"Error checking/creating test-audios bucket: {e}")
                return False
                    
        except Exception as e:
            logger.error(f"Error ensuring test-audios bucket: {e}")
            return False
    
    def _upload_audio_to_supabase(self, audio_data: bytes, voice_id: str, language: str, user_id: str) -> Optional[str]:
        """Upload audio data to Supabase storage and return public URL"""
        try:
            # Ensure bucket exists
            if not self._ensure_test_audios_bucket():
                logger.error("Failed to ensure test-audios bucket exists")
                return None
            
            # Create unique filename
            filename = f"{voice_id}_{language}_{uuid.uuid4().hex[:8]}.mp3"
            file_path = f"test-audios/{filename}"
            
            logger.info(f"Uploading audio to Supabase storage: {file_path}")
            
            # Upload to Supabase storage
            result = supabase_manager.client.storage.from_('test-audios').upload(
                path=file_path,
                file=audio_data,
                file_options={"content-type": "audio/mpeg"}
            )
            
            # Check for upload errors
            if hasattr(result, 'error') and result.error:
                logger.error(f"Failed to upload audio: {result.error}")
                return None
            
            # Get public URL
            public_url = supabase_manager.client.storage.from_('test-audios').get_public_url(file_path)
            
            logger.info(f"Successfully uploaded audio to: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"Error uploading audio to Supabase: {e}")
            return None
    
    def get_test_audio(self, request: TestAudioRequest) -> TestAudioResponse:
        """Get or generate test audio for the given voice and language"""
        logger.info(
            "test_audio request | voice_id=%s language=%s user_id=%s",
            request.voice_id, request.language, request.user_id,
        )
        try:
            # Validate language
            if not self._validate_language(request.language):
                logger.warning(
                    "test_audio validation failed | voice_id=%s language=%s unsupported",
                    request.voice_id, request.language,
                )
                resp = TestAudioResponse(
                    voice_id=request.voice_id,
                    language=request.language,
                    audio_url="",
                    user_id=request.user_id,
                    created_at=datetime.now(timezone.utc),
                    is_cached=False,
                    message=f"Unsupported language: {request.language}. Supported languages: {', '.join(self.supported_languages)}",
                    test_text=""
                )
                logger.info("test_audio response | status=validation_failed voice_id=%s", request.voice_id)
                return resp
            
            # Check if test audio already exists
            cached_audio = self._get_cached_audio(request.voice_id, request.language)
            if cached_audio:
                logger.info(
                    "test_audio cache hit | voice_id=%s language=%s audio_url=%s",
                    request.voice_id, request.language, cached_audio.get('audio_url', '')[:80],
                )
                test_text = self._get_test_text(request.language)
                resp = TestAudioResponse(
                    voice_id=request.voice_id,
                    language=request.language,
                    audio_url=cached_audio['audio_url'],
                    user_id=request.user_id,
                    created_at=cached_audio['created_at'],
                    is_cached=True,
                    message="Test audio retrieved from cache",
                    test_text=test_text
                )
                logger.info("test_audio response | status=cached voice_id=%s", request.voice_id)
                return resp

            logger.info("test_audio cache miss | voice_id=%s language=%s generating", request.voice_id, request.language)
            # Generate new test audio
            audio_url = self._generate_test_audio(request.voice_id, request.language, request.user_id)
            if not audio_url:
                logger.warning("test_audio generation failed | voice_id=%s language=%s", request.voice_id, request.language)
                test_text = self._get_test_text(request.language)
                resp = TestAudioResponse(
                    voice_id=request.voice_id,
                    language=request.language,
                    audio_url="",
                    user_id=request.user_id,
                    created_at=datetime.now(timezone.utc),
                    is_cached=False,
                    message="Failed to generate test audio",
                    test_text=test_text
                )
                logger.info("test_audio response | status=failed_generation voice_id=%s", request.voice_id)
                return resp

            logger.info("test_audio saving to cache | voice_id=%s language=%s", request.voice_id, request.language)
            # Save to MongoDB for future use
            self._save_audio_to_mongodb(request.voice_id, request.language, audio_url, request.user_id)

            test_text = self._get_test_text(request.language)
            resp = TestAudioResponse(
                voice_id=request.voice_id,
                language=request.language,
                audio_url=audio_url,
                user_id=request.user_id,
                created_at=datetime.now(timezone.utc),
                is_cached=False,
                message="Test audio generated successfully",
                test_text=test_text
            )
            logger.info(
                "test_audio response | status=generated voice_id=%s audio_url=%s",
                request.voice_id, audio_url[:80] if audio_url else "",
            )
            return resp

        except Exception as e:
            logger.error("test_audio error | voice_id=%s language=%s error=%s", request.voice_id, request.language, e)
            test_text = self._get_test_text(request.language)
            resp = TestAudioResponse(
                voice_id=request.voice_id,
                language=request.language,
                audio_url="",
                user_id=request.user_id,
                created_at=datetime.now(timezone.utc),
                is_cached=False,
                message=f"Error: {str(e)}",
                test_text=test_text
            )
            logger.info("test_audio response | status=error voice_id=%s", request.voice_id)
            return resp


# Create singleton instance
test_audio_service = TestAudioService()
