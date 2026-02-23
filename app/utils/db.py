"""
PostgreSQL access for Prisma schema (shorts, video_scenes, audio_info).

Uses DATABASE_URL from config. Tables: shorts, video_scenes, audio_info.
"""

import json
from typing import Any, Dict, List, Optional

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def _get_connection():
    """Get a psycopg2 connection. Requires DATABASE_URL to be set."""
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set; cannot connect to PostgreSQL")
    import psycopg2
    return psycopg2.connect(settings.DATABASE_URL)


def fetch_video_scenes(short_id: str) -> List[Dict[str, Any]]:
    """
    Fetch video scenes for a short, ordered by scene_number.
    Returns list of dicts with: id, short_id, scene_number, duration, generated_video_url, status.
    """
    conn = None
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, short_id, scene_number, duration, generated_video_url, status
                FROM video_scenes
                WHERE short_id = %s
                ORDER BY scene_number
                """,
                (short_id,),
            )
            rows = cur.fetchall()
        columns = ["id", "short_id", "scene_number", "duration", "generated_video_url", "status"]
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"fetch_video_scenes failed for short_id={short_id}: {e}")
        raise
    finally:
        if conn:
            conn.close()


def fetch_audio_info(short_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch audio info for a short (one row per short).
    Returns dict with: id, short_id, generated_audio_url, subtitles (JSON), status, etc.
    """
    conn = None
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, short_id, generated_audio_url, subtitles, status
                FROM audio_info
                WHERE short_id = %s
                LIMIT 1
                """,
                (short_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        subtitles = row[3]
        if isinstance(subtitles, str):
            try:
                subtitles = json.loads(subtitles) if subtitles else []
            except json.JSONDecodeError:
                subtitles = []
        if subtitles is None:
            subtitles = []
        return {
            "id": row[0],
            "short_id": row[1],
            "generated_audio_url": row[2],
            "subtitles": subtitles,
            "status": row[4],
        }
    except Exception as e:
        logger.error(f"fetch_audio_info failed for short_id={short_id}: {e}")
        raise
    finally:
        if conn:
            conn.close()


def update_short_final_video(short_id: str, final_video_url: str) -> None:
    """Update shorts.final_video_url for the given short."""
    conn = None
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE shorts
                SET final_video_url = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (final_video_url, short_id),
            )
            if cur.rowcount == 0:
                logger.warning(f"update_short_final_video: no row updated for short_id={short_id}")
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"update_short_final_video failed for short_id={short_id}: {e}")
        raise
    finally:
        if conn:
            conn.close()
