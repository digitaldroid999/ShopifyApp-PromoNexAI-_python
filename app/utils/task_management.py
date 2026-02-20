"""Task management — in-memory implementation for async/polling patterns."""

from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import uuid
import threading

class TaskType(str, Enum):
    SCRAPING = "scraping"
    VIDEO_GENERATION = "video_generation"
    MEDIA_PROCESSING = "media_processing"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskManager:
    """In-memory task store. Replace with DB/Redis when needed."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, task_type: str, **kwargs: Any) -> str:
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": TaskStatus.PENDING.value,
                "task_type": task_type,
                "user_id": kwargs.get("user_id"),
                "scene_id": kwargs.get("scene_id"),
                "short_id": kwargs.get("short_id"),
                "message": "",
                "created_at": now,
                "updated_at": now,
                "error_message": None,
                "metadata": {},
            }
        return task_id

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._tasks.get(task_id)

    def start(self, task_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = TaskStatus.RUNNING.value
                self._tasks[task_id]["updated_at"] = now

    def complete(self, task_id: str, **kwargs: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = TaskStatus.COMPLETED.value
                self._tasks[task_id]["updated_at"] = now
                if "metadata" in kwargs:
                    self._tasks[task_id]["metadata"] = kwargs["metadata"]

    def fail(self, task_id: str, error_message: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = TaskStatus.FAILED.value
                self._tasks[task_id]["updated_at"] = now
                self._tasks[task_id]["error_message"] = error_message


task_manager = TaskManager()


def create_task(task_type: TaskType, **kwargs: Any) -> str:
    return task_manager.create(task_type.value, **kwargs)


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    return task_manager.get(task_id)


def start_task(task_id: str) -> None:
    task_manager.start(task_id)


def complete_task(task_id: str, **kwargs: Any) -> None:
    task_manager.complete(task_id, **kwargs)


def fail_task(task_id: str, error_message: str = "") -> None:
    task_manager.fail(task_id, error_message)
