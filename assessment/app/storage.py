from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
KEYFRAME_DIR = ROOT / "outputs" / "keyframes"
TASKS_PATH = DATA_DIR / "tasks.json"


def ensure_directories() -> None:
    for path in (UPLOAD_DIR, RESULT_DIR, KEYFRAME_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if not TASKS_PATH.exists():
        TASKS_PATH.write_text("{}", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_tasks() -> dict[str, Any]:
    ensure_directories()
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def write_tasks(tasks: dict[str, Any]) -> None:
    ensure_directories()
    TASKS_PATH.write_text(json.dumps(tasks, indent=2), encoding="utf-8")


def create_task(task_id: str, filename: str, upload_path: Path) -> dict[str, Any]:
    tasks = read_tasks()
    task = {
        "task_id": task_id,
        "status": "queued",
        "filename": filename,
        "upload_path": str(upload_path),
        "result_path": str(RESULT_DIR / f"{task_id}.json"),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "error": None,
    }
    tasks[task_id] = task
    write_tasks(tasks)
    return task


def update_task(task_id: str, **fields: Any) -> dict[str, Any]:
    tasks = read_tasks()
    if task_id not in tasks:
        raise KeyError(task_id)
    tasks[task_id].update(fields)
    tasks[task_id]["updated_at"] = utc_now()
    write_tasks(tasks)
    return tasks[task_id]


def get_task(task_id: str) -> dict[str, Any] | None:
    return read_tasks().get(task_id)


def save_result(task_id: str, result: dict[str, Any]) -> Path:
    ensure_directories()
    result_path = RESULT_DIR / f"{task_id}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result_path


def load_result(task_id: str) -> dict[str, Any] | None:
    result_path = RESULT_DIR / f"{task_id}.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))
