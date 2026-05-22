from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.processor import process_video_task
from app.storage import UPLOAD_DIR, create_task, ensure_directories, get_task, load_result


app = FastAPI(
    title="Object Identification in a Video MVP",
    description="Upload a video, process it with YOLO, and retrieve object motion and interaction JSON.",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    ensure_directories()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/videos")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    task_id = uuid4().hex
    suffix = Path(file.filename).suffix or ".mp4"
    upload_path = UPLOAD_DIR / f"{task_id}{suffix}"

    async with aiofiles.open(upload_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            await out_file.write(chunk)

    task = create_task(task_id, file.filename, upload_path)
    background_tasks.add_task(process_video_task, task_id, str(upload_path))

    return JSONResponse(
        status_code=202,
        content={
            "task_id": task_id,
            "status": task["status"],
            "status_url": f"/tasks/{task_id}",
            "result_url": f"/tasks/{task_id}/result",
        },
    )


@app.get("/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, object]:
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@app.get("/tasks/{task_id}/result")
def task_result(task_id: str) -> dict[str, object]:
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Task is {task['status']}, not completed.")

    result = load_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result JSON not found.")
    return result
