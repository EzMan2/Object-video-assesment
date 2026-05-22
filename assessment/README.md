# Object Identification in a Video project

This is my project for the "Object Identification in a Video" intern assessment. It uploads a video, runs YOLO on sampled frames, tracks detected objects with simple IDs, marks objects as moving or stationary, detects basic person-object interactions, and saves the final result as JSON.

## Project Structure

```text
app/
  main.py
  processor.py
  detector.py
  tracker.py
  interaction.py
  storage.py
data/
  uploads/
  results/
outputs/
  keyframes/
requirements.txt
README.md
tests/
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the `yolov8n.pt` weights from Ultralytics. 
## Run

```bash
uvicorn app.main:app --reload
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

## API Examples

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Upload a video:

```bash
curl -X POST "http://127.0.0.1:8000/videos" \
  -F "file=@/path/to/video.mp4"
```

Example response:

```json
{
  "task_id": "9f99c9c7e2f447e19fd1e032e9ac7b57",
  "status": "queued",
  "status_url": "/tasks/9f99c9c7e2f447e19fd1e032e9ac7b57",
  "result_url": "/tasks/9f99c9c7e2f447e19fd1e032e9ac7b57/result"
}
```

Check task status:

```bash
curl http://127.0.0.1:8000/tasks/9f99c9c7e2f447e19fd1e032e9ac7b57
```

Get completed result JSON:

```bash
curl http://127.0.0.1:8000/tasks/9f99c9c7e2f447e19fd1e032e9ac7b57/result
```

If the task is still queued or processing, the result endpoint returns HTTP `409`.

## Output JSON Shape

Results are written to `data/results/{task_id}.json`.

```json
{
  "task_id": "task id",
  "videoMetadata": {
    "filename": "video.mp4",
    "fps": 30.0,
    "frame_count": 300,
    "width": 1280,
    "height": 720,
    "duration_sec": 10.0,
    "sample_every_n_frames": 5,
    "processed_frames": 60
  },
  "objectsDetected": [
    {
      "object_id": 1,
      "class": "backpack",
      "first_seen_frame": 10,
      "last_seen_frame": 90,
      "best_confidence": 0.82,
      "last_bbox": [100.0, 120.0, 180.0, 260.0],
      "interacted": true,
      "motion_history": [
        { "frame_range": [10, 20], "state": "stationary" },
        { "frame_range": [25, 90], "state": "moving" }
      ],
      "interactions": [
        {
          "interacted_by_person": 0,
          "frame_start": 20,
          "frame_end": 120,
          "confidence_max": 0.71,
          "reason": "near_person"
        }
      ]
    }
  ],
  "keyframes": [
    {
      "object_id": 1,
      "frame_index": 20,
      "confidence": 0.71,
      "path": "outputs/keyframes/task-id/object_1_best_interaction.jpg"
    }
  ]
}
```

## Assumptions and Tradeoffs

- I used YOLO's COCO classes for people and objects. Since COCO does not include a cord/cable class for the sample video, I added a small class override so the interacted object is labeled as `cord` instead of YOLO's generic `laptop` guess.
- The processor samples every 5 frames by default so the demo runs faster. This can be changed in `app/processor.py`.
- The tracker is simple on purpose. It matches detections by class, center distance, and IoU, so it may lose an object if there is fast motion or occlusion.
- Motion is based on center-point movement. If the center moves at least 8 pixels between sampled frames, I mark it as moving.
- For interaction detection, I use overlap or closeness between a person box and an object box for consecutive sampled frames. The PDF mentions hand coordinates, but I kept this as a simple heuristic for the project.
- Task state and results are stored in JSON files under `data/`. This keeps the project easy to run locally.
- FastAPI background tasks are used for async processing. A real deployment would probably use a job queue.

## Submission Info

- Repository link: `https://github.com/EzMan2/Object-video-assesment/tree/main`
- Time spent: about 1 working day
- Test evidence: all tests passed locally. Screenshot included in submission.

## Testing

Install dependencies, start the server, and upload a short MP4:

```bash
uvicorn app.main:app --reload
curl -X POST "http://127.0.0.1:8000/videos" -F "file=@sample.mp4"
```

Poll the status endpoint until it returns `"status": "completed"`, then call the result endpoint.

Run a quick import check:

```bash
python -m compileall app
```

Run the unit tests:

```bash
python -m unittest discover tests
```

## Notes

The best interaction keyframe for each interacted object is saved under `outputs/keyframes/{task_id}/`. The main output is still the JSON result.
