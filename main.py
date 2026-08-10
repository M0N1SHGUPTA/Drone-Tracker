"""
STAGE 3: Wrap the tracking logic in a FastAPI service.

Upload a video -> get back an annotated video (with locked target + tracking)
and a JSON telemetry log.

Run: uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs to test it via the browser (Swagger UI).
"""
import json
import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO

app = FastAPI(title="Aerial Target Lock & Track API")

# Load the model once when the server starts (not on every request -- that'd be slow)
model = YOLO("yolov8n.pt")

# Folders for temp storage of uploads/outputs
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def process_video(input_path: Path, job_id: str):
    """
    Same logic as stage2_track.py, just wrapped as a function so the API
    can call it. Returns paths to the output video + telemetry JSON.
    """
    output_video_path = OUTPUT_DIR / f"{job_id}_tracked.mp4"
    output_log_path = OUTPUT_DIR / f"{job_id}_log.json"

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center = (width // 2, height // 2)

    out = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    ret, first_frame = cap.read()
    if not ret:
        raise HTTPException(status_code=400, detail="Could not read uploaded video")

    results = model(first_frame, classes=[0], verbose=False)
    boxes = results[0].boxes.xywh.cpu().numpy()

    if len(boxes) == 0:
        raise HTTPException(status_code=422, detail="No person detected in the video")

    areas = boxes[:, 2] * boxes[:, 3]
    target_idx = areas.argmax()
    tx, ty, tw, th = boxes[target_idx]
    bbox = (int(tx - tw / 2), int(ty - th / 2), int(tw), int(th))

    tracker = cv2.TrackerCSRT.create()
    tracker.init(first_frame, bbox)

    x, y, w, h = bbox
    cv2.rectangle(first_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.putText(first_frame, "LOCKED TARGET", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.circle(first_frame, frame_center, 5, (0, 255, 0), -1)
    out.write(first_frame)

    frame_count = 1
    log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            target_center = (x + w // 2, y + h // 2)
            offset_x = target_center[0] - frame_center[0]
            offset_y = target_center[1] - frame_center[1]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, "TRACKING", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, frame_center, 5, (0, 255, 0), -1)
            cv2.line(frame, frame_center, target_center, (255, 255, 0), 2)
            cv2.putText(frame, f"offset: x={offset_x:+d} y={offset_y:+d}",
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            log.append({"frame": frame_count, "offset_x": offset_x, "offset_y": offset_y, "locked": True})
        else:
            cv2.putText(frame, "TARGET LOST", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            log.append({"frame": frame_count, "offset_x": None, "offset_y": None, "locked": False})

        out.write(frame)

    cap.release()
    out.release()

    with open(output_log_path, "w") as f:
        json.dump(log, f, indent=2)

    return output_video_path, output_log_path


@app.post("/track")
async def track_video(file: UploadFile = File(...)):
    """
    Upload a video. Returns a job_id you use to fetch the results.
    """
    job_id = str(uuid.uuid4())[:8]
    input_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    video_path, log_path = process_video(input_path, job_id)

    return {
        "job_id": job_id,
        "message": "Processing complete",
        "video_url": f"/download/video/{job_id}",
        "log_url": f"/download/log/{job_id}",
    }


@app.get("/download/video/{job_id}")
async def download_video(job_id: str):
    path = OUTPUT_DIR / f"{job_id}_tracked.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", filename="tracked_output.mp4")


@app.get("/download/log/{job_id}")
async def download_log(job_id: str):
    path = OUTPUT_DIR / f"{job_id}_log.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    return FileResponse(path, media_type="application/json", filename="tracking_log.json")


@app.get("/")
async def root():
    return {
        "message": "Aerial Target Lock & Track API",
        "usage": "POST a video to /track, then GET /download/video/{job_id} and /download/log/{job_id}",
    }