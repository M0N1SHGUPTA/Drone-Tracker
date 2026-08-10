
from ultralytics import YOLO
import cv2

VIDEO_PATH = "test_video.mp4"
OUTPUT_PATH = "output_tracked.mp4"

# ---- Step A: Load YOLO, but we only use it ONCE on the first frame ----
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center = (width // 2, height // 2)

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# ---- Step B: Read the first frame and detect all people in it ----
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read video")

results = model(first_frame, classes=[0], verbose=False)  # class 0 = person
boxes = results[0].boxes.xywh.cpu().numpy()  # x_center, y_center, w, h

if len(boxes) == 0:
    raise RuntimeError("No person detected in first frame -- try a different video")

# ---- Step C: Pick the target to lock onto ----
# For now: auto-pick the LARGEST box (assume biggest = closest/most relevant).
# Later you could make this a click-to-select UI instead.
areas = boxes[:, 2] * boxes[:, 3]  # width * height
target_idx = areas.argmax()
tx, ty, tw, th = boxes[target_idx]

# Convert center-format (x_center, y_center, w, h) to top-left format (x, y, w, h)
# because that's what OpenCV trackers expect.
bbox = (int(tx - tw / 2), int(ty - th / 2), int(tw), int(th))

print(f"Locked onto target at {bbox}")

# ---- Step D: Initialize the tracker on that target ----
# CSRT is slower but more accurate; good enough for our purposes here.
tracker = cv2.TrackerCSRT.create()
tracker.init(first_frame, bbox)

# Draw the lock on frame 1 and write it
x, y, w, h = bbox
cv2.rectangle(first_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
cv2.putText(first_frame, "LOCKED TARGET", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.circle(first_frame, frame_center, 5, (0, 255, 0), -1)  # frame center marker
out.write(first_frame)

# ---- Step E: Track the target through the rest of the video ----
frame_count = 1
log = []  # we'll save offset data here -- this is your "telemetry"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        target_center = (x + w // 2, y + h // 2)

        # Offset = how far target is from frame center.
        # A real gimbal controller would use this to know which way to turn.
        offset_x = target_center[0] - frame_center[0]
        offset_y = target_center[1] - frame_center[1]

        # Draw the tracked box + a line from center to target (visualizes the "follow" logic)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(frame, "TRACKING", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.circle(frame, frame_center, 5, (0, 255, 0), -1)
        cv2.line(frame, frame_center, target_center, (255, 255, 0), 2)
        cv2.putText(frame, f"offset: x={offset_x:+d} y={offset_y:+d}",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        log.append({"frame": frame_count, "offset_x": offset_x, "offset_y": offset_y, "locked": True})
    else:
        # Tracker lost the target (e.g. they walked behind something)
        cv2.putText(frame, "TARGET LOST", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        log.append({"frame": frame_count, "offset_x": None, "offset_y": None, "locked": False})

    out.write(frame)

cap.release()
out.release()

# ---- Step F: Save the telemetry log as JSON ----
import json
with open("tracking_log.json", "w") as f:
    json.dump(log, f, indent=2)

print(f"Done! {frame_count} frames processed.")
print(f"Video: {OUTPUT_PATH}")
print(f"Telemetry log: tracking_log.json")
