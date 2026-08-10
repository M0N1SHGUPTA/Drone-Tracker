"""
STAGE 4: Live feed version — same lock-and-track logic, but reading from
a live camera instead of a video file.

This proves the pipeline works on a continuous live stream, not just
pre-recorded video. On an actual drone, you'd swap cv2.VideoCapture(0)
for the drone's camera source (e.g. an RTSP URL like
cv2.VideoCapture("rtsp://drone-ip:port/stream")) -- the rest of the
pipeline is identical. That's the point: the processing logic doesn't
care where frames come from.

Controls:
  - Press 'l' to LOCK onto the largest detected person in the current frame
  - Press 'r' to RESET (unlock, go back to just detecting)
  - Press 'q' to QUIT

Run: python stage4_live.py
"""
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("test_video.mp4")  # 0 = default webcam. Swap for RTSP URL for real drone feed.
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Try a different index, e.g. cv2.VideoCapture(1)")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center = (width // 2, height // 2)

tracker = None
locked = False

# Make the display window resizable and set it to a comfortable fixed size,
# regardless of the actual video's resolution. This only affects how it's
# DISPLAYED -- the underlying frame processing/detection still happens at
# full resolution, so accuracy isn't affected.
WINDOW_NAME = "Live Target Lock & Track"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)  # change these numbers if you want bigger/smaller

print("Live feed started.")
print("  Press 'l' to lock onto a target")
print("  Press 'r' to reset")
print("  Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    if not locked:
        # Not locked yet -> just run detection every frame so user can see
        # what's detectable, and wait for the 'l' key to lock on.
        results = model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xywh.cpu().numpy()

        for box in boxes:
            bx, by, bw, bh = box
            x, y = int(bx - bw / 2), int(by - bh / 2)
            cv2.rectangle(frame, (x, y), (int(x + bw), int(y + bh)), (0, 255, 0), 2)

        cv2.putText(frame, "Press 'l' to LOCK onto largest target",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if key == ord('l') and len(boxes) > 0:
            areas = boxes[:, 2] * boxes[:, 3]
            target_idx = areas.argmax()
            tx, ty, tw, th = boxes[target_idx]
            bbox = (int(tx - tw / 2), int(ty - th / 2), int(tw), int(th))

            tracker = cv2.TrackerCSRT.create()
            tracker.init(frame, bbox)
            locked = True
            print(f"Locked onto target at {bbox}")

    else:
        # Locked -> track instead of re-detecting (cheaper, faster)
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            target_center = (x + w // 2, y + h // 2)
            offset_x = target_center[0] - frame_center[0]
            offset_y = target_center[1] - frame_center[1]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, "TRACKING (LOCKED)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, frame_center, 5, (0, 255, 0), -1)
            cv2.line(frame, frame_center, target_center, (255, 255, 0), 2)
            cv2.putText(frame, f"offset: x={offset_x:+d} y={offset_y:+d}",
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "TARGET LOST - press 'r' to reset",
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if key == ord('r'):
            locked = False
            tracker = None
            print("Reset. Detecting again...")

    cv2.imshow(WINDOW_NAME, frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()