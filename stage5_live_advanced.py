"""
Controls:
  - Press 'l' to LOCK onto the largest detected person in the current frame
  - Press 'r' to RESET (unlock, go back to just detecting)
  - Press 'q' to QUIT

Run: python stage5_live_advanced.py
"""
import time
import cv2
from ultralytics import YOLO

# ---- Config (pulled to the top so it's easy to tune without digging through logic) ----
CAMERA_SOURCE = 0          # 0 = webcam. Swap for a file path or RTSP URL.
HORIZONTAL_FOV_DEG = 78.0  # typical webcam/action-cam horizontal field of view.
                            # Change this to match your actual camera/drone camera spec.
RE_VERIFY_INTERVAL_SEC = 3.0  # how often to double-check the lock with a fresh detection
CONFIDENCE_THRESHOLD = 0.4

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("test_video.mp4")
if not cap.isOpened():
    raise RuntimeError("Could not open camera/video source.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center = (width // 2, height // 2)


deg_per_pixel_x = HORIZONTAL_FOV_DEG / width
vertical_fov_deg = HORIZONTAL_FOV_DEG * (height / width)  # approximate, assumes same aspect scaling
deg_per_pixel_y = vertical_fov_deg / height

tracker = None
locked = False
last_verify_time = 0

WINDOW_NAME = "Live Target Lock & Track (v2)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)

print("Live feed started.")
print("  Press 'l' to lock onto a target")
print("  Press 'r' to reset")
print("  Press 'q' to quit")


def detect_people(frame):
    """Run YOLO and return boxes in (x_center, y_center, w, h) format."""
    results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
    return results[0].boxes.xywh.cpu().numpy()


def pick_largest(boxes):
    """Pick the largest detected box (assume biggest = closest/main subject)."""
    areas = boxes[:, 2] * boxes[:, 3]
    idx = areas.argmax()
    tx, ty, tw, th = boxes[idx]
    return (int(tx - tw / 2), int(ty - th / 2), int(tw), int(th))


def draw_text_with_outline(frame, text, pos, scale=0.6, color=(255, 255, 255), thickness=2):
    """
    Draws text with a black outline behind it so it stays readable
    regardless of whether the background is bright or dark. Plain
    cv2.putText alone disappears on light backgrounds (e.g. sky, sand).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, text, pos, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)

    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_gimbal_readout(frame, offset_x, offset_y):
    """
    Converts pixel offset into simulated yaw/pitch degrees and draws a
    small readout. This is the number a real gimbal motor controller
    would receive: "rotate yaw by this many degrees, pitch by this many."
    """
    yaw_deg = offset_x * deg_per_pixel_x
    pitch_deg = -offset_y * deg_per_pixel_y

    text = f"GIMBAL CMD  yaw: {yaw_deg:+.1f} deg   pitch: {pitch_deg:+.1f} deg"
    draw_text_with_outline(frame, text, (10, 60), scale=0.65, color=(0, 255, 255))

    cx, cy, r = width - 80, 80, 50
    cv2.circle(frame, (cx, cy), r, (200, 200, 200), 2)
    cv2.line(frame, (cx - r, cy), (cx + r, cy), (100, 100, 100), 1)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), (100, 100, 100), 1)
    dot_x = max(-r, min(r, int(offset_x * deg_per_pixel_x)))
    dot_y = max(-r, min(r, int(-offset_y * deg_per_pixel_y)))
    cv2.circle(frame, (cx + dot_x, cy - dot_y), 6, (0, 0, 255), -1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    now = time.time()

    if not locked:
        boxes = detect_people(frame)

        for box in boxes:
            bx, by, bw, bh = box
            x, y = int(bx - bw / 2), int(by - bh / 2)
            cv2.rectangle(frame, (x, y), (int(x + bw), int(y + bh)), (0, 255, 0), 2)

        draw_text_with_outline(frame, "Press 'l' to LOCK onto largest target",
                                (10, 30), scale=0.7, color=(0, 255, 0))

        if key == ord('l') and len(boxes) > 0:
            bbox = pick_largest(boxes)
            tracker = cv2.TrackerCSRT.create()
            tracker.init(frame, bbox)
            locked = True
            last_verify_time = now
            print(f"Locked onto target at {bbox}")

    else:
        success, bbox = tracker.update(frame)

        due_for_recheck = (now - last_verify_time) > RE_VERIFY_INTERVAL_SEC

        if not success or due_for_recheck:
            boxes = detect_people(frame)

            if len(boxes) > 0 and success:
                tx, ty, tw, th = [int(v) for v in bbox]
                tracked_center = (tx + tw // 2, ty + th // 2)

                def dist_to_tracked(box):
                    bx, by = box[0], box[1]
                    return (bx - tracked_center[0]) ** 2 + (by - tracked_center[1]) ** 2

                closest = min(boxes, key=dist_to_tracked)
                bx, by, bw, bh = closest
                new_bbox = (int(bx - bw / 2), int(by - bh / 2), int(bw), int(bh))

                tracker = cv2.TrackerCSRT.create()
                tracker.init(frame, new_bbox)
                bbox = new_bbox
                success = True
                print("Re-anchored tracker via fresh detection (drift correction)")

            elif len(boxes) > 0 and not success:
                new_bbox = pick_largest(boxes)
                tracker = cv2.TrackerCSRT.create()
                tracker.init(frame, new_bbox)
                bbox = new_bbox
                success = True
                print("Target reacquired via re-detection after loss")

            last_verify_time = now

        if success:
            x, y, w, h = [int(v) for v in bbox]
            target_center = (x + w // 2, y + h // 2)
            offset_x = target_center[0] - frame_center[0]
            offset_y = target_center[1] - frame_center[1]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            draw_text_with_outline(frame, "TRACKING (LOCKED)", (x, y - 10),
                                    scale=0.7, color=(0, 0, 255))
            cv2.circle(frame, frame_center, 5, (0, 255, 0), -1)
            cv2.line(frame, frame_center, target_center, (255, 255, 0), 2)
            draw_text_with_outline(frame, f"offset: x={offset_x:+d}px y={offset_y:+d}px",
                                    (10, height - 20), scale=0.6, color=(255, 255, 0))

            draw_gimbal_readout(frame, offset_x, offset_y)
        else:
            draw_text_with_outline(frame, "TARGET LOST - searching...",
                                    (10, height - 20), scale=0.7, color=(0, 0, 255))

        if key == ord('r'):
            locked = False
            tracker = None
            print("Reset. Detecting again...")

    cv2.imshow(WINDOW_NAME, frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()