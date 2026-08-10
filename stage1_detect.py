
from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8 model (nano = smallest/fastest, good for "on-drone" framing)
model = YOLO("yolov8n.pt")  # auto-downloads first time you run it

# Open your test video
cap = cv2.VideoCapture("test_video.mp4")

# Get video properties so we can save output in the same format
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Writer for the output video (with boxes drawn on it)
out = cv2.VideoWriter(
    "output_detected.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # video ended

    frame_count += 1

    # Run YOLO detection on this single frame
    # classes=[0, 2] means: only detect "person" (0) and "car" (2)
    # (COCO dataset class ids: 0=person, 2=car, 5=bus, 7=truck)
    results = model(frame, classes=[0, 2, 5, 7], verbose=False)

    # results[0].plot() draws the boxes + labels directly on the frame for us
    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print("Done! Check output_detected.mp4")
