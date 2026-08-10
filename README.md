# Drone Tracker

This project simulates how a drone would lock onto a target and follow it. The idea is that this code runs on the drone's onboard computer something like a Jetson Nano or a Raspberry Pi which receives a live video feed from a USB camera mounted on the drone. The system finds a person in the frame, locks onto them, and tracks them continuously. Every frame, it calculates how far off-center the target is. Those offset values or the equivalent yaw and pitch degrees can then be sent to the flight controller over MAVLink, which is the protocol most drones use to receive commands. The flight controller then adjusts the drone's heading and altitude to fly toward and stay centered on the target.

On your laptop, you run it on a pre-recorded video just to see it work. In the real deployment, you swap the video file for a live camera feed and connect the output to MAVLink instead of just printing it on screen.

---

## What each file does

**stage1_detect.py** — The starting point. Runs object detection on every single frame of a video and draws boxes around anything it finds (people, cars, etc.). This was just to confirm the detection model works.

**stage2_track.py** — The core idea. Detects in the first frame only, picks the largest person as the target, and then tracks that one person through the rest of the video using a tracker instead of re-running detection every frame. Much faster, which is how a real drone system would have to work given the limited compute on board.

**main.py** — Wraps the stage2 logic into an API. You POST a video file to it, and it gives you back the annotated video and a JSON log of the tracking data.

**stage4_live.py** — Same lock-and-track logic, but running on a live feed instead of a saved video. You watch the feed, press `l` to lock onto the biggest detected person, `r` to reset, `q` to quit.

**stage5_live_advanced.py** — The most complete version. Adds three things on top of stage4: it periodically re-runs detection to correct any tracker drift, it can reacquire the target if the tracker loses it, and it converts the pixel offset into actual yaw and pitch degrees like a real gimbal controller would output.

---

## How to run it

You need Python installed. Then install the dependencies:

```
pip install -r requirements.txt
```

To run the live demo (stage 5), make sure `test_video.mp4` and `yolov8n.pt` are in the same folder, then:

```
python stage5_live_advanced.py
```

A window will open showing the video. Press `l` to lock onto a target, `r` to reset, `q` to quit.

---

## How the tracking works

Detection (YOLOv8) is only run on the first frame. It finds all the people in that frame and picks the largest bounding box — the assumption being that the biggest box is the closest or most relevant target.

From that point on, an OpenCV CSRT tracker takes over. It does not use the neural network at all. It just looks at where the target was in the previous frame and figures out where it moved to in the current frame. This is computationally cheap, which matters a lot on hardware with limited processing power.

In stage5, every few seconds a fresh detection is also run in the background to correct any drift the tracker might have accumulated. If the tracker loses the target entirely, the detection result is used to reacquire and start tracking again.

---

## Files you need to run this

- `stage5_live_advanced.py` (or whichever stage you want to run)
- `yolov8n.pt` — the YOLOv8 model weights. If this file is missing, ultralytics will download it automatically on first run.
- `test_video.mp4` — the input video. Swap this out for any video you want to test with.
