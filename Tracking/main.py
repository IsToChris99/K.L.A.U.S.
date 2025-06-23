import time
import cv2
from collections import deque

from config import video_path, use_webcam, orangeLower, orangeUpper, display_fps
from video.stream import VideoStream
from utils.smoother import Smoother
from processing.preprocessing import preprocess_frame
from processing.tracker import find_ball, estimate_position, draw_path

video_source = 0 if use_webcam else video_path
stream = VideoStream(video_source)

smoother = Smoother(window_size=20)
smoothed_pts = deque(maxlen=64)
missing_counter = 0

start_time = time.time()
last_display_time = time.time()
frame_counter = 0
display_interval = 1.0 / display_fps
display_counter = 0
last_display_fps_time = time.time()
display_fps_value = 0.0

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        break

    frame, mask = preprocess_frame(frame, (orangeLower, orangeUpper))
    center = find_ball(mask)

    if center is None:
        missing_counter += 1
        if missing_counter <= 120 and len(smoothed_pts) >= 2:
            est = estimate_position(smoothed_pts[1], smoothed_pts[0])
            smoothed_pts.appendleft(smoother.update(est))
        else:
            smoothed_pts.appendleft(None)
    else:
        smoothed_pts.appendleft(smoother.update(center))
        missing_counter = 0

    draw_path(frame, smoothed_pts)

    now = time.time()
    if now - last_display_time >= display_interval:
        elapsed = time.time() - start_time
        fps_display = frame_counter / elapsed
        cv2.putText(frame, f"Verarbeitung: {fps_display:.1f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Anzeige: {display_fps_value:.1f} fps", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Echtzeitanzeige", frame)
        key = cv2.waitKey(1)
        last_display_time = now
        if key & 0xFF == ord("q"):
            break

        display_counter += 1
        if (time.time() - last_display_fps_time) >= 1.0:
            display_fps_value = display_counter / (time.time() - last_display_fps_time)
            display_counter = 0
            last_display_fps_time = time.time()

    frame_counter += 1

cv2.destroyAllWindows()
stream.stop()
print(f"Verarbeitungszeit: {time.time() - start_time:.2f} s für {frame_counter} Frames ({frame_counter / (time.time() - start_time):.1f} fps)")
