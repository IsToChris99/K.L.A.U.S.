import cv2
import numpy as np

def find_ball(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            return (int(x), int(y))
    return None

def estimate_position(prev, curr):
    if prev is None or curr is None:
        return None
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    return (curr[0] + dx, curr[1] + dy)

def draw_path(frame, pts, color=(0, 0, 255)):
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], color, thickness)
