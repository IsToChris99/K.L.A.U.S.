import cv2
import numpy as np
import json
import os

class PlayerDetector:
    def __init__(self, color_config_path=None):
        # Falls kein Pfad übergeben wird, nimm die colors.json im selben Ordner wie diese Datei
        if color_config_path is None:
            color_config_path = os.path.join(os.path.dirname(__file__), "colors.json")
        
        with open(color_config_path, "r") as f:
            data = json.load(f)
        self.team1_ranges = data.get("team1", {}).get("ranges", [])
        self.team2_ranges = data.get("team2", {}).get("ranges", [])

    def non_max_suppression_fast(self, boxes, overlap_thresh=0.4):
        if not boxes:
            return []
        boxes = np.array(boxes)
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(y2)
        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)
            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / area[idxs[:-1]]
            idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1],
                                                   np.where(overlap > overlap_thresh)[0])))
        return boxes[pick].astype("int").tolist()

    def create_mask(self, hsv, ranges):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            lower = np.array(r["lower"], dtype=np.uint8)
            upper = np.array(r["upper"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        return mask

    def smooth_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def detect_players(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        boxes_t1 = self._detect_team(hsv, self.team1_ranges)
        boxes_t2 = self._detect_team(hsv, self.team2_ranges)
        return boxes_t1, boxes_t2

    def _detect_team(self, hsv, team_ranges):
        mask = self.smooth_mask(self.create_mask(hsv, team_ranges))
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 250 < area < 1700:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, w, h))
        return self.non_max_suppression_fast(boxes)
