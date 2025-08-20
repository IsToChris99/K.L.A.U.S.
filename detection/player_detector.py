import cv2
import numpy as np
import json
import os

class PlayerDetector:
    def __init__(self, color_config_path=None, field_calibration_config_path=None):
        # Farben laden
        if color_config_path is None:
            color_config_path = os.path.join(os.path.dirname(__file__), "colors.json")
        
        with open(color_config_path, "r") as f:
            data = json.load(f)
        self.team1_ranges = data.get("team1", {}).get("ranges", [])
        self.team2_ranges = data.get("team2", {}).get("ranges", [])

        # Feld-Ecken aus Config laden
        if field_calibration_config_path is None:
            field_calibration_config_path = os.path.join(os.path.dirname(__file__), "field_calibration.json")
        
        with open(field_calibration_config_path, "r") as f:
            field_data = json.load(f)
        self.field_corners = np.array(field_data.get("field_corners", []), dtype=np.int32)

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

        # Maske aus gespeicherten Feld-Ecken erstellen mit % Freiraum
        if self.field_corners.size > 0:
            # Mittelpunkt des Feldes berechnen
            center = np.mean(self.field_corners, axis=0)
            expanded_corners = []
            for corner in self.field_corners:
                vector = corner - center
                expanded_corner = center + vector * 1.20  # % anpassen
                expanded_corners.append(expanded_corner)
            expanded_corners = np.array(expanded_corners, dtype=np.int32)

            field_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(field_mask, [expanded_corners], 255)
        else:
            field_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        # Spieler nur innerhalb des Feldes erkennen
        boxes_t1 = self._detect_team(hsv, self.team1_ranges, field_mask)
        boxes_t2 = self._detect_team(hsv, self.team2_ranges, field_mask)
        return boxes_t1, boxes_t2


    def _detect_team(self, hsv, team_ranges, field_mask):
        mask = self.smooth_mask(self.create_mask(hsv, team_ranges))
        mask = cv2.bitwise_and(mask, field_mask)

        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 250 < area < 1700:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, w, h))
        return self.non_max_suppression_fast(boxes)
