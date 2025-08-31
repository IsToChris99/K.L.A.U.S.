import cv2
import numpy as np
import json
import os
import time

class PlayerDetector:
    def __init__(self, color_config_path=None):
        # path saving
        if color_config_path is None:
            color_config_path = os.path.join(os.path.dirname(__file__), "colors.json")
        self.color_config_path = color_config_path
        
        # load colors
        self.load_colors()
        
        # timestamp last modified
        self.last_modified = 0
        self.auto_reload = True  # automatical reload on change

    
    def load_colors(self):
        """Loads the color configuration from the JSON file"""
        try:
            with open(self.color_config_path, "r") as f:
                data = json.load(f)
            self.team1_ranges = data.get("team1", {}).get("ranges", [])
            self.team2_ranges = data.get("team2", {}).get("ranges", [])
            
            # timestamp last modified reload
            self.last_modified = os.path.getmtime(self.color_config_path)
            
            print(f"Colors loaded successfully: Team1={len(self.team1_ranges)} areas, Team2={len(self.team2_ranges)} areas")
        except Exception as e:
            print(f"Error loading colors: {e}")
            self.team1_ranges = []
            self.team2_ranges = []
    
    def check_and_reload_colors(self):
        """Checks if the colors.json file has been changed and reloads it"""
        if not self.auto_reload:
            return
            
        try:
            current_modified = os.path.getmtime(self.color_config_path)
            if current_modified > self.last_modified:
                print("colors.json has been changed - reload colors...")
                self.load_colors()
        except Exception as e:
            print(f"Error checking file: {e}")

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

    def detect_players(self, frame, field_corners):
        
        # Automatically checks for changes to colors.json
        self.check_and_reload_colors()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # creates mask from saved field corners with % free space
        if field_corners.size > 0:
            # Calculate the center of the field
            center = np.mean(field_corners, axis=0)
            expanded_corners = []
            for corner in field_corners:
                vector = corner - center
                expanded_corner = center + vector * 1.0 
                expanded_corners.append(expanded_corner)
            expanded_corners = np.array(expanded_corners, dtype=np.int32)

            field_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(field_mask, [expanded_corners], 255)
        else:
            field_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        # Detect players only within the field
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
