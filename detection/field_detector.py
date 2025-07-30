import cv2
import numpy as np
import json
import os
import time
from config import (
    FIELD_GREEN_LOWER, FIELD_GREEN_UPPER,
    FIELD_GREEN_LOWER_ALT, FIELD_GREEN_UPPER_ALT,
    GOAL_LOWER, GOAL_UPPER,
    FIELD_MIN_AREA, FIELD_STABILITY_FRAMES,
    GOAL_DETECTION_CONFIDENCE, MIN_GOAL_AREA,
    FIELD_CALIBRATION_FILE,
    FIELD_CLOSE_KERNEL_SIZE, FIELD_OPEN_KERNEL_SIZE,
    GOAL_KERNEL_SIZE, GOAL_THIN_FILTER_KERNEL_SIZE,
    DEBUG_GOAL_DETECTION,
    COLOR_FIELD_CONTOUR, COLOR_FIELD_CORNERS, COLOR_FIELD_BOUNDS, COLOR_GOALS
)

class FieldDetector:
    """Automatic field detection with calibration"""
    def __init__(self):

        self.field_lower = FIELD_GREEN_LOWER
        self.field_upper = FIELD_GREEN_UPPER
        
        self.field_lower_alt = FIELD_GREEN_LOWER_ALT
        self.field_upper_alt = FIELD_GREEN_UPPER_ALT

        self.goal_lower = GOAL_LOWER
        self.goal_upper = GOAL_UPPER
        
        # Field properties
        self.field_contour = None
        self.field_corners = None
        self.field_bounds = None
        self.field_min_area_rect = None
        self.field_rect_points = None
        self.goals = []
        self.field_transform_matrix = None
        self.calibrated = False
        
        self.min_field_area = FIELD_MIN_AREA
        self.goal_detection_confidence = GOAL_DETECTION_CONFIDENCE
        self.field_stability_frames = FIELD_STABILITY_FRAMES
        self.stable_detection_counter = 0
        
        self.debug_goal_detection = DEBUG_GOAL_DETECTION
        
        # Calibration data save/load
        self.calibration_file = FIELD_CALIBRATION_FILE
        
    def detect_field(self, frame):
        """Detects field and returns contours"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, self.field_lower, self.field_upper)
        mask2 = cv2.inRange(hsv, self.field_lower_alt, self.field_upper_alt)
        field_mask = cv2.bitwise_or(mask1, mask2)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, FIELD_CLOSE_KERNEL_SIZE)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, FIELD_OPEN_KERNEL_SIZE)
        
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel_close)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Find contours
        contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            field_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(field_contour) > self.min_field_area:
                return field_contour, field_mask
        
        return None, field_mask
    
    def detect_goals(self, frame, field_contour):
        """Detects goals based on bright areas at the field edges"""
        if field_contour is None:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        goal_mask = cv2.inRange(hsv, self.goal_lower, self.goal_upper)
        
        field_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(field_mask, [field_contour], 255)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        extended_field_mask = cv2.dilate(field_mask, kernel)
        
        goal_search_mask = cv2.bitwise_and(goal_mask, extended_field_mask)
        goal_search_mask = cv2.bitwise_xor(goal_search_mask, 
                                          cv2.bitwise_and(goal_search_mask, field_mask))
        
        goal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, GOAL_KERNEL_SIZE)
        goal_search_mask = cv2.morphologyEx(goal_search_mask, cv2.MORPH_CLOSE, goal_kernel)
        
        # Additional thin filter for better goal detection
        thin_filter_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, GOAL_THIN_FILTER_KERNEL_SIZE)
        goal_search_mask = cv2.morphologyEx(goal_search_mask, cv2.MORPH_OPEN, thin_filter_kernel)
        goal_search_mask = cv2.morphologyEx(goal_search_mask, cv2.MORPH_CLOSE, goal_kernel)
        
        goal_contours, _ = cv2.findContours(goal_search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug visualization
        if hasattr(self, 'debug_goal_detection') and self.debug_goal_detection:
            cv2.imshow("Goal Search Mask", goal_search_mask)
            cv2.imshow("Original Goal Mask", goal_mask)
            cv2.imshow("Extended Field Mask", extended_field_mask)
            cv2.imshow("Field Mask", field_mask)
            
            debug_frame = frame.copy()
            cv2.drawContours(debug_frame, goal_contours, -1, (0, 255, 255), 2)
            for i, contour in enumerate(goal_contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(debug_frame, f"C{i}: A={area:.0f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("All Goal Contours (Before Filter)", debug_frame)
        
        goals = []
        min_goal_area = MIN_GOAL_AREA
        
        for contour in goal_contours:
            area = cv2.contourArea(contour)
            if area > min_goal_area:
                # Bounding box for goal contour
                x, y, w, h = cv2.boundingRect(contour)

                field_bounds = cv2.boundingRect(field_contour)
                fx, fy, fw, fh = field_bounds
                
                # Classify goal type based on position
                goal_type = "unknown"
                if y + h < fy + fh * 0.2:
                    goal_type = "top"
                elif y > fy + fh * 0.8:
                    goal_type = "bottom"
                elif x + w < fx + fw * 0.2:
                    goal_type = "left"
                elif x > fx + fw * 0.8:
                    goal_type = "right"
                
                goals.append({
                    'contour': contour,
                    'bounds': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'area': area,
                    'type': goal_type
                })
        
        return goals
    
    def calculate_field_metrics(self, field_contour):
        """Calculates field metrics and transformation matrix"""
        if field_contour is None:
            return None
        
        self.field_bounds = cv2.boundingRect(field_contour)
        
        self.field_min_area_rect = cv2.minAreaRect(field_contour)
        self.field_rect_points = cv2.boxPoints(self.field_min_area_rect)
        self.field_rect_points = np.intp(self.field_rect_points)
        
        final_corners = self.field_rect_points.copy()

        # Sort corners
        if len(final_corners) >= 4:
            center = np.mean(final_corners, axis=0)
            
            top_left = final_corners[np.argmin(final_corners[:, 0] + final_corners[:, 1])]
            top_right = final_corners[np.argmin(-final_corners[:, 0] + final_corners[:, 1])]
            bottom_right = final_corners[np.argmax(final_corners[:, 0] + final_corners[:, 1])]
            bottom_left = final_corners[np.argmax(-final_corners[:, 0] + final_corners[:, 1])]
            
            final_corners = np.array([top_left, top_right, bottom_right, bottom_left])
            self.field_corners = final_corners
        
        # Define destination points for transformation
        field_width = 118
        field_height = 68
        dst_points = np.array([
                                [0, 0],
                                [field_width, 0],
                                [field_width, field_height],
                                [0, field_height]
                            ], dtype=np.float32)
        
        # Create transformation matrix
        src_points = final_corners.astype(np.float32)
        if len(src_points) == 4:
            self.field_transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        return {
            'bounds': self.field_bounds,
            'corners': final_corners,
            'area': cv2.contourArea(field_contour)
        }
    
    def transform_point_to_field_coords(self, point):
        """Transforms a point to field coordinates using the transformation matrix"""
        if self.field_transform_matrix is None or point is None:
            return None
        
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.field_transform_matrix)
        
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))
    
    def calibrate(self, frame):
        """Performs field calibration"""
        field_contour, field_mask = self.detect_field(frame)
        
        if field_contour is not None:

            self.stable_detection_counter += 1
            
            if self.stable_detection_counter >= self.field_stability_frames:
                metrics = self.calculate_field_metrics(field_contour)
                
                goals = self.detect_goals(frame, field_contour)
                
                self.field_contour = field_contour
                self.goals = goals
                self.calibrated = True
                
                self.save_calibration()
                
                # print(f"Spielfeld kalibriert! Erkannte Tore: {len(goals)}")
                # print(f"Spielfeld-Bereich: {self.field_bounds}")
                
                return True
        else:
            self.stable_detection_counter = max(0, self.stable_detection_counter - 2)
        
        return False
    
    def save_calibration(self):
        """Saves calibration data"""
        if not self.calibrated:
            return
        
        calibration_data = {
            'field_bounds': self.field_bounds,
            'field_corners': self.field_corners.tolist() if self.field_corners is not None else None,
            'field_min_area_rect': {
                'center': self.field_min_area_rect[0] if self.field_min_area_rect else None,
                'size': self.field_min_area_rect[1] if self.field_min_area_rect else None,
                'angle': self.field_min_area_rect[2] if self.field_min_area_rect else None
            } if self.field_min_area_rect else None,
            'field_rect_points': self.field_rect_points.tolist() if self.field_rect_points is not None else None,
            'goals': [{
                'bounds': goal['bounds'],
                'center': goal['center'],
                'area': goal['area'],
                'type': goal['type']
            } for goal in self.goals],
            'transform_matrix': self.field_transform_matrix.tolist() if self.field_transform_matrix is not None else None
        }
        
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            # print(f"Calibration saved to {self.calibration_file}")
        except Exception as e:
            print(f"Error saving calibration: {e}")
    
    def load_calibration(self):
        """Loads saved calibration data"""
        if not os.path.exists(self.calibration_file):
            return False
        
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
            
            self.field_bounds = tuple(data['field_bounds']) if data['field_bounds'] else None
            self.field_corners = np.array(data['field_corners']) if data['field_corners'] else None
            self.field_transform_matrix = np.array(data['transform_matrix']) if data['transform_matrix'] else None
            
            # Minimal Area Rectangle laden
            if 'field_min_area_rect' in data and data['field_min_area_rect']:
                mar_data = data['field_min_area_rect']
                if mar_data['center'] and mar_data['size'] and mar_data['angle'] is not None:
                    self.field_min_area_rect = (
                        tuple(mar_data['center']),
                        tuple(mar_data['size']),
                        mar_data['angle']
                    )
                else:
                    self.field_min_area_rect = None
            else:
                self.field_min_area_rect = None
                
            # Minimal Area Rectangle Punkte laden
            if 'field_rect_points' in data and data['field_rect_points']:
                self.field_rect_points = np.array(data['field_rect_points'], dtype=np.int0)
            else:
                self.field_rect_points = None
            
            # Tore rekonstruieren
            self.goals = []
            for goal_data in data['goals']:
                self.goals.append({
                    'bounds': tuple(goal_data['bounds']),
                    'center': tuple(goal_data['center']),
                    'area': goal_data['area'],
                    'type': goal_data['type'],
                    'contour': None
                })
            
            self.calibrated = True
            # print(f"Calibration loaded from {self.calibration_file}")
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False