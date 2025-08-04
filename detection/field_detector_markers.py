import cv2
import numpy as np
import json
import os
import time
from config import (
    GOAL_LOWER, GOAL_UPPER,
    FIELD_MIN_AREA, FIELD_STABILITY_FRAMES,
    GOAL_DETECTION_CONFIDENCE, MIN_GOAL_AREA,
    FIELD_CALIBRATION_FILE,
    FIELD_CLOSE_KERNEL_SIZE, FIELD_OPEN_KERNEL_SIZE,
    WIDTH_RATIO, HEIGHT_RATIO, FIELD_MARKER_LOWER, FIELD_MARKER_UPPER,
    FIELD_MARKER_LOWER_ALT, FIELD_MARKER_UPPER_ALT
)

class FieldDetector:
    """Automatic field detection with calibration"""
    def __init__(self):

        self.field_lower = FIELD_MARKER_LOWER
        self.field_upper = FIELD_MARKER_UPPER

        self.field_lower_alt = FIELD_MARKER_LOWER_ALT
        self.field_upper_alt = FIELD_MARKER_UPPER_ALT

        # self.goal_lower = GOAL_LOWER
        # self.goal_upper = GOAL_UPPER
        
        # Field properties
        self.field_contour = None
        self.field_corners = None
        self.goals = []
        self.field_transform_matrix = None
        self.perspective_transform_matrix = None
        self.calibrated = False
        
        self.min_field_area = FIELD_MIN_AREA
        self.goal_detection_confidence = GOAL_DETECTION_CONFIDENCE
        self.field_stability_frames = FIELD_STABILITY_FRAMES
        self.stable_detection_counter = 0
        
        # Calibration data save/load
        self.calibration_file = FIELD_CALIBRATION_FILE
        
    def detect_field(self, frame):
        """Detects field and returns contours"""

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # Maske für grüne Marker erstellen
        mask1 = cv2.inRange(hsv, self.field_lower, self.field_upper)
        mask2 = cv2.inRange(hsv, self.field_lower_alt, self.field_upper_alt)
        marker_mask = cv2.bitwise_or(mask1, mask2)

        # Morphologische Operationen anwenden
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_CLOSE, kernel_close)
        marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, kernel_open)

        # Alle weißen Pixel finden
        white_pixels = np.column_stack(np.where(marker_mask == 255))
        
        if len(white_pixels) < 4:
            return None, marker_mask
        
        # Die vier Extrempunkte finden (am weitesten voneinander entfernt)
        # Koordinaten sind in (y, x) Format von np.where, also umkehren zu (x, y)
        white_pixels_xy = white_pixels[:, [1, 0]]  # Von (y,x) zu (x,y)
        
        # Top-left: minimale Summe von x+y
        top_left_idx = np.argmin(white_pixels_xy[:, 0] + white_pixels_xy[:, 1])
        top_left = white_pixels_xy[top_left_idx]
        
        # Top-right: minimale Differenz von y-x (maximales x bei kleinem y)
        top_right_idx = np.argmin(white_pixels_xy[:, 1] - white_pixels_xy[:, 0])
        top_right = white_pixels_xy[top_right_idx]
        
        # Bottom-right: maximale Summe von x+y
        bottom_right_idx = np.argmax(white_pixels_xy[:, 0] + white_pixels_xy[:, 1])
        bottom_right = white_pixels_xy[bottom_right_idx]
        
        # Bottom-left: maximale Differenz von y-x (minimales x bei großem y)
        bottom_left_idx = np.argmax(white_pixels_xy[:, 1] - white_pixels_xy[:, 0])
        bottom_left = white_pixels_xy[bottom_left_idx]
        
        # Die vier Eckpunkte als field_corners setzen
        self.field_corners = np.array([top_left, top_right, bottom_right, bottom_left])

        # # Debug: Zeige die Maske an
        # cv2.imshow('Field Mask Debug', marker_mask)
        # cv2.waitKey(1)  # Kurzes Warten für die Anzeige
        
        return self.field_corners
    
    def detect_goals(self, frame, field_corners):
        """Detects goals positioned at the center of the shorter sides of the field"""
        if field_corners is None:
            return []
        
        if len(field_corners) < 4:
            return []
        
        top_left, top_right, bottom_right, bottom_left = field_corners
        
        top_length = np.linalg.norm(top_right - top_left)
        right_length = np.linalg.norm(bottom_right - top_right)
        bottom_length = np.linalg.norm(bottom_left - bottom_right)
        left_length = np.linalg.norm(top_left - bottom_left)

        horizontal_avg = (top_length + bottom_length) / 2
        vertical_avg = (left_length + right_length) / 2
        
        goals = []
        
        def create_goal_contour(center, direction_vector, goal_width, goal_depth):
            """Creates a goal contour aligned with the field edge"""
            # Normalisiere den Richtungsvektor
            direction_norm = np.linalg.norm(direction_vector)
            if direction_norm == 0:
                return None
            direction_unit = direction_vector / direction_norm
            
            # Senkrechter Vektor (90 Grad gedreht)
            perpendicular = np.array([-direction_unit[1], direction_unit[0]])
            
            # Berechne die vier Eckpunkte des Tors
            half_width = goal_width / 2
            p1 = center + direction_unit * half_width + perpendicular * goal_depth
            p2 = center - direction_unit * half_width + perpendicular * goal_depth
            p3 = center - direction_unit * half_width - perpendicular * goal_depth
            p4 = center + direction_unit * half_width - perpendicular * goal_depth
            
            return np.array([p1, p2, p3, p4], dtype=np.int32)
        
        if horizontal_avg < vertical_avg:
            # Horizontale Seiten sind kürzer - Tore oben und unten
            top_goal_center = ((top_left + top_right) / 2).astype(int)
            top_direction = (top_right - top_left)  # Richtung der oberen Kante
            top_contour = create_goal_contour(top_goal_center, top_direction, 80 * WIDTH_RATIO, 20 * HEIGHT_RATIO)
            
            if top_contour is not None:
                # Berechne Bounding Box für das ausgerichtete Tor
                x, y, w, h = cv2.boundingRect(top_contour)
                goals.append({
                    'center': tuple(top_goal_center),
                    'type': 'top',
                    'bounds': (x, y, w, h),
                    'area': 80 * WIDTH_RATIO * 20 * HEIGHT_RATIO,
                    'contour': top_contour
                })
            
            bottom_goal_center = ((bottom_left + bottom_right) / 2).astype(int)
            bottom_direction = (bottom_right - bottom_left)  # Richtung der unteren Kante
            bottom_contour = create_goal_contour(bottom_goal_center, bottom_direction, 80 * WIDTH_RATIO, 20 * HEIGHT_RATIO)
            
            if bottom_contour is not None:
                x, y, w, h = cv2.boundingRect(bottom_contour)
                goals.append({
                    'center': tuple(bottom_goal_center),
                    'type': 'bottom',
                    'bounds': (x, y, w, h),
                    'area': 80 * WIDTH_RATIO * 20 * HEIGHT_RATIO,
                    'contour': bottom_contour
                })
        else:
            # Vertikale Seiten sind kürzer - Tore links und rechts
            left_goal_center = ((top_left + bottom_left) / 2).astype(int)
            left_direction = (bottom_left - top_left)  # Richtung der linken Kante
            left_contour = create_goal_contour(left_goal_center, left_direction, 80 * HEIGHT_RATIO, 20 * WIDTH_RATIO)
            
            if left_contour is not None:
                x, y, w, h = cv2.boundingRect(left_contour)
                goals.append({
                    'center': tuple(left_goal_center),
                    'type': 'left',
                    'bounds': (x, y, w, h),
                    'area': 80 * HEIGHT_RATIO * 20 * WIDTH_RATIO,
                    'contour': left_contour
                })
            
            right_goal_center = ((top_right + bottom_right) / 2).astype(int)
            right_direction = (bottom_right - top_right)  # Richtung der rechten Kante
            right_contour = create_goal_contour(right_goal_center, right_direction, 80 * HEIGHT_RATIO, 20 * WIDTH_RATIO)
            
            if right_contour is not None:
                x, y, w, h = cv2.boundingRect(right_contour)
                goals.append({
                    'center': tuple(right_goal_center),
                    'type': 'right',
                    'bounds': (x, y, w, h),
                    'area': 80 * HEIGHT_RATIO * 20 * WIDTH_RATIO,
                    'contour': right_contour
                })
        


        # """Detects goals based on bright areas at the field edges"""
        # if field_contour is None:
        #     return []
        
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # goal_mask = cv2.inRange(hsv, self.goal_lower, self.goal_upper)
        
        # field_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # cv2.fillPoly(field_mask, [field_contour], 255)
        
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        # extended_field_mask = cv2.dilate(field_mask, kernel)
        
        # goal_search_mask = cv2.bitwise_and(goal_mask, extended_field_mask)
        # goal_search_mask = cv2.bitwise_xor(goal_search_mask, 
        #                                   cv2.bitwise_and(goal_search_mask, field_mask))
        
        # goal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, GOAL_KERNEL_SIZE)
        # goal_search_mask = cv2.morphologyEx(goal_search_mask, cv2.MORPH_CLOSE, goal_kernel)
        
        # # Additional thin filter for better goal detection
        # thin_filter_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, GOAL_THIN_FILTER_KERNEL_SIZE)
        # goal_search_mask = cv2.morphologyEx(goal_search_mask, cv2.MORPH_OPEN, thin_filter_kernel)
        # goal_search_mask = cv2.morphologyEx(goal_search_mask, cv2.MORPH_CLOSE, goal_kernel)
        
        # goal_contours, _ = cv2.findContours(goal_search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # goals = []
        # min_goal_area = MIN_GOAL_AREA
        
        # for contour in goal_contours:
        #     area = cv2.contourArea(contour)
        #     if area > min_goal_area:
        #         # Bounding box for goal contour
        #         x, y, w, h = cv2.boundingRect(contour)

        #         field_bounds = cv2.boundingRect(field_contour)
        #         fx, fy, fw, fh = field_bounds
                
        #         # Classify goal type based on position
        #         goal_type = "unknown"
        #         if y + h < fy + fh * 0.2:
        #             goal_type = "top"
        #         elif y > fy + fh * 0.8:
        #             goal_type = "bottom"
        #         elif x + w < fx + fw * 0.2:
        #             goal_type = "left"
        #         elif x > fx + fw * 0.8:
        #             goal_type = "right"
                
        #         goals.append({
        #             'contour': contour,
        #             'bounds': (x, y, w, h),
        #             'center': (x + w//2, y + h//2),
        #             'area': area,
        #             'type': goal_type
        #         })
        
        return goals
    
    def calculate_field_metrics(self, field_corners, frame):
        """Calculates field metrics and transformation matrix"""
        if field_corners is None:
            return None

        # Define destination points for perspective transformation
        field_width = frame.shape[0] * 0.9
        field_height = field_width * (68 / 118)
        dst_points_perspective = np.array([
                                [frame.shape[0] * 0.05, (frame.shape[1] - field_height)/2],
                                [frame.shape[0] * 0.95, (frame.shape[1] - field_height)/2],
                                [frame.shape[0] * 0.95, frame.shape[1] - ((frame.shape[1] - field_height) /2)],
                                [frame.shape[0] * 0.05, frame.shape[1] - ((frame.shape[1] - field_height) /2)]
                            ], dtype=np.float32)
        
        # Create transformation matrix
        src_points_perspective = field_corners.astype(np.float32)
        if len(src_points_perspective) == 4:
            self.perspective_transform_matrix = cv2.getPerspectiveTransform(src_points_perspective, dst_points_perspective)

        # Define destination points for transformation
        dst_points_field = np.array([
                                [-field_width/2, field_height/2],
                                [field_width/2, field_height/2],
                                [field_width/2, -field_height/2],
                                [-field_width/2, -field_height/2]
                            ], dtype=np.float32)
        
        # Create transformation matrix
        src_points_field = dst_points_perspective
        if len(src_points_field) == 4:
            self.field_transform_matrix = cv2.getPerspectiveTransform(src_points_field, dst_points_field)
        
        return {
            'corners': field_corners,
            'perspective_transform_matrix': self.perspective_transform_matrix,
            'field_transform_matrix': self.field_transform_matrix
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
        field_corners = self.detect_field(frame)

        if field_corners is not None:

            self.stable_detection_counter += 1
            
            if self.stable_detection_counter >= self.field_stability_frames:
                metrics = self.calculate_field_metrics(field_corners, frame)
                
                goals = self.detect_goals(frame, metrics['corners'])

                self.field_corners = field_corners
                self.goals = goals
                self.calibrated = True
                
                self.save_calibration()
                
                # print(f"Spielfeld kalibriert! Erkannte Tore: {len(goals)}")
                
                return True
        else:
            self.stable_detection_counter = max(0, self.stable_detection_counter - 2)
        
        return False
    
    def save_calibration(self):
        """Saves calibration data"""
        if not self.calibrated:
            return
        
        # Helper function to convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        calibration_data = {
            'field_corners': convert_to_native(self.field_corners),
            'goals': [{
                'bounds': convert_to_native(goal['bounds']),
                'center': convert_to_native(goal['center']),
                'area': convert_to_native(goal['area']),
                'type': goal['type']
            } for goal in self.goals],
            'field_transform_matrix': convert_to_native(self.field_transform_matrix),
            'perspective_transform_matrix': convert_to_native(getattr(self, 'perspective_transform_matrix', None))
        }
        
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"Calibration saved to {self.calibration_file}")
        except Exception as e:
            print(f"Error saving calibration: {e}")
    
    def load_calibration(self):
        """Loads saved calibration data"""
        if not os.path.exists(self.calibration_file):
            return False
        
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
            
            # Load field corners
            self.field_corners = np.array(data['field_corners']) if data.get('field_corners') else None
            
            # Load transformation matrices
            self.field_transform_matrix = np.array(data['field_transform_matrix']) if data.get('field_transform_matrix') else None
            
            # Load perspective transform matrix (for backward compatibility)
            transform_key = 'perspective_transform_matrix' if 'perspective_transform_matrix' in data else 'transform_matrix'
            if data.get(transform_key):
                self.perspective_transform_matrix = np.array(data[transform_key])
            
            # Load goals
            self.goals = []
            for goal_data in data.get('goals', []):
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