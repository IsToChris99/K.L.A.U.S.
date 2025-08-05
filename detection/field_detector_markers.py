import cv2
import numpy as np
import json
import os
import time
from collections import deque
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
        
        # Calibration data save/load
        self.calibration_file = FIELD_CALIBRATION_FILE

        # Queue für Eckpunkt-Stabilisierung
        self.corner_queue = deque(maxlen=20)
        
        # EMA: Speicher für vorherigen EMA-Wert
        self.previous_ema_corners = None
        
        # Pre-compute morphological kernels (Performance-Optimierung)
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5 * int(WIDTH_RATIO), 5 * int(HEIGHT_RATIO)))
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3 * int(WIDTH_RATIO), 3 * int(HEIGHT_RATIO)))

        self.counter = 0

    def detect_field(self, frame):
        """Detects field and returns contours"""

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # Maske für grüne Marker erstellen
        mask1 = cv2.inRange(hsv, self.field_lower, self.field_upper)
        mask2 = cv2.inRange(hsv, self.field_lower_alt, self.field_upper_alt)
        marker_mask = cv2.bitwise_or(mask1, mask2)

        # Morphologische Operationen anwenden (verwende vorkompilierte Kernel)
        marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_CLOSE, self.kernel_close)
        marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, self.kernel_open)

        contours = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # Debug: Zeige die Maske an (DEAKTIVIERT für Performance)
        cv2.imshow('Field Mask Debug', marker_mask)
        cv2.waitKey(1)  # Kurzes Warten für die Anzeige

        if len(contours) != 4:
            self.counter += 1
            print(f"\rDetected {len(contours)} contours, expected 4. Counter: {self.counter}", end="")
            return None

        # Alle weißen Pixel finden
        white_pixels = np.column_stack(np.where(marker_mask == 255))
        
        if len(white_pixels) < 4:
            return None
        
        # Optimierte Extrempunkt-Berechnung (vektorisiert)
        # Koordinaten sind in (y, x) Format von np.where, also umkehren zu (x, y)
        white_pixels_xy = white_pixels[:, [1, 0]]  # Von (y,x) zu (x,y)
        
        # Alle Berechnungen auf einmal (vektorisiert für bessere Performance)
        x_coords = white_pixels_xy[:, 0]
        y_coords = white_pixels_xy[:, 1]
        
        # Top-left: minimale Summe von x+y
        top_left = white_pixels_xy[np.argmin(x_coords + y_coords)]
        
        # Top-right: minimale Differenz von y-x
        top_right = white_pixels_xy[np.argmin(y_coords - x_coords)]
        
        # Bottom-right: maximale Summe von x+y
        bottom_right = white_pixels_xy[np.argmax(x_coords + y_coords)]
        
        # Bottom-left: maximale Differenz von y-x
        bottom_left = white_pixels_xy[np.argmax(y_coords - x_coords)]
        
        # Die vier Eckpunkte als field_corners setzen
        field_corners = np.array([top_left, top_right, bottom_right, bottom_left])


        
        return field_corners
    
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

            if len(self.corner_queue) < 5:
                self.corner_queue.append(field_corners)
                self.previous_ema_corners = field_corners
                self.new_field_metrics = False
                return False
            
            else:
                # Verwende EMA statt einfachen Mittelwert für current_avg
                if self.previous_ema_corners is not None:
                    ema_factor = 10
                    ema_field_corners = (field_corners * (ema_factor / (1 + len(self.corner_queue)))) + (self.previous_ema_corners * (1 - (ema_factor / (1 + len(self.corner_queue)))))
                # else:
                #     # Beim ersten Mal: verwende aktuellen Wert
                #     ema_field_corners = field_corners
                
                max_deviation = 0
                for i in range(4):  # For each corner
                    deviation = np.linalg.norm(self.previous_ema_corners[i] - ema_field_corners[i])
                    max_deviation = max(max_deviation, deviation)
                
                if (max_deviation > 20 or max_deviation <= 1) and self.calibrated:
                    self.new_field_metrics = False
                    return True
                
                self.corner_queue.append(field_corners)
            
            # Speichere aktuellen EMA für nächste Iteration
            self.previous_ema_corners = ema_field_corners

            avg_field_corners = ema_field_corners

            metrics = self.calculate_field_metrics(avg_field_corners, frame)
            goals = self.detect_goals(frame, metrics['corners'])

            self.field_corners = avg_field_corners
            self.goals = goals
            self.calibrated = True
            self.new_field_metrics = True

            return True
        return False