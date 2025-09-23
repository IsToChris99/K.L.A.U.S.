"""
Visualisierungs-Engine für die GUI
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path to find config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class VisualizationEngine:
    """Engine für die Visualisierung von Ball- und Felddaten"""
    
    def __init__(self):
        # Visualization modes
        self.BALL_ONLY = 1
        self.FIELD_ONLY = 2  
        self.PLAYER_ONLY = 3
        self.COMBINED = 4

        self.M_persp = None

        # Toggle states for showing/hiding detections
        self.show_detections = True

    def draw_ball_visualization(self, frame, ball_result):
        """Draws ball visualization"""

        if ball_result is None:
            return

        detection = ball_result['detection']
        smoothed_pts = ball_result['smoothed_pts']
        transformed_smoothed_pts = []
        if smoothed_pts is not None:
            # Filter out None values before creating numpy array
            valid_pts = [pt for pt in smoothed_pts if pt is not None]
            if valid_pts:
                transformed_smoothed_pts = self._transform_points(np.array(valid_pts), self.M_persp)
        missing_counter = ball_result['missing_counter']

        # Draw ball info
        if detection[0] is not None:
            center, radius, confidence, velocity = detection
            transformed_center = self._transform_points(np.array([center]), self.M_persp)[0]
            center_int = (int(transformed_center[0]), int(transformed_center[1]))

            # Color selection based on confidence
            if confidence >= 0.8:
                color = config.COLOR_BALL_HIGH_CONFIDENCE  # Green
            elif confidence >= 0.6:
                color = config.COLOR_BALL_MED_CONFIDENCE   # Yellow
            else:
                color = config.COLOR_BALL_LOW_CONFIDENCE   # Orange

            cv2.circle(frame, center_int, 3, color, -1)
            cv2.circle(frame, center_int, int(radius), color, 2)

            cv2.putText(frame, f"R: {radius:.1f}", (center_int[0] + 15, center_int[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (center_int[0] + 15, center_int[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Show Kalman velocity
            if velocity is not None:
                cv2.arrowedLine(frame, transformed_center,
                            (int(transformed_center[0] + velocity[0]*30), int(transformed_center[1] + velocity[1]*30)),
                            (255, 0, 255), 2)
                
        # Draw smoothed points trail
        for i in range(1, len(transformed_smoothed_pts)):
            thickness = int(np.sqrt(config.BALL_TRAIL_MAX_LENGTH / float(i + 1)) * config.BALL_TRAIL_THICKNESS_FACTOR)
            cv2.line(frame, tuple(transformed_smoothed_pts[i - 1]), tuple(transformed_smoothed_pts[i]), config.COLOR_BALL_TRAIL, thickness)
    
    def draw_field_visualization(self, frame, field_data):
        """Draws field visualization"""

        if field_data is None or field_data['field_corners'] is None:
            return

        transformed_corners = self._transform_points(field_data['field_corners'], self.M_persp)

        # Field corners
        for i, corner in enumerate(transformed_corners):
            corner_int = (int(corner[0]), int(corner[1]))
            cv2.circle(frame, corner_int, 2, config.COLOR_FIELD_CORNERS, -1)
            cv2.putText(frame, f"{i+1}", (int(corner[0])+10, int(corner[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_FIELD_CORNERS, 2)

        # Goals
        for i, goal in enumerate(field_data['goals']):
            # Zeichne die ausgerichtete Tor-Kontur wenn vorhanden
            if goal.get('contour') is not None:
                transformed_contour = self._transform_points(goal['contour'], self.M_persp)
                cv2.drawContours(frame, [transformed_contour], -1, config.COLOR_GOALS, 2)
            else:
                # Fallback auf rechteckige Bounds
                x, y, w, h = goal['bounds']
                transformed_bounds = self._transform_points(np.array([[x, y], [x + w, y + h]]), self.M_persp)
                cv2.rectangle(frame, (int(transformed_bounds[0][0]), int(transformed_bounds[0][1])),
                                      (int(transformed_bounds[1][0]), int(transformed_bounds[1][1])),
                                      config.COLOR_GOALS, 2)

            # Zeichne Tor-Center und Label
            center_x, center_y = goal['center']
            transformed_center = self._transform_points(np.array([[center_x, center_y]]), self.M_persp)[0]
            cv2.circle(frame, (int(transformed_center[0]), int(transformed_center[1])), 5, config.COLOR_GOALS, -1)
            cv2.putText(frame, f"Goal {i+1} ({goal['type']})", (int(transformed_center[0])-30, int(transformed_center[1])-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GOALS, 2)

        # Field limits with corners
        field_corners_int = np.array(transformed_corners, dtype=np.int32)
        cv2.drawContours(frame, [field_corners_int], -1, config.COLOR_FIELD_BOUNDS, 1)

    def draw_player_visualization(self, frame, player_data):
        """Zeichnet Spieler-Visualisierung auf den Frame"""
        if player_data is None:
            return

        # if player_data is None or self.M_persp is None:
        #     return
        
        # Transform team1 bounding boxes
        for box in player_data['team1_boxes']:
            x, y, w, h = box
            # Create corner points of the bounding box
            corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            # Transform the corners
            transformed_corners = self._transform_points(corners, self.M_persp)
            # Draw the transformed bounding box as a polygon
            cv2.polylines(frame, [transformed_corners], True, (0, 0, 255), 2)
            
        # Transform team2 bounding boxes
        for box in player_data['team2_boxes']:
            x, y, w, h = box
            # Create corner points of the bounding box
            corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            # Transform the corners
            transformed_corners = self._transform_points(corners, self.M_persp)
            # Draw the transformed bounding box as a polygon
            cv2.polylines(frame, [transformed_corners], True, (255, 0, 0), 2)

    def apply_visualizations(self, frame, visualization_mode, ball_data=None, field_data=None, player_data=None, M_persp=None):
        """Wendet Visualisierungen basierend auf dem aktuellen Modus an"""
        if frame is None or M_persp is None:
            return None
        
        #display_frame = frame.copy()
        self.M_persp = M_persp

        display_frame = cv2.warpPerspective(frame, M_persp, (frame.shape[1], frame.shape[0]))

        # Only draw visualizations if enabled
        if self.show_detections:
            # Add visualizations based on current mode
            if visualization_mode in [self.BALL_ONLY]:
                self.draw_ball_visualization(display_frame, ball_data)

            elif visualization_mode in [self.FIELD_ONLY]:
                self.draw_field_visualization(display_frame, field_data)

            elif visualization_mode in [self.PLAYER_ONLY]:
                self.draw_player_visualization(display_frame, player_data)

            elif visualization_mode in [self.COMBINED]:
                self.draw_ball_visualization(display_frame, ball_data)
                self.draw_field_visualization(display_frame, field_data)
                self.draw_player_visualization(display_frame, player_data)

        return display_frame
    

    def _transform_points(self, points_array, M):
        if points_array.size == 0:
            return np.array([], dtype=np.int32).reshape(0, 2)
        points_np_float = points_array.astype(np.float32)
        points_reshaped = points_np_float.reshape(-1, 1, 2)
        transformed_np = cv2.perspectiveTransform(points_reshaped, M)
        return transformed_np.reshape(-1, 2).astype(int)
    
    def toggle_detections(self):
        """Schaltet die Anzeige der Detections ein/aus"""
        self.show_detections = not self.show_detections
        return self.show_detections
