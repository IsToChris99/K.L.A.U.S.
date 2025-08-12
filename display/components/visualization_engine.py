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
        
        # Toggle states for showing/hiding detections
        self.show_detections = True
    
    def draw_ball_visualization(self, frame, ball_data):
        """Zeichnet Ball-Visualisierung auf den Frame"""
        if ball_data is None:
            return
            
        # Get ball detection data
        detection = ball_data.get('detection')
        smoothed_pts = ball_data.get('smoothed_pts', [])
        missing_counter = ball_data.get('missing_counter', 0)
        
        if detection[0] is not None:
            center, radius, confidence, velocity = detection
            center_int = (int(center[0]), int(center[1]))

            # Color selection based on confidence
            if confidence >= 0.8:
                color = config.COLOR_BALL_HIGH_CONFIDENCE  # Green
            elif confidence >= 0.6:
                color = config.COLOR_BALL_MED_CONFIDENCE   # Yellow
            else:
                color = config.COLOR_BALL_LOW_CONFIDENCE   # Orange

            cv2.circle(frame, center, 3, color, -1)
            cv2.circle(frame, center, int(radius), color, 2)

            cv2.putText(frame, f"R: {radius:.1f}", (center_int[0] + 15, center_int[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (center_int[0] + 15, center_int[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Show Kalman velocity
            if velocity is not None:
                cv2.arrowedLine(frame, center,
                            (int(center[0] + velocity[0]*30), int(center[1] + velocity[1]*30)),
                            (255, 0, 255), 2)

        # Ball trail drawing
        if smoothed_pts:
            for i in range(1, len(smoothed_pts)):
                if smoothed_pts[i - 1] is None or smoothed_pts[i] is None:
                    continue
                thickness = int(np.sqrt(config.BALL_TRAIL_MAX_LENGTH / float(i + 1)) * config.BALL_TRAIL_THICKNESS_FACTOR)
                cv2.line(frame, smoothed_pts[i - 1], smoothed_pts[i], config.COLOR_BALL_TRAIL, thickness)
    
    def draw_field_visualization(self, frame, field_data):
        """Zeichnet Feld-Visualisierung auf den Frame"""
        if field_data is None:
            return

        # Field corners
        if field_data.get('field_corners') is not None:
            for i, corner in enumerate(field_data['field_corners']):
                corner_int = (int(corner[0]), int(corner[1]))
                cv2.circle(frame, corner_int, 2, config.COLOR_FIELD_CORNERS, -1)
                cv2.putText(frame, f"{i+1}", (int(corner[0])+10, int(corner[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_FIELD_CORNERS, 2)

        # Goals
        goals = field_data.get('goals', [])
        for i, goal in enumerate(goals):
            if goal.get('contour') is not None:
                cv2.drawContours(frame, [goal['contour']], -1, config.COLOR_GOALS, 2)
            else:
                # Fallback to bounding box if contour is not available
                x, y, w, h = goal.get('bounds', (0, 0, 0, 0))
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), config.COLOR_GOALS, 2)

            # Draw Goal-Center and Label
            center_x, center_y = goal.get('center', (0, 0))
            center_int = (int(center_x), int(center_y))
            cv2.circle(frame, center_int, 5, config.COLOR_GOALS, -1)  

        # Field limits
        if (field_data.get('calibrated') and field_data.get('field_corners') is not None):
            field_corners_int = np.array(field_data['field_corners'], dtype=np.int32)
            cv2.drawContours(frame, [field_corners_int], -1, config.COLOR_FIELD_BOUNDS, 1)
    
    def draw_player_visualization(self, frame, player_data):
        """Zeichnet Spieler-Visualisierung auf den Frame"""
        if player_data is None:
            return

        # Team 1 players (e.g., blue)
        team1_boxes = player_data.get('team1_boxes', [])
        for box in team1_boxes:
            x, y, w, h = box
            # Draw bounding box for team 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue
            # Draw team label
            cv2.putText(frame, "T1", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Team 2 players (e.g., red)  
        team2_boxes = player_data.get('team2_boxes', [])
        for box in team2_boxes:
            x, y, w, h = box
            # Draw bounding box for team 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red
            # Draw team label
            cv2.putText(frame, "T2", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw total player count
        total_players = player_data.get('total_players', 0)
        cv2.putText(frame, f"Players: {total_players}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def apply_visualizations(self, frame, visualization_mode, ball_data=None, field_data=None, player_data=None):
        """Wendet Visualisierungen basierend auf dem aktuellen Modus an"""
        if frame is None:
            return None
            
        display_frame = frame.copy()
        
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
    
    def toggle_detections(self):
        """Schaltet die Anzeige der Detections ein/aus"""
        self.show_detections = not self.show_detections
        return self.show_detections
