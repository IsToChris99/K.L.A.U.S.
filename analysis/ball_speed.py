import numpy as np
import time

class BallSpeed:
    
    def __init__(self, decay_factor=0.85):

        self.speed = 0.0
        self.last_position = None
        self.last_timestamp_ns = None
        self.decay_factor = decay_factor
        self.px_to_cm_ratio = 0.0
    
    def update(self, ball_position, timestamp_ns=None, px_to_cm_ratio=None):

        if timestamp_ns is None:
            return 0.0
            
        if px_to_cm_ratio is not None:
            self.px_to_cm_ratio = px_to_cm_ratio
            
        if ball_position is not None:
            # Ball detected, calculate speed normally
            dt = None
            if self.last_timestamp_ns is not None:
                dt = (timestamp_ns - self.last_timestamp_ns) / 1e9  # Convert to seconds
                
            if self.last_position is not None and dt and dt > 0:
                dx = ball_position[0] - self.last_position[0]
                dy = ball_position[1] - self.last_position[1]
                distance_px = np.sqrt(dx**2 + dy**2)
                speed_px_per_sec = distance_px / dt
                
                # Convert to m/s (px/s * cm/px * m/cm)
                self.speed = speed_px_per_sec * self.px_to_cm_ratio / 100.0
            else:
                self.speed = 0.0
                
            # Update tracking variables
            self.last_position = ball_position
            self.last_timestamp_ns = timestamp_ns
        else:
            # Ball lost, apply decay to last speed
            self.speed *= self.decay_factor
            # Don't update last_position or last_timestamp_ns when ball is lost
            
        return self.speed
    
    def get_speed(self):

        return self.speed
    
    def reset(self):
        self.speed = 0.0
        self.last_position = None
        self.last_timestamp_ns = None
    
    def get_speed_kmh(self):

        return self.speed * 3.6