import numpy as np
from collections import deque

class BallSpeed:
    
    def __init__(self, decay_factor=0.85, speed_buffer_size=10):

        self.speed = 0.0
        self.last_position = None
        self.last_timestamp_ns = None
        self.decay_factor = decay_factor
        self.px_to_cm_ratio = 0.0
        
        # Buffer for storing last N actual (non-decayed) speed values
        self.speed_buffer = deque(maxlen=speed_buffer_size)
        self.last_actual_speed = 0.0  # Store the last calculated speed before decay
    
    
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
                calculated_speed = speed_px_per_sec * self.px_to_cm_ratio / 100.0
                self.speed = calculated_speed
                self.last_actual_speed = self.speed
                
                # Add to buffer only when we have an actual calculation
                self.speed_buffer.append(calculated_speed)
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
    
    def get_max_recent_speed(self):
        """Get the maximum speed from the recent speed buffer."""
        if len(self.speed_buffer) == 0:
            return 0.0
        return max(self.speed_buffer)
    
    def clear_speed_buffer(self):
        """Clear the speed buffer (useful after a goal is scored)."""
        self.speed_buffer.clear()

    def reset(self):
        self.speed = 0.0
        self.last_position = None
        self.last_timestamp_ns = None
        self.last_actual_speed = 0.0
        self.speed_buffer.clear()
    
    def get_speed_kmh(self):
        return self.speed * 3.6
    
    def get_max_recent_speed_kmh(self):
        """Get the maximum recent speed in km/h."""
        return self.get_max_recent_speed() * 3.6