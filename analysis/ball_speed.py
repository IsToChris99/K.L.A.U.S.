import numpy as np
import config
from collections import deque
from typing import Tuple, Optional, List

# Calculate the pixel to centimeter ratio based on field width and detection width
pixel_to_cm_ratio = (config.FIELD_WIDTH_M * 100) / config.DETECTION_WIDTH

def calculate_ball_speed(
    current_position: tuple[float, float] | None,
    last_position: tuple[float, float] | None,
    fps: float,
    pixel_to_cm_ratio: float | None = pixel_to_cm_ratio  # Default parameter value set to calculated ratio
) -> tuple[float, tuple[float, float] | None]:
    
    # Case 1: Cannot calculate speed (first frame, ball lost, or invalid FPS).
    if last_position is None or current_position is None or fps <= 0:
        # The speed is 0, and the "new last position" is whatever the current one is.
        # If the ball is lost (current_position=None), the state is correctly reset for the next frame.
        return 0.0, current_position

    # --- Calculation Logic ---
    # Time between frames
    dt = 1.0 / fps

    # Distance in pixels using Pythagorean theorem
    dx = current_position[0] - last_position[0]
    dy = current_position[1] - last_position[1]
    distance_px = np.sqrt(dx**2 + dy**2)

    # Speed in pixels per second
    speed_px_per_sec = distance_px / dt

    # Convert to centimeters per second if a ratio is provided
    if pixel_to_cm_ratio is not None:
        speed_cms = speed_px_per_sec * pixel_to_cm_ratio
    else:
        speed_cms = speed_px_per_sec # Fallback to pixels/sec

    # Return the calculated speed and the new position to be stored for the next frame.
    return speed_cms, current_position


def calculate_ball_speed_with_timestamp(
    current_position: tuple[float, float] | None,
    current_timestamp_ns: int | None,
    last_position: tuple[float, float] | None,
    last_timestamp_ns: int | None,
    pixel_to_cm_ratio: float | None = pixel_to_cm_ratio  # Default parameter value set to calculated ratio
) -> tuple[float, tuple[float, float] | None, int | None]:
    """
    Calculate ball speed using actual timestamps instead of FPS.
    
    Args:
        current_position: Current ball position (x, y) in pixels or None if ball not detected
        current_timestamp_ns: Current frame timestamp in nanoseconds
        last_position: Previous ball position (x, y) in pixels or None
        last_timestamp_ns: Previous frame timestamp in nanoseconds
        pixel_to_cm_ratio: Conversion ratio from pixels to centimeters
        
    Returns:
        Tuple of (speed_cm/s, new_last_position, new_last_timestamp)
    """
    
    # Case 1: Cannot calculate speed (first frame, ball lost, or invalid timestamps).
    if (last_position is None or current_position is None or 
        current_timestamp_ns is None or last_timestamp_ns is None):
        # The speed is 0, and the "new last position/timestamp" is whatever the current ones are.
        # If the ball is lost (current_position=None), the state is correctly reset for the next frame.
        return 0.0, current_position, current_timestamp_ns
    print(f"\rCurrent Timestamp: {current_timestamp_ns}, Last Timestamp: {last_timestamp_ns}", f"Position: {current_position}", end="")
    # --- Calculation Logic ---
    # Time between frames in seconds (convert from nanoseconds)
    dt = (current_timestamp_ns - last_timestamp_ns) / 1_000_000_000.0
    
    # Prevent division by zero or negative time differences
    if dt <= 0:
        return 0.0, current_position, current_timestamp_ns

    # Distance in pixels using Pythagorean theorem
    dx = current_position[0] - last_position[0]
    dy = current_position[1] - last_position[1]
    distance_px = np.sqrt(dx**2 + dy**2)

    # Speed in pixels per second
    speed_px_per_sec = distance_px / dt

    # Convert to centimeters per second if a ratio is provided
    if pixel_to_cm_ratio is not None:
        speed_cms = speed_px_per_sec * pixel_to_cm_ratio
    else:
        speed_cms = speed_px_per_sec # Fallback to pixels/sec

    # Return the calculated speed and the new position/timestamp to be stored for the next frame.
    return speed_cms, current_position, current_timestamp_ns


class AdaptiveBallSpeedCalculator:
    """
    Adaptive ball speed calculator that adjusts smoothing based on movement characteristics.
    
    Features:
    - Adaptive window size based on speed
    - Outlier detection for jitter reduction
    - Momentum-based smoothing
    - Different behavior for slow/medium/fast movements
    """
    
    def __init__(self, max_history_size: int = 20, pixel_to_cm_ratio: float = None):
        """
        Initialize the adaptive ball speed calculator.
        
        Args:
            max_history_size: Maximum number of position/timestamp pairs to keep
            pixel_to_cm_ratio: Conversion ratio from pixels to centimeters
        """
        self.pixel_to_cm_ratio = pixel_to_cm_ratio or ((config.FIELD_WIDTH_M * 100) / config.DETECTION_WIDTH)
        self.max_history_size = max_history_size
        
        # History storage: (position, timestamp_ns, raw_speed)
        self.position_history: deque = deque(maxlen=max_history_size)
        self.speed_history: deque = deque(maxlen=max_history_size)
        
        # Speed thresholds for different behaviors (in cm/s)
        # Adjust these based on your specific use case:
        self.SLOW_THRESHOLD = 50.0     # Below this: maximum smoothing (reduces jitter when ball is stationary/slow)
        self.MEDIUM_THRESHOLD = 300.0  # Medium smoothing (normal ball movement)
        self.FAST_THRESHOLD = 800.0    # Above this: minimal smoothing (fast shots, quick reactions needed)
        
        # Adaptive window sizes for different speed ranges
        # These control how many frames are used for smoothing:
        self.SLOW_WINDOW = 15     # Large window for slow movement/jitter (strong smoothing)
        self.MEDIUM_WINDOW = 8    # Medium window for normal movement (balanced)
        self.FAST_WINDOW = 3      # Small window for fast movement (minimal lag)
        
        # Outlier detection parameters
        self.OUTLIER_FACTOR = 2.0  # How many standard deviations for outlier detection
        
        self.last_smoothed_speed = 0.0
        
    def update(self, current_position: Optional[Tuple[float, float]], 
               current_timestamp_ns: Optional[int]) -> float:
        """
        Update the speed calculation with a new position and timestamp.
        
        Args:
            current_position: Current ball position (x, y) in pixels or None if ball not detected
            current_timestamp_ns: Current frame timestamp in nanoseconds
            
        Returns:
            Smoothed ball speed in cm/s
        """
        # Reset if ball is lost
        if current_position is None or current_timestamp_ns is None:
            if current_position is None:  # Ball lost
                self.position_history.clear()
                self.speed_history.clear()
                self.last_smoothed_speed = 0.0
            return 0.0
            
        # Add current position to history
        self.position_history.append((current_position, current_timestamp_ns))
        
        # Need at least 2 positions to calculate speed
        if len(self.position_history) < 2:
            return 0.0
            
        # Calculate raw speed between last two positions
        raw_speed = self._calculate_raw_speed()
        
        # Add to speed history
        self.speed_history.append(raw_speed)
        
        # Apply adaptive smoothing
        smoothed_speed = self._adaptive_smoothing(raw_speed)
        
        self.last_smoothed_speed = smoothed_speed
        return smoothed_speed
    
    def _calculate_raw_speed(self) -> float:
        """Calculate raw speed between the last two positions."""
        if len(self.position_history) < 2:
            return 0.0
            
        # Get last two entries
        (pos1, ts1) = self.position_history[-2]
        (pos2, ts2) = self.position_history[-1]
        
        # Time difference in seconds
        dt = (ts2 - ts1) / 1_000_000_000.0
        if dt <= 0:
            return 0.0
            
        # Distance in pixels
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance_px = np.sqrt(dx**2 + dy**2)
        
        # Speed in pixels per second, then convert to cm/s
        speed_px_per_sec = distance_px / dt
        return speed_px_per_sec * self.pixel_to_cm_ratio
    
    def _adaptive_smoothing(self, raw_speed: float) -> float:
        """
        Apply adaptive smoothing based on speed characteristics.
        
        Args:
            raw_speed: Raw calculated speed in cm/s
            
        Returns:
            Smoothed speed in cm/s
        """
        if len(self.speed_history) < 2:
            return raw_speed
            
        # Determine speed category and window size
        window_size = self._get_adaptive_window_size(raw_speed)
        
        # Get relevant speed history
        relevant_speeds = list(self.speed_history)[-window_size:]
        
        # Remove outliers for better smoothing
        filtered_speeds = self._remove_outliers(relevant_speeds)
        
        if not filtered_speeds:
            return raw_speed
            
        # Apply different smoothing strategies based on speed range
        if raw_speed < self.SLOW_THRESHOLD:
            # Slow movement: Strong smoothing to reduce jitter
            return self._weighted_average(filtered_speeds, emphasis_recent=False)
        elif raw_speed < self.MEDIUM_THRESHOLD:
            # Medium movement: Balanced smoothing
            return self._weighted_average(filtered_speeds, emphasis_recent=True)
        else:
            # Fast movement: Minimal smoothing, high responsiveness
            return self._responsive_smoothing(filtered_speeds, raw_speed)
    
    def _get_adaptive_window_size(self, current_speed: float) -> int:
        """Determine window size based on current speed."""
        if current_speed < self.SLOW_THRESHOLD:
            return self.SLOW_WINDOW
        elif current_speed < self.FAST_THRESHOLD:
            return self.MEDIUM_WINDOW
        else:
            return self.FAST_WINDOW
    
    def _remove_outliers(self, speeds: List[float]) -> List[float]:
        """Remove outliers using statistical method."""
        if len(speeds) < 3:
            return speeds
            
        speeds_array = np.array(speeds)
        mean_speed = np.mean(speeds_array)
        std_speed = np.std(speeds_array)
        
        # Filter out outliers
        filtered = []
        for speed in speeds:
            if abs(speed - mean_speed) <= self.OUTLIER_FACTOR * std_speed:
                filtered.append(speed)
                
        return filtered if filtered else speeds  # Fallback to original if all filtered out
    
    def _weighted_average(self, speeds: List[float], emphasis_recent: bool = True) -> float:
        """Calculate weighted average with optional emphasis on recent values."""
        if not speeds:
            return 0.0
            
        if not emphasis_recent:
            return sum(speeds) / len(speeds)
            
        # Exponential weighting favoring recent values
        weights = np.exp(np.linspace(0, 1, len(speeds)))
        weights = weights / np.sum(weights)
        
        return np.sum(np.array(speeds) * weights)
    
    def _responsive_smoothing(self, speeds: List[float], raw_speed: float) -> float:
        """
        Responsive smoothing for fast movements.
        Uses momentum detection to minimize lag during rapid changes.
        """
        if len(speeds) < 2:
            return raw_speed
            
        # Check if we're in acceleration phase (momentum building)
        recent_speeds = speeds[-3:] if len(speeds) >= 3 else speeds
        is_accelerating = all(recent_speeds[i] <= recent_speeds[i+1] 
                            for i in range(len(recent_speeds)-1))
        
        if is_accelerating or raw_speed > self.FAST_THRESHOLD:
            # High responsiveness: weight recent measurements heavily
            alpha = 0.7  # Strong emphasis on current measurement
            return alpha * raw_speed + (1 - alpha) * self.last_smoothed_speed
        else:
            # Normal smoothing for fast but stable movement
            return self._weighted_average(speeds[-self.FAST_WINDOW:], emphasis_recent=True)
    
    def reset(self):
        """Reset the calculator state."""
        self.position_history.clear()
        self.speed_history.clear()
        self.last_smoothed_speed = 0.0
    
    def configure_thresholds(self, slow_threshold: float = None, 
                           medium_threshold: float = None, 
                           fast_threshold: float = None,
                           slow_window: int = None,
                           medium_window: int = None, 
                           fast_window: int = None):
        """
        Configure speed thresholds and window sizes for adaptive behavior.
        
        Args:
            slow_threshold: Speed threshold for slow movement (cm/s)
            medium_threshold: Speed threshold for medium movement (cm/s)  
            fast_threshold: Speed threshold for fast movement (cm/s)
            slow_window: Window size for slow movement smoothing
            medium_window: Window size for medium movement smoothing
            fast_window: Window size for fast movement smoothing
        """
        if slow_threshold is not None:
            self.SLOW_THRESHOLD = slow_threshold
        if medium_threshold is not None:
            self.MEDIUM_THRESHOLD = medium_threshold
        if fast_threshold is not None:
            self.FAST_THRESHOLD = fast_threshold
        if slow_window is not None:
            self.SLOW_WINDOW = slow_window
        if medium_window is not None:
            self.MEDIUM_WINDOW = medium_window
        if fast_window is not None:
            self.FAST_WINDOW = fast_window
    
    def get_debug_info(self) -> dict:
        """Get debug information about current state."""
        current_speed = self.speed_history[-1] if self.speed_history else 0.0
        window_size = self._get_adaptive_window_size(current_speed)
        
        return {
            'current_raw_speed': current_speed,
            'smoothed_speed': self.last_smoothed_speed,
            'window_size': window_size,
            'history_length': len(self.position_history),
            'speed_category': (
                'slow' if current_speed < self.SLOW_THRESHOLD else
                'medium' if current_speed < self.FAST_THRESHOLD else 'fast'
            )
        }
