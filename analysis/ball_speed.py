import numpy as np

class BallSpeed:
    def __init__(self) -> None:
        """
        Initialize the BallSpeed calculator.

        Args:
            fps (float): Camera frame rate (250).
            pixel_to_meter_ratio (float | None): Conversion factor from pixels to meters (optional).
        """
        self.last_position: tuple[float, float] | None = None
        self.last_speed: float = 0.0
        self.last_timestamp = 0.0



    def update(self, current_position: tuple[float, float] | None, current_timestamp, px_to_cm_ratio: float | None = None) -> float:
        """
        Calculate the speed based on the current and last position.

        Args:
            current_position (tuple[float, float] | None): Current position as (x, y) or None.

        Returns:
            float: Speed in m/s if calibrated, else pixels/s.
        """
        if self.last_position is None or current_position is None:
            self.last_position = current_position
            self.last_timestamp = current_timestamp
            return 0.0

        x1, y1 = self.last_position
        x2, y2 = current_position

        self.dt = current_timestamp - self.last_timestamp
        dx = x2 - x1
        dy = y2 - y1
        distance_px = np.sqrt(dx ** 2 + dy ** 2)
        
        if self.dt <= 0:
            return 0.0
        
        speed_px_per_sec = distance_px / self.dt

        if px_to_cm_ratio is not None:
            speed = speed_px_per_sec * px_to_cm_ratio
        else:
            speed = speed_px_per_sec

        self.last_position = current_position
        self.last_speed = speed
        print(f"\rLast Position: {self.last_position}, Current Position: {current_position}, Speed: {(speed * 10**9):.2f} m/s", end="")
        return speed
    def reset(self) -> None:
        """
        Reset the internal state (e.g., when the ball is lost).
        """
        self.last_position = None
        self.last_speed = 0.0

    def get_last_speed(self) -> float:
        """
        Get the last calculated speed.

        Returns:
            float: Last speed value.
        """
        return self.last_speed
