# kickerklaus/analysis/ball_speed.py

import numpy as np

class BallSpeed:
    def __init__(self, initial_fps: float = 30.0) -> None:
        self.fps = initial_fps
        self.dt = 1.0 / self.fps if self.fps > 0 else 0.0
        self.pixel_to_m_ratio: float | None = None
        self.last_position: tuple[float, float] | None = None
        self.last_speed_ms: float = 0.0

    def update_parameters(self, fps: float | None = None, pixel_to_m_ratio: float | None = None) -> None:
        if fps is not None and fps > 0:
            self.fps = fps
            self.dt = 1.0 / fps
        if pixel_to_m_ratio is not None:
            self.pixel_to_m_ratio = pixel_to_m_ratio

    def update(self, current_position: tuple[float, float] | None) -> float:
        if self.last_position is None or current_position is None or self.dt == 0.0:
            self.last_position = current_position
            return 0.0

        dx = current_position[0] - self.last_position[0]
        dy = current_position[1] - self.last_position[1]
        distance_px = np.sqrt(dx**2 + dy**2)

        speed_px_per_sec = distance_px / self.dt

        if self.pixel_to_m_ratio is not None:
            speed = speed_px_per_sec * self.pixel_to_m_ratio
        else:
            speed = speed_px_per_sec

        self.last_position = current_position
        self.last_speed_ms = speed
        return speed

    def reset(self) -> None:
        self.last_position = None
        self.last_speed_ms = 0.0

    def get_last_speed(self) -> float:
        return self.last_speed_ms