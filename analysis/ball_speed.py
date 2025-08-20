
import numpy as np

def calculate_ball_speed(
    current_position: tuple[float, float] | None,
    last_position: tuple[float, float] | None,
    fps: float,
    pixel_to_m_ratio: float | None
) -> tuple[float, tuple[float, float] | None]:
    """
    Calculates the speed of the ball based on its current and last position.

    This function is stateless, meaning it doesn't remember anything between calls.
    You must provide the 'last_position' from the previous call and store the
    new position that this function returns for the next call.

    Args:
        current_position (tuple | None): The new (x, y) position of the ball.
                                         If None, it means the ball was not detected.
        last_position (tuple | None): The position of the ball from the previous frame.
        fps (float): The frames per second to use for the time calculation.
        pixel_to_m_ratio (float | None): The conversion factor from pixels to meters.

    Returns:
        tuple[float, tuple | None]: A tuple containing:
            - The calculated speed in m/s (or 0.0 if it cannot be calculated).
            - The new 'last_position' to be used in the next call (this will be the current_position).
    """
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

    # Convert to meters per second if a ratio is provided
    if pixel_to_m_ratio is not None:
        speed_ms = speed_px_per_sec * pixel_to_m_ratio
    else:
        speed_ms = speed_px_per_sec # Fallback to pixels/sec

    # Return the calculated speed and the new position to be stored for the next frame.
    return speed_ms, current_position
