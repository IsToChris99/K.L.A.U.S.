last_ball_pos = None
last_time = None

def compute_speed(ball_pos, timestamp):
    global last_ball_pos, last_time
    if last_ball_pos is None or last_time is None:
        last_ball_pos = ball_pos
        last_time = timestamp
        return 0.0

    dx = ball_pos[0] - last_ball_pos[0]
    dy = ball_pos[1] - last_ball_pos[1]
    dt = timestamp - last_time
    if dt == 0:
        return 0.0

    pixel_speed = (dx**2 + dy**2) ** 0.5 / dt 
    last_ball_pos = ball_pos
    last_time = timestamp
    return pixel_speed