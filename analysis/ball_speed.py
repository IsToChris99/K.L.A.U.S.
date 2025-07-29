last_ball_pos = None # Startwerte für vorherige Ballposition und Zeit
last_time = None

def compute_speed(ball_pos, timestamp):
    global last_ball_pos, last_time # Zugriff auf die globalen Variablen

        # Wenn noch keine vorherigen Werte gespeichert sind → ersten Wert merken, 0 zurückgeben
    if last_ball_pos is None or last_time is None:
        last_ball_pos = ball_pos
        last_time = timestamp
        return 0.0

     # Berechne Abstand (Delta X/Y) und Zeitdifferenz
    dx = ball_pos[0] - last_ball_pos[0]
    dy = ball_pos[1] - last_ball_pos[1]
    dt = timestamp - last_time
    if dt == 0:
        return 0.0

    # Geschwindigkeit in Pixeln pro Sekunde (euklidische Distanz durch Zeit)
    pixel_speed = (dx**2 + dy**2) ** 0.5 / dt 

    # Update der alten Werte für den nächsten Aufruf
    last_ball_pos = ball_pos
    last_time = timestamp
    return pixel_speed