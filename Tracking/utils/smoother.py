from collections import deque

class Smoother:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.x_vals = deque()
        self.y_vals = deque()
        self.sum_x = 0
        self.sum_y = 0

    def update(self, point):
        if point is None:
            return None
        x, y = point
        self.x_vals.append(x)
        self.y_vals.append(y)
        self.sum_x += x
        self.sum_y += y
        if len(self.x_vals) > self.window_size:
            self.sum_x -= self.x_vals.popleft()
            self.sum_y -= self.y_vals.popleft()
        avg_x = int(self.sum_x / len(self.x_vals))
        avg_y = int(self.sum_y / len(self.y_vals))
        return (avg_x, avg_y)
