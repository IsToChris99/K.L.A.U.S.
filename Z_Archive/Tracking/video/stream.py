import cv2
from threading import Thread

class VideoStream:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                self.stop()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()
