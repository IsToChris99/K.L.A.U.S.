import cv2
from threading import Thread, Lock
from processing.cpu_preprocessor import Preprocessor
from config import CAMERA_CALIBRATION_FILE

class VideoStream:
    """Video stream class for reading video files"""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.frame_lock = Lock()

        # Strart thread
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            
            with self.frame_lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.frame_lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()