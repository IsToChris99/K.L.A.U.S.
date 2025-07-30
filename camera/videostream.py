import cv2
from threading import Thread, Lock
from processing.preprocessor import Preprocessor
from config import CAMERA_CALIBRATION_FILE

class VideoStream:
    """Video stream class for reading video files or webcam input"""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.frame_lock = Lock()
        
        # Camera calibration
        self.camera_calibration = Preprocessor(CAMERA_CALIBRATION_FILE)

        # Strart thread
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            
            # Use camera calibration
            if self.camera_calibration.calibrated and frame is not None:
                frame = self.camera_calibration.undistort_frame(frame)
            
            with self.frame_lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.frame_lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()
    
    def is_calibrated(self):
        """Returns whether the camera is calibrated"""
        return self.camera_calibration.calibrated
    
    def get_camera_info(self):
        """Returns camera information"""
        if self.camera_calibration.calibrated:
            performance_info = self.camera_calibration.get_performance_info()
            return {
                'camera_matrix': self.camera_calibration.camera_matrix,
                'dist_coeffs': self.camera_calibration.dist_coeffs,
                'calibrated': True,
                'maps_initialized': performance_info['maps_initialized'],
                'optimization_active': performance_info['optimization_active'],
                'image_size': performance_info['image_size']
            }
        return {'calibrated': False}