import cv2
import numpy as np
import json
import os


class Preprocessor:
    """Class for camera calibration and undistortion"""
    def __init__(self, calibration_file="calibration_data.json"):
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        self.map1 = None
        self.map2 = None
        self.image_size = None
        
        self.load_calibration()
    
    def load_calibration(self):
        """Loads calibration data from JSON file"""
        if not os.path.exists(self.calibration_file):
            print(f"Calibration file {self.calibration_file} not found.")
            return False
        
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data['cameraMatrix'], dtype=np.float32)
            self.dist_coeffs = np.array(data['distCoeffs'], dtype=np.float32)
            self.calibrated = True
            
            # print(f"Camera calibration loaded from {self.calibration_file}")
            # print(f"Kamera-Matrix: {self.camera_matrix.shape}")
            # print(f"Distortioncoefficients: {self.dist_coeffs.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def _initialize_remap_maps(self, frame_size):
        """Initializes the remap maps for optimized undistortion"""
        if self.image_size == frame_size and self.map1 is not None:
            return
            
        # print(f"Initialisiere Entzerrung für Bildgröße: {frame_size}")

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, frame_size, alpha=1.0)
        
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, 
            new_camera_matrix, frame_size, cv2.CV_16SC2)
        
        self.image_size = frame_size
        print("Remap-Maps berechnet.")
    
    def undistort_frame(self, frame):
        """Undistorts an image based on calibration data"""
        if not self.calibrated or frame is None:
            return frame
        
        frame_size = (frame.shape[1], frame.shape[0])
        
        if self.map1 is None or self.image_size != frame_size:
            self._initialize_remap_maps(frame_size)
        
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
    
    def get_optimal_camera_matrix(self, frame_size, alpha=1.0):
        """Calculates optimal camera matrix for undistortion"""
        if not self.calibrated:
            return self.camera_matrix, None
        
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, frame_size, alpha)
        return new_camera_matrix, roi
    
    def get_performance_info(self):
        """Resturns performance information about the calibration"""
        return {
            'calibrated': self.calibrated,
            'maps_initialized': self.map1 is not None,
            'image_size': self.image_size,
            'optimization_active': self.map1 is not None
        }