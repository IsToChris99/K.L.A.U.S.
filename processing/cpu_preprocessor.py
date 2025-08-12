import cv2
import numpy as np
import json
import os
import config


class CPUPreprocessor:
    """Class for camera calibration and undistortion"""
    def __init__(self, bayer_size = (config.CAM_WIDTH, config.CAM_HEIGHT), target_size=(config.DETECTION_WIDTH, config.DETECTION_HEIGHT), calibration_file=config.CAMERA_CALIBRATION_FILE):
        self.bayer_width = bayer_size[0]
        self.bayer_height = bayer_size[1]
        self.target_width = target_size[0]
        self.target_height = target_size[1]
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        self.map1 = None
        self.map2 = None
        self.image_size = None
        
        self.load_calibration()
    
    def initialize_for_size(self, target_size):
        """Pre-initialize remap maps for a specific target size"""
        if self.calibrated and target_size:
            self._initialize_remap_maps(target_size)
            print(f"Preprocessor initialized for size: {target_size}")
    
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
    
    def process_frame(self, bayer_frame):
        """Processes a single Bayer frame through the entire pipeline.
        Returns the undistorted RGB frame in resized and original resolution."""
        rgb_frame = cv2.cvtColor(bayer_frame, cv2.COLOR_BayerRG2RGB)
        # Apply undistortion
        undist_frame = self.undistort_frame(rgb_frame)
        # Resize to target size
        resized_frame = cv2.resize(undist_frame, (self.target_width, self.target_height))
        return resized_frame, undist_frame

    def process_display_frame(self, bayer_frame, perspective_transform_matrix):
        """Processes a single Bayer frame for display purposes.
        Returns the undistorted RGB frame (with perspective distortion)in original resolution."""
        rgb_frame = cv2.cvtColor(bayer_frame, cv2.COLOR_BayerRG2RGB)
        # Apply undistortion
        undist_frame = self.undistort_frame(rgb_frame)

        if perspective_transform_matrix is not None:
            undist_frame = cv2.warpPerspective(undist_frame, perspective_transform_matrix, (undist_frame.shape[1], undist_frame.shape[0]))

        return undist_frame