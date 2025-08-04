import cv2
import numpy as np
import time
from threading import Thread, Lock
from queue import Queue, LifoQueue, Empty
import sys
from PySide6.QtWidgets import QApplication

# Local imports
from detection.ball_detector import BallDetector  
from detection.field_detector import FieldDetector
from analysis.goal_scorer import GoalScorer
from input.ids_camera import IDS_Camera
from processing.cpu_preprocessor import CPUPreprocessor
#from processing.gpu_preprocessor import GPUPreprocessor    #import in qt_window.py
from display.qt_window import KickerMainWindow
import config
from match_modes.match_modes import MatchModes  # Import match modes class

# ================== COMBINED TRACKER ==================

class CombinedTracker:
    """Combined Ball and Field Tracker with Multithreading"""
    
    def __init__(self):
        self.count = 0
        
        self.ball_tracker = BallDetector()
        self.field_detector = FieldDetector()
        self.goal_scorer = GoalScorer()
        
        # Try to load saved calibration
        if not self.field_detector.load_calibration():
            print("No calibration file found. Perform manual calibration...")
        
        # Calibration mode - only activate on key press
        self.calibration_mode = False
        self.calibration_requested = False
        
        # Initialize IDS Camera later when needed (not in constructor)
        self.camera = None
        self.camera_available = False
        
        # Visualization modes
        self.BALL_ONLY = 1
        self.FIELD_ONLY = 2  
        self.COMBINED = 3
        self.visualization_mode = self.COMBINED
        
        # Threading variables
        self.ball_thread = None
        self.field_thread = None
        self.frame_reader_thread = None
        self.running = False
        self.frame_queue = LifoQueue(maxsize=1) 
        self.result_lock = Lock()
        
        self.current_frame = None
        self.current_bayer_frame = None
        self.ball_result = None
        self.field_data = None
        
        # Display control variables
        self.frame_count = 0
        self.processing_fps = 0
        self.last_fps_time = time.time()
        self.last_frame_count = 0

        # Camera calibration
        self.camera_calibration = CPUPreprocessor(config.CAMERA_CALIBRATION_FILE)
        self.gpu_preprocessor = None  # Will be created lazily in tracking thread
        self.cpu_preprocessor = CPUPreprocessor(config.CAMERA_CALIBRATION_FILE)
        self.camera_calibration.initialize_for_size((config.DETECTION_WIDTH, config.DETECTION_HEIGHT))

        self.enable_undistortion = True
        self.use_gpu_processing = False  # Use CPU by default
        
    def initialize_camera(self):
        """Initializes the camera with error handling"""
        try:
            # If camera already exists, stop it first
            if self.camera is not None:
                try:
                    print("Stopping existing camera before reinitialization...")
                    self.camera.stop()
                    time.sleep(0.2)  # Wait briefly
                except:
                    pass  # Ignore errors when stopping
                self.camera = None
            
            print("Initializing new camera...")
            self.camera = IDS_Camera()
            self.camera_available = True
            print("Camera successfully initialized")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.camera_available = False
            self.camera = None
            return False
    
    def frame_reader_thread_method(self):
        """Frame reading thread - reads from camera"""
        while self.running:
            # Camera mode
            if not self.camera_available or self.camera is None:
                time.sleep(0.1)
                continue
                
            try:
                bayer_frame, metadata = self.camera.get_frame()
                if bayer_frame is None:
                    continue

                # Store raw Bayer frame
                with self.result_lock:
                    self.current_bayer_frame = bayer_frame

                self.count += 1
            except Exception as e:
                print(f"Error reading frame: {e}")
                time.sleep(0.1)
        
    def ball_tracking_thread(self):
        """Thread for Ball-Tracking"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break
            except Empty:
                continue
                
            # Field bounds for restricted ball search
            field_bounds = None
            goals = []
            if self.field_data and self.field_data['calibrated']:
                if self.field_data['field_bounds']:
                    field_bounds = self.field_data['field_bounds']
                if self.field_data['goals']:
                    goals = self.field_data['goals']

            # Ball detection with field_bounds
            detection_result = self.ball_tracker.detect_ball(frame, field_bounds)
            self.ball_tracker.update_tracking(detection_result, field_bounds)
            
            # Goal scoring system update
            ball_position = detection_result[0] if detection_result[0] is not None else None
            self.goal_scorer.update_ball_tracking(
                ball_position, 
                goals, 
                field_bounds, 
                self.ball_tracker.missing_counter
            )
            
            with self.result_lock:
                self.ball_result = {
                    'detection': detection_result,
                    'smoothed_pts': list(self.ball_tracker.smoothed_pts),
                    'missing_counter': self.ball_tracker.missing_counter
                }
                
    def field_tracking_thread(self):
        """Thread for Field-Tracking"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break
            except Empty:
                continue
            
            # Use FieldDetector's calibration logic
            if (self.calibration_requested and 
                self.calibration_mode and 
                not self.field_detector.calibrated):
                self.field_detector.calibrate(frame)
            
            # Store current field data
            with self.result_lock:
                self.field_data = {
                    'calibrated': self.field_detector.calibrated,
                    'field_contour': self.field_detector.field_contour,
                    'field_corners': self.field_detector.field_corners,
                    'field_bounds': self.field_detector.field_bounds,
                    'field_rect_points': self.field_detector.field_rect_points,
                    'goals': self.field_detector.goals,
                    'stable_counter': self.field_detector.stable_detection_counter,
                    'calibration_mode': self.calibration_mode,
                    'calibration_requested': self.calibration_requested
                }
    
    def draw_ball_visualization(self, frame):
        """Draws ball visualization"""
        with self.result_lock:
            ball_result_copy = self.ball_result.copy() if self.ball_result else None
        
        if ball_result_copy is None:
            return

        detection = ball_result_copy['detection']
        smoothed_pts = ball_result_copy['smoothed_pts']
        missing_counter = ball_result_copy['missing_counter']

        # Draw ball info
        if detection[0] is not None:
            center, radius, confidence, velocity = detection

            # Color selection based on confidence
            if confidence >= 0.8:
                color = config.COLOR_BALL_HIGH_CONFIDENCE
            elif confidence >= 0.6:
                color = config.COLOR_BALL_MED_CONFIDENCE
            else:
                color = config.COLOR_BALL_LOW_CONFIDENCE

            cv2.circle(frame, center, 3, color, -1)
            cv2.circle(frame, center, int(radius), color, 2)

            # Show Kalman velocity
            if velocity is not None:
                cv2.arrowedLine(frame, center,
                            (int(center[0] + velocity[0]*30), int(center[1] + velocity[1]*30)),
                            (255, 0, 255), 2)

        # Ball trail drawing
        for i in range(1, len(smoothed_pts)):
            if smoothed_pts[i - 1] is None or smoothed_pts[i] is None:
                continue
            thickness = int(np.sqrt(config.BALL_TRAIL_MAX_LENGTH / float(i + 1)) * config.BALL_TRAIL_THICKNESS_FACTOR)
            cv2.line(frame, smoothed_pts[i - 1], smoothed_pts[i], config.COLOR_BALL_TRAIL, thickness)
    
    def draw_field_visualization(self, frame):
        """Draws field visualization"""
        with self.result_lock:
            field_data_copy = self.field_data.copy() if self.field_data else None
        
        if field_data_copy is None:
            return

        # Field contour
        if field_data_copy['calibrated'] and field_data_copy['field_contour'] is not None:
            cv2.drawContours(frame, [field_data_copy['field_contour']], -1, config.COLOR_FIELD_CONTOUR, 3)

        # Field corners
        if field_data_copy['field_corners'] is not None:
            for i, corner in enumerate(field_data_copy['field_corners']):
                cv2.circle(frame, tuple(corner), 8, config.COLOR_FIELD_CORNERS, -1)

        # Goals
        for i, goal in enumerate(field_data_copy['goals']):
            x, y, w, h = goal['bounds']
            cv2.rectangle(frame, (x, y), (x+w, y+h), config.COLOR_GOALS, 2)

        # Field limits
        if (field_data_copy['calibrated'] and 
            field_data_copy.get('field_rect_points') is not None):
            cv2.drawContours(frame, [field_data_copy['field_rect_points']], -1, config.COLOR_FIELD_BOUNDS, 2)
        elif field_data_copy['calibrated'] and field_data_copy['field_bounds']:
            x, y, w, h = field_data_copy['field_bounds']
            cv2.rectangle(frame, (x, y), (x+w, y+h), config.COLOR_FIELD_BOUNDS, 2)

        # Calibration progress
        if (field_data_copy['calibration_requested'] and 
            field_data_copy['calibration_mode'] and 
            not field_data_copy['calibrated']):
            progress = min(field_data_copy['stable_counter'] / 30, 1.0)
            progress_width = int(300 * progress)
            
            cv2.rectangle(frame, (10, 130), (310, 160), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 130), (10 + progress_width, 160), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 130), (310, 160), (255, 255, 255), 2)
    
    def start_threads(self):
        """Starts the tracking threads"""
        if self.running:
            return
            
        self.running = True
        
        self.frame_reader_thread = Thread(target=self.frame_reader_thread_method, daemon=True)
        self.frame_reader_thread.start()
        
        self.ball_thread = Thread(target=self.ball_tracking_thread, daemon=True)
        self.ball_thread.start()
            
        self.field_thread = Thread(target=self.field_tracking_thread, daemon=True)
        self.field_thread.start()

    def stop_threads(self):
        """Stops the tracking threads"""
        print("Stopping all tracker threads...")
        self.running = False
        
        # Send termination signals to worker threads
        try:
            self.frame_queue.put(None)
            self.frame_queue.put(None)
        except:
            pass
        
        # Wait for frame reader thread to stop (important for camera)
        if self.frame_reader_thread and self.frame_reader_thread.is_alive():
            print("Waiting for frame reader thread...")
            self.frame_reader_thread.join(timeout=2.0)
            if self.frame_reader_thread.is_alive():
                print("Warning: Frame reader thread could not be stopped")
            else:
                print("Frame reader thread stopped")
        
        if self.ball_thread and self.ball_thread.is_alive():
            print("Waiting for ball thread...")
            self.ball_thread.join(timeout=1.0)
            if self.ball_thread.is_alive():
                print("Warning: Ball thread could not be stopped")
            else:
                print("Ball thread stopped")
            
        if self.field_thread and self.field_thread.is_alive():
            print("Waiting for field thread...")
            self.field_thread.join(timeout=1.0)
            if self.field_thread.is_alive():
                print("Warning: Field thread could not be stopped")
            else:
                print("Field thread stopped")
            
        # Stop camera if available
        if self.camera_available and self.camera is not None:
            try:
                print("Stopping camera...")
                self.camera.stop()
                print("Camera stopped")
                # Wait briefly for all camera threads to terminate
                time.sleep(0.5)
                # Reset camera object for clean restart
                self.camera = None
                self.camera_available = False
                print("Camera object reset")
            except Exception as e:
                print(f"Error stopping camera: {e}")
                # Reset camera even on error
                self.camera = None
                self.camera_available = False
    
    def toggle_processing_mode(self):
        """Toggles between CPU and GPU processing"""
        self.use_gpu_processing = not self.use_gpu_processing
        print(f"Processing mode switched to: {'GPU' if self.use_gpu_processing else 'CPU'}")


# ================== MAIN PROGRAM ==================

def main_gui():
    """Main function to start the GUI"""
    app = QApplication(sys.argv)
    
    # Start GUI even without camera
    try:
        # Create tracker
        tracker = CombinedTracker()
        matcher = MatchModes()  # Initialize match modes

        # Create main window and pass tracker
        window = KickerMainWindow(tracker, matcher)
        window.show()
        window.add_log_message("GUI started - testing camera status...")
        
        # Initial camera test (without aborting on error)
        app.processEvents()  # Show GUI before test
        window.test_camera()
        
        return app.exec()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        return 1
    
def main():
    sys.exit(main_gui())
    
if __name__ == "__main__":
    main()
