import cv2
import numpy as np
import time
from threading import Thread, Lock

# Local imports
from detection.ball_detector import BallDetector  
from detection.field_detector import FieldDetector
from analysis.goal_scorer import GoalScorer
from camera.videostream import VideoStream
from config import (
    VIDEO_PATH, USE_WEBCAM, FRAME_WIDTH, FRAME_HEIGHT,
    DISPLAY_FPS, DISPLAY_INTERVAL,
    COLOR_BALL_HIGH_CONFIDENCE, COLOR_BALL_MED_CONFIDENCE, 
    COLOR_BALL_LOW_CONFIDENCE, COLOR_BALL_TRAIL,
    COLOR_FIELD_CONTOUR, COLOR_FIELD_CORNERS, COLOR_FIELD_BOUNDS, COLOR_GOALS
)

# ================== COMBINED TRACKER ==================

class CombinedTracker:
    """Combined Ball and Field Tracker with Multithreading"""
    
    def __init__(self, video_path, use_webcam=False):
        
        self.video_path = video_path
        self.use_webcam = use_webcam
        
        self.ball_tracker = BallDetector(video_path, use_webcam)
        self.field_detector = FieldDetector()
        self.goal_scorer = GoalScorer()
        
        # Try to load saved calibration
        if not self.field_detector.load_calibration():
            print("No calibration file found. Perform manual calibration...")
        
        # Calibration mode - only activate on key press
        self.calibration_mode = False
        self.calibration_requested = False
        
        video_source = 0 if use_webcam else video_path
        self.stream = VideoStream(video_source)
        
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
        self.frame_lock = Lock()
        
        self.current_frame = None
        self.ball_result = None
        self.field_data = None
        
        # Display control variables
        self.frame_count = 0
        self.processing_fps = 0
        self.last_fps_time = time.time()
        self.last_frame_count = 0
        
    def frame_reader_thread_method(self):
        """Centralized frame reading thread"""
        while self.running:
            ret, frame = self.stream.read()
            if not ret or frame is None:
                break
                
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            with self.frame_lock:
                self.current_frame = frame.copy()
        
    def ball_tracking_thread(self):
        """Thread for Ball-Tracking"""
        while self.running:
            # Get current frame from centralized reader
            with self.frame_lock:
                if self.current_frame is None:
                    
                    continue
                frame = self.current_frame.copy()
            
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
            
            with self.frame_lock:
                self.ball_result = {
                    'detection': detection_result,
                    'smoothed_pts': list(self.ball_tracker.smoothed_pts),
                    'missing_counter': self.ball_tracker.missing_counter
                }
                
            
            
    def field_tracking_thread(self):
        """Thread for Field-Tracking"""
        while self.running:
            # Get current frame from centralized reader
            with self.frame_lock:
                if self.current_frame is None:
                    
                    continue
                frame = self.current_frame.copy()
            
            # Use FieldDetector's calibration logic
            if (self.calibration_requested and 
                self.calibration_mode and 
                not self.field_detector.calibrated):
                self.field_detector.calibrate(frame)
            
            # Store current field data
            with self.frame_lock:
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
        if self.ball_result is None:
            return
            
        detection = self.ball_result['detection']
        smoothed_pts = self.ball_result['smoothed_pts']
        missing_counter = self.ball_result['missing_counter']
        
        # Draw ball info
        if detection[0] is not None:
            center, radius, confidence = detection

            # Color selection based on confidence
            if confidence >= 0.8:
                color = COLOR_BALL_HIGH_CONFIDENCE  # Green
            elif confidence >= 0.6:
                color = COLOR_BALL_MED_CONFIDENCE   # Yellow
            else:
                color = COLOR_BALL_LOW_CONFIDENCE   # Orange
                
            cv2.circle(frame, center, 3, color, -1)
            cv2.circle(frame, center, int(radius), color, 2)
            
            cv2.putText(frame, f"R: {radius:.1f}", (center[0] + 15, center[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (center[0] + 15, center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Ball trail drawing
        for i in range(1, len(smoothed_pts)):
            if smoothed_pts[i - 1] is None or smoothed_pts[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, smoothed_pts[i - 1], smoothed_pts[i], COLOR_BALL_TRAIL, thickness)
        
        # Missing Counter
        cv2.putText(frame, f"Missing: {missing_counter}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_field_visualization(self, frame):
        """Draws field visualization"""
        if self.field_data is None:
            return

        # Field contour
        if self.field_data['calibrated'] and self.field_data['field_contour'] is not None:
            cv2.drawContours(frame, [self.field_data['field_contour']], -1, COLOR_FIELD_CONTOUR, 3)

        # Field corners
        if self.field_data['field_corners'] is not None:
            for i, corner in enumerate(self.field_data['field_corners']):
                cv2.circle(frame, tuple(corner), 8, COLOR_FIELD_CORNERS, -1)
                cv2.putText(frame, f"{i+1}", (corner[0]+10, corner[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FIELD_CORNERS, 2)

        # Goals
        for i, goal in enumerate(self.field_data['goals']):
            x, y, w, h = goal['bounds']
            cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_GOALS, 2)
            cv2.putText(frame, f"Goal {i+1} ({goal['type']})", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GOALS, 2)

        # Field limits - Uses minAreaRect
        if (self.field_data['calibrated'] and 
            self.field_data.get('field_rect_points') is not None):
            cv2.drawContours(frame, [self.field_data['field_rect_points']], -1, COLOR_FIELD_BOUNDS, 2)
        elif self.field_data['calibrated'] and self.field_data['field_bounds']:
            # Fallback: normal bounding box rectangle
            x, y, w, h = self.field_data['field_bounds']
            cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_FIELD_BOUNDS, 2)

        # Calibration info
        if (self.field_data['calibration_requested'] and 
            self.field_data['calibration_mode'] and 
            not self.field_data['calibrated']):
            progress = min(self.field_data['stable_counter'] / 30, 1.0)
            progress_width = int(300 * progress)
            
            cv2.rectangle(frame, (10, 130), (310, 160), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 130), (10 + progress_width, 160), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 130), (310, 160), (255, 255, 255), 2)
            
            cv2.putText(frame, f"Calibration: {progress*100:.1f}%", (10, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def draw_status_info(self, frame):
        """Draws status information"""
        # Mode display
        mode_text = {
            self.BALL_ONLY: "Ball Tracking",
            self.FIELD_ONLY: "Field Tracking", 
            self.COMBINED: "Combined Tracking"
        }
        
        cv2.putText(frame, f"Mode: {mode_text[self.visualization_mode]}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # FPS display
        cv2.putText(frame, f"Processing: {self.processing_fps:.1f} FPS", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Camera calibration status
        if self.stream.is_calibrated():
            cv2.putText(frame, "Camera: Undistorted", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Camera: Not Calibrated", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show key commands
        cv2.putText(frame, "Keys: 1=Ball, 2=Field, 3=Both, r=Calibration, s=Screenshot, d=Debug, g=Reset Score, h=Help", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def print_camera_info(self):
        """Shows detailed camera calibration info in the console"""
        camera_info = self.stream.get_camera_info()
        
        print("\n" + "=" * 70)
        print("Camera Calibration Info:")
        print("=" * 70)
        
        if camera_info['calibrated']:
            print("✓ Camera is calibrated and undistortion is active")

            # Performance-Status
            if camera_info.get('optimization_active', False):
                print("Performance optimization active:")
                print("   ✓ Remap maps precomputed - very fast undistortion!")
                if camera_info.get('image_size'):
                    size = camera_info['image_size']
                    print(f"   ✓ Optimized for image size: {size[0]}x{size[1]}")
            else:
                print("⚠  PERFORMANCE-WARNING:")
                print("   ✗ Remap maps not initialized")
                print("   → First frames will be processed more slowly")

            print(f"\nCamera Matrix:")
            for i, row in enumerate(camera_info['camera_matrix']):
                print(f"  [{row[0]:10.2f}, {row[1]:10.2f}, {row[2]:10.2f}]")

            print(f"\nDistortion Coefficients:")
            dist_coeffs = camera_info['dist_coeffs']
            print(f"  k1: {dist_coeffs[0]:10.6f}  (radial distortion)")
            print(f"  k2: {dist_coeffs[1]:10.6f}  (radial distortion)")
            print(f"  p1: {dist_coeffs[2]:10.6f}  (tangential distortion)")
            print(f"  p2: {dist_coeffs[3]:10.6f}  (tangential distortion)")
            print(f"  k3: {dist_coeffs[4]:10.6f}  (radial distortion)")

            print(f"\nSource: calibration_data.json")
            print("Algorithm: cv2.remap() with precomputed maps")
            print("All images will be automatically undistorted!")
        else:
            print("✗ No camera calibration found")
            print("Create a calibration_data.json file with:")
            print("  - cameraMatrix: 3x3 camera matrix")
            print("  - distCoeffs: 5 distortion coefficients")
            print("No undistortion active - lens distortion will remain!")
        
        print("=" * 70)
    
    def start_threads(self):
        """Starts the tracking threads"""
        self.running = True
        
        # Always start the frame reader thread
        self.frame_reader_thread = Thread(target=self.frame_reader_thread_method, daemon=True)
        self.frame_reader_thread.start()
        
        if self.visualization_mode in [self.BALL_ONLY, self.COMBINED]:
            self.ball_thread = Thread(target=self.ball_tracking_thread, daemon=True)
            self.ball_thread.start()
            
        if self.visualization_mode in [self.FIELD_ONLY, self.COMBINED]:
            self.field_thread = Thread(target=self.field_tracking_thread, daemon=True)
            self.field_thread.start()

    def stop_threads(self):
        """Stops the tracking threads"""
        self.running = False
        
        if self.frame_reader_thread and self.frame_reader_thread.is_alive():
            self.frame_reader_thread.join(timeout=1.0)
        
        if self.ball_thread and self.ball_thread.is_alive():
            self.ball_thread.join(timeout=1.0)
            
        if self.field_thread and self.field_thread.is_alive():
            self.field_thread.join(timeout=1.0)
    
    def run(self):
        """Main loop for combined tracker"""
        print("=" * 60)

        # Camera status display
        if self.stream.is_calibrated():
            print("✓ Camera calibration loaded - undistortion active")
        else:
            print("⚠ No camera calibration - undistortion disabled")

        print("=" * 60)
        print("Control commands:")
        print("  'q' - Quit")
        print("  '1' - Show only ball tracking")
        print("  '2' - Show only field tracking")
        print("  '3' - Combined view (default)")
        print("  'r' - Start/recalibrate field calibration")
        print("  's' - Save screenshot (with ball curve if available)")
        print("  'd' - Toggle debug goal detection")
        print("  'c' - Show camera calibration info")
        print("  'h' - Show help")
        print("=" * 60)
        
        self.start_threads()
        
        try:
            while True:
                # Get current frame for processing from centralized reader
                with self.frame_lock:
                    if self.current_frame is None:
                        continue
                    frame = self.current_frame.copy()
                
                self.frame_count += 1
                current_time = time.time()
                
                # Calculate processing FPS every second
                if current_time - self.last_fps_time >= 1.0:
                    frames_processed = self.frame_count - self.last_frame_count
                    self.processing_fps = frames_processed / (current_time - self.last_fps_time)
                    self.last_fps_time = current_time
                    self.last_frame_count = self.frame_count
                
                # Check if we should display this frame (every 8.333 frames for 30 FPS display at 250 FPS processing)
                should_display = (self.frame_count % 8 == 0)
                
                if should_display:
                    # Create display frame only when needed
                    display_frame = frame.copy()
                    
                    # Visualization based on mode
                    if self.visualization_mode in [self.BALL_ONLY, self.COMBINED]:
                        self.draw_ball_visualization(display_frame)
                        
                    if self.visualization_mode in [self.FIELD_ONLY, self.COMBINED]:
                        self.draw_field_visualization(display_frame)
                    
                    self.goal_scorer.draw_score_info(display_frame)
                    self.draw_status_info(display_frame)

                    cv2.imshow("Combined Tracker", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF if self.frame_count % 50 == 0 else 255
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    print("Switching to: Ball tracking only")
                    self.stop_threads()
                    self.visualization_mode = self.BALL_ONLY
                    self.start_threads()
                elif key == ord('2'):
                    print("Switching to: Field tracking only")
                    self.stop_threads()
                    self.visualization_mode = self.FIELD_ONLY
                    self.start_threads()
                elif key == ord('3'):
                    print("Switching to: Combined view")
                    self.stop_threads()
                    self.visualization_mode = self.COMBINED
                    self.start_threads()
                elif key == ord('r'):
                    print("Calibrate field")
                    self.field_detector.calibrated = False
                    self.field_detector.stable_detection_counter = 0
                    self.calibration_mode = True
                    self.calibration_requested = True
                elif key == ord('s'):
                    # Screenshot with ball curve
                    # Use the current processing frame for screenshot (not display frame)
                    screenshot_frame = frame.copy()
                    
                    if self.visualization_mode in [self.BALL_ONLY, self.COMBINED]:
                        self.draw_ball_visualization(screenshot_frame)
                        
                    if self.visualization_mode in [self.FIELD_ONLY, self.COMBINED]:
                        self.draw_field_visualization(screenshot_frame)
                    
                    self.goal_scorer.draw_score_info(screenshot_frame)
                    self.draw_status_info(screenshot_frame)
                    
                    timestamp = int(time.time())
                    cv2.imwrite(f"combined_screenshot_{timestamp}.jpg", screenshot_frame)
                    print(f"Screenshot saved: combined_screenshot_{timestamp}.jpg")
                elif key == ord('h'):
                    print("\n" + "=" * 60)
                    print("KEY COMMANDS:")
                    print("  'q' - Quit")
                    print("  '1' - Show ball tracking only")
                    print("  '2' - Show field tracking only")
                    print("  '3' - Show combined view")
                    print("  'r' - Start/recalibrate field calibration")
                    print("  's' - Save screenshot (with ball curve if available)")
                    print("  'd' - Toggle debug goal detection")
                    print("  'c' - Show camera calibration info")
                    print("  'g' - Reset score to 0-0")
                    print("  'h' - Show help")
                    print("=" * 60)
                elif key == ord('c'):
                    # Show camera information
                    self.print_camera_info()
                elif key == ord('d'):
                    # Toggle debug goal detection
                    self.field_detector.debug_goal_detection = not self.field_detector.debug_goal_detection
                    status = "enabled" if self.field_detector.debug_goal_detection else "disabled"
                    print(f"Debug Goal Detection {status}")
                    if not self.field_detector.debug_goal_detection:
                        # Close debug window
                        try:
                            cv2.destroyWindow("Goal Search Mask")
                            cv2.destroyWindow("Original Goal Mask")
                            cv2.destroyWindow("Extended Field Mask")
                            cv2.destroyWindow("Field Mask")
                            cv2.destroyWindow("All Goal Contours (Before Filter)")
                        except:
                            pass
                elif key == ord('g'):
                    # Reset score
                    self.goal_scorer.reset_score()
                    print("Score reset!")
                
        finally:
            # Cleanup
            self.stop_threads()
            self.stream.stop()
            cv2.destroyAllWindows()
            
            print(f"\nCombined Tracker finished.")


# ================== MAIN PROGRAM ==================

if __name__ == "__main__":
    # Load configuration from config file
    video_path = VIDEO_PATH
    use_webcam = USE_WEBCAM

    # Create and start combined tracker
    tracker = CombinedTracker(video_path, use_webcam)
    tracker.run()