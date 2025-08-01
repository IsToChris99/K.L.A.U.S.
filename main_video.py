import cv2
import numpy as np
import time
from threading import Thread, Lock
from queue import Queue, LifoQueue, Empty

# Local imports
from detection.ball_detector import BallDetector  
from detection.field_detector import FieldDetector
from analysis.goal_scorer import GoalScorer
from input.videostream import VideoStream
from processing.preprocessor import Preprocessor
from config import (
    VIDEO_PATH, USE_WEBCAM, FRAME_WIDTH, FRAME_HEIGHT,
    DISPLAY_FPS, DISPLAY_INTERVAL,
    COLOR_BALL_HIGH_CONFIDENCE, COLOR_BALL_MED_CONFIDENCE, 
    COLOR_BALL_LOW_CONFIDENCE, COLOR_BALL_TRAIL,
    COLOR_FIELD_CONTOUR, COLOR_FIELD_CORNERS, COLOR_FIELD_BOUNDS, COLOR_GOALS, CAMERA_CALIBRATION_FILE, BALL_TRAIL_MAX_LENGTH, BALL_TRAIL_THICKNESS_FACTOR
)

# ================== COMBINED TRACKER ==================

class CombinedTracker:

    

    """Combined Ball and Field Tracker with Multithreading"""
    
    def __init__(self, video_path=None):
        self.count = 0

        self.video_path = video_path
        
        self.ball_tracker = BallDetector(video_path, USE_WEBCAM)
        self.field_detector = FieldDetector()
        self.goal_scorer = GoalScorer()
        
        # Try to load saved calibration
        if not self.field_detector.load_calibration():
            print("No calibration file found. Perform manual calibration...")
        
        # Calibration mode - only activate on key press
        self.calibration_mode = False
        self.calibration_requested = False

        self.stream = VideoStream(video_path)
        
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
        self.ball_result = None
        self.field_data = None
        
        # Display control variables
        self.frame_count = 0
        self.processing_fps = 0
        self.last_fps_time = time.time()
        self.last_frame_count = 0

        # Camera calibration
        self.camera_calibration = Preprocessor(CAMERA_CALIBRATION_FILE)
        
    def frame_reader_thread_method(self):
        """Centralized frame reading thread"""
        
        while self.running:
            t_start = time.perf_counter()
            ret, frame = self.stream.read()
            t_read = time.perf_counter()
            
            if not ret or frame is None:
                self.frame_queue.put(None)
                break

            # Apply camera calibration (undistortion) using optimized Preprocessor
            frame = self.camera_calibration.undistort_frame(frame)

            # Resize frame to target size
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            t_resize = time.perf_counter()
            
            with self.result_lock:
                self.current_frame = frame.copy()

            self.count += 1
            if self.count % 250 == 0: # Alle 250 Frames
                read_duration = (t_read - t_start) * 1000  # in ms
                resize_duration = (t_resize - t_read) * 1000 # in ms
                # Diese print-Anweisung ist für Debugging gedacht und sollte später entfernt werden.
                #print(f"\rTimestamp: {metadata.get('timestamp_ns', 'N/A')}", end="")

            # Versuche, den neuen Frame in die Queue zu legen.
            # Wenn die Queue voll ist (Analyse-Threads sind zu langsam),
            # leere sie zuerst, um den alten Frame zu verwerfen und Platz für den neuen zu machen.
            if self.frame_queue.full():
                try:
                    # Entferne den alten, nicht verarbeiteten Frame
                    self.frame_queue.get_nowait()
                except Empty:
                    pass # Kann passieren, wenn ein anderer Thread ihn genau jetzt geholt hat
            
            # Lege den neusten Frame in die nun freie Queue
            self.frame_queue.put(frame)

        # Signal zum Beenden an die Worker senden
        self.frame_queue.put(None)
        self.frame_queue.put(None)
        
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

            # Show Kalman velocity
            if velocity is not None:
                cv2.arrowedLine(frame, center,
                            (int(center[0] + velocity[0]*30), int(center[1] + velocity[1]*30)),
                            (255, 0, 255), 2)
        
        # Ball trail drawing
        
        # if len(smoothed_pts) >= 64:
        #     print(f"\rDrawing ball trail with {len(smoothed_pts)} points", end="")
        #     cv2.line(frame, smoothed_pts[0], smoothed_pts[63], COLOR_BALL_TRAIL, 4)  # Dummy point for trail

        for i in range(1, len(smoothed_pts)):
            if smoothed_pts[i - 1] is None or smoothed_pts[i] is None:
                continue
            thickness = int(np.sqrt(BALL_TRAIL_MAX_LENGTH / float(i + 1)) * BALL_TRAIL_THICKNESS_FACTOR)
            cv2.line(frame, smoothed_pts[i - 1], smoothed_pts[i], COLOR_BALL_TRAIL, thickness)

        # Missing Counter
        cv2.putText(frame, f"Missing: {missing_counter}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_field_visualization(self, frame):
        """Draws field visualization"""
        with self.result_lock:
            field_data_copy = self.field_data.copy() if self.field_data else None
        
        if field_data_copy is None:
            return

        # Field contour
        if field_data_copy['calibrated'] and field_data_copy['field_contour'] is not None:
            cv2.drawContours(frame, [field_data_copy['field_contour']], -1, COLOR_FIELD_CONTOUR, 3)

        # Field corners
        if field_data_copy['field_corners'] is not None:
            for i, corner in enumerate(field_data_copy['field_corners']):
                cv2.circle(frame, tuple(corner), 8, COLOR_FIELD_CORNERS, -1)
                cv2.putText(frame, f"{i+1}", (corner[0]+10, corner[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FIELD_CORNERS, 2)

        # Goals
        for i, goal in enumerate(field_data_copy['goals']):
            x, y, w, h = goal['bounds']
            cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_GOALS, 2)
            cv2.putText(frame, f"Goal {i+1} ({goal['type']})", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GOALS, 2)

        # Field limits - Uses minAreaRect
        if (field_data_copy['calibrated'] and 
            field_data_copy.get('field_rect_points') is not None):
            cv2.drawContours(frame, [field_data_copy['field_rect_points']], -1, COLOR_FIELD_BOUNDS, 2)
        elif field_data_copy['calibrated'] and field_data_copy['field_bounds']:
            # Fallback: normal bounding box rectangle
            x, y, w, h = field_data_copy['field_bounds']
            cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_FIELD_BOUNDS, 2)

        # Calibration info
        if (field_data_copy['calibration_requested'] and 
            field_data_copy['calibration_mode'] and 
            not field_data_copy['calibrated']):
            progress = min(field_data_copy['stable_counter'] / 30, 1.0)
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
        
        # Camera calibration status - IDS Camera doesn't have built-in calibration
        cv2.putText(frame, "Camera: IDS Live Stream", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show key commands
        cv2.putText(frame, "Keys: 1=Ball, 2=Field, 3=Both, r=Calibration, s=Screenshot, g=Reset Score, h=Help", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
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

        # Camera status display - IDS Camera doesn't have calibration info like VideoStream
        print("✓ IDS Camera initialized - live capture active")

        print("=" * 60)
        print("Control commands:")
        print("  'q' - Quit")
        print("  '1' - Show only ball tracking")
        print("  '2' - Show only field tracking")
        print("  '3' - Combined view (default)")
        print("  'r' - Start/recalibrate field calibration")
        print("  's' - Save screenshot (with ball curve if available)")
        print("  'c' - Show camera calibration info")
        print("  'h' - Show help")
        print("=" * 60)
        
        self.start_threads()
        
        try:
            while True:
                # Get current frame for processing from centralized reader
                with self.result_lock:
                    if self.current_frame is None:
                        continue
                    frame = self.current_frame.copy()
                
                self.frame_count += 1
                current_time = time.time()
                measure_time = time.time()
                
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
                    print("  'c' - Show camera calibration info")
                    print("  'g' - Reset score to 0-0")
                    print("  'h' - Show help")
                    print("=" * 60)

                elif key == ord('g'):
                    # Reset score
                    self.goal_scorer.reset_score()
                    print("Score reset!")

                #print(f"\r{(time.time() - measure_time) * 1000000}", end="")

        finally:
            # Cleanup
            self.stop_threads()
            cv2.destroyAllWindows()
            
            print(f"\nCombined Tracker finished.")


# ================== MAIN PROGRAM ==================

if __name__ == "__main__":

    video_path = VIDEO_PATH

    tracker = CombinedTracker(video_path=video_path)
    tracker.run()