import cv2
import numpy as np
import time
from threading import Thread, Lock
from queue import Queue, LifoQueue, Empty

# Local imports
from detection.ball_detector import BallDetector
from detection.field_detector_markers import FieldDetector
from analysis.goal_scorer import GoalScorer
from analysis.ball_speed import calculate_ball_speed
from detection.player_detector import PlayerDetector
from input.ids_camera import IDS_Camera
from processing.cpu_preprocessor import CPUPreprocessor
from processing.gpu_preprocessor import GPUPreprocessor
import config

# ================== COMBINED TRACKER ==================

class CombinedTracker:

    """Combined Ball and Field Tracker with Multithreading"""
    
    def __init__(self, video_path=None, use_webcam=False):
        self.count = 0
        # IDS Camera doesn't use video_path or use_webcam parameters
        # but we keep them for compatibility
        
        
        self.ball_tracker = BallDetector()
        self.field_detector = FieldDetector()
        self.goal_scorer = GoalScorer()
        self.player_detector = PlayerDetector()

        #for Ball Speed
        self.ball_speed = 0.0
        self.last_ball_position = None
        self.pixel_to_m_ratio = config.FIELD_WIDTH_M / config.DETECTION_WIDTH

        self.player_result = None
        self.player_thread = None
        
        # Calibration mode - only activate on key press
        self.calibration_mode = True
        self.calibration_requested = True
        
        # Initialize IDS Camera instead of VideoStream
        self.camera = IDS_Camera()
        
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
        self.current_bayer_frame = None  # Store raw Bayer frame from camera thread
        self.ball_result = None
        self.field_data = None
        self.M_persp = None  # Perspective transform matrix
        self.M_field = None  # Field transform matrix
        
        # Display control variables
        self.frame_count = 0
        self.processing_fps = 0
        self.last_fps_time = time.time()
        self.last_frame_count = 0
        self.last_frame_time = None #delete if it does not work

        # Camera calibration - make undistortion optional for performance
        self.camera_calibration = CPUPreprocessor(config.CAMERA_CALIBRATION_FILE)
        self.gpu_preprocessor = GPUPreprocessor((1440, 1080), (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
        self.cpu_preprocessor = CPUPreprocessor(config.CAMERA_CALIBRATION_FILE)
        # Pre-initialize for target resolution to avoid runtime overhead
        self.camera_calibration.initialize_for_size((config.DETECTION_WIDTH, config.DETECTION_HEIGHT))

        # Performance setting: disable undistortion if not critical
        self.enable_undistortion = True  # Set to False to skip undistortion for max speed
        
        # Processing mode control
        self.use_gpu_processing = True  # Start with GPU processing by default
        
        # Ball speed calculator
        self.timestamp_ns = 0
        self.velocity = 0.0
        self.px_to_cm_ratio = 0
        #self.ball_speed = BallSpeed()
        
    def frame_reader_thread_method(self):
        """Frame reading thread - only reads raw Bayer frames"""
        read_duration = 0.0
        while self.running:
            t_start = time.perf_counter()
            
            # Get frame from IDS camera
            bayer_frame, metadata = self.camera.get_frame()
            t_read = time.perf_counter()
            # if metadata is not None:
            #     print(f"\rOffsets: {metadata.get('offset_x', 0)}, {metadata.get('offset_y', 0)}", end="")
            if bayer_frame is None:
                continue

            read_duration += (t_read - t_start)

            # Store raw Bayer frame (processing will happen in main thread)
            with self.result_lock:
                self.current_bayer_frame = bayer_frame

            self.count += 1
            if self.count % 250 == 0:  # Every 250 frames
                read_duration_avg = (read_duration / self.count) * 1000000  # in µs
                fps = self.count / read_duration if read_duration > 0 else 0
                #print(f"\rRead: {read_duration_avg:.2f} µs, FPS: {fps:.2f}", end="")

            # Bayer frame is now stored and will be processed in main thread
            # No queue operations needed here anymore
        
    def ball_tracking_thread(self):
        """Thread for Ball-Tracking"""
        count = 0
        while self.running:
            count += 1
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break
            except Empty:
                continue
                
            
            # Field corners for restricted ball search
            field_corners = None
            goals = []
            if self.field_data and self.field_data['calibrated']:
                if self.field_data['field_corners'] is not None:
                    field_corners = self.field_data['field_corners']
                if self.field_data['goals']:
                    goals = self.field_data['goals']

                # Ball detection with field_corners
                detection_result = self.ball_tracker.detect_ball(frame, field_corners)
                self.ball_tracker.update_tracking(detection_result, field_corners)


                # if count % 8 == 0:  # Every 10 frames
                #     self.velocity = self.ball_speed.update(detection_result[0], self.timestamp_ns if self.timestamp_ns > 0 else time.perf_counter_ns(), self.px_to_cm_ratio)
                #print(f"\rBall Velocity: {self.velocity:.2f} cm/s", end="")

                # Goal scoring system update
                ball_position = detection_result[0] if detection_result[0] is not None else None
                ball_velocity = None
                
                # If no velocity from detection, get it from Kalman tracker
                if self.ball_tracker.kalman_tracker.initialized:
                    ball_velocity = self.ball_tracker.kalman_tracker.get_velocity()                
                self.goal_scorer.update_ball_tracking(
                    ball_position, 
                    goals, 
                    field_corners, 
                    self.ball_tracker.missing_counter,
                    ball_velocity
                )

                # Calculate ball speed using the imported function
                current_time = time.time() # Current time in seconds
                fps = 1.0 / (current_time - self.last_frame_time) if (current_time - self.last_frame_time) > 0 else config.FRAME_RATE_TARGET #Change to actual frame rate
                self.ball_speed, self.last_ball_position = calculate_ball_speed(
                    ball_position,
                    self.last_ball_position,
                    fps,
                    self.pixel_to_m_ratio
                )

                with self.result_lock:
                    self.ball_result = {
                        'detection': detection_result,
                        'smoothed_pts': list(self.ball_tracker.smoothed_pts),
                        'missing_counter': self.ball_tracker.missing_counter,
                        'ball_speed': self.ball_speed 
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
            
            # Optimierte Kalibrierung - jetzt schnell genug für jeden Frame
            if self.calibration_mode:
                before_calibration = time.perf_counter_ns()
                self.field_detector.calibrate(frame)
                after_calibration = time.perf_counter_ns()
                calibration_time = (after_calibration - before_calibration) / 1e9
                # print(f'\rCalibration attempt took: {calibration_time:.4f} seconds', end='')
            
            self.M_persp = self.field_detector.perspective_transform_matrix
            self.M_field = self.field_detector.field_transform_matrix

            # Store current field data
            with self.result_lock:
                self.field_data = {
                    'calibrated': self.field_detector.calibrated,
                    'field_corners': self.field_detector.field_corners,
                    'goals': self.field_detector.goals,
                    'calibration_mode': self.calibration_mode,
                    'calibration_requested': self.calibration_requested,
                    'perspective_transform_matrix': self.field_detector.perspective_transform_matrix,
                    'field_transform_matrix': self.field_detector.field_transform_matrix
                }


            self.field_width = (self.field_data['field_corners'][1][0] - self.field_data['field_corners'][0][0] 
                                if self.field_data['field_corners'] is not None and len(self.field_data['field_corners']) >= 2 
                                and self.field_data['field_corners'][0] is not None 
                                and self.field_data['field_corners'][1] is not None 
                                else 0)
            self.px_to_cm_ratio = self.field_width / 118 if self.field_width != 0 else 0
            # print(f"Field Width: {self.field_width} px, Pixel to Meter Ratio: {self.px_to_cm_ratio:.4f}")


    def player_tracking_thread(self):
        """Thread for Player Detection"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break
            except Empty:
                continue

            # Get field corners safely, use empty array if field_data is None or not calibrated
            field_corners = np.array([])
            if self.field_data is not None and self.field_data.get('field_corners') is not None:
                field_corners = self.field_data['field_corners']

            boxes_t1, boxes_t2 = self.player_detector.detect_players(frame, field_corners)
            with self.result_lock:
                self.player_result = {
                    'team1': boxes_t1,
                    'team2': boxes_t2
                }


    def draw_player_visualization(self, frame):
        with self.result_lock:
            player_result_copy = self.player_result.copy() if self.player_result else None
        
        if player_result_copy is None or self.M_persp is None:
            return
            
        # Transform team1 bounding boxes
        for box in player_result_copy['team1']:
            x, y, w, h = box
            # Create corner points of the bounding box
            corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            # Transform the corners
            transformed_corners = self._transform_points(corners, self.M_persp)
            # Draw the transformed bounding box as a polygon
            cv2.polylines(frame, [transformed_corners], True, (0, 0, 255), 2)
            
        # Transform team2 bounding boxes  
        for box in player_result_copy['team2']:
            x, y, w, h = box
            # Create corner points of the bounding box
            corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            # Transform the corners
            transformed_corners = self._transform_points(corners, self.M_persp)
            # Draw the transformed bounding box as a polygon
            cv2.polylines(frame, [transformed_corners], True, (255, 0, 0), 2)



    def _transform_points(self, points_array, M):
        if points_array.size == 0:
            return np.array([], dtype=np.int32).reshape(0, 2)
        points_np_float = points_array.astype(np.float32)
        points_reshaped = points_np_float.reshape(-1, 1, 2)
        transformed_np = cv2.perspectiveTransform(points_reshaped, M)
        return transformed_np.reshape(-1, 2).astype(int)

    def draw_ball_visualization(self, frame):
        """Draws ball visualization"""
        with self.result_lock:
            ball_result_copy = self.ball_result.copy() if self.ball_result else None
        
        if ball_result_copy is None:
            return

        detection = ball_result_copy['detection']
        smoothed_pts = ball_result_copy['smoothed_pts']
        transformed_smoothed_pts = []
        if smoothed_pts is not None:
            # Filter out None values before creating numpy array
            valid_pts = [pt for pt in smoothed_pts if pt is not None]
            if valid_pts:
                transformed_smoothed_pts = self._transform_points(np.array(valid_pts), self.M_persp)
        missing_counter = ball_result_copy['missing_counter']
        ball_speed = ball_result_copy.get('ball_speed', 0.0)

        # Draw ball info
        if detection[0] is not None:
            center, radius, confidence, velocity = detection
            transformed_center = self._transform_points(np.array([center]), self.M_persp)[0]
            center_int = (int(transformed_center[0]), int(transformed_center[1]))

            # Color selection based on confidence
            if confidence >= 0.8:
                color = config.COLOR_BALL_HIGH_CONFIDENCE  # Green
            elif confidence >= 0.6:
                color = config.COLOR_BALL_MED_CONFIDENCE   # Yellow
            else:
                color = config.COLOR_BALL_LOW_CONFIDENCE   # Orange

            cv2.circle(frame, center_int, 3, color, -1)
            cv2.circle(frame, center_int, int(radius), color, 2)

            cv2.putText(frame, f"R: {radius:.1f}", (center_int[0] + 15, center_int[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (center_int[0] + 15, center_int[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Show ball speed
            cv2.putText(frame, f"Speed: {self.ball_speed:.2f} m/s", (center_int[0] + 15, center_int[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Show Kalman velocity
            if velocity is not None:
                cv2.arrowedLine(frame, transformed_center,
                            (int(transformed_center[0] + velocity[0]*30), int(transformed_center[1] + velocity[1]*30)),
                            (255, 0, 255), 2)
                
        # Draw smoothed points trail
        for i in range(1, len(transformed_smoothed_pts)):
            thickness = int(np.sqrt(config.BALL_TRAIL_MAX_LENGTH / float(i + 1)) * config.BALL_TRAIL_THICKNESS_FACTOR)
            cv2.line(frame, tuple(transformed_smoothed_pts[i - 1]), tuple(transformed_smoothed_pts[i]), config.COLOR_BALL_TRAIL, thickness)


        # Missing Counter
        cv2.putText(frame, f"Missing: {missing_counter}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_field_visualization(self, frame):
        """Draws field visualization"""
        with self.result_lock:
            field_data_copy = self.field_data.copy() if self.field_data else None
        
        if field_data_copy is None or field_data_copy['field_corners'] is None:
            return
        transformed_corners = self._transform_points(field_data_copy['field_corners'], self.M_persp)

        # Field corners
        for i, corner in enumerate(transformed_corners):
            corner_int = (int(corner[0]), int(corner[1]))
            cv2.circle(frame, corner_int, 2, config.COLOR_FIELD_CORNERS, -1)
            cv2.putText(frame, f"{i+1}", (int(corner[0])+10, int(corner[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_FIELD_CORNERS, 2)

        # Goals
        for i, goal in enumerate(field_data_copy['goals']):
            # Zeichne die ausgerichtete Tor-Kontur wenn vorhanden
            if goal.get('contour') is not None:
                transformed_contour = self._transform_points(goal['contour'], self.M_persp)
                cv2.drawContours(frame, [transformed_contour], -1, config.COLOR_GOALS, 2)
            else:
                # Fallback auf rechteckige Bounds
                x, y, w, h = goal['bounds']
                transformed_bounds = self._transform_points(np.array([[x, y], [x + w, y + h]]), self.M_persp)
                cv2.rectangle(frame, (int(transformed_bounds[0][0]), int(transformed_bounds[0][1])),
                                      (int(transformed_bounds[1][0]), int(transformed_bounds[1][1])),
                                      config.COLOR_GOALS, 2)

            # Zeichne Tor-Center und Label
            center_x, center_y = goal['center']
            transformed_center = self._transform_points(np.array([[center_x, center_y]]), self.M_persp)[0]
            cv2.circle(frame, (int(transformed_center[0]), int(transformed_center[1])), 5, config.COLOR_GOALS, -1)
            cv2.putText(frame, f"Goal {i+1} ({goal['type']})", (int(transformed_center[0])-30, int(transformed_center[1])-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GOALS, 2)

        # Field limits with corners
        field_corners_int = np.array(transformed_corners, dtype=np.int32)
        cv2.drawContours(frame, [field_corners_int], -1, config.COLOR_FIELD_BOUNDS, 1)

    def draw_status_info(self, frame):
        """Draws status information"""
        # Mode display
        mode_text = {
            self.BALL_ONLY: "Ball Tracking",
            self.FIELD_ONLY: "Field Tracking", 
            self.COMBINED: "Combined Tracking"
        }
        
        cv2.putText(frame, f"Mode: {mode_text[self.visualization_mode]}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Camera calibration status - IDS Camera doesn't have built-in calibration
        cv2.putText(frame, "Camera: IDS Live Stream", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Processing mode display
        mode_text = "GPU" if self.use_gpu_processing else "CPU"
        mode_color = (0, 255, 0) if self.use_gpu_processing else (255, 255, 0)
        cv2.putText(frame, f"Preprocessing: {mode_text}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # FPS display
        cv2.putText(frame, f"Processing: {self.processing_fps:.1f} FPS", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show key commands
        cv2.putText(frame, "Keys: 1=Ball, 2=Field, 3=Both, r=Calibration, s=Screenshot, g=Reset Score, r=Reset Field Calibration, x=Reload GPU, c=Toggle CPU/GPU, h=Help", 
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

        if self.visualization_mode in [self.COMBINED]: 
            self.player_thread = Thread(target=self.player_tracking_thread, daemon=True)
            self.player_thread.start()

    def stop_threads(self):
        """Stops the tracking threads"""
        self.running = False
        
        # Send termination signals to worker threads
        try:
            self.frame_queue.put(None)
            self.frame_queue.put(None)
        except:
            pass  # Queue might be full or closed
        
        if self.frame_reader_thread and self.frame_reader_thread.is_alive():
            self.frame_reader_thread.join(timeout=1.0)
        
        if self.ball_thread and self.ball_thread.is_alive():
            self.ball_thread.join(timeout=1.0)
            
        if self.field_thread and self.field_thread.is_alive():
            self.field_thread.join(timeout=1.0)

        if self.player_thread and self.player_thread.is_alive():
            self.player_thread.join(timeout=1.0)
    
    def toggle_processing_mode(self):
        """Toggles between CPU and GPU processing"""
        self.use_gpu_processing = not self.use_gpu_processing
        mode_text = "GPU" if self.use_gpu_processing else "CPU"
        print(f"\nSwitched to {mode_text} processing")
        
        # Reinitialize GPU if switching back to GPU mode
        if self.use_gpu_processing:
            try:
                self.gpu_preprocessor.force_reinitialize()
                print("GPU preprocessor reinitialized")
            except Exception as e:
                print(f"Failed to reinitialize GPU, falling back to CPU: {e}")
                self.use_gpu_processing = False
    

    def _show_help(self):
        """Displays help information"""
        print("\n" + "=" * 60)
        print("KEY COMMANDS:")
        print("  'q' - Quit")
        print("  '1' - Show ball tracking only")
        print("  '2' - Show field tracking only")
        print("  '3' - Show combined view")
        print("  's' - Save screenshot (with ball curve if available)")
        print("  'g' - Reset score to 0-0")
        print("  'r' - Reset field calibration")
        print("  'x' - Force GPU shader reload")
        print("  'c' - Toggle CPU/GPU processing")
        print("  'h' - Show help")
        print("=" * 60)

    def run(self):
        """Main loop for combined tracker"""
        print("=" * 60)

        # Camera status display - IDS Camera doesn't have calibration info like VideoStream
        print("✓ IDS Camera initialized - live capture active")

        self._show_help()
        
        # Start IDS camera acquisition
        self.camera.start()
        self.start_threads()
        
        try:
            # Enhanced preprocessing time measurement
            from collections import deque
            processing_times = deque(maxlen=100)  # Keep last 100 measurements
            count = 0
            last_stats_time = time.time()
            
            while True:
                # Get current Bayer frame and process it on GPU in main thread
                with self.result_lock:
                    if self.current_bayer_frame is None:
                        continue
                    bayer_frame = self.current_bayer_frame.copy()
                
                count += 1
                t_start = time.perf_counter()
                
                # Process frame based on selected mode
                if self.use_gpu_processing:
                    try:
                        frame, _ = self.gpu_preprocessor.process_frame(bayer_frame)
                    except Exception as e:
                        print(f"\nGPU processing failed, falling back to CPU: {e}")
                        self.use_gpu_processing = False
                        frame, _ = self.cpu_preprocessor.process_frame(bayer_frame)
                else:
                    frame, _ = self.cpu_preprocessor.process_frame(bayer_frame)
                
                t_process = time.perf_counter()
                delta_t_sec = t_process - t_start
                
                # Add current measurement to rolling window
                processing_times.append(delta_t_sec * 1000)  # Convert to ms
                
                # Display detailed statistics every second
                current_time = time.time()
                if current_time - last_stats_time >= 1.0:
                    if processing_times:
                        avg_time_ms = sum(processing_times) / len(processing_times)
                        min_time_ms = min(processing_times)
                        max_time_ms = max(processing_times)
                        # Calculate theoretical max FPS based on preprocessing time
                        theoretical_fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
                        #print(f"\rPreprocessing: {avg_time_ms:.2f}ms avg (min: {min_time_ms:.2f}, max: {max_time_ms:.2f}) | Theoretical FPS: {theoretical_fps:.1f} | Samples: {len(processing_times)}", end="")
                    last_stats_time = current_time

                # Store processed frame for display
                with self.result_lock:
                    self.current_frame = frame
                
                # Put processed frame into queue for analysis threads
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                    self.frame_queue.put(frame)
                
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

                    if self.M_persp is not None:
                        display_frame = cv2.warpPerspective(
                            display_frame, 
                            self.M_persp, 
                            (config.DETECTION_WIDTH, config.DETECTION_HEIGHT)
                        )

                    # Visualization based on mode
                    if self.visualization_mode in [self.BALL_ONLY, self.COMBINED]:
                        self.draw_ball_visualization(display_frame)
                        
                    if self.visualization_mode in [self.FIELD_ONLY, self.COMBINED]:
                       self.draw_field_visualization(display_frame)

                    if self.visualization_mode in [self.COMBINED]:
                        self.draw_player_visualization(display_frame)
                    
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
                    self._show_help()

                elif key == ord('g'):
                    # Reset score
                    self.goal_scorer.reset_score()
                    print("Score reset!")
                    
                elif key == ord('r'):
                    # Reset field calibration
                    print("Resetting field calibration...")
                    self.field_detector.reset_calibration()
                    print("Field detector calibration has been reset successfully")

                elif key == ord('x'):
                    print("Forcing GPU preprocessor reinitialization...")
                    self.gpu_preprocessor.force_reinitialize()
                
                elif key == ord('c'):
                    self.toggle_processing_mode()

                #print(f"\r{(time.time() - measure_time) * 1000000}", end="")

        finally:
            # Cleanup
            self.stop_threads()
            self.camera.stop()
            cv2.destroyAllWindows()
            
            print(f"\nCombined Tracker finished.")


# ================== MAIN PROGRAM ==================

if __name__ == "__main__":
    # Create and start combined tracker with IDS camera
    # video_path and quse_webcam parameters are kept for compatibility but not used
    tracker = CombinedTracker()
    tracker.run()