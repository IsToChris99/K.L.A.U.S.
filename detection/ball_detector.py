import cv2
import numpy as np
from collections import deque
from config import (
    BALL_LOWER, BALL_UPPER,
    BALL_SMOOTHER_WINDOW_SIZE, BALL_MAX_MISSING_FRAMES,
    BALL_CONFIDENCE_THRESHOLD, BALL_TRAIL_MAX_LENGTH,
    DISPLAY_FPS, WIDTH_RATIO, AREA_RATIO
)

class BallDetector:
    """Main class for ball tracking"""
    def __init__(self):

        self.lower = BALL_LOWER
        self.upper = BALL_UPPER

        self.display_interval = 1.0 / DISPLAY_FPS

        # Tracking-Variablen
        self.smoother = Smoother(window_size=BALL_SMOOTHER_WINDOW_SIZE)
        self.kalman_tracker = KalmanBallTracker()
        self.smoothed_pts = deque(maxlen=BALL_TRAIL_MAX_LENGTH)
        self.all_ball_positions = []
        self.missing_counter = 0
        self.final_frame = None
        self.max_missing_frames = BALL_MAX_MISSING_FRAMES

        self.last_good_detection = None
        self.confidence_threshold = BALL_CONFIDENCE_THRESHOLD
        self.recent_detections = deque(maxlen=10)
        
        # Flag to track if we just recovered from a long pause
        self.just_recovered = False

        pass

    def detect_ball(self, frame, field_corners=None):
        """Ball detection with multi-criteria evaluation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower, self.upper)
        
        # When field corners are provided, create a mask for the field area
        if field_corners is not None:
            field_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(field_mask, [field_corners.astype(np.int32)], 255)
            mask = cv2.bitwise_and(mask, field_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        if len(cnts) == 0:
            return None, 0, 0.0
            
        # Contour candidates
        candidates = []
        kalman_prediction = None
        
        # Kalman prediction for weighting
        if self.kalman_tracker.initialized:
            kalman_prediction = self.kalman_tracker.get_velocity()
        
        for contour in cnts:
            score, center, radius = self._evaluate_ball_candidate(contour, frame, kalman_prediction)
            if score > 0:
                candidates.append({
                    'score': score,
                    'center': center,
                    'radius': radius,
                    'contour': contour
                })
        
        if not candidates:
            return None, 0, 0.0

        best = max(candidates, key=lambda x: x['score'])
        confidence = min(best['score'] / 100.0, 1.0)

        # Lower confidence threshold after long missing period
        effective_threshold = self.confidence_threshold
        if hasattr(self, 'missing_counter') and self.missing_counter > self.max_missing_frames:
            effective_threshold = max(0.3, self.confidence_threshold - 0.2)

        if confidence >= effective_threshold:
            return best['center'], best['radius'], confidence, kalman_prediction
        else:
            return None, 0, confidence, kalman_prediction

    def _evaluate_ball_candidate(self, contour, frame, kalman_prediction):
        """Evaluates a contour candidate for ball detection"""

        area = cv2.contourArea(contour)
        if area < 10 * AREA_RATIO:
            return 0, None, 0
            
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))

        if not (3 * WIDTH_RATIO < radius < 15 * WIDTH_RATIO):
            return 0, center, radius
            
        score = 0
        
        # 1. Scale
        if 5 * WIDTH_RATIO <= radius <= 9 * WIDTH_RATIO:
            score += 30
        elif 4 * WIDTH_RATIO <= radius <= 11 * WIDTH_RATIO:
            score += 20
        else:
            score += 10
            
        # 2. Circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.7 <= circularity <= 1.3:
                score += 25 * (1.0 - abs(1.0 - circularity))
                
        # 3. Fill ratio
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0
        if 0.6 <= fill_ratio <= 1.0:
            score += 20 * fill_ratio
            
        # 4. Movement consistency with last detections
        if len(self.recent_detections) >= 2:
            avg_pos = np.mean(self.recent_detections, axis=0)
            distance_to_trend = np.sqrt((center[0] - avg_pos[0])**2 + (center[1] - avg_pos[1])**2)
            if distance_to_trend < 30:
                score += 15 * (1.0 - min(distance_to_trend / 30.0, 1.0))
        
        # If ball was missing for long time, be more lenient with movement consistency
        elif hasattr(self, 'missing_counter') and self.missing_counter > self.max_missing_frames:
            # Give bonus for any reasonable detection after long absence
            if 3 * WIDTH_RATIO <= radius <= 12 * WIDTH_RATIO:
                score += 10
                
        # 5. Kalman prediction consistency
        if kalman_prediction and self.last_good_detection:
            predicted_pos = (
                self.last_good_detection[0] + kalman_prediction[0],
                self.last_good_detection[1] + kalman_prediction[1]
            )
            kalman_distance = np.sqrt((center[0] - predicted_pos[0])**2 + (center[1] - predicted_pos[1])**2)
            if kalman_distance < 25 * WIDTH_RATIO:
                score += 10 * (1.0 - min(kalman_distance / (25 * WIDTH_RATIO), 1.0))

        # 6. Color intensity in the region
        roi = frame[max(0, int(y-radius)):min(frame.shape[0], int(y+radius)),
                   max(0, int(x-radius)):min(frame.shape[1], int(x+radius))]
        if roi.size > 0:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            color_pixels = cv2.inRange(hsv_roi, self.lower, self.upper)
            color_ratio = np.sum(color_pixels > 0) / color_pixels.size
            score += 10 * color_ratio

        return score, center, radius

    def _constrain_to_field(self, position, field_corners):
        """Constrains the position to the field boundaries defined by corners"""
        if position is None or field_corners is None:
            return position
            
        x, y = position
        
        # Check if point is inside the field polygon
        point_in_field = cv2.pointPolygonTest(field_corners.astype(np.float32), (x, y), False)
        
        # If point is inside the field, return original position
        if point_in_field >= 0:
            return position
        
        # If outside field, simply return original position (let Kalman handle it)
        # This is much simpler but less precise
        return position
    
    def _adjust_kalman_velocity(self, position, field_corners):
        """Reduces Kalman velocity near field boundaries"""
        if position is None or field_corners is None or not self.kalman_tracker.initialized:
            return
            
        x, y = position
        
        # Calculate distance to the field boundary
        distance_to_boundary = cv2.pointPolygonTest(field_corners.astype(np.float32), (x, y), True)
        
        # If close to the boundary (within 20 pixels), reduce velocity
        if distance_to_boundary < 20 and distance_to_boundary >= 0:
            velocity_factor = max(0.1, distance_to_boundary / 20.0)

            current_state = self.kalman_tracker.kalman.statePost.copy()
            
            current_state[2] *= velocity_factor
            current_state[3] *= velocity_factor
            
            self.kalman_tracker.kalman.statePost = current_state

    def update_tracking(self, detection_result, field_corners=None):
        """Extended tracking logic with Kalman filter and field boundaries"""
        center, radius, confidence, velocity = detection_result if detection_result[0] is not None else (None, 0, 0, None)

        kalman_pos = self.kalman_tracker.update(center)

        # Kalman prediction constrained to field
        if kalman_pos is not None and field_corners is not None:
            kalman_pos = self._constrain_to_field(kalman_pos, field_corners)
            self._adjust_kalman_velocity(kalman_pos, field_corners)
        
        if center is not None:
            # Good detection - use real measurement
            
            # Check if we're recovering from a long pause
            was_long_missing = self.missing_counter > self.max_missing_frames
            
            # If ball was missing for a long time, use direct position without heavy smoothing
            if was_long_missing:
                # Reset smoother to respond faster
                self.smoother = Smoother(window_size=max(1, BALL_SMOOTHER_WINDOW_SIZE // 2))
                self.just_recovered = True
                
                # For first detection after long pause, use direct measurement
                if self.missing_counter > self.max_missing_frames * 1.5:
                    smoothed_center = center  # Use direct measurement
                    self.kalman_tracker.soft_reset(center)
                else:
                    smoothed_center = self.smoother._adaptive_ema_update(center)
            else:
                # Normal tracking - gradually restore normal smoothing after recovery
                if self.just_recovered and len(self.recent_detections) >= 3:
                    self.smoother = Smoother(window_size=BALL_SMOOTHER_WINDOW_SIZE)
                    self.just_recovered = False
                
                smoothed_center = self.smoother._adaptive_ema_update(center)
            
            self.missing_counter = 0
            self.last_good_detection = center
            self.recent_detections.append(center)
            
            self.smoothed_pts.appendleft(smoothed_center)
            
            if smoothed_center is not None:
                self.all_ball_positions.append(smoothed_center)
                
        else:
            # Else, use Kalman prediction
            self.missing_counter += 1
            
            if self.missing_counter <= self.max_missing_frames:
                if kalman_pos is not None:
                    smoothed_center = self.smoother._adaptive_ema_update(kalman_pos)
                    self.smoothed_pts.appendleft(smoothed_center)
                    
                    if smoothed_center is not None:
                        self.all_ball_positions.append(smoothed_center)
                else:
                    self.smoothed_pts.appendleft(None)
            else:
                self.smoothed_pts.appendleft(None)
                if self.missing_counter > self.max_missing_frames:
                    self.all_ball_positions.append(None)
                    
        # Tracking reset if ball is missing for too long
        if self.missing_counter > self.max_missing_frames * 2:
            # Don't completely reset - try soft reset first if we have recent detections
            if len(self.recent_detections) > 0 and self.kalman_tracker.initialized:
                # Store the current velocity before any reset
                current_velocity = self.kalman_tracker.get_velocity()
                self.kalman_tracker.last_velocity = current_velocity
                
            self.smoother = Smoother(window_size=max(1, BALL_SMOOTHER_WINDOW_SIZE))
            # Only hard reset Kalman if we have no recent information
            if len(self.recent_detections) == 0:
                self.kalman_tracker = KalmanBallTracker()
            else:
                # Keep the tracker but mark as uninitialized for soft reset
                self.kalman_tracker.initialized = False
            self.recent_detections.clear()

    def create_screenshot(self):
        """Creates a screenshot with the complete ball trajectory and interruptions"""
        if self.final_frame is not None and len(self.all_ball_positions) > 1:
            screenshot = self.final_frame.copy()
            
            for i in range(1, len(self.all_ball_positions)):
                prev_pos = self.all_ball_positions[i-1]
                curr_pos = self.all_ball_positions[i]
                
                if prev_pos is not None and curr_pos is not None:
                    # Additional check for realistic distance
                    distance = np.sqrt((prev_pos[0] - curr_pos[0])**2 + (prev_pos[1] - curr_pos[1])**2)
                    if distance < 150:
                        cv2.line(screenshot, prev_pos, curr_pos, (0, 0, 255), 2)
            
            segments = self._get_curve_segments()
            colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            
            for idx, segment in enumerate(segments):
                if len(segment) > 0:
                    color = colors[idx % len(colors)]
                    # Startpoint
                    cv2.circle(screenshot, segment[0], 8, color, -1)
                    cv2.putText(screenshot, f"S{idx+1}", 
                               (segment[0][0] + 10, segment[0][1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    # Endpoint
                    if len(segment) > 1:
                        cv2.circle(screenshot, segment[-1], 8, color, -1)
                        cv2.putText(screenshot, f"E{idx+1}", 
                                   (segment[-1][0] + 10, segment[-1][1] + 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show screenshot
            cv2.imshow("Ballcurve", screenshot)
            print("Press any key to continue...")
            cv2.waitKey(0)
        else:
            print("No sufficient ballpositions found.")
            print(f"Found Positions: {len(self.all_ball_positions)}")
    
    def _get_curve_segments(self):
        """Splits the ball positions into separate curve segments"""
        segments = []
        current_segment = []
        
        for pos in self.all_ball_positions:
            if pos is None:
                # Interruption found
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(pos)
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    

class Smoother:
    """Smoothing for ball positions"""
    def __init__(self, window_size=20):
        self.window_size = window_size
        
        # EMA variables (Exponential Moving Average)
        self.ema_alpha = min(0.8, 2.0 / (window_size + 1))  # Cap alpha at 0.8 for responsiveness
        self.ema_x = None
        self.ema_y = None
        
        # Adaptive EMA variables
        self.prev_point = None
    
    def _adaptive_ema_update(self, center):
        """Adaptive EMA that adjusts alpha based on movement speed"""
        x, y = center
        if self.ema_x is None:
            self.ema_x = float(x)
            self.ema_y = float(y)
            self.prev_point = (x, y)
            return (int(x), int(y))
        
        # Calculate movement speed
        if self.prev_point:
            speed = np.sqrt((x - self.prev_point[0])**2 + (y - self.prev_point[1])**2)
            # Adaptive alpha: higher speed = higher alpha (less smoothing)
            max_speed = 30.0  # Reduced from 50 for more responsiveness
            speed_factor = min(speed / max_speed, 1.0)
            # More aggressive adaptation: start higher, reach higher maximum
            adaptive_alpha = max(0.3, self.ema_alpha + speed_factor * (0.9 - self.ema_alpha))
        else:
            adaptive_alpha = max(0.3, self.ema_alpha)  # Minimum alpha of 0.3
        
        self.ema_x = adaptive_alpha * x + (1 - adaptive_alpha) * self.ema_x
        self.ema_y = adaptive_alpha * y + (1 - adaptive_alpha) * self.ema_y
        
        self.prev_point = (x, y)
        return (int(self.ema_x), int(self.ema_y))


class KalmanBallTracker:
    """Kalman Filter for precise ball tracking"""
    def __init__(self):
        # Kalman Filter for 2D position with velocity
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 5 * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        
        # Store last known velocity for re-initialization
        self.last_velocity = (0, 0)
        self.frames_since_last_update = 0
        
    def update(self, measurement):
        """Updates Kalman Filter with new measurement"""
        if not self.initialized and measurement is not None:
            # Initialization with first measurement
            # Use stored velocity if available (from previous tracking session)
            self.kalman.statePre = np.array([measurement[0], measurement[1], 
                                           self.last_velocity[0], self.last_velocity[1]], np.float32)
            self.kalman.statePost = np.array([measurement[0], measurement[1], 
                                            self.last_velocity[0], self.last_velocity[1]], np.float32)
            self.initialized = True
            self.frames_since_last_update = 0
            return measurement
            
        if not self.initialized:
            return None
            
        prediction = self.kalman.predict()
        predicted_pos = (int(prediction[0]), int(prediction[1]))
        
        if measurement is not None:
            # Correction with measurement
            self.kalman.correct(np.array([measurement[0], measurement[1]], np.float32))
            # Store current velocity for potential re-initialization
            self.last_velocity = self.get_velocity()
            self.frames_since_last_update = 0
            
            # Reset process noise after adaptation period (10 frames)
            if hasattr(self, '_adaptation_frames'):
                self._adaptation_frames += 1
                if self._adaptation_frames > 10:
                    self.reset_adaptation_noise()
                    delattr(self, '_adaptation_frames')
            
            return measurement
        else:
            # Use prediction only
            self.frames_since_last_update += 1
            
            # Gradually reduce velocity for long missing periods
            if self.frames_since_last_update > 30:
                velocity_decay = max(0.5, 1.0 - (self.frames_since_last_update - 30) * 0.01)
                current_state = self.kalman.statePost.copy()
                current_state[2] *= velocity_decay
                current_state[3] *= velocity_decay
                self.kalman.statePost = current_state
                
            return predicted_pos
    
    def soft_reset(self, new_position):
        """Soft reset that preserves some velocity information"""
        if new_position is not None:
            # Keep some of the last known velocity, but reduce it
            reduced_velocity = (self.last_velocity[0] * 0.3, self.last_velocity[1] * 0.3)
            
            self.kalman.statePre = np.array([new_position[0], new_position[1], 
                                           reduced_velocity[0], reduced_velocity[1]], np.float32)
            self.kalman.statePost = np.array([new_position[0], new_position[1], 
                                            reduced_velocity[0], reduced_velocity[1]], np.float32)
            
            # Temporarily increase process noise for faster adaptation
            self.kalman.processNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
            self._adaptation_frames = 0  # Track adaptation period
            
            self.frames_since_last_update = 0
            self.initialized = True
    
    def reset_adaptation_noise(self):
        """Reset process noise to normal after adaptation period"""
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
            
    def get_velocity(self):
        """Returns current velocity"""
        if self.initialized:
            state = self.kalman.statePost
            return (state[2], state[3])
        return (0, 0)