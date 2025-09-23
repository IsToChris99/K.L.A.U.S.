import cv2
import numpy as np
import time
from collections import deque
from config import (
    DEBUG_VERBOSE_OUTPUT,
    COLOR_GOALS, COLOR_SCORE_TEXT, COLOR_GOAL_ALERT, COLOR_BALL_IN_GOAL,
    GOAL_DISAPPEAR_FRAMES, GOAL_REVERSAL_TIME_WINDOW,
    GOAL_DIRECTION_THRESHOLD_DISTANCE
)

class GoalScorer:
    """Class for scoring system"""
    
    def __init__(self):
        self.player1_goals = 0
        self.player2_goals = 0
        self.max_goals = 10
        
        self.player1_goal_types = ['left']
        self.player2_goal_types = ['right']
        
        # Ball status tracking
        self.ball_in_goal = False
        self.ball_in_goal_type = None
        self.ball_in_goal_start_time = None
        self.ball_missing_counter = 0
        self.last_ball_position = None
        self.last_ball_velocity = None
        self.seen_before = False
        
        # Configuration for direction-based goal detection
        self.goal_direction_threshold_distance = GOAL_DIRECTION_THRESHOLD_DISTANCE
        
        self.goal_disappear_frames = GOAL_DISAPPEAR_FRAMES
        self.goal_reversal_time_window = GOAL_REVERSAL_TIME_WINDOW

        self.goal_scored_recently = False
        self.goal_scored_time = None
        self.goal_scored_type = None
        
        self.debug_verbose = DEBUG_VERBOSE_OUTPUT
    
    def is_ball_in_goal(self, ball_position, goals):
        """Checks if the ball is in one of the goals"""
        if ball_position is None or not goals:
            return False, None
        
        ball_x, ball_y = ball_position
        
        for goal in goals:
            goal_x, goal_y, goal_w, goal_h = goal['bounds']
            
            padding = 10  # Padding for better detection
            if (goal_x - padding <= ball_x <= goal_x + goal_w + padding and
                goal_y - padding <= ball_y <= goal_y + goal_h + padding):
                return True, goal['type']
        
        return False, None
    
    def is_ball_in_field(self, ball_position, field_corners):
        """Checks if the ball is in the field using corner points"""
        if ball_position is None or field_corners is None:
            return False
        
        ball_x, ball_y = ball_position
        
        # Use cv2.pointPolygonTest to check if ball is inside the field polygon
        result = cv2.pointPolygonTest(field_corners.astype(np.float32), (ball_x, ball_y), False)
        return result >= 0  # >= 0 means inside or on the boundary
    
    def is_ball_heading_towards_goal(self, ball_position, ball_velocity, goals):
        """Checks if the ball is heading towards any goal based on velocity from Kalman tracker"""
        if ball_position is None or ball_velocity is None or not goals:
            return False, None
        
        ball_position = np.array(ball_position)
        ball_velocity = np.array(ball_velocity)
        
        # Calculate speed from velocity components
        speed = np.linalg.norm(ball_velocity)

        # Only consider significant movement (minimum speed threshold)
        if speed < 3.0:  # Minimum speed threshold
            return False, None
        
        # Normalize velocity to get direction
        dir = ball_velocity / speed

        for goal in goals:
            goal_x, goal_y, goal_w, goal_h = goal['bounds']
            goal_pos = np.array([goal_x, goal_y])
            goal_dim = np.array([goal_w, goal_h])

            goal_center = goal_pos + goal_dim / 2
            
            # Vector from ball to goal center
            to_goal = goal_center - ball_position
            goal_distance = np.linalg.norm(to_goal)

            # # Only consider goals within threshold distance
            # if goal_distance > self.goal_direction_threshold_distance:
            #     continue
            
            # Normalize vector to goal
            normalized_to_goal = to_goal / np.linalg.norm(to_goal) if np.linalg.norm(to_goal) > 0 else to_goal

            # Calculate dot product (cosine of angle between direction and goal vector)
            dot_product = np.dot(dir, normalized_to_goal)
            
            # If dot product > 0.5, ball is heading roughly towards goal (angle < 60°)
            if dot_product > 0.5:
                if self.debug_verbose:
                    print(f"Ball heading towards {goal['type']} goal (dot product: {dot_product:.2f}, distance: {goal_distance:.1f}, speed: {speed:.1f})")
                return True, goal['type']
        
        return False, None
    
    def update_ball_tracking(self, ball_position, goals, field_corners, ball_missing_counter, ball_velocity=None):
        """Main function for updating ball tracking and goal detection
        
        Args:
            ball_position: Current ball position (x, y) or None if not detected
            goals: List of goal dictionaries with 'bounds' and 'type'
            field_corners: Field corner points for boundary detection
            ball_missing_counter: Number of consecutive frames without ball detection
            ball_velocity: Ball velocity (vx, vy) from Kalman tracker, or None
        """
        self.ball_missing_counter = ball_missing_counter
        current_time = time.time()
         
        
        if ball_position is not None:
            self.last_ball_position = ball_position
            self.last_ball_velocity = ball_velocity
            self.seen_before = True

            in_goal, goal_type = self.is_ball_in_goal(ball_position, goals)
            
            if in_goal and not self.ball_in_goal:
                # Ball entered goal
                self.ball_in_goal = True
                self.ball_in_goal_type = goal_type
                self.ball_in_goal_start_time = current_time
                
                if self.debug_verbose:
                    print(f"Ball entered goal: {goal_type}")
            
            elif in_goal and self.ball_in_goal and self.ball_in_goal_type == goal_type:
                # Ball is still in same goal
                pass
            
            elif not in_goal and self.ball_in_goal:
                # Ball left the goal
                if self.is_ball_in_field(ball_position, field_corners):
                    # Ball returned to field - no goal!
                    if self.debug_verbose:
                        print(f"Ball returned to field from {self.ball_in_goal_type} goal - NO GOAL")
                    self._reset_goal_tracking()
                else:
                    # Ball is out of the goal, but not in the field - possibly disappeared
                    pass
        
        else:
            # Ball not detected
            if self.ball_in_goal and self.ball_missing_counter >= self.goal_disappear_frames:
                # Ball was in goal and is now missing - GOAL!
                self._score_goal(self.ball_in_goal_type)
            elif not self.ball_in_goal and not self.goal_scored_recently and self.ball_missing_counter >= self.goal_disappear_frames:

                if self.seen_before and self.last_ball_velocity is not None and self.last_ball_position is not None:
                    heading_to_goal, target_goal_type = self.is_ball_heading_towards_goal(
                        self.last_ball_position, self.last_ball_velocity, goals
                    )

                    if heading_to_goal:
                        # Ball was heading towards goal and disappeared - GOAL!
                        if self.debug_verbose:
                            print(f"Ball was heading towards {target_goal_type} goal and disappeared - GOAL!")
                        self._score_goal(target_goal_type)
                    else:
                        # Reset tracking data if ball wasn't heading to goal and disappeared
                        self.seen_before = False

        if self.goal_scored_recently:
            self._check_for_goal_return(ball_position, field_corners, current_time)
    
    def _score_goal(self, goal_type):
        """Counts a goal for the corresponding player"""
        if goal_type in self.player1_goal_types:
            if self.player1_goals < self.max_goals:
                self.player1_goals += 1
                scoring_player = "Player 1"
                goal_counted = True
            else:
                # print(f"Player 1 has already reached maximum goals ({self.max_goals}). Goal not counted.")
                return
        elif goal_type in self.player2_goal_types:
            if self.player2_goals < self.max_goals:
                self.player2_goals += 1
                scoring_player = "Player 2"
                goal_counted = True
            else:
                # print(f"Player 2 has already reached maximum goals ({self.max_goals}). Goal not counted.")
                return
        else:
            scoring_player = "Unknown"
            goal_counted = False
        
        if goal_counted:
            print(f"GOAL! {scoring_player} scored in {goal_type} goal!")
            print(f"Score: Player 1| {self.player1_goals} - {self.player2_goals} |Player 2")
            
            # Check if game is won
            if self.player1_goals == self.max_goals:
                print(f"GAME WON! Player 1 reached {self.max_goals} goals!")
            elif self.player2_goals == self.max_goals:
                print(f"GAME WON! Player 2 reached {self.max_goals} goals!")

            self.goal_scored_recently = True
            self.goal_scored_time = time.time()
            self.goal_scored_type = goal_type
            
            # Reset all ball tracking data to prevent duplicate goals
            self._reset_goal_tracking()
            self._reset_ball_tracking()
    
    def _reset_goal_tracking(self):
        """Resets goal tracking"""
        self.ball_in_goal = False
        self.ball_in_goal_type = None
        self.ball_in_goal_start_time = None
    
    def _reset_ball_tracking(self):
        """Resets ball tracking data to prevent duplicate goal detection"""
        self.last_ball_position = None
        self.last_ball_velocity = None
        self.seen_before = False
        self.ball_missing_counter = 0
    
    def _check_for_goal_return(self, ball_position, field_corners, current_time):
        """Checks if the ball returns to the playing field after a goal"""
        if current_time - self.goal_scored_time > self.goal_reversal_time_window:
            self.goal_scored_recently = False
            return
        
        if ball_position and self.is_ball_in_field(ball_position, field_corners):
            # Ball is back in the field - reverse goal
            if self.goal_scored_type in self.player1_goal_types:
                self.player1_goals = max(0, self.player1_goals - 1)
                reversed_player = "Player 1"
            elif self.goal_scored_type in self.player2_goal_types:
                self.player2_goals = max(0, self.player2_goals - 1)
                reversed_player = "Player 2"
            else:
                reversed_player = "Unknown"
            
            print(f"GOAL REVERSED! Ball returned to field. {reversed_player} goal cancelled.")
            print(f"New Score: Player 1: {self.player1_goals} - Player 2: {self.player2_goals}")
            
            self.goal_scored_recently = False
    
    def get_score(self):
        """Returns the current score"""
        return {
            'player1': self.player1_goals,
            'player2': self.player2_goals,
            'total_goals': self.player1_goals + self.player2_goals,
            'max_goals': self.max_goals,
            'winner': 1 if self.player1_goals >= self.max_goals else (2 if self.player2_goals >= self.max_goals else None)
        }
    
    def reset_score(self):
        """Resets the score"""
        self.player1_goals = 0
        self.player2_goals = 0
        self._reset_goal_tracking()
        self._reset_ball_tracking()
        self.goal_scored_recently = False
        print("Score reset to 0-0")
    
    def draw_score_info(self, frame):
        """Draws score information on the frame"""
        # Score
        score_text = f"Score - P1: {self.player1_goals}  P2: {self.player2_goals}"
        # cv2.putText(frame, score_text, (10, frame.shape[0] - 80),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SCORE_TEXT, 2)
        
        # Check for game won
        if self.player1_goals >= self.max_goals or self.player2_goals >= self.max_goals:
            winner = "Player 1" if self.player1_goals >= self.max_goals else "Player 2"
            win_text = f"{winner} WINS!"
            # Center the text
            text_size = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] // 2
            # cv2.putText(frame, win_text, (text_x, text_y),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_GOAL_ALERT, 3)
        
        # Ball status
        if self.ball_in_goal:
            status_text = f"Ball in {self.ball_in_goal_type} goal"
            color = COLOR_BALL_IN_GOAL
        elif self.goal_scored_recently:
            time_since_goal = time.time() - self.goal_scored_time
            status_text = f"Goal scored! ({time_since_goal:.1f}s ago)"
            color = COLOR_GOAL_ALERT
        else:
            status_text = "Tracking..."
            color = COLOR_SCORE_TEXT
        
        cv2.putText(frame, status_text, (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def update_score(self, idx=0, amount=1):
        """Updates the score for a player"""
        if idx == 0:
            if self.max_goals > self.player1_goals > 0:
                self.player1_goals = min(self.player1_goals + amount, self.max_goals)
            elif self.player1_goals > self.max_goals:
                self.player1_goals = self.max_goals
        elif idx == 1:
            if self.max_goals > self.player2_goals > 0:
                self.player2_goals = min(self.player2_goals + amount, self.max_goals)
                self.player2_goals = min(self.player2_goals + amount, self.max_goals)
            elif self.player2_goals > self.max_goals:
                self.player2_goals = self.max_goals
        else:
            print("Invalid player index")

    def set_max_goals(self, max_goals=10, is_infinity=False):
        """Sets the maximum goals for both players"""
        if is_infinity:
            max_goals = 9999
        self.max_goals = max_goals
        print(f"Maximum goals set to: {self.max_goals}")
