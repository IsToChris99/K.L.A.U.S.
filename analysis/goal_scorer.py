import cv2
import numpy as np
import time
from collections import deque
from config import (
    DEBUG_VERBOSE_OUTPUT,
    COLOR_GOALS, COLOR_SCORE_TEXT, COLOR_GOAL_ALERT, COLOR_BALL_IN_GOAL,
    GOAL_DISAPPEAR_FRAMES, GOAL_REVERSAL_TIME_WINDOW
)

class GoalScorer:
    """Class for scoring system"""
    
    def __init__(self):
        self.player1_goals = 0
        self.player2_goals = 0
        
        self.player1_goal_types = ['left']
        self.player2_goal_types = ['right']
        
        # Ball status tracking
        self.ball_in_goal = False
        self.ball_in_goal_type = None
        self.ball_in_goal_start_time = None
        self.ball_missing_counter = 0
        self.last_ball_position = None
        
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
    
    def is_ball_in_field(self, ball_position, field_bounds):
        """Checks if the ball is in the field"""
        if ball_position is None or field_bounds is None:
            return False
        
        ball_x, ball_y = ball_position
        field_x, field_y, field_w, field_h = field_bounds
        
        return (field_x <= ball_x <= field_x + field_w and
                field_y <= ball_y <= field_y + field_h)
    
    def update_ball_tracking(self, ball_position, goals, field_bounds, ball_missing_counter):
        """Main function for updating ball tracking and goal detection"""
        self.ball_missing_counter = ball_missing_counter
        current_time = time.time()
        
        ball_detected = ball_position is not None
        
        if ball_detected:
            self.last_ball_position = ball_position

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
                if self.is_ball_in_field(ball_position, field_bounds):
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
                
        if self.goal_scored_recently:
            self._check_for_goal_return(ball_position, field_bounds, current_time)
    
    def _score_goal(self, goal_type):
        """Counts a goal for the corresponding player"""
        if goal_type in self.player1_goal_types:
            self.player1_goals += 1
            scoring_player = "Player 1"
        elif goal_type in self.player2_goal_types:
            self.player2_goals += 1
            scoring_player = "Player 2"
        else:
            scoring_player = "Unknown"
        
        print(f"GOAL! {scoring_player} scored in {goal_type} goal!")
        print(f"Score: Player 1| {self.player1_goals} - {self.player2_goals} |Player 2")

        self.goal_scored_recently = True
        self.goal_scored_time = time.time()
        self.goal_scored_type = goal_type
        
        self._reset_goal_tracking()
    
    def _reset_goal_tracking(self):
        """Resets goal tracking"""
        self.ball_in_goal = False
        self.ball_in_goal_type = None
        self.ball_in_goal_start_time = None
    
    def _check_for_goal_return(self, ball_position, field_bounds, current_time):
        """Checks if the ball returns to the playing field after a goal"""
        if current_time - self.goal_scored_time > self.goal_reversal_time_window:
            self.goal_scored_recently = False
            return
        
        if ball_position and self.is_ball_in_field(ball_position, field_bounds):
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
            'total_goals': self.player1_goals + self.player2_goals
        }
    
    def reset_score(self):
        """Resets the score"""
        self.player1_goals = 0
        self.player2_goals = 0
        self._reset_goal_tracking()
        self.goal_scored_recently = False
        print("Score reset to 0-0")
    
    def draw_score_info(self, frame):
        """Draws score information on the frame"""
        # Score
        score_text = f"Score - P1: {self.player1_goals}  P2: {self.player2_goals}"
        cv2.putText(frame, score_text, (10, frame.shape[0] - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SCORE_TEXT, 2)
        
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
