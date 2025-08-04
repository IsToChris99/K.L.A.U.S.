import time
import json

class MatchModes():    
    """
    A class to handle match modes in the Kicker Klaus application.
    """
    def __init__(self, mode="normal"):
        self.modes = {
            "normal": "Normal Mode",
            "practice": "Practice Mode",
            "tournament": "Tournament Mode"
        }
        self.mode_descriptions = {
            "normal": "Standard mode for regular matches.",
            "practice": "Mode for practicing skills without scoring.",
            "tournament": "Mode optimized for tournament play with specific rules."
        }
        if mode not in self.modes:
            print(f"Unknown mode: {mode} Set to 'normal' by default.")
            self.current_mode = "normal"
        else:
            self.current_mode = mode
            print(f"Match mode set to: {self.current_mode}")
        
        print(f"Match mode description: {self.get_mode_description()}")

        self.score_team1 = 0
        self.score_team2 = 0
        self.match_time = 0  # in seconds
        self.match_started = False
        self.match_ended = False
        self.match_won = False
        
        
    def set_mode(self, mode):
        """
        Set the current match mode.
        """
        if mode in self.modes:
            self.current_mode = mode
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_mode(self):
        """
        Get the current match mode.
        """
        return self.current_mode
    
    def get_mode_description(self):
        """
        Get the description of the current match mode.
        """
        return self.mode_descriptions.get(self.current_mode, "No description available.")
    
    def reset_scores(self):
        """
        Resets the scores for both teams.
        """
        self.score_team1 = 0
        self.score_team2 = 0
        
    def update_score(self, team, points):
        """
        Update the score for a specific team.
        """
        if team == 1:
            self.score_team1 += points
        elif team == 2:
            self.score_team2 += points
        else:
            raise ValueError(f"Unknown team: {team}")
        
        if self.modes[self.current_mode] == "Tournament Mode":
            if self.score_team1 >= 10 or self.score_team2 >= 10:
                self.match_won = True
                self.match_ended = True
        
        return self.match_won
        
    def get_scores(self):
        """
        Get the current scores for both teams.
        """
        return {
            "team1": self.score_team1,
            "team2": self.score_team2
        }

    def start_match(self):
        """
        Start the match timer.
        """
        self.match_started = True
        self.match_ended = False
        self.match_time = time.time()  # Reset match time at start
        
    def end_match(self):
        """
        End the match and calculate total match time.
        """
        self.match_ended = True
        self.match_duration = time.time() - self.match_time
        self.match_time = 0  # Reset match time after ending
        return self.match_duration, self.get_scores()

    def is_match_ongoing(self):
        """
        Check if the match is currently ongoing.
        """
        return self.match_started and not self.match_ended
    
    def save_match_results(self, file_path):
        """
        Save the match results to a file.
        """
        results = {
            "mode": self.current_mode,
            "scores": self.get_scores(),
        }
        with open(file_path, 'w') as f:
            json.dump(results, f)
        return True
    
    