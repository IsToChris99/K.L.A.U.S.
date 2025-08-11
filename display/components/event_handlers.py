"""
Event-Handler für die GUI
"""

import time
from PySide6.QtCore import Slot
from PySide6.QtGui import QColor


class EventHandlers:
    """Zentrale Event-Handler für die GUI"""
    
    def __init__(self, main_window):
        self.main_window = main_window
    
    @Slot()
    def set_visualization_mode(self, mode):
        """Setzt den Visualisierungsmodus"""
        self.main_window.visualization_mode = mode
        mode_names = {1: "Ball", 2: "Field", 3: "Combined"}
        
        # Reset button highlighting
        for btn in [self.main_window.ball_only_btn, self.main_window.field_only_btn, self.main_window.combined_btn]:
            btn.setStyleSheet("")
        
        # Highlight active button
        if mode == 1:
            self.main_window.ball_only_btn.setStyleSheet("background-color: lightgreen;")
        elif mode == 2:
            self.main_window.field_only_btn.setStyleSheet("background-color: lightgreen;")
        elif mode == 3:
            self.main_window.combined_btn.setStyleSheet("background-color: lightgreen;")

        self.main_window.add_log_message(f"Visualization mode set to: {mode_names.get(mode, 'Unknown')}")
    
    @Slot()
    def reset_score_placeholder(self):
        """Placeholder für das Zurücksetzen des Punktestands"""
        # Reset local score
        self.main_window.player1_goals = 0
        self.main_window.player2_goals = 0
        self.main_window.update_score("0:0")
        
        # Send command to processing process to reset its score too
        try:
            self.main_window.command_queue.put({'type': 'reset_score'})
            self.main_window.add_log_message("Score reset (local and processing)")
        except:
            self.main_window.add_log_message("Score reset (local only - could not communicate with processing)")
    
    @Slot()
    def toggle_processing_mode(self, checked):
        """Umschalten zwischen CPU und GPU-Vorverarbeitung"""
        try:
            self.main_window.command_queue.put({'type': 'toggle_processing_mode'})
            mode = "GPU" if checked else "CPU"
            self.main_window.add_log_message(f"Switched to {mode} processing")
            
            # Update status display
            self.main_window.processing_status_label.setText(mode)
            color = "green" if mode == "GPU" else "orange"
            self.main_window.processing_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        except Exception as e:
            self.main_window.add_log_message(f"Failed to toggle processing mode: {e}")
            # Revert checkbox state on failure
            self.main_window.processing_mode_checkbox.setChecked(not checked)
    
    @Slot()
    def team1_score_plus_placeholder(self):
        """Placeholder für Erhöhung des Punktestands von Spieler 1"""
        self.main_window.player1_goals += 1
        self.main_window.update_score(f"{self.main_window.player1_goals}:{self.main_window.player2_goals}")
        self.main_window.add_log_message("Team 1 score +1 (manual)")

    @Slot()
    def team1_score_minus_placeholder(self):
        """Placeholder für Verringerung des Punktestands von Spieler 1"""
        if self.main_window.player1_goals > 0:
            self.main_window.player1_goals -= 1
            self.main_window.update_score(f"{self.main_window.player1_goals}:{self.main_window.player2_goals}")
            self.main_window.add_log_message("Team 1 score -1 (manual)")
        else:
            self.main_window.add_log_message("Team 1 score cannot go below 0")

    @Slot()
    def team2_score_plus_placeholder(self):
        """Placeholder für Erhöhung des Punktestands von Spieler 2"""
        self.main_window.player2_goals += 1
        self.main_window.update_score(f"{self.main_window.player1_goals}:{self.main_window.player2_goals}")
        self.main_window.add_log_message("Team 2 score +1 (manual)")

    @Slot()
    def team2_score_minus_placeholder(self):
        """Placeholder für Verringerung des Punktestands von Spieler 2"""
        if self.main_window.player2_goals > 0:
            self.main_window.player2_goals -= 1
            self.main_window.update_score(f"{self.main_window.player1_goals}:{self.main_window.player2_goals}")
            self.main_window.add_log_message("Team 2 score -1 (manual)")
        else:
            self.main_window.add_log_message("Team 2 score cannot go below 0")

    @Slot()
    def start_match_placeholder(self):
        """Placeholder für das Starten eines Spiels"""
        self.main_window.start_match_btn.hide()
        
        self.main_window.reset_score_btn.show()
        self.main_window.cancel_match_btn.show()

        self.main_window.add_log_message("Match started (placeholder)")

    @Slot()
    def cancel_match_placeholder(self):
        """Placeholder für das Abbrechen eines Spiels"""
        self.main_window.reset_score_btn.hide()
        self.main_window.cancel_match_btn.hide()
        
        self.main_window.player1_goals = 0
        self.main_window.player2_goals = 0
        
        self.main_window.update_score("0:0")
        
        self.main_window.start_match_btn.show()
        
        self.main_window.add_log_message("Match canceled (placeholder)")
        
    @Slot()
    def reset_goal_limit_placeholder(self):
        """Placeholder für das Zurücksetzen des Tor-Limits auf den Standardwert"""
        self.main_window.goal_limit_input.setValue(9)
        self.main_window.add_log_message("Goal limit reset to 9")
    
    @Slot(int, int, int)
    def on_calibration_color_picked(self, r, g, b):
        """Behandelt die Farbauswahl aus dem Kalibrierungs-Tab-Video"""       
        # Calculate and display HSV values
        qcolor = QColor(r, g, b)
        h, s, v = qcolor.hsvHue(), qcolor.hsvSaturation(), qcolor.value()
        # Handle special case where HSV hue is -1 (grayscale)
        h_display = h if h != -1 else 0
        self.main_window.hsv_label.setText(f"HSV: ({h_display}, {s}, {v})")
        self.main_window.color_preview.setText("")
        self.main_window.color_preview.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")

        # Calculate HSV range suggestion (±10 for hue, ±30 for saturation, ±30 for value)
        h_min = max(0, h_display - 10)
        h_max = min(179, h_display + 10)  # OpenCV uses 0-179 for hue
        s_min = max(0, s - 30)
        s_max = min(255, s + 30)
        v_min = max(0, v - 30)
        v_max = min(255, v + 30)
        
        range_text = f"HSV Range:\nLower: ({h_min}, {s_min}, {v_min})\nUpper: ({h_max}, {s_max}, {v_max})"
        self.main_window.hsv_range_label.setText(range_text)
        self.main_window.hsv_range_label.setStyleSheet("color: #4CAF50;")
        
        # Enable the save buttons
        self.main_window.save_field_color_btn.setEnabled(True)
        self.main_window.save_ball_color_btn.setEnabled(True)
        
        # Log the color selection
        self.main_window.add_log_message(f"Calibration color selected: RGB({r}, {g}, {b}) HSV({h_display}, {s}, {v})")
    
    @Slot()
    def apply_settings(self):
        """Kamera- und Verarbeitungseinstellungen anwenden"""
        # Collect camera settings
        camera_settings = {
            'exposure_time': self.main_window.exposure_time_input.value(),
            'gain': self.main_window.gain_input.value(),
            'wb_red': self.main_window.wb_red_input.value(),
            'wb_blue': self.main_window.wb_blue_input.value(),
            'brightness': self.main_window.brightness_input.value(),
            'contrast': self.main_window.contrast_input.value(),
            'gamma': self.main_window.gamma_input.value(),
            'framerate': self.main_window.framerate_input.value()
        }
        
        # Collect processing settings
        processing_settings = {
            'ball_sensitivity': self.main_window.ball_sensitivity_input.value(),
            'ball_size_min': self.main_window.ball_size_min_input.value(),
            'ball_size_max': self.main_window.ball_size_max_input.value(),
            'field_threshold': self.main_window.field_threshold_input.value(),
            'kalman_strength': self.main_window.kalman_strength_input.value(),
            'goal_sensitivity': self.main_window.goal_sensitivity_input.value(),
            'processing_resolution': self.main_window.processing_resolution_input.currentText()
        }
        
        # Send settings to processing process
        try:
            self.main_window.command_queue.put({
                'type': 'update_settings',
                'camera_settings': camera_settings,
                'processing_settings': processing_settings
            })
            self.main_window.add_log_message("Settings applied successfully")
            
            # Log the settings for debugging
            self.main_window.add_log_message(f"Camera: Exposure={camera_settings['exposure_time']}ms, Gain={camera_settings['gain']}dB")
            self.main_window.add_log_message(f"Processing: Ball Sensitivity={processing_settings['ball_sensitivity']}%")
            
        except Exception as e:
            self.main_window.add_log_message(f"Failed to apply settings: {e}")
    
    @Slot()
    def reset_settings(self):
        """Alle Einstellungen auf Standardwerte zurücksetzen"""
        # Reset camera settings
        self.main_window.exposure_time_input.setValue(10.0)
        self.main_window.gain_input.setValue(1.0)
        self.main_window.wb_red_input.setValue(1.0)
        self.main_window.wb_blue_input.setValue(1.0)
        self.main_window.brightness_input.setValue(0)
        self.main_window.contrast_input.setValue(1.0)
        self.main_window.gamma_input.setValue(1.0)
        self.main_window.framerate_input.setValue(30)
        
        # Reset processing settings
        self.main_window.ball_sensitivity_input.setValue(50)
        self.main_window.ball_size_min_input.setValue(5)
        self.main_window.ball_size_max_input.setValue(50)
        self.main_window.field_threshold_input.setValue(128)
        self.main_window.kalman_strength_input.setValue(75)
        self.main_window.goal_sensitivity_input.setValue(60)
        self.main_window.processing_resolution_input.setCurrentIndex(1)
        
        self.main_window.add_log_message("Settings reset to defaults")
