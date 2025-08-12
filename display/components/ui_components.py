"""
UI-Komponenten für die GUI
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QPushButton, QLabel, QTextEdit, QCheckBox, 
    QSpinBox, QFormLayout
)

class ScoreSection:
    """Klasse für die Punkte-Anzeige"""
    
    def __init__(self, parent_window):
        self.parent_window = parent_window
    
    def create_score_section(self):
        """Erstellt die Punkte-Sektion"""
        score_group = QGroupBox("SCORE")
        score_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 3px solid #4CAF50;
                border-radius: 10px;
                margin: 10px;
                padding-top: 0px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4CAF50;
            }
        """)
        score_layout = QHBoxLayout(score_group)

        # Max Goal Selection (left)
        max_goal_widget = self.create_max_goal_selection()

        # Score Display with embedded buttons (center)
        score_widget = self.create_score_display_with_buttons()
        
        # Match Control Buttons (right)
        self.parent_window.match_buttons_widget = self.create_match_buttons()

        score_layout.addWidget(max_goal_widget, stretch=1)
        score_layout.addWidget(score_widget, stretch=5)
        score_layout.addWidget(self.parent_window.match_buttons_widget, stretch=1)
        
        return score_group
    
    def create_score_display_with_buttons(self):
        """Erstellt die Punkte-Anzeige mit integrierten manuellen Steuerungs-Buttons"""
        score_widget = QWidget()
        score_widget.setStyleSheet("""
            QWidget {
                background-color: #E8F5E8;
                border: 2px solid #4CAF50;
                border-radius: 15px;
                margin: 10px;
            }
        """)
        
        main_layout = QHBoxLayout(score_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Team 1 buttons (left)
        team1_layout = QVBoxLayout()
        team1_layout.setSpacing(3)
        team1_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.parent_window.team1_plus_btn = QPushButton("+")
        self.parent_window.team1_minus_btn = QPushButton("-")
        
        team_button_style = """
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                color: black;
                background-color: #f5f5f5;
                padding: 3px 10px;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                min-width: 25px;
                max-width: 25px;
                min-height: 20px;
                max-height: 20px;
            }
            QPushButton:hover {
                background-color: #b7c2b7;
                border: 1px solid #45a049;
            }
            QPushButton:pressed {
                background-color: #868f86;
                border: 1px solid #3d8b40;
            }
        """
        
        self.parent_window.team1_plus_btn.setStyleSheet(team_button_style)
        self.parent_window.team1_minus_btn.setStyleSheet(team_button_style)
        
        team1_layout.addWidget(self.parent_window.team1_plus_btn)
        team1_layout.addWidget(self.parent_window.team1_minus_btn)
        
        # Score display (center)
        self.parent_window.big_score_label = QLabel("0 : 0")
        self.parent_window.big_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.parent_window.big_score_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: #2E7D32;
                background-color: transparent;
                border: none;
                padding: 10px;
            }
        """)
        self.parent_window.big_score_label.setMinimumHeight(80)
        
        # Team 2 buttons (right)
        team2_layout = QVBoxLayout()
        team2_layout.setSpacing(3)
        team2_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.parent_window.team2_plus_btn = QPushButton("+")
        self.parent_window.team2_minus_btn = QPushButton("-")
        
        self.parent_window.team2_plus_btn.setStyleSheet(team_button_style)
        self.parent_window.team2_minus_btn.setStyleSheet(team_button_style)
        
        team2_layout.addWidget(self.parent_window.team2_plus_btn)
        team2_layout.addWidget(self.parent_window.team2_minus_btn)
        
        # Add to main horizontal layout
        main_layout.addLayout(team1_layout)
        main_layout.addWidget(self.parent_window.big_score_label, stretch=1)
        main_layout.addLayout(team2_layout)
        
        return score_widget
    
    def create_max_goal_selection(self):
        """Erstellt das Tor-Limit-Eingabe-Widget"""
        goal_limit_widget = QWidget()
        goal_limit_layout = QVBoxLayout(goal_limit_widget)
        goal_limit_layout.setContentsMargins(0, 15, 0, 0)
        
        # Label for Goal Limit
        limit_label = QLabel("Max Goals")
        limit_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #4CAF50;
                margin-bottom: 0;
                margin-top: 5px;
            }
        """)
        limit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Input field for goal limit with slider
        self.parent_window.goal_limit_input = QSpinBox()
        self.parent_window.goal_limit_input.setRange(1, 999999)
        self.parent_window.goal_limit_input.setValue(10)  # Default value
        self.parent_window.goal_limit_input.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                margin: 0;
                padding-top: 0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 0 0 0;
            }
        """)

        # Default button
        self.parent_window.default_goal_limit_btn = QPushButton("Default")
        self.parent_window.default_goal_limit_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 5px 10px;
                background-color: #58ad57;
                color: white;
                border: none;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        # Infinity button
        self.parent_window.infinity_goal_limit_btn = QPushButton("Infinity")
        self.parent_window.infinity_goal_limit_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 5px 10px;
                background-color: #58ad57;
                color: white;
                border: none;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        # Group for Two Buttons: "Default" and "Infinity"
        self.parent_window.goal_limit_buttons = QHBoxLayout()
        self.parent_window.goal_limit_buttons.addWidget(self.parent_window.default_goal_limit_btn)
        self.parent_window.goal_limit_buttons.addWidget(self.parent_window.infinity_goal_limit_btn)
        
        
        
        goal_limit_layout.addWidget(limit_label)
        goal_limit_layout.addWidget(self.parent_window.goal_limit_input)
        goal_limit_layout.addLayout(self.parent_window.goal_limit_buttons)
        goal_limit_layout.addStretch()
        
        return goal_limit_widget
    
    def create_match_buttons(self):
        """Erstellt die Match-Steuerungs-Buttons"""
        match_buttons_widget = QWidget()
        match_buttons_layout = QVBoxLayout(match_buttons_widget)
        match_buttons_layout.setContentsMargins(0, 15, 0, 0)
        
        self.parent_window.reset_score_btn = QPushButton("Reset Score")
        self.parent_window.reset_score_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 14px 20px;
                background-color: #FFA726;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #FF9800;
            }
            QPushButton:pressed {
                background-color: #F57C00;
            }
        """)
        self.parent_window.reset_score_btn.setMaximumWidth(200)
        
        match_buttons_layout.addWidget(self.parent_window.reset_score_btn)
        match_buttons_layout.addStretch()
        
        return match_buttons_widget

class ControlSection:
    """Klasse für die Steuerungssektion"""
    
    def __init__(self, parent_window):
        self.parent_window = parent_window
    
    def create_control_section(self):
        """Erstellt die Steuerungssektion"""
        control_layout = QVBoxLayout()
        
        # Visualization Mode
        viz_group = self.create_visualization_controls()
        
        # Processing Settings (simplified for multiprocessing)
        settings_group = self.create_settings_controls()
        
        # Status
        status_group = self.create_status_section()
        
        # Log Output
        log_group = self.create_log_section()
        
        control_layout.addWidget(viz_group)
        control_layout.addWidget(settings_group)
        control_layout.addWidget(status_group)
        control_layout.addWidget(log_group)
        control_layout.addStretch()
        
        return control_layout
    
    def create_visualization_controls(self):
        """Erstellt die Visualisierungs-Buttons"""
        viz_group = QGroupBox("Display Mode")
        viz_layout = QHBoxLayout(viz_group)  # Changed to horizontal layout
        
        # Left side: Display mode buttons (2/3 of space)
        mode_buttons_widget = QWidget()
        mode_buttons_layout = QVBoxLayout(mode_buttons_widget)
        mode_buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.parent_window.ball_only_btn = QPushButton("Ball Only")
        self.parent_window.field_only_btn = QPushButton("Field Only")
        self.parent_window.player_only_btn = QPushButton("Player Only")

        mode_buttons_layout.addWidget(self.parent_window.ball_only_btn)
        mode_buttons_layout.addWidget(self.parent_window.field_only_btn)
        mode_buttons_layout.addWidget(self.parent_window.player_only_btn)
        
        # Right side: Combined button and Toggle detections button (1/3 of space)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.parent_window.combined_btn = QPushButton("Combined")
        self.parent_window.combined_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 12px;
                background-color: lightgreen;
                color: white;
                margin-top: 5px;
            }
        """)
        
        self.parent_window.toggle_detections_btn = QPushButton("Hide Detections")
        self.parent_window.toggle_detections_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 12px;
                background-color: #FFA726;
                color: white;
            }
            QPushButton:hover {
                background-color: #FF9800;
            }
            QPushButton:pressed {
                background-color: #F57C00;
            }
        """)
        
        right_layout.addWidget(self.parent_window.combined_btn)
        right_layout.addWidget(self.parent_window.toggle_detections_btn)
        right_layout.addStretch()  # Add stretch to push buttons to top
        
        # Add widgets to main layout with 2:1 ratio
        viz_layout.addWidget(mode_buttons_widget, 2)  # 2/3 of space
        viz_layout.addWidget(right_widget, 1)         # 1/3 of space
        
        return viz_group
    
    def create_settings_controls(self):
        """Erstellt die Einstellungssteuerung"""
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Processing mode toggle
        processing_layout = QHBoxLayout()
        processing_label = QLabel("Processing Mode:")
        processing_layout.addWidget(processing_label)
        
        self.parent_window.processing_mode_checkbox = QCheckBox("Use GPU")
        self.parent_window.processing_mode_checkbox.setChecked(True)  # Default to GPU
        self.parent_window.processing_mode_checkbox.toggled.connect(self.parent_window.toggle_processing_mode)
        processing_layout.addWidget(self.parent_window.processing_mode_checkbox)
        
        # Status label for current processing mode
        self.parent_window.processing_status_label = QLabel("GPU")
        self.parent_window.processing_status_label.setStyleSheet("color: green; font-weight: bold;")
        processing_layout.addWidget(self.parent_window.processing_status_label)
        
        settings_layout.addLayout(processing_layout)
        
        return settings_group
    
    def create_status_section(self):
        """Erstellt die Status-Sektion mit detaillierten FPS-Anzeigen in zwei Spalten mit ausgerichteten Zahlen"""
        status_group = QGroupBox("Status and Performance")
        status_layout = QHBoxLayout(status_group)  # Hauptlayout horizontal
        
        # Individual FPS labels for different components
        self.parent_window.camera_fps_label = QLabel("0.0 FPS")
        self.parent_window.preprocessing_fps_label = QLabel("0.0 FPS")
        self.parent_window.ball_detection_fps_label = QLabel("0.0 FPS")
        self.parent_window.field_detection_fps_label = QLabel("0.0 FPS")
        self.parent_window.player_detection_fps_label = QLabel("0.0 FPS")
        self.parent_window.display_fps_label = QLabel("0.0 FPS")
        
        # Style the FPS labels - right aligned for better number alignment
        fps_style = "color: #2E7D32; text-align: right; min-width: 60px;"
        self.parent_window.camera_fps_label.setStyleSheet(fps_style)
        self.parent_window.preprocessing_fps_label.setStyleSheet(fps_style)
        self.parent_window.ball_detection_fps_label.setStyleSheet(fps_style)
        self.parent_window.field_detection_fps_label.setStyleSheet(fps_style)
        self.parent_window.player_detection_fps_label.setStyleSheet(fps_style)
        self.parent_window.display_fps_label.setStyleSheet(fps_style)
        
        # Set alignment for all FPS labels
        self.parent_window.camera_fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.parent_window.preprocessing_fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.parent_window.ball_detection_fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.parent_window.field_detection_fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.parent_window.player_detection_fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.parent_window.display_fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Left column: Camera, Preprocessing, Display
        left_column = QFormLayout()
        left_column.addRow("Camera:", self.parent_window.camera_fps_label)
        left_column.addRow("Preprocessing:", self.parent_window.preprocessing_fps_label)
        left_column.addRow("Display:", self.parent_window.display_fps_label)
        
        # Right column: Ball Detection, Field Detection, Player Detection
        right_column = QFormLayout()
        right_column.addRow("Ball Detection:", self.parent_window.ball_detection_fps_label)
        right_column.addRow("Field Detection:", self.parent_window.field_detection_fps_label)
        right_column.addRow("Player Detection:", self.parent_window.player_detection_fps_label)

        middle_empty_place = QFormLayout()

        # Add both columns to the main horizontal layout
        status_layout.addLayout(left_column, 2)
        status_layout.addLayout(middle_empty_place, 1)
        status_layout.addLayout(right_column, 2)
        
        return status_group
    
    def create_log_section(self):
        """Erstellt die Log-Sektion"""
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout(log_group)
        self.parent_window.log_text = QTextEdit()
        self.parent_window.log_text.setMaximumHeight(200)
        self.parent_window.log_text.setReadOnly(True)
        log_layout.addWidget(self.parent_window.log_text)
        
        return log_group

class VideoSection:
    """Klasse für die Video-Sektion"""
    
    def __init__(self, parent_window):
        self.parent_window = parent_window
    
    def create_video_section(self):
        """Erstellt den Video-Display-Bereich"""
        video_layout = QVBoxLayout()
        
        # Use simple QLabel for main video display (no color picking)
        self.parent_window.video_label = QLabel("Processing started - Waiting for video stream...")
        self.parent_window.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.parent_window.video_label.setStyleSheet("color: white; border: 1px solid #9f9f9f; border-radius: 5px;")
        self.parent_window.video_label.setMinimumSize(720, 540)
        self.parent_window.video_label.setScaledContents(False)  # Important for aspect ratio
        video_layout.addWidget(self.parent_window.video_label)
        
        return video_layout
