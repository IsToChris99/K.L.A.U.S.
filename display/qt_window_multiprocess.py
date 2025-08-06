import cv2
import numpy as np
import time
import config
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor, QMouseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSizePolicy,
    QGroupBox, QCheckBox, QComboBox
)
from match_modes.match_modes import MatchModes


class ClickableVideoLabel(QLabel):
    """Custom QLabel that can detect mouse clicks and extract pixel colors"""
    color_picked = Signal(int, int, int)  # RGB values
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pixmap = None
        self.original_frame = None
        
    def set_frame_data(self, frame, pixmap):
        """Store both the original frame and the displayed pixmap"""
        self.original_frame = frame
        self.current_pixmap = pixmap
        self.setPixmap(pixmap)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse clicks to extract pixel color"""
        if event.button() == Qt.MouseButton.LeftButton and self.original_frame is not None:
            # Get click position
            click_pos = event.position().toPoint()
            
            # Get the current pixmap and its dimensions
            if self.current_pixmap is not None:
                pixmap_rect = self.current_pixmap.rect()
                label_rect = self.rect()
                
                # Calculate scaling factor and offsets for centered image
                scale_x = pixmap_rect.width() / label_rect.width() if label_rect.width() > 0 else 1
                scale_y = pixmap_rect.height() / label_rect.height() if label_rect.height() > 0 else 1
                scale = max(scale_x, scale_y)
                
                # Calculate actual displayed image size
                displayed_width = int(pixmap_rect.width() / scale)
                displayed_height = int(pixmap_rect.height() / scale)
                
                # Calculate offset for centering
                offset_x = (label_rect.width() - displayed_width) // 2
                offset_y = (label_rect.height() - displayed_height) // 2
                
                # Convert click position to image coordinates
                img_x = int((click_pos.x() - offset_x) * scale)
                img_y = int((click_pos.y() - offset_y) * scale)
                
                # Check if click is within image bounds
                if (0 <= img_x < pixmap_rect.width() and 0 <= img_y < pixmap_rect.height()):
                    # Map to original frame coordinates
                    frame_height, frame_width = self.original_frame.shape[:2]
                    orig_x = int(img_x * frame_width / pixmap_rect.width())
                    orig_y = int(img_y * frame_height / pixmap_rect.height())
                    
                    # Ensure coordinates are within frame bounds
                    orig_x = max(0, min(orig_x, frame_width - 1))
                    orig_y = max(0, min(orig_y, frame_height - 1))
                    
                    # Extract pixel color (BGR format in OpenCV)
                    try:
                        pixel_bgr = self.original_frame[orig_y, orig_x]
                        # Convert BGR to RGB
                        r, g, b = int(pixel_bgr[2]), int(pixel_bgr[1]), int(pixel_bgr[0])
                        
                        # Emit the color values
                        self.color_picked.emit(r, g, b)
                        
                    except IndexError:
                        print(f"Index error: trying to access pixel at ({orig_x}, {orig_y}) in frame of size {frame_width}x{frame_height}")
        
        super().mousePressEvent(event)


class KickerMainWindow(QMainWindow):
    """Main window of the Kicker Klaus application for multiprocessing architecture"""
    
    def __init__(self, results_queue, command_queue, running_event):
        super().__init__()
        self.setWindowTitle("Kicker Klaus - Tracking System (Multi-Process)")
        self.resize(1400, 900)
        
        # Multi-processing communication
        self.results_queue = results_queue
        self.command_queue = command_queue
        self.running_event = running_event
        
        # Current states
        self.current_display_frame = None
        self.current_ball_data = None
        self.current_player_data = None
        self.current_M_field = None
        
        # Visualization modes
        self.BALL_ONLY = 1
        self.FIELD_ONLY = 2  
        self.COMBINED = 3
        self.visualization_mode = self.COMBINED
        
        # Statistics and scoring
        self.matcher = MatchModes()
        self.player1_goals = 0
        self.player2_goals = 0
        
        # FPS tracking for different components
        self.display_frame_count = 0
        self.display_last_fps_time = time.time()
        
        # Store latest FPS values from processing
        self.camera_fps = 0.0
        self.preprocessing_fps = 0.0
        self.ball_detection_fps = 0.0
        self.field_detection_fps = 0.0
        
        # Field calibration is now automatic - no manual state needed
        
        self.setup_ui()
        self.connect_signals()
        
        # Timer to poll the results queue
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.poll_results_queue)
        self.update_timer.start(16)  # ~60 FPS update rate for UI
        
        self.add_log_message("GUI initialized with multi-processing architecture")
        
    def setup_ui(self):
        """Creates the complete user interface"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        
        # Score section
        score_group = self.create_score_section()
        
        # Content area: Video and Controls
        content_layout = QHBoxLayout()
        
        # Left side: Video
        video_layout = self.create_video_section()
        
        # Right side: Controls
        control_layout = self.create_control_section()
        
        content_layout.addLayout(video_layout, stretch=3)
        content_layout.addLayout(control_layout, stretch=1)
        
        # Main layout assembly
        main_layout.addWidget(score_group)
        main_layout.addLayout(content_layout, stretch=4)
        
    def create_score_section(self):
        """Creates the score section"""
        score_group = QGroupBox("SCORE")
        score_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 3px solid #4CAF50;
                border-radius: 10px;
                margin: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4CAF50;
            }
        """)
        score_layout = QHBoxLayout(score_group)
        
        # Game Mode Selection (left)
        game_mode_widget = self.create_game_mode_selection()
        
        # Score Display with embedded buttons (center)
        score_widget = self.create_score_display_with_buttons()
        
        # Match Control Buttons (right)
        self.match_buttons_widget = self.create_match_buttons()
        
        score_layout.addWidget(game_mode_widget, stretch=1)
        score_layout.addWidget(score_widget, stretch=5)
        score_layout.addWidget(self.match_buttons_widget, stretch=1)
        
        return score_group
    
    def create_score_display_with_buttons(self):
        """Creates the score display with embedded manual control buttons"""
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
        
        self.team1_plus_btn = QPushButton("+")
        self.team1_minus_btn = QPushButton("-")
        
        team_button_style = """
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                padding: 3px 10px;
                background-color: palette(button);
                color: palette(button-text);
                border: 1px solid palette(mid);
                border-radius: 4px;
                min-width: 25px;
                max-width: 25px;
                min-height: 20px;
                max-height: 20px;
            }
            QPushButton:hover {
                background-color: palette(light);
                border: 1px solid palette(highlight);
            }
            QPushButton:pressed {
                background-color: palette(dark);
                color: palette(highlighted-text);
            }
        """
        
        self.team1_plus_btn.setStyleSheet(team_button_style)
        self.team1_minus_btn.setStyleSheet(team_button_style)
        
        team1_layout.addWidget(self.team1_plus_btn)
        team1_layout.addWidget(self.team1_minus_btn)
        
        # Score display (center)
        self.big_score_label = QLabel("0 : 0")
        self.big_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.big_score_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: #2E7D32;
                background-color: transparent;
                border: none;
                padding: 10px;
            }
        """)
        self.big_score_label.setMinimumHeight(80)
        
        # Team 2 buttons (right)
        team2_layout = QVBoxLayout()
        team2_layout.setSpacing(3)
        team2_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.team2_plus_btn = QPushButton("+")
        self.team2_minus_btn = QPushButton("-")
        
        self.team2_plus_btn.setStyleSheet(team_button_style)
        self.team2_minus_btn.setStyleSheet(team_button_style)
        
        team2_layout.addWidget(self.team2_plus_btn)
        team2_layout.addWidget(self.team2_minus_btn)
        
        # Add to main horizontal layout
        main_layout.addLayout(team1_layout)
        main_layout.addWidget(self.big_score_label, stretch=1)
        main_layout.addLayout(team2_layout)
        
        return score_widget
    
    def create_game_mode_selection(self):
        """Creates the game mode selection widget"""
        game_mode_widget = QWidget()
        game_mode_layout = QVBoxLayout(game_mode_widget)
        game_mode_layout.setContentsMargins(0, 15, 0, 0)
        
        mode_label = QLabel("Game Mode")
        mode_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2E7D32;
                margin-bottom: 10px;
            }
        """)
        mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.game_mode_combo = QComboBox()
        self.game_mode_combo.addItems(["Normal Mode", "Practice Mode", "Tournament Mode"])
        self.game_mode_combo.setCurrentText("Normal Mode")
        self.game_mode_combo.setStyleSheet("""
            QComboBox {
                font-size: 12px;
                padding: 8px;
                background-color: palette(button);
                color: palette(button-text);
                border: 1px solid palette(mid);
                border-radius: 4px;
                min-width: 120px;
            }
            QComboBox:hover {
                background-color: palette(light);
                border: 1px solid palette(highlight);
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border: 2px solid palette(button-text);
                border-top: none;
                border-right: none;
                width: 6px;
                height: 6px;
                margin-right: 8px;
                transform: rotate(-45deg);
            }
        """)
        
        game_mode_layout.addWidget(mode_label)
        game_mode_layout.addWidget(self.game_mode_combo)
        game_mode_layout.addStretch()
        
        return game_mode_widget
    
    def create_match_buttons(self):
        """Creates the match control buttons"""
        match_buttons_widget = QWidget()
        match_buttons_layout = QVBoxLayout(match_buttons_widget)
        match_buttons_layout.setContentsMargins(0, 15, 0, 0)
        
        self.start_match_btn = QPushButton("Start Match")
        self.start_match_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                padding: 40px 40px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.start_match_btn.setMaximumWidth(200)
        
        self.reset_score_btn = QPushButton("Reset Score")
        self.reset_score_btn.setStyleSheet("""
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
        self.reset_score_btn.setMaximumWidth(200)
        self.reset_score_btn.hide()
        
        self.cancel_match_btn = QPushButton("Cancel Match")
        self.cancel_match_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 14px 20px;
                background-color: #FF6B6B;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #FF5252;
            }
            QPushButton:pressed {
                background-color: #E53935;
            }
        """)
        self.cancel_match_btn.setMaximumWidth(200)
        self.cancel_match_btn.hide()
        
        match_buttons_layout.addWidget(self.start_match_btn)
        match_buttons_layout.addWidget(self.reset_score_btn)
        match_buttons_layout.addWidget(self.cancel_match_btn)
        match_buttons_layout.addStretch()
        
        return match_buttons_widget
    
    def create_video_section(self):
        """Creates the video display area"""
        video_layout = QVBoxLayout()
        self.video_label = ClickableVideoLabel()
        self.video_label.setText("Processing gestartet - Warten auf Video-Stream...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        self.video_label.setMinimumSize(800, 600)
        video_layout.addWidget(self.video_label)
        
        return video_layout
    
    def create_control_section(self):
        """Creates the control section"""
        control_layout = QVBoxLayout()
        
        # Tracking Controls (modified for multiprocessing)
        tracking_group = self.create_tracking_controls()
        
        # Visualization Mode
        viz_group = self.create_visualization_controls()
        
        # Processing Settings (simplified for multiprocessing)
        settings_group = self.create_settings_controls()
        
        # Status
        status_group = self.create_status_section()
        
        # Log Output
        log_group = self.create_log_section()
        
        control_layout.addWidget(tracking_group)
        control_layout.addWidget(viz_group)
        control_layout.addWidget(settings_group)
        control_layout.addWidget(status_group)
        control_layout.addWidget(log_group)
        control_layout.addStretch()
        
        return control_layout
    
    def create_tracking_controls(self):
        """Creates the tracking control buttons (adapted for multiprocessing)"""
        tracking_group = QGroupBox("Processing Controls")
        tracking_layout = QVBoxLayout(tracking_group)
        
        self.stop_btn = QPushButton("Stop Processing")
        # Removed calibration button - field detection is now automatic
        
        tracking_layout.addWidget(self.stop_btn)
        
        return tracking_group
    
    def create_visualization_controls(self):
        """Creates the visualization buttons"""
        viz_group = QGroupBox("Display Mode")
        viz_layout = QVBoxLayout(viz_group)
        
        self.ball_only_btn = QPushButton("Ball Only")
        self.field_only_btn = QPushButton("Field Only")
        self.combined_btn = QPushButton("Combined")
        
        self.combined_btn.setStyleSheet("background-color: lightgreen;")
        
        viz_layout.addWidget(self.ball_only_btn)
        viz_layout.addWidget(self.field_only_btn)
        viz_layout.addWidget(self.combined_btn)
        
        return viz_group
    
    def create_settings_controls(self):
        """Creates the settings controls"""
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Processing mode toggle
        processing_layout = QHBoxLayout()
        processing_label = QLabel("Processing Mode:")
        processing_layout.addWidget(processing_label)
        
        self.processing_mode_checkbox = QCheckBox("Use GPU")
        self.processing_mode_checkbox.setChecked(True)  # Default to GPU
        self.processing_mode_checkbox.toggled.connect(self.toggle_processing_mode)
        processing_layout.addWidget(self.processing_mode_checkbox)
        
        # Status label for current processing mode
        self.processing_status_label = QLabel("GPU")
        self.processing_status_label.setStyleSheet("color: green; font-weight: bold;")
        processing_layout.addWidget(self.processing_status_label)
        
        settings_layout.addLayout(processing_layout)
        
        info_label = QLabel("Processing läuft in\nseparatem Prozess")
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        settings_layout.addWidget(info_label)
        
        return settings_group
    
    def create_status_section(self):
        """Creates the status section with detailed FPS displays"""
        status_group = QGroupBox("Status & Performance")
        status_layout = QVBoxLayout(status_group)
        
        self.process_status_label = QLabel("Process: Running")
        
        # Individual FPS labels for different components
        self.camera_fps_label = QLabel("Camera: 0.0 FPS")
        self.preprocessing_fps_label = QLabel("Preprocessing: 0.0 FPS")
        self.ball_detection_fps_label = QLabel("Ball Detection: 0.0 FPS")
        self.field_detection_fps_label = QLabel("Field Detection: 0.0 FPS")
        self.display_fps_label = QLabel("Display: 0.0 FPS")
        
        # Style the FPS labels
        fps_style = "color: #2E7D32; font-size: 11px; font-family: monospace;"
        self.camera_fps_label.setStyleSheet(fps_style)
        self.preprocessing_fps_label.setStyleSheet(fps_style)
        self.ball_detection_fps_label.setStyleSheet(fps_style)
        self.field_detection_fps_label.setStyleSheet(fps_style)
        self.display_fps_label.setStyleSheet(fps_style)
        
        status_layout.addWidget(self.process_status_label)
        status_layout.addWidget(self.camera_fps_label)
        status_layout.addWidget(self.preprocessing_fps_label)
        status_layout.addWidget(self.ball_detection_fps_label)
        status_layout.addWidget(self.field_detection_fps_label)
        status_layout.addWidget(self.display_fps_label)
        
        return status_group
    
    def create_log_section(self):
        """Creates the log section"""
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        return log_group
        
    def connect_signals(self):
        """Connects all button signals to their functions"""
        self.stop_btn.clicked.connect(self.stop_processing)
        
        self.ball_only_btn.clicked.connect(lambda: self.set_visualization_mode(1))
        self.field_only_btn.clicked.connect(lambda: self.set_visualization_mode(2))
        self.combined_btn.clicked.connect(lambda: self.set_visualization_mode(3))
        
        self.reset_score_btn.clicked.connect(self.reset_score)
        self.start_match_btn.clicked.connect(self.start_match)
        self.cancel_match_btn.clicked.connect(self.cancel_match)
        
        # Manual score control buttons
        self.team1_plus_btn.clicked.connect(self.team1_score_plus)
        self.team1_minus_btn.clicked.connect(self.team1_score_minus)
        self.team2_plus_btn.clicked.connect(self.team2_score_plus)
        self.team2_minus_btn.clicked.connect(self.team2_score_minus)
        
        # Game mode selection
        self.game_mode_combo.currentTextChanged.connect(self.change_game_mode)
        
        # Color picking from video
        self.video_label.color_picked.connect(self.on_color_picked)
    
    def poll_results_queue(self):
        """Polls the results queue for new data from the processing process"""
        try:
            while not self.results_queue.empty():
                result_package = self.results_queue.get_nowait()
                
                # Update current state
                self.current_display_frame = result_package.get("display_frame")
                self.current_ball_data = result_package.get("ball_data")
                self.current_player_data = result_package.get("player_data")
                self.current_M_field = result_package.get("M_field")
                
                # Update score from processing if goals were detected
                score_data = result_package.get("score")
                if score_data:
                    # Update local score if it changed
                    new_score1 = score_data.get('player1', 0)
                    new_score2 = score_data.get('player2', 0)
                    
                    if new_score1 != self.player1_goals or new_score2 != self.player2_goals:
                        self.player1_goals = new_score1
                        self.player2_goals = new_score2
                        self.update_score(f"{self.player1_goals}:{self.player2_goals}")
                        
                        # Log automatic goal detection
                        if new_score1 > self.player1_goals or new_score2 > self.player2_goals:
                            self.add_log_message("Goal detected automatically!")
                
                # Update FPS data from processing if available
                fps_data = result_package.get("fps_data")
                if fps_data:
                    self.camera_fps = fps_data.get('camera', 0.0)
                    self.preprocessing_fps = fps_data.get('preprocessing', 0.0)
                    self.ball_detection_fps = fps_data.get('ball_detection', 0.0)
                    self.field_detection_fps = fps_data.get('field_detection', 0.0)
                    self.update_processing_fps()
                
                # Update field data
                self.current_field_data = result_package.get("field_data")
                
                # Update the display
                self.update_display()
                
                # Update Display FPS (how fast GUI receives and displays frames)
                self.display_frame_count += 1
                current_time = time.time()
                if current_time - self.display_last_fps_time >= 1.0:
                    display_fps = self.display_frame_count / (current_time - self.display_last_fps_time)
                    self.update_display_fps(display_fps)
                    self.display_frame_count = 0
                    self.display_last_fps_time = current_time
                    
        except:
            # Queue is empty or other error - ignore
            pass
    
    def update_display(self):
        """Updates the video display with the latest frame and visualizations"""
        if self.current_display_frame is None:
            return
            
        display_frame = self.current_display_frame.copy()
        
        # Add visualizations based on current mode
        if self.visualization_mode in [self.BALL_ONLY, self.COMBINED]:
            self.draw_ball_visualization(display_frame)
            
        if self.visualization_mode in [self.FIELD_ONLY, self.COMBINED]:
            self.draw_field_visualization(display_frame)
        
        # Convert frame to Qt format and display
        self.update_frame(display_frame)
    
    def draw_ball_visualization(self, frame):
        """Draws ball visualization on the frame"""
        if self.current_ball_data is None:
            return
            
        # Get ball detection data
        detection = self.current_ball_data.get('detection')
        smoothed_pts = self.current_ball_data.get('smoothed_pts', [])
        missing_counter = self.current_ball_data.get('missing_counter', 0)
        
        if detection[0] is not None:
            center, radius, confidence, velocity = detection
            center_int = (int(center[0]), int(center[1]))

            # Color selection based on confidence
            if confidence >= 0.8:
                color = config.COLOR_BALL_HIGH_CONFIDENCE  # Green
            elif confidence >= 0.6:
                color = config.COLOR_BALL_MED_CONFIDENCE   # Yellow
            else:
                color = config.COLOR_BALL_LOW_CONFIDENCE   # Orange

            cv2.circle(frame, center, 3, color, -1)
            cv2.circle(frame, center, int(radius), color, 2)

            cv2.putText(frame, f"R: {radius:.1f}", (center_int[0] + 15, center_int[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (center_int[0] + 15, center_int[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Show Kalman velocity
            if velocity is not None:
                cv2.arrowedLine(frame, center,
                            (int(center[0] + velocity[0]*30), int(center[1] + velocity[1]*30)),
                            (255, 0, 255), 2)

        # Ball trail drawing
        if smoothed_pts:
            for i in range(1, len(smoothed_pts)):
                if smoothed_pts[i - 1] is None or smoothed_pts[i] is None:
                    continue
                thickness = int(np.sqrt(config.BALL_TRAIL_MAX_LENGTH / float(i + 1)) * config.BALL_TRAIL_THICKNESS_FACTOR)
                cv2.line(frame, smoothed_pts[i - 1], smoothed_pts[i], config.COLOR_BALL_TRAIL, thickness)
    
    def draw_field_visualization(self, frame):
        """Draws field visualization on the frame"""
        if not hasattr(self, 'current_field_data') or self.current_field_data is None:
            return
            
        field_data = self.current_field_data

        # Field corners
        if field_data.get('field_corners') is not None:
            for i, corner in enumerate(field_data['field_corners']):
                corner_int = (int(corner[0]), int(corner[1]))
                cv2.circle(frame, corner_int, 2, config.COLOR_FIELD_CORNERS, -1)
                cv2.putText(frame, f"{i+1}", (int(corner[0])+10, int(corner[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_FIELD_CORNERS, 2)

        # Goals
        goals = field_data.get('goals', [])
        for i, goal in enumerate(goals):
            if goal.get('contour') is not None:
                cv2.drawContours(frame, [goal['contour']], -1, config.COLOR_GOALS, 2)
            else:
                # Fallback to bounding box if contour is not available
                x, y, w, h = goal.get('bounds', (0, 0, 0, 0))
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), config.COLOR_GOALS, 2)

            # Draw Goal-Center and Label
            center_x, center_y = goal.get('center', (0, 0))
            center_int = (int(center_x), int(center_y))
            cv2.circle(frame, center_int, 5, config.COLOR_GOALS, -1)  
            #cv2.putText(frame, f"Goal {i+1} ({goal['type']})", (int(center_x)+10, int(center_y)-10),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GOALS, 2)
        # Field limits
        if (field_data.get('calibrated') and field_data.get('field_corners') is not None):
            field_corners_int = np.array(field_data['field_corners'], dtype=np.int32)
            cv2.drawContours(frame, [field_corners_int], -1, config.COLOR_FIELD_BOUNDS, 1)
    
    # ============= SIGNAL HANDLER METHODS =============
    
    @Slot()
    def stop_processing(self):
        """Stops the processing by clearing the running event"""
        self.running_event.clear()
        self.process_status_label.setText("Process: Stopping...")
        self.add_log_message("Stopping processing...")

    
    @Slot()
    def set_visualization_mode(self, mode):
        """Sets the visualization mode"""
        self.visualization_mode = mode
        mode_names = {1: "Ball", 2: "Field", 3: "Combined"}
        
        # Reset button highlighting
        for btn in [self.ball_only_btn, self.field_only_btn, self.combined_btn]:
            btn.setStyleSheet("")
        
        # Highlight active button
        if mode == 1:
            self.ball_only_btn.setStyleSheet("background-color: lightgreen;")
        elif mode == 2:
            self.field_only_btn.setStyleSheet("background-color: lightgreen;")
        elif mode == 3:
            self.combined_btn.setStyleSheet("background-color: lightgreen;")

        self.add_log_message(f"Visualization mode set to: {mode_names.get(mode, 'Unknown')}")
    
    @Slot()
    def reset_score(self):
        """Resets the score"""
        if self.matcher.get_mode() == "practice":
            self.add_log_message("Score reset disabled in Practice Mode")
            return
            
        # Reset local score
        self.player1_goals = 0
        self.player2_goals = 0
        self.matcher.reset_scores()
        self.update_score("0:0")
        
        # Send command to processing process to reset its score too
        try:
            self.command_queue.put({'type': 'reset_score'})
            self.add_log_message("Score reset (local and processing)")
        except:
            self.add_log_message("Score reset (local only - could not communicate with processing)")
    
    @Slot()
    def toggle_processing_mode(self, checked):
        """Toggle between CPU and GPU preprocessing"""
        try:
            self.command_queue.put({'type': 'toggle_processing_mode'})
            mode = "GPU" if checked else "CPU"
            self.add_log_message(f"Switched to {mode} processing")
            
            # Update status display
            self.processing_status_label.setText(mode)
            color = "green" if mode == "GPU" else "orange"
            self.processing_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        except Exception as e:
            self.add_log_message(f"Failed to toggle processing mode: {e}")
            # Revert checkbox state on failure
            self.processing_mode_checkbox.setChecked(not checked)
    
    @Slot()
    def team1_score_plus(self):
        """Increases Player 1 score by 1"""
        if self.matcher.get_mode() == "practice":
            self.add_log_message("Score changes disabled in Practice Mode")
            return
            
        self.player1_goals += 1
        self.update_score(f"{self.player1_goals}:{self.player2_goals}")
        if self.matcher.update_score(1, 1):
            self.add_log_message("Team 1 won!")
        self.add_log_message("Team 1 score +1 (manual)")

    @Slot()
    def team1_score_minus(self):
        """Decreases Player 1 score by 1"""
        if self.matcher.get_mode() == "practice":
            self.add_log_message("Score changes disabled in Practice Mode")
            return
            
        if self.player1_goals > 0:
            self.player1_goals -= 1
            self.update_score(f"{self.player1_goals}:{self.player2_goals}")
            self.matcher.update_score(1, -1)
            self.add_log_message("Team 1 score -1 (manual)")
        else:
            self.add_log_message("Team 1 score cannot go below 0")

    @Slot()
    def team2_score_plus(self):
        """Increases Player 2 score by 1"""
        if self.matcher.get_mode() == "practice":
            self.add_log_message("Score changes disabled in Practice Mode")
            return
            
        self.player2_goals += 1
        self.update_score(f"{self.player1_goals}:{self.player2_goals}")
        if self.matcher.update_score(2, 1):
            self.add_log_message("Team 2 won!")
        self.add_log_message("Team 2 score +1 (manual)")

    @Slot()
    def team2_score_minus(self):
        """Decreases Player 2 score by 1"""
        if self.matcher.get_mode() == "practice":
            self.add_log_message("Score changes disabled in Practice Mode")
            return
            
        if self.player2_goals > 0:
            self.player2_goals -= 1
            self.update_score(f"{self.player1_goals}:{self.player2_goals}")
            self.matcher.update_score(2, -1)
            self.add_log_message("Team 2 score -1 (manual)")
        else:
            self.add_log_message("Team 2 score cannot go below 0")

    @Slot()
    def start_match(self):
        """Starts a match and shows the match buttons"""
        self.start_match_btn.hide()
        
        self.matcher.reset_scores()
        self.matcher.start_match()
        
        self.reset_score_btn.show()
        self.cancel_match_btn.show()

        self.add_log_message("Match started")

    @Slot()
    def cancel_match(self):
        """Cancels the current match and resets the score"""
        self.reset_score_btn.hide()
        self.cancel_match_btn.hide()
        
        self.matcher.end_match()
        self.player1_goals = 0
        self.player2_goals = 0
        
        self.update_score("0:0")
        
        self.start_match_btn.show()
        
        self.add_log_message("Match canceled")
    
    @Slot(str)
    def change_game_mode(self, mode_text):
        """Changes the game mode based on ComboBox selection"""
        mode_mapping = {
            "Normal Mode": "normal",
            "Practice Mode": "practice", 
            "Tournament Mode": "tournament"
        }
        
        if mode_text in mode_mapping:
            mode_key = mode_mapping[mode_text]
            self.matcher.set_mode(mode_key)
            
            description = self.matcher.get_mode_description()
            self.add_log_message(f"Game mode changed to: {mode_text}")
            self.add_log_message(f"Description: {description}")
        else:
            self.add_log_message(f"Unknown game mode: {mode_text}")
    
    @Slot(int, int, int)
    def on_color_picked(self, r, g, b):
        """Handle color picking from video"""
        self.add_log_message(f"Color picked: RGB({r}, {g}, {b})")
        
        # Also log HSV values for better color analysis
        qcolor = QColor(r, g, b)
        h, s, v = qcolor.hsvHue(), qcolor.hsvSaturation(), qcolor.value()
        self.add_log_message(f"Color picked: HSV({h}, {s}, {v})")
        
        # Also log the hex representation
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.add_log_message(f"Color picked: HEX({hex_color})")
        
        self.add_log_message("Tip: Diese Farbe könnte für die Spielfelderkennung verwendet werden")
    
    # ============= UPDATE METHODS =============
    
    @Slot(np.ndarray)
    def update_frame(self, frame):
        """Updates the video display with a new frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Store frame data in the clickable video label for color picking
        self.video_label.set_frame_data(frame, pixmap)
    
    def add_log_message(self, message):
        """Adds a log message"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_processing_fps(self):
        """Updates the processing FPS displays"""
        self.camera_fps_label.setText(f"Camera: {self.camera_fps:.1f} FPS")
        self.preprocessing_fps_label.setText(f"Preprocessing: {self.preprocessing_fps:.1f} FPS")
        self.ball_detection_fps_label.setText(f"Ball Detection: {self.ball_detection_fps:.1f} FPS")
        self.field_detection_fps_label.setText(f"Field Detection: {self.field_detection_fps:.1f} FPS")
    
    def update_display_fps(self, fps):
        """Updates the display FPS"""
        self.display_fps_label.setText(f"Display: {fps:.1f} FPS")
    
    def update_fps(self, fps):
        """Legacy method - now redirects to display FPS for compatibility"""
        self.update_display_fps(fps)
    
    def update_score(self, score):
        """Updates the score display"""
        self.big_score_label.setText(score.replace(":", " : "))
    
    def closeEvent(self, event):
        """Called when the window is closed"""
        print("GUI closeEvent aufgerufen...")
        
        # Stoppe den Update-Timer
        if hasattr(self, 'update_timer') and self.update_timer:
            self.update_timer.stop()
        
        # Signal an alle Prozesse zum Beenden
        self.running_event.clear()
        self.add_log_message("GUI closing - stopping all processes...")
        
        # Gib der Anwendung etwas Zeit zum Cleanup
        import time
        time.sleep(0.1)
        
        event.accept()
        print("GUI closeEvent abgeschlossen.")
