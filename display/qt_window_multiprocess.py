import cv2
import numpy as np
import time
import config
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor, QMouseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSizePolicy,
    QGroupBox, QCheckBox, QComboBox, QTabWidget, QLineEdit,
    QFormLayout, QSpinBox, QDoubleSpinBox, QSlider
)


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
        """Creates the complete user interface with tab system"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.tracking_tab = self.create_tracking_tab()
        self.settings_tab = self.create_settings_tab()
        self.about_tab = self.create_about_tab()
        
        # Add tabs to widget
        self.tab_widget.addTab(self.tracking_tab, "🎯 Tracking")
        self.tab_widget.addTab(self.settings_tab, "⚙️ Settings")
        self.tab_widget.addTab(self.about_tab, "ℹ️ About")

        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
    def create_tracking_tab(self):
        """Creates the main tracking interface tab"""
        tracking_widget = QWidget()
        main_layout = QVBoxLayout(tracking_widget)
        
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
        
        return tracking_widget
    
    def create_settings_tab(self):
        """Creates the camera settings interface tab"""
        settings_widget = QWidget()
        main_layout = QVBoxLayout(settings_widget)
        
        # Title
        title_label = QLabel("Camera & System Settings")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
                margin: 20px;
                text-align: center;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Settings content in horizontal layout
        settings_content = QHBoxLayout()
        
        # Left column: Camera Settings
        camera_group = self.create_camera_settings_group()
        
        # Right column: Processing Settings
        processing_group = self.create_processing_settings_group()
        
        settings_content.addWidget(camera_group, stretch=1)
        settings_content.addWidget(processing_group, stretch=1)
        
        main_layout.addLayout(settings_content)
        
        # Bottom: Action buttons
        button_layout = QHBoxLayout()
        
        self.apply_settings_btn = QPushButton("Apply Settings")
        self.apply_settings_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 15px 30px;
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
        
        self.reset_settings_btn = QPushButton("Reset to Defaults")
        self.reset_settings_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 15px 30px;
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
        
        button_layout.addStretch()
        button_layout.addWidget(self.reset_settings_btn)
        button_layout.addWidget(self.apply_settings_btn)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        main_layout.addStretch()
        
        
        video_layout = QVBoxLayout()
        self.video_label = ClickableVideoLabel()
        self.video_label.setText("Processing gestartet - Warten auf Video-Stream...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        self.video_label.setMinimumSize(500, 350)
        video_layout.addWidget(self.video_label)

        main_layout.addLayout(video_layout)
        
        return settings_widget

    def create_about_tab(self):
        """Creates the about section with centered labels"""
        about_widget = QWidget()
        about_layout = QVBoxLayout(about_widget)
        
        # Headline
        about_headline = QLabel("KLAUS - Kicker Live Analytics Universal System\nVersion 1.0")
        about_headline.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
                margin: 20px;
            }
        """)
        about_headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        about_layout.addWidget(about_headline)
        
        # Team info in a nice container
        team_container = QGroupBox("Development Team")
        team_container.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                margin: 20px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 10px 0 10px;
                color: #4CAF50;
            }
        """)
        
        team_layout = QVBoxLayout(team_container)
        
        # Team members
        developers_label = QLabel("Tim Vesper • Roman Heck • Jose Pineda • Joshua Siemer • Christian Gunter")
        developers_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        developers_label.setStyleSheet("font-size: 16px; margin: 10px; font-weight: bold;")
        
        # Roles
        roles = [
            ("Team Leader", "Tim Vesper"),
            ("Processing Architecture", "Tim Vesper"),
            ("GUI Design & Implementation", "Christian Gunter"),
            ("IDS Camera Integration", "Tim Vesper"),
            ("Camera Calibration", "Christian Gunter"),
            ("Field & Ball Detection", "Joshua Siemer"),
            ("Player Detection", "Roman Heck"),
            ("Ball Heatmap", "Jose Pineda")
        ]
        
        team_layout.addWidget(developers_label)
        team_layout.addWidget(QLabel())  # Spacer
        
        for role, person in roles:
            role_label = QLabel(f"{role}: {person}")
            role_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            role_label.setStyleSheet("font-size: 14px; margin: 5px;")
            team_layout.addWidget(role_label)
        
        about_layout.addWidget(team_container)
        about_layout.addStretch()
        
        return about_widget

    def create_camera_settings_group(self):
        """Creates the camera settings group"""
        camera_group = QGroupBox("📹 Camera Settings")
        camera_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #4CAF50;
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
        
        layout = QFormLayout(camera_group)
        layout.setSpacing(15)
        
        # Exposure Time
        self.exposure_time_input = QDoubleSpinBox()
        self.exposure_time_input.setRange(0.1, 1000.0)
        self.exposure_time_input.setValue(10.0)
        self.exposure_time_input.setSuffix(" ms")
        self.exposure_time_input.setDecimals(1)
        layout.addRow("Exposure Time:", self.exposure_time_input)
        
        # Gain
        self.gain_input = QDoubleSpinBox()
        self.gain_input.setRange(0.0, 40.0)
        self.gain_input.setValue(1.0)
        self.gain_input.setSuffix(" dB")
        self.gain_input.setDecimals(1)
        layout.addRow("Gain:", self.gain_input)
        
        # White Balance Red
        self.wb_red_input = QDoubleSpinBox()
        self.wb_red_input.setRange(0.5, 3.0)
        self.wb_red_input.setValue(1.0)
        self.wb_red_input.setDecimals(2)
        layout.addRow("White Balance Red:", self.wb_red_input)
        
        # White Balance Blue
        self.wb_blue_input = QDoubleSpinBox()
        self.wb_blue_input.setRange(0.5, 3.0)
        self.wb_blue_input.setValue(1.0)
        self.wb_blue_input.setDecimals(2)
        layout.addRow("White Balance Blue:", self.wb_blue_input)
        
        # Brightness
        self.brightness_input = QSpinBox()
        self.brightness_input.setRange(-100, 100)
        self.brightness_input.setValue(0)
        layout.addRow("Brightness:", self.brightness_input)
        
        # Contrast
        self.contrast_input = QDoubleSpinBox()
        self.contrast_input.setRange(0.1, 3.0)
        self.contrast_input.setValue(1.0)
        self.contrast_input.setDecimals(2)
        layout.addRow("Contrast:", self.contrast_input)
        
        # Gamma
        self.gamma_input = QDoubleSpinBox()
        self.gamma_input.setRange(0.1, 3.0)
        self.gamma_input.setValue(1.0)
        self.gamma_input.setDecimals(2)
        layout.addRow("Gamma:", self.gamma_input)
        
        # Frame Rate
        self.framerate_input = QSpinBox()
        self.framerate_input.setRange(1, 250)
        self.framerate_input.setValue(30)
        self.framerate_input.setSuffix(" fps")
        layout.addRow("Target Frame Rate:", self.framerate_input)
        
        return camera_group
    
    def create_processing_settings_group(self):
        """Creates the processing settings group"""
        processing_group = QGroupBox("🔧 Processing Settings")
        processing_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #4CAF50;
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
        
        layout = QFormLayout(processing_group)
        layout.setSpacing(15)
        
        # Ball Detection Sensitivity
        self.ball_sensitivity_input = QSlider(Qt.Orientation.Horizontal)
        self.ball_sensitivity_input.setRange(1, 100)
        self.ball_sensitivity_input.setValue(50)
        self.ball_sensitivity_label = QLabel("50%")
        self.ball_sensitivity_input.valueChanged.connect(
            lambda v: self.ball_sensitivity_label.setText(f"{v}%")
        )
        
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(self.ball_sensitivity_input)
        sensitivity_layout.addWidget(self.ball_sensitivity_label)
        layout.addRow("Ball Detection Sensitivity:", sensitivity_layout)
        
        # Ball Size Range Min
        self.ball_size_min_input = QSpinBox()
        self.ball_size_min_input.setRange(1, 50)
        self.ball_size_min_input.setValue(5)
        self.ball_size_min_input.setSuffix(" px")
        layout.addRow("Min Ball Size:", self.ball_size_min_input)
        
        # Ball Size Range Max
        self.ball_size_max_input = QSpinBox()
        self.ball_size_max_input.setRange(10, 200)
        self.ball_size_max_input.setValue(50)
        self.ball_size_max_input.setSuffix(" px")
        layout.addRow("Max Ball Size:", self.ball_size_max_input)
        
        # Field Detection Threshold
        self.field_threshold_input = QSlider(Qt.Orientation.Horizontal)
        self.field_threshold_input.setRange(1, 255)
        self.field_threshold_input.setValue(128)
        self.field_threshold_label = QLabel("128")
        self.field_threshold_input.valueChanged.connect(
            lambda v: self.field_threshold_label.setText(str(v))
        )
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.field_threshold_input)
        threshold_layout.addWidget(self.field_threshold_label)
        layout.addRow("Field Detection Threshold:", threshold_layout)
        
        # Kalman Filter Strength
        self.kalman_strength_input = QSlider(Qt.Orientation.Horizontal)
        self.kalman_strength_input.setRange(1, 100)
        self.kalman_strength_input.setValue(75)
        self.kalman_strength_label = QLabel("75%")
        self.kalman_strength_input.valueChanged.connect(
            lambda v: self.kalman_strength_label.setText(f"{v}%")
        )
        
        kalman_layout = QHBoxLayout()
        kalman_layout.addWidget(self.kalman_strength_input)
        kalman_layout.addWidget(self.kalman_strength_label)
        layout.addRow("Kalman Filter Strength:", kalman_layout)
        
        # Goal Detection Sensitivity
        self.goal_sensitivity_input = QSlider(Qt.Orientation.Horizontal)
        self.goal_sensitivity_input.setRange(1, 100)
        self.goal_sensitivity_input.setValue(60)
        self.goal_sensitivity_label = QLabel("60%")
        self.goal_sensitivity_input.valueChanged.connect(
            lambda v: self.goal_sensitivity_label.setText(f"{v}%")
        )
        
        goal_layout = QHBoxLayout()
        goal_layout.addWidget(self.goal_sensitivity_input)
        goal_layout.addWidget(self.goal_sensitivity_label)
        layout.addRow("Goal Detection Sensitivity:", goal_layout)
        
        # Processing Resolution
        self.processing_resolution_input = QComboBox()
        self.processing_resolution_input.addItems([
            "720x540 (Fast)",
            "960x720 (Balanced)",
            "1440x1080 (Quality)"
        ])
        self.processing_resolution_input.setCurrentIndex(1)
        layout.addRow("Processing Resolution:", self.processing_resolution_input)
        
        return processing_group
        
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

        # Max Goal Selection (left)
        max_goal_widget = self.create_max_goal_selection()

        # Score Display with embedded buttons (center)
        score_widget = self.create_score_display_with_buttons()
        
        # Match Control Buttons (right)
        self.match_buttons_widget = self.create_match_buttons()

        score_layout.addWidget(max_goal_widget, stretch=1)
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
    
    def create_max_goal_selection(self):
        """Creates the goal limit input widget"""
        goal_limit_widget = QWidget()
        goal_limit_layout = QVBoxLayout(goal_limit_widget)
        goal_limit_layout.setContentsMargins(0, 15, 0, 0)
        
        # Label for Goal Limit
        limit_label = QLabel("Max Goals")
        limit_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2E7D32;
                margin-bottom: 10px;
            }
        """)
        limit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Input field for goal limit
        self.goal_limit_input = QSpinBox()
        self.goal_limit_input.setRange(1, 1000)
        self.goal_limit_input.setValue(9)  # Default value
        self.goal_limit_input.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #4CAF50;
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
        
        # Reset button
        self.reset_goal_limit_btn = QPushButton("Reset max. Goals")
        self.reset_goal_limit_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 5px 10px;
                background-color: #FFA726;
                color: white;
                border: none;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #FF9800;
            }
            QPushButton:pressed {
                background-color: #F57C00;
            }
        """)
        
        goal_limit_layout.addWidget(limit_label)
        goal_limit_layout.addWidget(self.goal_limit_input)
        goal_limit_layout.addWidget(self.reset_goal_limit_btn)
        goal_limit_layout.addStretch()
        
        return goal_limit_widget
    
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
        self.video_label.setMinimumSize(720, 540)
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
        
        self.reset_score_btn.clicked.connect(self.reset_score_placeholder)
        self.start_match_btn.clicked.connect(self.start_match_placeholder)
        self.cancel_match_btn.clicked.connect(self.cancel_match_placeholder)
        
        # Manual score control buttons
        self.team1_plus_btn.clicked.connect(self.team1_score_plus_placeholder)
        self.team1_minus_btn.clicked.connect(self.team1_score_minus_placeholder)
        self.team2_plus_btn.clicked.connect(self.team2_score_plus_placeholder)
        self.team2_minus_btn.clicked.connect(self.team2_score_minus_placeholder)
        
        # Goal limit controls
        self.reset_goal_limit_btn.clicked.connect(self.reset_goal_limit_placeholder)
        
        # Color picking from video
        self.video_label.color_picked.connect(self.on_color_picked)
        
        # Settings tab buttons
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        self.reset_settings_btn.clicked.connect(self.reset_settings)
    
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
    def reset_score_placeholder(self):
        """Placeholder for resetting the score - logic to be implemented later"""
        # Reset local score
        self.player1_goals = 0
        self.player2_goals = 0
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
    def team1_score_plus_placeholder(self):
        """Placeholder for increasing Player 1 score - logic to be implemented later"""
        self.player1_goals += 1
        self.update_score(f"{self.player1_goals}:{self.player2_goals}")
        self.add_log_message("Team 1 score +1 (manual)")

    @Slot()
    def team1_score_minus_placeholder(self):
        """Placeholder for decreasing Player 1 score - logic to be implemented later"""
        if self.player1_goals > 0:
            self.player1_goals -= 1
            self.update_score(f"{self.player1_goals}:{self.player2_goals}")
            self.add_log_message("Team 1 score -1 (manual)")
        else:
            self.add_log_message("Team 1 score cannot go below 0")

    @Slot()
    def team2_score_plus_placeholder(self):
        """Placeholder for increasing Player 2 score - logic to be implemented later"""
        self.player2_goals += 1
        self.update_score(f"{self.player1_goals}:{self.player2_goals}")
        self.add_log_message("Team 2 score +1 (manual)")

    @Slot()
    def team2_score_minus_placeholder(self):
        """Placeholder for decreasing Player 2 score - logic to be implemented later"""
        if self.player2_goals > 0:
            self.player2_goals -= 1
            self.update_score(f"{self.player1_goals}:{self.player2_goals}")
            self.add_log_message("Team 2 score -1 (manual)")
        else:
            self.add_log_message("Team 2 score cannot go below 0")

    @Slot()
    def start_match_placeholder(self):
        """Placeholder for starting a match - logic to be implemented later"""
        self.start_match_btn.hide()
        
        self.reset_score_btn.show()
        self.cancel_match_btn.show()

        self.add_log_message("Match started (placeholder)")

    @Slot()
    def cancel_match_placeholder(self):
        """Placeholder for canceling a match - logic to be implemented later"""
        self.reset_score_btn.hide()
        self.cancel_match_btn.hide()
        
        self.player1_goals = 0
        self.player2_goals = 0
        
        self.update_score("0:0")
        
        self.start_match_btn.show()
        
        self.add_log_message("Match canceled (placeholder)")
        
    @Slot()
    def reset_goal_limit_placeholder(self):
        """Placeholder for resetting goal limit to default value"""
        self.goal_limit_input.setValue(9)
        self.add_log_message("Goal limit reset to 9")
    
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
    
    @Slot()
    def apply_settings(self):
        """Apply camera and processing settings"""
        # Collect camera settings
        camera_settings = {
            'exposure_time': self.exposure_time_input.value(),
            'gain': self.gain_input.value(),
            'wb_red': self.wb_red_input.value(),
            'wb_blue': self.wb_blue_input.value(),
            'brightness': self.brightness_input.value(),
            'contrast': self.contrast_input.value(),
            'gamma': self.gamma_input.value(),
            'framerate': self.framerate_input.value()
        }
        
        # Collect processing settings
        processing_settings = {
            'ball_sensitivity': self.ball_sensitivity_input.value(),
            'ball_size_min': self.ball_size_min_input.value(),
            'ball_size_max': self.ball_size_max_input.value(),
            'field_threshold': self.field_threshold_input.value(),
            'kalman_strength': self.kalman_strength_input.value(),
            'goal_sensitivity': self.goal_sensitivity_input.value(),
            'processing_resolution': self.processing_resolution_input.currentText()
        }
        
        # Send settings to processing process
        try:
            self.command_queue.put({
                'type': 'update_settings',
                'camera_settings': camera_settings,
                'processing_settings': processing_settings
            })
            self.add_log_message("Settings applied successfully")
            
            # Log the settings for debugging
            self.add_log_message(f"Camera: Exposure={camera_settings['exposure_time']}ms, Gain={camera_settings['gain']}dB")
            self.add_log_message(f"Processing: Ball Sensitivity={processing_settings['ball_sensitivity']}%")
            
        except Exception as e:
            self.add_log_message(f"Failed to apply settings: {e}")
    
    @Slot()
    def reset_settings(self):
        """Reset all settings to default values"""
        # Reset camera settings
        self.exposure_time_input.setValue(10.0)
        self.gain_input.setValue(1.0)
        self.wb_red_input.setValue(1.0)
        self.wb_blue_input.setValue(1.0)
        self.brightness_input.setValue(0)
        self.contrast_input.setValue(1.0)
        self.gamma_input.setValue(1.0)
        self.framerate_input.setValue(30)
        
        # Reset processing settings
        self.ball_sensitivity_input.setValue(50)
        self.ball_size_min_input.setValue(5)
        self.ball_size_max_input.setValue(50)
        self.field_threshold_input.setValue(128)
        self.kalman_strength_input.setValue(75)
        self.goal_sensitivity_input.setValue(60)
        self.processing_resolution_input.setCurrentIndex(1)
        
        self.add_log_message("Settings reset to defaults")
    
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

