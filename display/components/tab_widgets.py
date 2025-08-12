"""
Tab Widgets für die Hauptanwendung
"""

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QPushButton, QLabel, QFormLayout, QSpinBox, 
    QDoubleSpinBox, QSlider, QComboBox, QCheckBox
)

from .clickable_video_label import ClickableVideoLabel
from config import (FRAME_RATE_TARGET, EXPOSURE_TIME, GAIN, BLACK_LEVEL, WHITE_BALANCE_AUTO)


class TrackingTab(QWidget):
    """Tab für das Haupttracking Interface"""
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the tracking tab UI"""
        main_layout = QVBoxLayout(self)
        
        # Score section
        score_group = self.parent_window.create_score_section()
        
        # Content area: Video and Controls
        content_layout = QHBoxLayout()
        
        # Left side: Video
        video_layout = self.parent_window.create_video_section()
        
        # Right side: Controls
        control_layout = self.parent_window.create_control_section()
        
        content_layout.addLayout(video_layout, stretch=2)
        content_layout.addLayout(control_layout, stretch=1)
        
        # Main layout assembly
        main_layout.addWidget(score_group)
        main_layout.addLayout(content_layout, stretch=4)

class StatisticsTab(QWidget):
    """Tab für die Statistiken"""

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()

    def setup_ui(self):
        """Setup the statistics tab UI"""
        main_layout = QVBoxLayout(self)
        
        

        # Add widgets for statistics here
        # Example: main_layout.addWidget(QLabel("Statistics will be displayed here"))

        self.setLayout(main_layout)


class CalibrationTab(QWidget):
    """Tab für die Farbkalibrierung"""
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the calibration tab UI"""
        main_layout = QVBoxLayout(self)
        
        # Main container for calibration
        calibration_container = QGroupBox("Color Calibration and Field Detection")
        calibration_container.setStyleSheet("""
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
        
        # Horizontal layout for video and info
        calibration_horizontal_layout = QHBoxLayout(calibration_container)
        
        # Left side: Video with color picker
        left_video_layout = QVBoxLayout()
        
        # Video label with color picker
        self.parent_window.calibration_video_label = ClickableVideoLabel()
        self.parent_window.calibration_video_label.setText("Click on video to pick colors - Waiting for video stream...")
        self.parent_window.calibration_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.parent_window.calibration_video_label.setStyleSheet("border: 1px solid #9f9f9f; border-radius: 5px;")
        self.parent_window.calibration_video_label.setMinimumSize(720, 540)  # 4:3 aspect ratio, larger for calibration
        self.parent_window.calibration_video_label.setScaledContents(False)
        
        # Connect color picker signal
        self.parent_window.calibration_video_label.color_picked.connect(self.parent_window.on_calibration_color_picked)
        
        left_video_layout.addWidget(self.parent_window.calibration_video_label)
        
        # Right side: Color information and controls
        right_info_layout = self.create_color_info_section()
        
        # Add layouts to horizontal layout
        calibration_horizontal_layout.addLayout(left_video_layout, stretch=2)
        calibration_horizontal_layout.addLayout(right_info_layout, stretch=1)
        
        main_layout.addWidget(calibration_container)
        main_layout.addStretch()
    
    def create_color_info_section(self):
        """Erstellt den Farbinformations-Bereich"""
        right_info_layout = QVBoxLayout()
        
        # HSV Color Info Box
        self.parent_window.color_info_group = QGroupBox("Selected Color Information")
        self.parent_window.color_info_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: normal;
                border: 1px solid #9f9f9f;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 5px 0 5px;
                color: white;
            }
        """)
        
        color_info_layout = QVBoxLayout(self.parent_window.color_info_group)
        
        # Color preview
        self.parent_window.color_preview = QLabel()
        self.parent_window.color_preview.setFixedSize(60, 60)
        self.parent_window.color_preview.setStyleSheet("border: 1px solid #9f9f9f; border-radius: 5px;")
        self.parent_window.color_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.parent_window.color_preview.setText("No color\nselected")
        color_info_layout.addWidget(self.parent_window.color_preview)        
        
        # HSV Values
        self.parent_window.hsv_label = QLabel("HSV: Not selected")
        self.parent_window.hsv_label.setStyleSheet("font-family: monospace; font-size: 12px; margin: 2px;")
        color_info_layout.addWidget(self.parent_window.hsv_label)
                
        # HSV Range suggestion
        self.parent_window.hsv_range_label = QLabel("HSV Range: Not calculated")
        self.parent_window.hsv_range_label.setStyleSheet("font-family: monospace; font-size: 10px; margin: 2px; color: #FF6B6B;")
        self.parent_window.hsv_range_label.setWordWrap(True)
        color_info_layout.addWidget(self.parent_window.hsv_range_label)
        
        right_info_layout.addWidget(self.parent_window.color_info_group)
        
        # Calibration controls
        calibration_controls_group = self.create_calibration_controls()
        right_info_layout.addWidget(calibration_controls_group)
        
        # Instructions
        instructions_label = QLabel("Click on the video to select colors for detectors.")
        instructions_label.setStyleSheet("color: lightgrey; font-style: italic; margin: 5px;")
        instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_info_layout.addWidget(instructions_label)
        right_info_layout.addStretch()
        
        return right_info_layout
    
    def create_calibration_controls(self):
        """Erstellt die Kalibrierungssteuerung"""
        calibration_controls_group = QGroupBox("Calibration Controls")
        calibration_controls_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: normal;
                border: 1px solid #9f9f9f;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 5px 0 5px;
                color: white;
            }
        """)
        
        controls_layout = QVBoxLayout(calibration_controls_group)
        
        # Placeholder buttons
        self.parent_window.save_field_color_btn = QPushButton("Save Field Color")
        self.parent_window.save_field_color_btn.setEnabled(False)  # Disabled until color is picked
        
        self.parent_window.save_ball_color_btn = QPushButton("Save Ball Color")
        self.parent_window.save_ball_color_btn.setEnabled(False)
        
        self.parent_window.reset_calibration_btn = QPushButton("Reset Calibration")        
        
        controls_layout.addWidget(self.parent_window.save_field_color_btn)
        controls_layout.addWidget(self.parent_window.save_ball_color_btn)
        controls_layout.addWidget(self.parent_window.reset_calibration_btn)
        
        return calibration_controls_group


class SettingsTab(QWidget):
    """Tab für die Kamera- und System-Einstellungen"""
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the settings tab UI"""
        main_layout = QVBoxLayout(self)

        # Frame for Camera & System Settings
        settings_container = QGroupBox("Camera and System Settings")
        settings_container.setStyleSheet("""
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

        # Horizontal layout for left (settings) and right (video) areas
        settings_horizontal_layout = QHBoxLayout(settings_container)
        
        # Left side: Settings groups
        left_settings_layout = QVBoxLayout()
        camera_group = self.create_camera_settings_group()
        processing_group = self.create_processing_settings_group()
        left_settings_layout.addWidget(camera_group)
        left_settings_layout.addWidget(processing_group)
        left_settings_layout.addStretch()
        
        # Right side: Video preview and free space
        right_preview_layout = self.create_preview_section()
        
        # Add left and right areas to horizontal layout
        settings_horizontal_layout.addLayout(left_settings_layout, stretch=1)
        settings_horizontal_layout.addLayout(right_preview_layout, stretch=1)

        main_layout.addWidget(settings_container)

        # Action Buttons below the frame
        button_layout = self.create_settings_buttons()
        main_layout.addLayout(button_layout)
        main_layout.addStretch()
    
    def create_camera_settings_group(self):
        """Erstellt die Kamera-Einstellungsgruppe"""
        camera_group = QGroupBox("Camera Settings")
        camera_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: normal;
                border: 1px solid #9f9f9f;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 5px 0 5px;
                color: white;
            }
        """)
        layout = QFormLayout(camera_group)
        layout.setSpacing(15)

        # ---- Werte aus config übernehmen ----
        # Frame Rate (bleibt SpinBox da es Integer-Werte sind)
        self.parent_window.framerate_input = QSpinBox()
        self.parent_window.framerate_input.setRange(1, 250)
        self.parent_window.framerate_input.setValue(FRAME_RATE_TARGET)
        self.parent_window.framerate_input.setSuffix(" fps")
        layout.addRow("Target Frame Rate:", self.parent_window.framerate_input)

        # Exposure Time - jetzt als Slider
        self.parent_window.exposure_time_input = QSlider(Qt.Orientation.Horizontal)
        self.parent_window.exposure_time_input.setRange(1000, 40000)  # 100.0 bis 4000.0 * 10 für bessere Auflösung
        self.parent_window.exposure_time_input.setValue(int(EXPOSURE_TIME * 10))
        self.parent_window.exposure_time_label = QLabel(f"{EXPOSURE_TIME} ms")
        self.parent_window.exposure_time_input.valueChanged.connect(
            lambda v: self.parent_window.exposure_time_label.setText(f"{v/10:.1f} ms")
        )
        
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(self.parent_window.exposure_time_input)
        exposure_layout.addWidget(self.parent_window.exposure_time_label)
        layout.addRow("Exposure Time:", exposure_layout)

        # Gain - jetzt als Slider
        self.parent_window.gain_input = QSlider(Qt.Orientation.Horizontal)
        self.parent_window.gain_input.setRange(0, 400)  # 0.0 bis 40.0 * 10 für bessere Auflösung
        self.parent_window.gain_input.setValue(int(GAIN * 10))
        self.parent_window.gain_label = QLabel(f"{GAIN} dB")
        self.parent_window.gain_input.valueChanged.connect(
            lambda v: self.parent_window.gain_label.setText(f"{v/10:.1f} dB")
        )
        
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(self.parent_window.gain_input)
        gain_layout.addWidget(self.parent_window.gain_label)
        layout.addRow("Gain:", gain_layout)

        # Black Level - jetzt als Slider
        self.parent_window.black_level_input = QSlider(Qt.Orientation.Horizontal)
        self.parent_window.black_level_input.setRange(0, 200)  # 0.0 bis 20.0 * 10 für bessere Auflösung
        self.parent_window.black_level_input.setValue(int(BLACK_LEVEL * 10))
        self.parent_window.black_level_label = QLabel(f"{BLACK_LEVEL}")
        self.parent_window.black_level_input.valueChanged.connect(
            lambda v: self.parent_window.black_level_label.setText(f"{v/10:.1f}")
        )
        
        black_level_layout = QHBoxLayout()
        black_level_layout.addWidget(self.parent_window.black_level_input)
        black_level_layout.addWidget(self.parent_window.black_level_label)
        layout.addRow("Black Level:", black_level_layout)

        # White Balance Toggle Off and Automatic - jetzt mit "Once" Button rechts neben Checkbox
        self.parent_window.wb_auto_checkbox = QCheckBox("Automatic")
        wb = WHITE_BALANCE_AUTO
        if wb == "Off" or wb == "Once":
            self.parent_window.wb_auto_checkbox.setChecked(False)
        elif wb == "Continuous":
            self.parent_window.wb_auto_checkbox.setChecked(True)
        
        self.parent_window.wb_one_time_checkbox = QPushButton("Once")
        
        wb_layout = QHBoxLayout()
        wb_layout.addWidget(self.parent_window.wb_auto_checkbox)
        wb_layout.addWidget(self.parent_window.wb_one_time_checkbox)
        wb_layout.addStretch()  # Fügt Freiraum rechts hinzu
        layout.addRow("White Balance:", wb_layout)

        return camera_group
    
    def create_processing_settings_group(self):
        """Erstellt die Verarbeitungseinstellungsgruppe"""
        processing_group = QGroupBox("Processing Settings")
        processing_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: normal;
                border: 1px solid #9f9f9f;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 5px 0 5px;
                color: white;
            }
        """)
        
        layout = QFormLayout(processing_group)
        layout.setSpacing(15)
        
        # Ball Detection Sensitivity
        self.parent_window.ball_sensitivity_input = QSlider(Qt.Orientation.Horizontal)
        self.parent_window.ball_sensitivity_input.setRange(1, 100)
        self.parent_window.ball_sensitivity_input.setValue(50)
        self.parent_window.ball_sensitivity_label = QLabel("50%")
        self.parent_window.ball_sensitivity_input.valueChanged.connect(
            lambda v: self.parent_window.ball_sensitivity_label.setText(f"{v}%")
        )
        
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(self.parent_window.ball_sensitivity_input)
        sensitivity_layout.addWidget(self.parent_window.ball_sensitivity_label)
        layout.addRow("Ball Detection Sensitivity:", sensitivity_layout)
        
        # Ball Size Range Min
        self.parent_window.ball_size_min_input = QSpinBox()
        self.parent_window.ball_size_min_input.setRange(1, 50)
        self.parent_window.ball_size_min_input.setValue(5)
        self.parent_window.ball_size_min_input.setSuffix(" px")
        layout.addRow("Min Ball Size:", self.parent_window.ball_size_min_input)
        
        # Ball Size Range Max
        self.parent_window.ball_size_max_input = QSpinBox()
        self.parent_window.ball_size_max_input.setRange(10, 200)
        self.parent_window.ball_size_max_input.setValue(50)
        self.parent_window.ball_size_max_input.setSuffix(" px")
        layout.addRow("Max Ball Size:", self.parent_window.ball_size_max_input)
        
        # Field Detection Threshold
        self.parent_window.field_threshold_input = QSlider(Qt.Orientation.Horizontal)
        self.parent_window.field_threshold_input.setRange(1, 255)
        self.parent_window.field_threshold_input.setValue(128)
        self.parent_window.field_threshold_label = QLabel("128")
        self.parent_window.field_threshold_input.valueChanged.connect(
            lambda v: self.parent_window.field_threshold_label.setText(str(v))
        )
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.parent_window.field_threshold_input)
        threshold_layout.addWidget(self.parent_window.field_threshold_label)
        layout.addRow("Field Detection Threshold:", threshold_layout)
        
        # Kalman Filter Strength
        self.parent_window.kalman_strength_input = QSlider(Qt.Orientation.Horizontal)
        self.parent_window.kalman_strength_input.setRange(1, 100)
        self.parent_window.kalman_strength_input.setValue(75)
        self.parent_window.kalman_strength_label = QLabel("75%")
        self.parent_window.kalman_strength_input.valueChanged.connect(
            lambda v: self.parent_window.kalman_strength_label.setText(f"{v}%")
        )
        
        kalman_layout = QHBoxLayout()
        kalman_layout.addWidget(self.parent_window.kalman_strength_input)
        kalman_layout.addWidget(self.parent_window.kalman_strength_label)
        layout.addRow("Kalman Filter Strength:", kalman_layout)
        
        # Goal Detection Sensitivity
        self.parent_window.goal_sensitivity_input = QSlider(Qt.Orientation.Horizontal)
        self.parent_window.goal_sensitivity_input.setRange(1, 100)
        self.parent_window.goal_sensitivity_input.setValue(60)
        self.parent_window.goal_sensitivity_label = QLabel("60%")
        self.parent_window.goal_sensitivity_input.valueChanged.connect(
            lambda v: self.parent_window.goal_sensitivity_label.setText(f"{v}%")
        )
        
        goal_layout = QHBoxLayout()
        goal_layout.addWidget(self.parent_window.goal_sensitivity_input)
        goal_layout.addWidget(self.parent_window.goal_sensitivity_label)
        layout.addRow("Goal Detection Sensitivity:", goal_layout)
        
        # Processing Resolution
        self.parent_window.processing_resolution_input = QComboBox()
        self.parent_window.processing_resolution_input.addItems([
            "720x540 (Fast)",
            "960x720 (Balanced)",
            "1440x1080 (Quality)"
        ])
        self.parent_window.processing_resolution_input.setCurrentIndex(1)
        layout.addRow("Processing Resolution:", self.parent_window.processing_resolution_input)
        
        return processing_group
    
    def create_preview_section(self):
        """Erstellt den Vorschaubereich"""
        right_preview_layout = QVBoxLayout()
        
        # Video preview (upper right)
        self.parent_window.settings_video_label = QLabel("Live Preview - Waiting for video stream...")
        self.parent_window.settings_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.parent_window.settings_video_label.setStyleSheet("border: 1px solid #9f9f9f; border-radius: 5px; margin-top: 1ex; padding-top: 10px;")
        self.parent_window.settings_video_label.setMinimumSize(360, 270)  # 4:3 aspect ratio, smaller than main video
        self.parent_window.settings_video_label.setScaledContents(False)
        
        # Free space (lower right) - placeholder for future features
        free_space_widget = QWidget()
        free_space_widget.setStyleSheet("border: 1px dashed #9f9f9f; border-radius: 5px; margin-bottom: 10px;")
        free_space_label = QLabel("Reserved for future features")
        free_space_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        free_space_label.setStyleSheet("color: #888888; font-style: italic;")
        free_space_layout = QVBoxLayout(free_space_widget)
        free_space_layout.addWidget(free_space_label)
        
        right_preview_layout.addWidget(self.parent_window.settings_video_label, stretch=3)
        right_preview_layout.addWidget(free_space_widget, stretch=1)
        
        return right_preview_layout
    
    def create_settings_buttons(self):
        """Erstellt die Einstellungs-Buttons"""
        button_layout = QHBoxLayout()
        self.parent_window.apply_settings_btn = QPushButton("Apply Settings")
        self.parent_window.apply_settings_btn.setStyleSheet("""
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
        self.parent_window.reset_settings_btn = QPushButton("Reset to Defaults")
        self.parent_window.reset_settings_btn.setStyleSheet("""
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
        button_layout.addWidget(self.parent_window.reset_settings_btn)
        button_layout.addWidget(self.parent_window.apply_settings_btn)
        button_layout.addStretch()
        
        return button_layout


class AboutTab(QWidget):
    """Tab für die About-Sektion"""
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the about tab UI"""
        about_layout = QVBoxLayout(self)
        
        # Headline
        about_headline = QLabel("K.L.A.U.S. - Kicker Live Analytics Universal System\nVersion 1.0")
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
        team_container = self.create_team_info()
        about_layout.addWidget(team_container)
        about_layout.addStretch()
    
    def create_team_info(self):
        """Erstellt den Team-Informationsbereich"""
        team_container = QGroupBox("Development Team")
        team_container.setStyleSheet("""
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
        
        team_layout = QVBoxLayout(team_container)
        
        # Team members
        developers_label = QLabel("Tim Vesper • Roman Heck • Jose Pineda • Joshua Siemer • Christian Gunter")
        developers_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        developers_label.setStyleSheet("font-size: 16px; margin: 10px; font-weight: bold;")
        
        # Roles
        roles = [
            ("Processing Architecture", "Tim Vesper"),
            ("Choice of Camera and Lens", "Christian Gunter"),
            ("IDS Camera Integration", "Tim Vesper"),
            ("Camera Calibration", "Christian Gunter"),
            ("GUI Design & Implementation", "Christian Gunter"),
            ("Color Picker and Calibration", "Roman Heck"),
            ("Field & Goal Detection", "Joshua Siemer"),
            ("Player Detection", "Roman Heck"),
            ("Ball Tracking", "Joshua Siemer"),
            ("Ball Speed", "Jose Pineda"),
            ("Ball Heatmap", "Jose Pineda"),
            (" ", " "),
            ("General Development and Improvement", "All Team Members")
        ]
        
        team_layout.addWidget(developers_label)
        team_layout.addWidget(QLabel())  # Spacer
        
        for role, person in roles:
            role_label = QLabel(f"{role}: {person}")
            role_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            role_label.setStyleSheet("font-size: 14px; margin: 5px;")
            team_layout.addWidget(role_label)
        
        return team_container
