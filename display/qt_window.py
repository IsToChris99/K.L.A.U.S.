import cv2
import numpy as np
import time
import config
from processing.gpu_preprocessor import GPUPreprocessor
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QColor, QMouseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSizePolicy,
    QGroupBox, QCheckBox, QComboBox
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


class KickerTrackingThread(QThread):
    """Thread for the entire kicker processing"""
    frame_ready = Signal(np.ndarray)
    log_message = Signal(str)
    fps_update = Signal(float)
    score_update = Signal(str)
    camera_status_update = Signal(str)
    
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self._running = False
        
    def run(self):
        """Executes the complete tracking logic"""
        self._running = True
        
        # Initialize camera
        if not self.tracker.initialize_camera():
            self.log_message.emit("Error: No camera available")
            self.camera_status_update.emit("Camera: Not available")
            return
        
        # Start camera
        try:
            self.tracker.camera.start()
            self.camera_status_update.emit("Camera: Active")
        except Exception as e:
            self.log_message.emit(f"Error starting camera: {e}")
            return
        
        # Start tracking threads
        try:
            self.tracker.start_threads()
            self.log_message.emit("Tracking started")
            
            frame_count = 0
            last_fps_time = time.time()
            display_counter = 0
            
            # Main loop
            while self._running:
                # Get frame from camera
                with self.tracker.result_lock:
                    if self.tracker.current_bayer_frame is None:
                        self.msleep(5)
                        continue
                    bayer_frame = self.tracker.current_bayer_frame.copy()
                
                frame_count += 1
                
                # Process frame - robust error handling
                frame = None
                
                if self.tracker.use_gpu_processing:
                    try:
                        # Create GPU preprocessor lazily in tracking thread
                        if self.tracker.gpu_preprocessor is None:
                            self.tracker.gpu_preprocessor = GPUPreprocessor((1440, 1080), (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
                            print("GPU preprocessor created in tracking thread")
                        
                        frame = self.tracker.gpu_preprocessor.process_frame(bayer_frame)
                    except Exception as e:
                        self.log_message.emit(f"GPU processing error: {e}")
                        self.tracker.use_gpu_processing = False
                        frame = None
                
                # CPU processing if GPU disabled or failed
                if frame is None:
                    try:
                        # CPU preprocessor expects a tuple (frame, undistorted_frame)
                        frame_result = self.tracker.cpu_preprocessor.process_frame(bayer_frame)
                        if isinstance(frame_result, tuple) and len(frame_result) >= 1:
                            frame = frame_result[0]  # First element is the processed frame
                        else:
                            frame = frame_result  # If only one frame is returned
                    except Exception as cpu_e:
                        self.log_message.emit(f"CPU processing error: {cpu_e}")
                        # Emergency fallback: use frame directly
                        try:
                            if len(bayer_frame.shape) == 3 and bayer_frame.shape[2] == 3:
                                # Frame is already BGR
                                frame = cv2.resize(bayer_frame, (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
                            elif len(bayer_frame.shape) == 2:
                                # Convert grayscale to BGR
                                bgr_frame = cv2.cvtColor(bayer_frame, cv2.COLOR_GRAY2BGR)
                                frame = cv2.resize(bgr_frame, (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
                            else:
                                # Convert Bayer to BGR
                                bgr_frame = cv2.cvtColor(bayer_frame, cv2.COLOR_BayerRG2BGR)
                                frame = cv2.resize(bgr_frame, (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
                        except Exception as fallback_e:
                            self.log_message.emit(f"Emergency fallback failed: {fallback_e}")
                            continue
                
                # Only continue if frame was processed successfully and is valid
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    continue
                
                # Additional validation: frame should be 3-channel BGR
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    self.log_message.emit(f"Warning: Frame has unexpected shape: {frame.shape}, skipping...")
                    continue
                
                # Provide frame for analysis threads
                with self.tracker.result_lock:
                    self.tracker.current_frame = frame
                
                # Put frame in queue for worker threads
                if not self.tracker.frame_queue.full():
                    self.tracker.frame_queue.put(frame)
                else:
                    try:
                        self.tracker.frame_queue.get_nowait()
                        self.tracker.frame_queue.put(frame)
                    except:
                        pass
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    self.fps_update.emit(fps)
                    frame_count = 0
                    last_fps_time = current_time
                
                # Adjust display rate for camera
                display_interval = 8  # Updates for camera
                
                # Only every nth frame for display
                display_counter += 1
                if display_counter % display_interval == 0:
                    try:
                        # Add visualizations
                        display_frame = frame.copy()
                        
                        if self.tracker.visualization_mode in [self.tracker.BALL_ONLY, self.tracker.COMBINED]:
                            self.tracker.draw_ball_visualization(display_frame)
                            
                        if self.tracker.visualization_mode in [self.tracker.FIELD_ONLY, self.tracker.COMBINED]:
                            self.tracker.draw_field_visualization(display_frame)
                        
                        # Send frame to GUI
                        self.frame_ready.emit(display_frame)
                        
                        # Score update
                        score_text = f"{self.tracker.goal_scorer.get_score()['player1']}:{self.tracker.goal_scorer.get_score()['player2']}"
                        self.score_update.emit(score_text)
                        
                    except Exception as vis_e:
                        self.log_message.emit(f"Visualization error: {vis_e}")
                        # Continue without visualization
                
        except Exception as e:
            self.log_message.emit(f"Critical tracking error: {e}")
        finally:
            # Cleanup in the correct order
            print("Starting cleanup...")
            
            # 1. First stop tracker threads
            self.tracker.stop_threads()
            print("Tracker threads stopped")
            
            # 2. GPU preprocessor cleanup
            if self.tracker.gpu_preprocessor is not None:
                try:
                    self.tracker.gpu_preprocessor.close()
                    self.tracker.gpu_preprocessor = None
                    print("GPU preprocessor closed")
                except Exception as e:
                    print(f"Error closing GPU preprocessor: {e}")
            
            # 3. Then stop camera
            if self.tracker.camera_available and self.tracker.camera is not None:
                try:
                    print("Stopping camera from tracking thread...")
                    self.tracker.camera.stop()
                    print("Camera stopped from tracking thread")
                    # Wait briefly for all camera threads to terminate
                    time.sleep(0.5)
                    # Reset camera object for clean restart
                    self.tracker.camera = None
                    self.tracker.camera_available = False
                    print("Camera object reset from tracking thread")
                except Exception as e:
                    self.log_message.emit(f"Error stopping camera: {e}")
                    # Reset camera even on error
                    self.tracker.camera = None
                    self.tracker.camera_available = False
            
            self.camera_status_update.emit("Camera: Stopped")
            self.log_message.emit("Tracking ended")
            print("Cleanup completed")
    
    def stop(self):
        self._running = False
        self.wait(2000)


class KickerMainWindow(QMainWindow):
    """Main window of the Kicker Klaus application"""
    
    def __init__(self, tracker, matcher):
        super().__init__()
        self.setWindowTitle("Kicker Klaus - Tracking System")
        self.resize(1400, 900)
        
        # Tracker received from outside
        self.tracker = tracker
        self.tracking_thread = None
        
        # Match modes instance received from outside
        self.matcher = matcher
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Creates the complete user interface"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        
        # Oberer Bereich: Score-Anzeige
        score_group = self.create_score_section()
        
        # Mittlerer Bereich: Video und Controls
        content_layout = QHBoxLayout()
        
        # Linke Seite: Video
        video_layout = self.create_video_section()
        
        # Rechte Seite: Controls
        control_layout = self.create_control_section()
        
        content_layout.addLayout(video_layout, stretch=3)
        content_layout.addLayout(control_layout, stretch=1)
        
        # Hauptlayout zusammensetzen
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
        
        # Game Mode Selection (links)
        game_mode_widget = self.create_game_mode_selection()
        
        # Score Display with embedded buttons (Mitte)
        score_widget = self.create_score_display_with_buttons()
        
        # Match Control Buttons (rechts)
        self.match_buttons_widget = self.create_match_buttons()
        
        score_layout.addWidget(game_mode_widget, stretch=1)  # Game mode selection
        score_layout.addWidget(score_widget, stretch=5)  # Main score area
        score_layout.addWidget(self.match_buttons_widget, stretch=1)  # Match controls
        
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
        
        # Use QHBoxLayout for horizontal arrangement (buttons on same level as score)
        main_layout = QHBoxLayout(score_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Team 1 buttons (left) - vertical arrangement
        team1_layout = QVBoxLayout()
        team1_layout.setSpacing(3)
        team1_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.team1_plus_btn = QPushButton("+")
        self.team1_minus_btn = QPushButton("-")
        
        # Team button styling
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
        
        # Team 2 buttons (right) - vertical arrangement
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
        main_layout.addWidget(self.big_score_label, stretch=1)  # Score takes most space
        main_layout.addLayout(team2_layout)
        
        return score_widget
    
    def create_game_mode_selection(self):
        """Creates the game mode selection widget"""
        game_mode_widget = QWidget()
        game_mode_layout = QVBoxLayout(game_mode_widget)
        game_mode_layout.setContentsMargins(0, 15, 0, 0)
        
        # Game Mode Label
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
        
        # Game Mode ComboBox
        self.game_mode_combo = QComboBox()
        self.game_mode_combo.addItems(["Normal Mode", "Practice Mode", "Tournament Mode"])
        self.game_mode_combo.setCurrentText("Normal Mode")  # Default selection
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
        
        # Start Match Button (initial sichtbar)
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
        
        # Score Reset Button (initial versteckt)
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
        self.reset_score_btn.hide()  # Initial versteckt
        
        # Cancel Match Button (initial versteckt)
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
        self.cancel_match_btn.hide()  # Initial versteckt
        
        # Buttons zum Layout hinzufügen
        match_buttons_layout.addWidget(self.start_match_btn)
        match_buttons_layout.addWidget(self.reset_score_btn)
        match_buttons_layout.addWidget(self.cancel_match_btn)
        match_buttons_layout.addStretch()
        
        return match_buttons_widget
    
    def create_video_section(self):
        """Erstellt den Video-Anzeige-Bereich"""
        video_layout = QVBoxLayout()
        self.video_label = ClickableVideoLabel()
        self.video_label.setText("Kamera nicht gestartet - Klicken Sie auf das Video, um Farben zu extrahieren")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        self.video_label.setMinimumSize(800, 600)
        video_layout.addWidget(self.video_label)
        
        return video_layout
    
    def create_control_section(self):
        """Erstellt den Control-Bereich"""
        control_layout = QVBoxLayout()
        
        # Tracking Controls
        tracking_group = self.create_tracking_controls()
        
        # Visualization Mode
        viz_group = self.create_visualization_controls()
        
        # Processing Settings
        settings_group = self.create_settings_controls()
        
        # Status
        status_group = self.create_status_section()
        
        # Log Output
        log_group = self.create_log_section()
        
        # Layouts zusammenfügen
        control_layout.addWidget(tracking_group)
        control_layout.addWidget(viz_group)
        control_layout.addWidget(settings_group)
        control_layout.addWidget(status_group)
        control_layout.addWidget(log_group)
        control_layout.addStretch()
        
        return control_layout
    
    def create_tracking_controls(self):
        """Creates the tracking control buttons"""
        tracking_group = QGroupBox("Tracking Controls")
        tracking_layout = QVBoxLayout(tracking_group)
        
        self.start_btn = QPushButton("Start Tracking")
        self.stop_btn = QPushButton("Stop Tracking")
        self.test_camera_btn = QPushButton("Test Camera")
        self.calibrate_btn = QPushButton("Start Calibration")
        
        self.stop_btn.setEnabled(False)
        
        tracking_layout.addWidget(self.start_btn)
        tracking_layout.addWidget(self.stop_btn)
        tracking_layout.addWidget(self.test_camera_btn)
        tracking_layout.addWidget(self.calibrate_btn)
        
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
        """Creates the settings buttons"""
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        self.gpu_checkbox = QCheckBox("GPU Processing")
        self.gpu_checkbox.setChecked(False)  # Use CPU by default
        settings_layout.addWidget(self.gpu_checkbox)
        
        return settings_group
    
    def create_status_section(self):
        """Creates the status section"""
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.camera_status_label = QLabel("Camera: Not initialized")
        self.fps_label = QLabel("FPS: 0.0")
        
        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.fps_label)
        
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
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.test_camera_btn.clicked.connect(self.test_camera)
        self.calibrate_btn.clicked.connect(self.start_calibration)
        
        self.ball_only_btn.clicked.connect(lambda: self.set_visualization_mode(1))
        self.field_only_btn.clicked.connect(lambda: self.set_visualization_mode(2))
        self.combined_btn.clicked.connect(lambda: self.set_visualization_mode(3))
        
        self.gpu_checkbox.toggled.connect(self.toggle_gpu_processing)
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
        
    # ============= SIGNAL HANDLER METHODEN =============
    
    @Slot()
    def start_tracking(self):
        """Starts the tracking"""
        if self.tracking_thread is None or not self.tracking_thread.isRunning():
            # Status zurücksetzen
            self.tracker.current_frame = None
            self.tracker.current_bayer_frame = None
            self.tracker.ball_result = None
            self.tracker.field_data = None
            
            self.tracking_thread = KickerTrackingThread(self.tracker)
            self.tracking_thread.frame_ready.connect(self.update_frame)
            self.tracking_thread.log_message.connect(self.add_log_message)
            self.tracking_thread.fps_update.connect(self.update_fps)
            self.tracking_thread.score_update.connect(self.update_score)
            self.tracking_thread.camera_status_update.connect(self.update_camera_status)
            self.tracking_thread.start()
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            self.add_log_message("Live Tracking started")
        
    @Slot()
    def stop_tracking(self):
        """Stops the tracking"""
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.video_label.setText("Tracking stopped")
            self.video_label.setPixmap(QPixmap())
            
            self.camera_status_label.setText("Camera: Stopped")
            self.add_log_message("Tracking stopped")

    @Slot()
    def test_camera(self):
        """Tests the camera"""
        if self.tracker.initialize_camera():
            self.camera_status_label.setText("Camera: Available")
            self.add_log_message("Camera successfully tested")
        else:
            self.camera_status_label.setText("Camera: Not available")
            self.add_log_message("Camera test failed")
    
    @Slot()
    def start_calibration(self):
        """Starts the calibration"""
        if self.tracker:
            self.tracker.field_detector.calibrated = False
            self.tracker.field_detector.stable_detection_counter = 0
            self.tracker.calibration_mode = True
            self.tracker.calibration_requested = True
            self.add_log_message("Calibration started")
    
    @Slot(int)
    def set_visualization_mode(self, mode):
        """Sets the visualization mode"""
        if self.tracker:
            self.tracker.visualization_mode = mode
            mode_names = {1: "Ball", 2: "Field", 3: "Combined"}
            
            # Button-Highlighting zurücksetzen
            for btn in [self.ball_only_btn, self.field_only_btn, self.combined_btn]:
                btn.setStyleSheet("")
            
            # Aktiven Button highlighten
            if mode == 1:
                self.ball_only_btn.setStyleSheet("background-color: lightgreen;")
            elif mode == 2:
                self.field_only_btn.setStyleSheet("background-color: lightgreen;")
            elif mode == 3:
                self.combined_btn.setStyleSheet("background-color: lightgreen;")

            self.add_log_message(f"Visualization mode set to: {mode_names.get(mode, 'Unknown')}")
    
    @Slot(bool)
    def toggle_gpu_processing(self, enabled):
        """Toggles between CPU and GPU processing"""
        if self.tracker:
            self.tracker.use_gpu_processing = enabled
            mode = "GPU" if enabled else "CPU"
            self.add_log_message(f"Processing changed to: {mode}")
    
    @Slot()
    def reset_score(self):
        """Resets the score"""
        if self.tracker:
            # Check if practice mode is active
            if self.matcher.get_mode() == "practice":
                self.add_log_message("Score reset disabled in Practice Mode")
                return
                
            self.tracker.goal_scorer.reset_score()
            self.update_score("0:0")
            self.add_log_message("Score resetted")
    
    @Slot()
    def team1_score_plus(self):
        """Increases Player 1 score by 1"""
        if self.tracker:
            # Check if practice mode is active
            if self.matcher.get_mode() == "practice":
                self.add_log_message("Score changes disabled in Practice Mode")
                return
                
            self.tracker.goal_scorer.player1_goals += 1
            self.update_score(f"{self.tracker.goal_scorer.player1_goals}:{self.tracker.goal_scorer.player2_goals}")
            if self.matcher.update_score(1, 1):
                self.add_log_message("Team 1 won!")
            self.add_log_message("Team 1 score +1 (manual)")

    @Slot()
    def team1_score_minus(self):
        """Decreases Player 1 score by 1"""
        if self.tracker:
            # Check if practice mode is active
            if self.matcher.get_mode() == "practice":
                self.add_log_message("Score changes disabled in Practice Mode")
                return
                
            if self.tracker.goal_scorer.player1_goals > 0:
                self.tracker.goal_scorer.player1_goals -= 1
                self.update_score(f"{self.tracker.goal_scorer.player1_goals}:{self.tracker.goal_scorer.player2_goals}")
                self.matcher.update_score(1, -1)
                self.add_log_message("Team 1 score -1 (manual)")
            else:
                self.add_log_message("Team 1 score cannot go below 0")

    @Slot()
    def team2_score_plus(self):
        """Increases Player 2 score by 1"""
        if self.tracker:
            # Check if practice mode is active
            if self.matcher.get_mode() == "practice":
                self.add_log_message("Score changes disabled in Practice Mode")
                return
            self.tracker.goal_scorer.player2_goals += 1
            self.update_score(f"{self.tracker.goal_scorer.player1_goals}:{self.tracker.goal_scorer.player2_goals}")
            if self.matcher.update_score(2, 1):
                self.add_log_message("Team 2 won!")
            self.add_log_message("Team 2 score +1 (manual)")

    @Slot()
    def team2_score_minus(self):
        """Decreases Player 2 score by 1"""
        if self.tracker:
            # Check if practice mode is active
            if self.matcher.get_mode() == "practice":
                self.add_log_message("Score changes disabled in Practice Mode")
                return
                
            if self.tracker.goal_scorer.player2_goals > 0:
                self.tracker.goal_scorer.player2_goals -= 1
                self.update_score(f"{self.tracker.goal_scorer.player1_goals}:{self.tracker.goal_scorer.player2_goals}")
                self.matcher.update_score(2, -1)
                self.add_log_message("Team 2 score -1 (manual)")
            else:
                self.add_log_message("Team 2 score cannot go below 0")

    @Slot()
    def start_match(self):
        """Starts a match and shows the match buttons"""
        # Start Match Button verstecken
        self.start_match_btn.hide()
        
        # Match-Modus initialisieren
        self.matcher.reset_scores()
        self.matcher.start_match()
        
        # Score Reset und Cancel Match Buttons anzeigen
        self.reset_score_btn.show()
        self.cancel_match_btn.show()

        self.add_log_message("Match started")

    @Slot()
    def cancel_match(self):
        """Cancels the current match and resets the score"""
        # Score Reset und Cancel Match Buttons verstecken
        self.reset_score_btn.hide()
        self.cancel_match_btn.hide()
        
        # Setzt den Score zurück
        self.matcher.end_match()
        self.tracker.goal_scorer.reset_score()
        
        # Setzt den Score-Anzeigetext zurück
        self.update_score("0:0")
        
        # Start Match Button wieder anzeigen
        self.start_match_btn.show()
        
        self.add_log_message("Match canceled")
    
    @Slot(str)
    def change_game_mode(self, mode_text):
        """Changes the game mode based on ComboBox selection"""
        # Map display text to internal mode keys
        mode_mapping = {
            "Normal Mode": "normal",
            "Practice Mode": "practice", 
            "Tournament Mode": "tournament"
        }
        
        if mode_text in mode_mapping:
            mode_key = mode_mapping[mode_text]
            self.matcher.set_mode(mode_key)
            
            # Get description from matcher class
            description = self.matcher.get_mode_description()
            self.add_log_message(f"Game mode changed to: {mode_text}")
            self.add_log_message(f"Description: {description}")
        else:
            self.add_log_message(f"Unknown game mode: {mode_text}")
    
    @Slot(int, int, int)
    def on_color_picked(self, r, g, b):
        """Handle color picking from video"""
        # Log the RGB values
        self.add_log_message(f"Color picked: RGB({r}, {g}, {b})")
        
        # Also log HSV values for better color analysis
        # Convert RGB to HSV for more intuitive color representation
        qcolor = QColor(r, g, b)
        h, s, v = qcolor.hsvHue(), qcolor.hsvSaturation(), qcolor.value()
        self.add_log_message(f"Color picked: HSV({h}, {s}, {v})")
        
        # Also log the hex representation
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.add_log_message(f"Color picked: HEX({hex_color})")
        
        # For field detection, we could store this color for later use
        # TODO: Implement field color calibration based on picked color
        if hasattr(self.tracker, 'field_detector'):
            # This could be extended to actually set the field color in the detector
            self.add_log_message("Tip: Diese Farbe könnte für die Spielfelderkennung verwendet werden")
    
    # ============= UPDATE METHODEN =============
    
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
    
    @Slot(str)
    def add_log_message(self, message):
        """Adds a log message"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    @Slot(float)
    def update_fps(self, fps):
        """Updates the FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    @Slot(str)
    def update_score(self, score):
        """Updates the score display"""
        self.big_score_label.setText(score.replace(":", " : "))
    
    @Slot(str)
    def update_camera_status(self, status):
        """Updates the camera status"""
        self.camera_status_label.setText(status)
    
    def closeEvent(self, event):
        """Called when the window is closed"""
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
        event.accept()
