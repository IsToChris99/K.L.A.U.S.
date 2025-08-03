import cv2
import numpy as np
import time
import config
from processing.gpu_preprocessor import GPUPreprocessor
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSizePolicy,
    QGroupBox, QCheckBox
)


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
    
    def __init__(self, tracker):
        super().__init__()
        self.setWindowTitle("Kicker Klaus - Tracking System")
        self.resize(1400, 900)
        
        # Tracker received from outside
        self.tracker = tracker
        self.tracking_thread = None
        
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
        """Erstellt den Score-Bereich"""
        score_group = QGroupBox("SPIELSTAND")
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
        
        self.big_score_label = QLabel("0 : 0")
        self.big_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.big_score_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: #2E7D32;
                background-color: #E8F5E8;
                border: 2px solid #4CAF50;
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
            }
        """)
        self.big_score_label.setMinimumHeight(120)
        
        # Match Control Buttons
        self.match_buttons_widget = self.create_match_buttons()
        
        score_layout.addWidget(self.big_score_label, stretch=4)
        score_layout.addWidget(self.match_buttons_widget, stretch=1)
        
        return score_group
    
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
        self.reset_score_btn = QPushButton("Score zurücksetzen")
        self.reset_score_btn.setStyleSheet("""
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
        self.reset_score_btn.setMaximumWidth(200)
        self.reset_score_btn.hide()  # Initial versteckt
        
        # Cancel Match Button (initial versteckt)
        self.cancel_match_btn = QPushButton("Match abbrechen")
        self.cancel_match_btn.setStyleSheet("""
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
        self.video_label = QLabel("Kamera nicht gestartet")
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
        self.combined_btn = QPushButton("Kombiniert")
        
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
        """Erstellt den Status-Bereich"""
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.camera_status_label = QLabel("Camera: Not initialized")
        self.fps_label = QLabel("FPS: 0.0")
        
        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.fps_label)
        
        return status_group
    
    def create_log_section(self):
        """Erstellt den Log-Bereich"""
        log_group = QGroupBox("System-Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        return log_group
        
    def connect_signals(self):
        """Verbindet alle Button-Signale mit ihren Funktionen"""
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
        
    # ============= SIGNAL HANDLER METHODEN =============
    
    @Slot()
    def start_tracking(self):
        """Startet das Tracking"""
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
            
            self.add_log_message("Live-Tracking gestartet")
        
    @Slot()
    def stop_tracking(self):
        """Stoppt das Tracking"""
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
            mode_names = {1: "Ball", 2: "Spielfeld", 3: "Kombiniert"}
            
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
            
            self.add_log_message(f"Anzeigemodus: {mode_names.get(mode, 'Unbekannt')}")
    
    @Slot(bool)
    def toggle_gpu_processing(self, enabled):
        """Schaltet zwischen CPU und GPU um"""
        if self.tracker:
            self.tracker.use_gpu_processing = enabled
            mode = "GPU" if enabled else "CPU"
            self.add_log_message(f"Verarbeitung umgeschaltet auf: {mode}")
    
    @Slot()
    def reset_score(self):
        """Setzt den Score zurück"""
        if self.tracker:
            self.tracker.goal_scorer.reset_score()
            self.add_log_message("Score zurückgesetzt")
    
    @Slot()
    def start_match(self):
        """Startet ein Match und zeigt die Match-Buttons an"""
        # Start Match Button verstecken
        self.start_match_btn.hide()
        
        # Score Reset und Cancel Match Buttons anzeigen
        self.reset_score_btn.show()
        self.cancel_match_btn.show()
        
        self.add_log_message("Match gestartet")
    
    @Slot()
    def cancel_match(self):
        """Bricht das Match ab und zeigt wieder den Start Match Button"""
        # Score Reset und Cancel Match Buttons verstecken
        self.reset_score_btn.hide()
        self.cancel_match_btn.hide()
        
        # Start Match Button wieder anzeigen
        self.start_match_btn.show()
        
        self.add_log_message("Match abgebrochen")
    
    # ============= UPDATE METHODEN =============
    
    @Slot(np.ndarray)
    def update_frame(self, frame):
        """Aktualisiert das angezeigte Frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)
    
    @Slot(str)
    def add_log_message(self, message):
        """Fügt eine Log-Nachricht hinzu"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    @Slot(float)
    def update_fps(self, fps):
        """Aktualisiert die FPS-Anzeige"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    @Slot(str)
    def update_score(self, score):
        """Aktualisiert die Score-Anzeige"""
        self.big_score_label.setText(score.replace(":", " : "))
    
    @Slot(str)
    def update_camera_status(self, status):
        """Aktualisiert den Kamera-Status"""
        self.camera_status_label.setText(status)
    
    def closeEvent(self, event):
        """Wird beim Schließen des Fensters aufgerufen"""
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
        event.accept()
