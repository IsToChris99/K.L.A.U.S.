import cv2
import numpy as np
import time
from threading import Thread, Lock
from queue import Queue, LifoQueue, Empty
import sys
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSizePolicy,
    QGroupBox, QCheckBox, QFileDialog, QRadioButton, QButtonGroup
)

# Local imports
from detection.ball_detector import BallDetector  
from detection.field_detector import FieldDetector
from analysis.goal_scorer import GoalScorer
from input.ids_camera import IDS_Camera
from processing.cpu_preprocessor import CPUPreprocessor
from processing.gpu_preprocessor import GPUPreprocessor
import config

# ================== COMBINED TRACKER ==================

class CombinedTracker:
    """Combined Ball and Field Tracker with Multithreading"""
    
    def __init__(self, video_path=None, use_webcam=False):
        self.count = 0
        
        self.ball_tracker = BallDetector()
        self.field_detector = FieldDetector()
        self.goal_scorer = GoalScorer()
        
        # Try to load saved calibration
        if not self.field_detector.load_calibration():
            print("No calibration file found. Perform manual calibration...")
        
        # Calibration mode - only activate on key press
        self.calibration_mode = False
        self.calibration_requested = False
        
        # Initialize IDS Camera later when needed (not in constructor)
        self.camera = None
        self.camera_available = False
        
        # Video file support
        self.video_capture = None
        self.video_path = video_path
        self.use_video_file = video_path is not None
        
        # Visualization modes
        self.BALL_ONLY = 1
        self.FIELD_ONLY = 2  
        self.COMBINED = 3
        self.visualization_mode = self.COMBINED
        
        # Threading variables
        self.ball_thread = None
        self.field_thread = None
        self.frame_reader_thread = None
        self.running = False
        self.frame_queue = LifoQueue(maxsize=1) 
        self.result_lock = Lock()
        
        self.current_frame = None
        self.current_bayer_frame = None
        self.ball_result = None
        self.field_data = None
        
        # Display control variables
        self.frame_count = 0
        self.processing_fps = 0
        self.last_fps_time = time.time()
        self.last_frame_count = 0

        # Camera calibration
        self.camera_calibration = CPUPreprocessor(config.CAMERA_CALIBRATION_FILE)
        self.gpu_preprocessor = GPUPreprocessor((1440, 1080), (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
        self.cpu_preprocessor = CPUPreprocessor(config.CAMERA_CALIBRATION_FILE)
        self.camera_calibration.initialize_for_size((config.DETECTION_WIDTH, config.DETECTION_HEIGHT))

        self.enable_undistortion = True
        self.use_gpu_processing = False  # Standardmäßig CPU verwenden
        
    def initialize_camera(self):
        """Initialisiert die Kamera mit Fehlerbehandlung"""
        try:
            if self.camera is None:
                self.camera = IDS_Camera()
            self.camera_available = True
            return True
        except Exception as e:
            print(f"Fehler beim Initialisieren der Kamera: {e}")
            self.camera_available = False
            return False
    
    def initialize_video(self, video_path):
        """Initialisiert Video-Capture für Datei-Wiedergabe"""
        try:
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                return False
            self.video_path = video_path
            self.use_video_file = True
            return True
        except Exception as e:
            print(f"Fehler beim Öffnen der Video-Datei: {e}")
            return False
        
    def frame_reader_thread_method(self):
        """Frame reading thread - reads from camera or video file"""
        while self.running:
            if self.use_video_file:
                # Video-Datei Modus
                if self.video_capture is None:
                    time.sleep(0.1)
                    continue
                    
                try:
                    ret, frame = self.video_capture.read()
                    if not ret:
                        # Video zu Ende - neu starten
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    # Store video frame directly (already BGR)
                    with self.result_lock:
                        self.current_bayer_frame = frame
                    
                    self.count += 1
                    #time.sleep(1/30)  # 30 FPS für Video-Wiedergabe
                except Exception as e:
                    print(f"Fehler beim Lesen des Video-Frames: {e}")
                    time.sleep(0.1)
                    
            else:
                # Kamera Modus
                if not self.camera_available or self.camera is None:
                    time.sleep(0.1)
                    continue
                    
                try:
                    bayer_frame, metadata = self.camera.get_frame()
                    if bayer_frame is None:
                        continue

                    # Store raw Bayer frame
                    with self.result_lock:
                        self.current_bayer_frame = bayer_frame

                    self.count += 1
                except Exception as e:
                    print(f"Fehler beim Lesen des Frames: {e}")
                    time.sleep(0.1)
        
    def ball_tracking_thread(self):
        """Thread for Ball-Tracking"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break
            except Empty:
                continue
                
            # Field bounds for restricted ball search
            field_bounds = None
            goals = []
            if self.field_data and self.field_data['calibrated']:
                if self.field_data['field_bounds']:
                    field_bounds = self.field_data['field_bounds']
                if self.field_data['goals']:
                    goals = self.field_data['goals']

            # Ball detection with field_bounds
            detection_result = self.ball_tracker.detect_ball(frame, field_bounds)
            self.ball_tracker.update_tracking(detection_result, field_bounds)
            
            # Goal scoring system update
            ball_position = detection_result[0] if detection_result[0] is not None else None
            self.goal_scorer.update_ball_tracking(
                ball_position, 
                goals, 
                field_bounds, 
                self.ball_tracker.missing_counter
            )
            
            with self.result_lock:
                self.ball_result = {
                    'detection': detection_result,
                    'smoothed_pts': list(self.ball_tracker.smoothed_pts),
                    'missing_counter': self.ball_tracker.missing_counter
                }
                
    def field_tracking_thread(self):
        """Thread for Field-Tracking"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break
            except Empty:
                continue
            
            # Use FieldDetector's calibration logic
            if (self.calibration_requested and 
                self.calibration_mode and 
                not self.field_detector.calibrated):
                self.field_detector.calibrate(frame)
            
            # Store current field data
            with self.result_lock:
                self.field_data = {
                    'calibrated': self.field_detector.calibrated,
                    'field_contour': self.field_detector.field_contour,
                    'field_corners': self.field_detector.field_corners,
                    'field_bounds': self.field_detector.field_bounds,
                    'field_rect_points': self.field_detector.field_rect_points,
                    'goals': self.field_detector.goals,
                    'stable_counter': self.field_detector.stable_detection_counter,
                    'calibration_mode': self.calibration_mode,
                    'calibration_requested': self.calibration_requested
                }
    
    def draw_ball_visualization(self, frame):
        """Draws ball visualization"""
        with self.result_lock:
            ball_result_copy = self.ball_result.copy() if self.ball_result else None
        
        if ball_result_copy is None:
            return

        detection = ball_result_copy['detection']
        smoothed_pts = ball_result_copy['smoothed_pts']
        missing_counter = ball_result_copy['missing_counter']

        # Draw ball info
        if detection[0] is not None:
            center, radius, confidence, velocity = detection

            # Color selection based on confidence
            if confidence >= 0.8:
                color = config.COLOR_BALL_HIGH_CONFIDENCE
            elif confidence >= 0.6:
                color = config.COLOR_BALL_MED_CONFIDENCE
            else:
                color = config.COLOR_BALL_LOW_CONFIDENCE

            cv2.circle(frame, center, 3, color, -1)
            cv2.circle(frame, center, int(radius), color, 2)

            # Show Kalman velocity
            if velocity is not None:
                cv2.arrowedLine(frame, center,
                            (int(center[0] + velocity[0]*30), int(center[1] + velocity[1]*30)),
                            (255, 0, 255), 2)

        # Ball trail drawing
        for i in range(1, len(smoothed_pts)):
            if smoothed_pts[i - 1] is None or smoothed_pts[i] is None:
                continue
            thickness = int(np.sqrt(config.BALL_TRAIL_MAX_LENGTH / float(i + 1)) * config.BALL_TRAIL_THICKNESS_FACTOR)
            cv2.line(frame, smoothed_pts[i - 1], smoothed_pts[i], config.COLOR_BALL_TRAIL, thickness)
    
    def draw_field_visualization(self, frame):
        """Draws field visualization"""
        with self.result_lock:
            field_data_copy = self.field_data.copy() if self.field_data else None
        
        if field_data_copy is None:
            return

        # Field contour
        if field_data_copy['calibrated'] and field_data_copy['field_contour'] is not None:
            cv2.drawContours(frame, [field_data_copy['field_contour']], -1, config.COLOR_FIELD_CONTOUR, 3)

        # Field corners
        if field_data_copy['field_corners'] is not None:
            for i, corner in enumerate(field_data_copy['field_corners']):
                cv2.circle(frame, tuple(corner), 8, config.COLOR_FIELD_CORNERS, -1)

        # Goals
        for i, goal in enumerate(field_data_copy['goals']):
            x, y, w, h = goal['bounds']
            cv2.rectangle(frame, (x, y), (x+w, y+h), config.COLOR_GOALS, 2)

        # Field limits
        if (field_data_copy['calibrated'] and 
            field_data_copy.get('field_rect_points') is not None):
            cv2.drawContours(frame, [field_data_copy['field_rect_points']], -1, config.COLOR_FIELD_BOUNDS, 2)
        elif field_data_copy['calibrated'] and field_data_copy['field_bounds']:
            x, y, w, h = field_data_copy['field_bounds']
            cv2.rectangle(frame, (x, y), (x+w, y+h), config.COLOR_FIELD_BOUNDS, 2)

        # Calibration progress
        if (field_data_copy['calibration_requested'] and 
            field_data_copy['calibration_mode'] and 
            not field_data_copy['calibrated']):
            progress = min(field_data_copy['stable_counter'] / 30, 1.0)
            progress_width = int(300 * progress)
            
            cv2.rectangle(frame, (10, 130), (310, 160), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 130), (10 + progress_width, 160), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 130), (310, 160), (255, 255, 255), 2)
    
    def start_threads(self):
        """Starts the tracking threads"""
        if self.running:
            return
            
        self.running = True
        
        self.frame_reader_thread = Thread(target=self.frame_reader_thread_method, daemon=True)
        self.frame_reader_thread.start()
        
        self.ball_thread = Thread(target=self.ball_tracking_thread, daemon=True)
        self.ball_thread.start()
            
        self.field_thread = Thread(target=self.field_tracking_thread, daemon=True)
        self.field_thread.start()

    def stop_threads(self):
        """Stops the tracking threads"""
        self.running = False
        
        # Send termination signals to worker threads
        try:
            self.frame_queue.put(None)
            self.frame_queue.put(None)
        except:
            pass
        
        if self.frame_reader_thread and self.frame_reader_thread.is_alive():
            self.frame_reader_thread.join(timeout=1.0)
        
        if self.ball_thread and self.ball_thread.is_alive():
            self.ball_thread.join(timeout=1.0)
            
        if self.field_thread and self.field_thread.is_alive():
            self.field_thread.join(timeout=1.0)
            
        # Kamera stoppen falls vorhanden
        if self.camera_available and self.camera is not None:
            try:
                self.camera.stop()
            except Exception as e:
                print(f"Fehler beim Stoppen der Kamera: {e}")
        
        # Video-Capture stoppen falls vorhanden
        if self.video_capture is not None:
            try:
                self.video_capture.release()
                self.video_capture = None
            except Exception as e:
                print(f"Fehler beim Stoppen des Videos: {e}")
    
    def toggle_processing_mode(self):
        """Toggles between CPU and GPU processing"""
        self.use_gpu_processing = not self.use_gpu_processing
        
        if self.use_gpu_processing:
            try:
                self.gpu_preprocessor.force_reinitialize()
            except Exception as e:
                print(f"Failed to reinitialize GPU, falling back to CPU: {e}")
                self.use_gpu_processing = False


# ================== GUI COMPONENTS ==================
    
class KickerTrackingThread(QThread):
    """Thread für die gesamte Kicker-Verarbeitung"""
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
        """Führt die gesamte Tracking-Logik aus"""
        self._running = True
        
        # Input-Quelle initialisieren (Kamera oder Video)
        if self.tracker.use_video_file:
            if not self.tracker.video_capture or not self.tracker.video_capture.isOpened():
                self.log_message.emit("Fehler: Video-Datei konnte nicht geöffnet werden")
                self.camera_status_update.emit("Video: Nicht verfügbar")
                return
            self.camera_status_update.emit("Video: Aktiv")
        else:
            # Kamera initialisieren
            if not self.tracker.initialize_camera():
                self.log_message.emit("Fehler: Keine Kamera verfügbar")
                self.camera_status_update.emit("Kamera: Nicht verfügbar")
                return
            
            # Kamera starten
            try:
                self.tracker.camera.start()
                self.camera_status_update.emit("Kamera: Aktiv")
            except Exception as e:
                self.log_message.emit(f"Fehler beim Starten der Kamera: {e}")
                return
        
        # Tracking-Threads starten
        try:
            self.tracker.start_threads()
            
            # Kurz warten bis die Quelle stabilisiert ist
            #self.msleep(500)  # 500ms warten
            
            self.log_message.emit("Tracking gestartet")
            
            frame_count = 0
            last_fps_time = time.time()
            display_counter = 0
            
            # Hauptschleife ohne try-catch, damit kleine Fehler nicht das Tracking beenden
            while self._running:
                # Frame von Kamera holen
                with self.tracker.result_lock:
                    if self.tracker.current_bayer_frame is None:
                        #self.msleep(5)
                        continue
                    bayer_frame = self.tracker.current_bayer_frame.copy()
                
                frame_count += 1
                
                # Frame verarbeiten - Robuste Fehlerbehandlung
                frame = None
                try:
                    if self.tracker.use_video_file:
                        # Bei Video-Dateien ist das Frame bereits RGB/BGR
                        frame = bayer_frame
                    else:
                        # Bei Kamera: Bayer-Frame verarbeiten
                        if self.tracker.use_gpu_processing:
                            frame = self.tracker.gpu_preprocessor.process_frame(bayer_frame)
                        else:
                            frame, _ = self.tracker.cpu_preprocessor.process_frame(bayer_frame)
                except Exception as e:
                    # Bei GPU-Fehler: Auf CPU umschalten und nochmal versuchen
                    if self.tracker.use_gpu_processing and not self.tracker.use_video_file:
                        self.log_message.emit(f"GPU-Fehler, wechsle zu CPU: {e}")
                        self.tracker.use_gpu_processing = False
                        try:
                            frame, _ = self.tracker.cpu_preprocessor.process_frame(bayer_frame)
                        except Exception as cpu_e:
                            self.log_message.emit(f"CPU-Verarbeitung fehlgeschlagen: {cpu_e}")
                            continue  # Frame überspringen, nicht das ganze Tracking beenden
                    else:
                        self.log_message.emit(f"Frame-Verarbeitung fehlgeschlagen: {e}")
                        continue  # Frame überspringen, nicht das ganze Tracking beenden
                
                # Nur weitermachen wenn Frame erfolgreich verarbeitet wurde
                if frame is None:
                    continue
                
                # Frame für Analyse-Threads bereitstellen
                with self.tracker.result_lock:
                    self.tracker.current_frame = frame
                
                # Frame in Queue für Worker-Threads
                if not self.tracker.frame_queue.full():
                    self.tracker.frame_queue.put(frame)
                else:
                    try:
                        self.tracker.frame_queue.get_nowait()
                        self.tracker.frame_queue.put(frame)
                    except:
                        pass
                
                # FPS berechnen
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    self.fps_update.emit(fps)
                    frame_count = 0
                    last_fps_time = current_time
                
                # Nur jeden 8. Frame für Anzeige (ca. 30 FPS bei 250 FPS Verarbeitung)
                display_counter += 1
                if display_counter % 8 == 0:
                    try:
                        # Visualisierungen hinzufügen
                        display_frame = frame.copy()
                        
                        if self.tracker.visualization_mode in [self.tracker.BALL_ONLY, self.tracker.COMBINED]:
                            self.tracker.draw_ball_visualization(display_frame)
                            
                        if self.tracker.visualization_mode in [self.tracker.FIELD_ONLY, self.tracker.COMBINED]:
                            self.tracker.draw_field_visualization(display_frame)
                        
                        # Frame an GUI senden
                        self.frame_ready.emit(display_frame)
                        
                        # Score-Update
                        score_text = f"{self.tracker.goal_scorer.score_left}:{self.tracker.goal_scorer.score_right}"
                        self.score_update.emit(score_text)
                    except Exception as vis_e:
                        self.log_message.emit(f"Visualisierungsfehler: {vis_e}")
                        # Weiter ohne Visualisierung
                
        except Exception as e:
            self.log_message.emit(f"Kritischer Tracking-Fehler: {e}")
        finally:
            # Cleanup
            self.tracker.stop_threads()
            if self.tracker.use_video_file:
                if self.tracker.video_capture is not None:
                    try:
                        self.tracker.video_capture.release()
                        self.tracker.video_capture = None
                    except Exception as e:
                        self.log_message.emit(f"Fehler beim Stoppen des Videos: {e}")
            else:
                if self.tracker.camera_available and self.tracker.camera is not None:
                    try:
                        self.tracker.camera.stop()
                    except Exception as e:
                        self.log_message.emit(f"Fehler beim Stoppen der Kamera: {e}")
            
            status_text = "Video: Gestoppt" if self.tracker.use_video_file else "Kamera: Gestoppt"
            self.camera_status_update.emit(status_text)
            self.log_message.emit("Tracking beendet")
    
    def stop(self):
        self._running = False
        self.wait(2000)

class KickerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kicker Klaus - Tracking System")
        self.resize(1400, 900)
        
        # Tracker initialisieren
        self.tracker = CombinedTracker()
        
        self.tracking_thread = None
        self.setup_ui()
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        
        # Oberer Bereich: Großer Score-Anzeige
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
        
        self.reset_score_btn = QPushButton("Score zurücksetzen")
        self.reset_score_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 10px 20px;
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
        
        score_layout.addWidget(self.big_score_label, stretch=4)
        score_layout.addWidget(self.reset_score_btn, stretch=1)
        
        # Mittlerer Bereich: Video und Controls
        content_layout = QHBoxLayout()
        
        # Linke Seite: Video
        video_layout = QVBoxLayout()
        self.video_label = QLabel("Kamera nicht gestartet")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        self.video_label.setMinimumSize(800, 600)
        video_layout.addWidget(self.video_label)
        
        # Rechte Seite: Controls
        control_layout = QVBoxLayout()
        
        # Tracking Controls
        tracking_group = QGroupBox("Tracking Controls")
        tracking_layout = QVBoxLayout(tracking_group)
        
        # Input Mode Selection
        mode_group = QGroupBox("Input-Modus")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_button_group = QButtonGroup()
        self.live_mode_radio = QRadioButton("Live-Kamera")
        self.video_mode_radio = QRadioButton("Video-Datei")
        self.live_mode_radio.setChecked(True)  # Standard: Live-Modus
        
        self.mode_button_group.addButton(self.live_mode_radio, 0)
        self.mode_button_group.addButton(self.video_mode_radio, 1)
        
        mode_layout.addWidget(self.live_mode_radio)
        mode_layout.addWidget(self.video_mode_radio)
        
        # Video file selection
        self.video_file_layout = QHBoxLayout()
        self.video_path_label = QLabel("Keine Datei ausgewählt")
        self.video_path_label.setStyleSheet("font-style: italic; color: gray;")
        self.select_video_btn = QPushButton("Video auswählen")
        self.select_video_btn.setEnabled(False)  # Standardmäßig deaktiviert
        
        self.video_file_layout.addWidget(self.video_path_label, stretch=3)
        self.video_file_layout.addWidget(self.select_video_btn, stretch=1)
        
        mode_layout.addLayout(self.video_file_layout)
        tracking_layout.addWidget(mode_group)
        
        # Control buttons
        self.start_btn = QPushButton("Start Tracking")
        self.stop_btn = QPushButton("Stop Tracking")
        self.test_camera_btn = QPushButton("Kamera testen")
        self.calibrate_btn = QPushButton("Kalibrierung starten")
        
        self.stop_btn.setEnabled(False)
        
        tracking_layout.addWidget(self.start_btn)
        tracking_layout.addWidget(self.stop_btn)
        tracking_layout.addWidget(self.test_camera_btn)
        tracking_layout.addWidget(self.calibrate_btn)
        
        # Visualization Mode
        viz_group = QGroupBox("Anzeigemodus")
        viz_layout = QVBoxLayout(viz_group)
        
        self.ball_only_btn = QPushButton("Nur Ball")
        self.field_only_btn = QPushButton("Nur Spielfeld")
        self.combined_btn = QPushButton("Kombiniert")
        
        self.combined_btn.setStyleSheet("background-color: lightgreen;")
        
        viz_layout.addWidget(self.ball_only_btn)
        viz_layout.addWidget(self.field_only_btn)
        viz_layout.addWidget(self.combined_btn)
        
        # Processing Settings
        settings_group = QGroupBox("Einstellungen")
        settings_layout = QVBoxLayout(settings_group)
        
        self.gpu_checkbox = QCheckBox("GPU-Verarbeitung")
        self.gpu_checkbox.setChecked(False)  # Standardmäßig CPU verwenden
        settings_layout.addWidget(self.gpu_checkbox)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.camera_status_label = QLabel("Kamera: Nicht initialisiert")
        self.fps_label = QLabel("FPS: 0.0")
        
        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.fps_label)
        
        # Log Output
        log_group = QGroupBox("System-Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Layouts zusammenfügen
        control_layout.addWidget(tracking_group)
        control_layout.addWidget(viz_group)
        control_layout.addWidget(settings_group)
        control_layout.addWidget(status_group)
        control_layout.addWidget(log_group)
        control_layout.addStretch()
        
        content_layout.addLayout(video_layout, stretch=3)
        content_layout.addLayout(control_layout, stretch=1)
        
        # Hauptlayout zusammensetzen
        main_layout.addWidget(score_group)
        main_layout.addLayout(content_layout, stretch=4)
        
        # Signal-Verbindungen
        self.connect_signals()
        
    def connect_signals(self):
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.test_camera_btn.clicked.connect(self.test_camera)
        self.calibrate_btn.clicked.connect(self.start_calibration)
        
        # Mode selection
        self.live_mode_radio.toggled.connect(self.on_mode_changed)
        self.video_mode_radio.toggled.connect(self.on_mode_changed)
        self.select_video_btn.clicked.connect(self.select_video_file)
        
        self.ball_only_btn.clicked.connect(lambda: self.set_visualization_mode(1))
        self.field_only_btn.clicked.connect(lambda: self.set_visualization_mode(2))
        self.combined_btn.clicked.connect(lambda: self.set_visualization_mode(3))
        
        self.gpu_checkbox.toggled.connect(self.toggle_gpu_processing)
        self.reset_score_btn.clicked.connect(self.reset_score)
        
    @Slot()
    def start_tracking(self):
        if self.tracking_thread is None or not self.tracking_thread.isRunning():
            # Prüfen ob Video-Modus und Datei ausgewählt
            if self.video_mode_radio.isChecked():
                if not hasattr(self.tracker, 'video_capture') or self.tracker.video_capture is None:
                    self.add_log_message("Fehler: Keine Video-Datei ausgewählt")
                    return
            
            self.tracking_thread = KickerTrackingThread(self.tracker)
            self.tracking_thread.frame_ready.connect(self.update_frame)
            self.tracking_thread.log_message.connect(self.add_log_message)
            self.tracking_thread.fps_update.connect(self.update_fps)
            self.tracking_thread.score_update.connect(self.update_score)
            self.tracking_thread.camera_status_update.connect(self.update_camera_status)
            self.tracking_thread.start()
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            mode_text = "Video-Tracking" if self.video_mode_radio.isChecked() else "Live-Tracking"
            self.add_log_message(f"{mode_text} gestartet")
        
    @Slot()
    def stop_tracking(self):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.video_label.setText("Tracking gestoppt")
            self.video_label.setPixmap(QPixmap())
            
            if self.video_mode_radio.isChecked():
                self.camera_status_label.setText("Video: Gestoppt")
            else:
                self.camera_status_label.setText("Kamera: Gestoppt")
            self.add_log_message("Tracking gestoppt")
    
    @Slot()
    def test_camera(self):
        if self.tracker.initialize_camera():
            self.camera_status_label.setText("Kamera: Verfügbar")
            self.add_log_message("Kamera erfolgreich getestet")
        else:
            self.camera_status_label.setText("Kamera: Nicht verfügbar")
            self.add_log_message("Kamera-Test fehlgeschlagen")
    
    @Slot()
    def on_mode_changed(self):
        """Wird aufgerufen wenn der Input-Modus geändert wird"""
        is_video_mode = self.video_mode_radio.isChecked()
        
        # Video-Auswahl-Button aktivieren/deaktivieren
        self.select_video_btn.setEnabled(is_video_mode)
        
        # Kamera-Test-Button aktivieren/deaktivieren
        self.test_camera_btn.setEnabled(not is_video_mode)
        
        # Modus im Tracker setzen
        if hasattr(self, 'tracker'):
            self.tracker.use_video_file = is_video_mode
            
        mode_text = "Video-Modus" if is_video_mode else "Live-Modus"
        self.add_log_message(f"Modus gewechselt zu: {mode_text}")
    
    @Slot()
    def select_video_file(self):
        """Öffnet Dialog zur Video-Datei-Auswahl"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Video-Datei auswählen",
            "",
            "Video-Dateien (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;Alle Dateien (*)"
        )
        
        if file_path:
            # Video-Datei im Tracker initialisieren
            if self.tracker.initialize_video(file_path):
                self.video_path_label.setText(f"Datei: {file_path.split('/')[-1]}")
                self.video_path_label.setStyleSheet("color: green;")
                self.add_log_message(f"Video-Datei ausgewählt: {file_path}")
                self.camera_status_label.setText("Video: Bereit")
            else:
                self.video_path_label.setText("Fehler beim Laden der Datei")
                self.video_path_label.setStyleSheet("color: red;")
                self.add_log_message("Fehler beim Laden der Video-Datei")
                self.camera_status_label.setText("Video: Fehler")
    
    @Slot()
    def start_calibration(self):
        if self.tracker:
            self.tracker.field_detector.calibrated = False
            self.tracker.field_detector.stable_detection_counter = 0
            self.tracker.calibration_mode = True
            self.tracker.calibration_requested = True
            self.add_log_message("Kalibrierung gestartet")
    
    @Slot(int)
    def set_visualization_mode(self, mode):
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
        if self.tracker:
            self.tracker.use_gpu_processing = enabled
            mode = "GPU" if enabled else "CPU"
            self.add_log_message(f"Verarbeitung umgeschaltet auf: {mode}")
    
    @Slot()
    def reset_score(self):
        if self.tracker:
            self.tracker.goal_scorer.reset_score()
            self.add_log_message("Score zurückgesetzt")
    
    @Slot(np.ndarray)
    def update_frame(self, frame):
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
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    @Slot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    @Slot(str)
    def update_score(self, score):
        self.big_score_label.setText(score.replace(":", " : "))
    
    @Slot(str)
    def update_camera_status(self, status):
        self.camera_status_label.setText(status)
    
    def closeEvent(self, event):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
        event.accept()

  

# ================== MAIN PROGRAM ==================

def main_gui():
    app = QApplication(sys.argv)
    
    # GUI auch ohne Kamera starten
    try:
        window = KickerMainWindow()
        window.show()
        window.add_log_message("GUI gestartet - Kamera-Status wird getestet...")
        
        # Initial Kamera-Test (ohne Fehler abzubrechen)
        QApplication.processEvents()  # GUI anzeigen bevor Test
        window.test_camera()
        
        return app.exec()
    except Exception as e:
        print(f"Fehler beim Starten der GUI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main_gui())