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
    
    def __init__(self):
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
        self.gpu_preprocessor = None  # Wird lazy im Tracking-Thread erstellt
        self.cpu_preprocessor = CPUPreprocessor(config.CAMERA_CALIBRATION_FILE)
        self.camera_calibration.initialize_for_size((config.DETECTION_WIDTH, config.DETECTION_HEIGHT))

        self.enable_undistortion = True
        self.use_gpu_processing = False  # Standardmäßig CPU verwenden
        
    def initialize_camera(self):
        """Initialisiert die Kamera mit Fehlerbehandlung"""
        try:
            # Wenn Kamera bereits vorhanden ist, zuerst stoppen
            if self.camera is not None:
                try:
                    print("Stoppe vorhandene Kamera vor Neuinitialisierung...")
                    self.camera.stop()
                    time.sleep(0.2)  # Kurz warten
                except:
                    pass  # Ignoriere Fehler beim Stoppen
                self.camera = None
            
            print("Initialisiere neue Kamera...")
            self.camera = IDS_Camera()
            self.camera_available = True
            print("Kamera erfolgreich initialisiert")
            return True
        except Exception as e:
            print(f"Fehler beim Initialisieren der Kamera: {e}")
            self.camera_available = False
            self.camera = None
            return False
    
    def frame_reader_thread_method(self):
        """Frame reading thread - reads from camera"""
        while self.running:
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
        print("Stoppe alle Tracker-Threads...")
        self.running = False
        
        # Send termination signals to worker threads
        try:
            self.frame_queue.put(None)
            self.frame_queue.put(None)
        except:
            pass
        
        # Warten bis Frame-Reader-Thread gestoppt ist (wichtig für Kamera)
        if self.frame_reader_thread and self.frame_reader_thread.is_alive():
            print("Warte auf Frame-Reader-Thread...")
            self.frame_reader_thread.join(timeout=2.0)
            if self.frame_reader_thread.is_alive():
                print("Warning: Frame-Reader-Thread konnte nicht gestoppt werden")
            else:
                print("Frame-Reader-Thread gestoppt")
        
        if self.ball_thread and self.ball_thread.is_alive():
            print("Warte auf Ball-Thread...")
            self.ball_thread.join(timeout=1.0)
            if self.ball_thread.is_alive():
                print("Warning: Ball-Thread konnte nicht gestoppt werden")
            else:
                print("Ball-Thread gestoppt")
            
        if self.field_thread and self.field_thread.is_alive():
            print("Warte auf Field-Thread...")
            self.field_thread.join(timeout=1.0)
            if self.field_thread.is_alive():
                print("Warning: Field-Thread konnte nicht gestoppt werden")
            else:
                print("Field-Thread gestoppt")
            
        # Kamera stoppen falls vorhanden
        if self.camera_available and self.camera is not None:
            try:
                print("Stoppe Kamera...")
                self.camera.stop()
                print("Kamera gestoppt")
                # Kurz warten damit alle Kamera-Threads beendet werden
                time.sleep(0.5)
                # Kamera-Objekt auf None setzen für sauberen Neustart
                self.camera = None
                self.camera_available = False
                print("Kamera-Objekt zurückgesetzt")
            except Exception as e:
                print(f"Fehler beim Stoppen der Kamera: {e}")
                # Auch bei Fehler Kamera zurücksetzen
                self.camera = None
                self.camera_available = False
    
    def toggle_processing_mode(self):
        """Toggles between CPU and GPU processing"""
        self.use_gpu_processing = not self.use_gpu_processing
        print(f"Processing mode switched to: {'GPU' if self.use_gpu_processing else 'CPU'}")


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
                        self.msleep(5)
                        continue
                    bayer_frame = self.tracker.current_bayer_frame.copy()
                
                frame_count += 1
                
                # Frame verarbeiten - Robuste Fehlerbehandlung
                frame = None
                
                if self.tracker.use_gpu_processing:
                    try:
                        # GPU-Preprocessor lazy im Tracking-Thread erstellen
                        if self.tracker.gpu_preprocessor is None:
                            self.tracker.gpu_preprocessor = GPUPreprocessor((1440, 1080), (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
                            print("GPU-Preprocessor im Tracking-Thread erstellt")
                        
                        frame = self.tracker.gpu_preprocessor.process_frame(bayer_frame)
                    except Exception as e:
                        self.log_message.emit(f"GPU-Verarbeitungsfehler: {e}")
                        self.tracker.use_gpu_processing = False
                        frame = None
                
                # CPU-Verarbeitung wenn GPU deaktiviert oder fehlgeschlagen
                if frame is None:
                    try:
                        # CPU-Preprocessor erwartet ein Tuple (frame, undistorted_frame)
                        frame_result = self.tracker.cpu_preprocessor.process_frame(bayer_frame)
                        if isinstance(frame_result, tuple) and len(frame_result) >= 1:
                            frame = frame_result[0]  # Erstes Element ist das verarbeitete Frame
                        else:
                            frame = frame_result  # Falls nur ein Frame zurückgegeben wird
                    except Exception as cpu_e:
                        self.log_message.emit(f"CPU-Verarbeitungsfehler: {cpu_e}")
                        # Notfall-Fallback: Frame direkt verwenden
                        try:
                            if len(bayer_frame.shape) == 3 and bayer_frame.shape[2] == 3:
                                # Frame ist bereits BGR
                                frame = cv2.resize(bayer_frame, (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
                            elif len(bayer_frame.shape) == 2:
                                # Grayscale zu BGR konvertieren
                                bgr_frame = cv2.cvtColor(bayer_frame, cv2.COLOR_GRAY2BGR)
                                frame = cv2.resize(bgr_frame, (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
                            else:
                                # Bayer zu BGR konvertieren
                                bgr_frame = cv2.cvtColor(bayer_frame, cv2.COLOR_BayerRG2BGR)
                                frame = cv2.resize(bgr_frame, (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
                        except Exception as fallback_e:
                            self.log_message.emit(f"Notfall-Fallback fehlgeschlagen: {fallback_e}")
                            continue
                
                # Nur weitermachen wenn Frame erfolgreich verarbeitet wurde und gültig ist
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    continue
                
                # Zusätzliche Validierung: Frame sollte 3-Kanal BGR sein
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    self.log_message.emit(f"Warning: Frame hat unerwartete Form: {frame.shape}, überspringe...")
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
                
                # Display-Rate für Kamera anpassen
                display_interval = 8  # Updates für Kamera
                
                # Nur jeden n-ten Frame für Anzeige
                display_counter += 1
                if display_counter % display_interval == 0:
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
                        score_text = f"{self.tracker.goal_scorer.get_score()['player1']}:{self.tracker.goal_scorer.get_score()['player2']}"
                        self.score_update.emit(score_text)
                        
                        
                    except Exception as vis_e:
                        self.log_message.emit(f"Visualisierungsfehler: {vis_e}")
                        # Weiter ohne Visualisierung
                
        except Exception as e:
            self.log_message.emit(f"Kritischer Tracking-Fehler: {e}")
        finally:
            # Cleanup in der richtigen Reihenfolge
            print("Beginne Cleanup...")
            
            # 1. Zuerst Tracker-Threads stoppen
            self.tracker.stop_threads()
            print("Tracker-Threads gestoppt")
            
            # 2. GPU-Preprocessor cleanup
            if self.tracker.gpu_preprocessor is not None:
                try:
                    self.tracker.gpu_preprocessor.close()
                    self.tracker.gpu_preprocessor = None
                    print("GPU-Preprocessor geschlossen")
                except Exception as e:
                    print(f"Fehler beim Schließen des GPU-Preprocessors: {e}")
            
            # 3. Dann Kamera stoppen
            if self.tracker.camera_available and self.tracker.camera is not None:
                try:
                    print("Stoppe Kamera aus Tracking-Thread...")
                    self.tracker.camera.stop()
                    print("Kamera aus Tracking-Thread gestoppt")
                    # Kurz warten damit alle Kamera-Threads beendet werden
                    time.sleep(0.5)
                    # Kamera-Objekt auf None setzen für sauberen Neustart
                    self.tracker.camera = None
                    self.tracker.camera_available = False
                    print("Kamera-Objekt aus Tracking-Thread zurückgesetzt")
                except Exception as e:
                    self.log_message.emit(f"Fehler beim Stoppen der Kamera: {e}")
                    # Auch bei Fehler Kamera zurücksetzen
                    self.tracker.camera = None
                    self.tracker.camera_available = False
            
            self.camera_status_update.emit("Kamera: Gestoppt")
            self.log_message.emit("Tracking beendet")
            print("Cleanup abgeschlossen")
    
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
        
        # Match Control Buttons - Container für Button-Verwaltung
        self.match_buttons_widget = QWidget()
        match_buttons_layout = QVBoxLayout(self.match_buttons_widget)
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
        
        score_layout.addWidget(self.big_score_label, stretch=4)
        score_layout.addWidget(self.match_buttons_widget, stretch=1)
        
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
        
        self.ball_only_btn.clicked.connect(lambda: self.set_visualization_mode(1))
        self.field_only_btn.clicked.connect(lambda: self.set_visualization_mode(2))
        self.combined_btn.clicked.connect(lambda: self.set_visualization_mode(3))
        
        self.gpu_checkbox.toggled.connect(self.toggle_gpu_processing)
        self.reset_score_btn.clicked.connect(self.reset_score)
        self.start_match_btn.clicked.connect(self.start_match)
        self.cancel_match_btn.clicked.connect(self.cancel_match)
        
    @Slot()
    def start_tracking(self):
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
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.video_label.setText("Tracking gestoppt")
            self.video_label.setPixmap(QPixmap())
            
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