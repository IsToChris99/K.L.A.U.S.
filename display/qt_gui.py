import sys
import cv2
import time
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSlider, QSizePolicy,
    QGroupBox, QCheckBox, QSpinBox
)

class KickerTrackingThread(QThread):
    """Thread für die gesamte Kicker-Verarbeitung"""
    frame_ready = Signal(np.ndarray)  # Verarbeitetes Frame für Anzeige
    log_message = Signal(str)
    fps_update = Signal(float)
    score_update = Signal(str)
    
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self._running = True
        
    def run(self):
        """Führt die gesamte Tracking-Logik aus"""
        try:
            # Kamera starten
            self.tracker.camera.start()
            self.tracker.start_threads()
            
            self.log_message.emit("Tracking gestartet")
            
            frame_count = 0
            last_fps_time = time.time()
            display_counter = 0
            
            while self._running:
                # Deine bestehende Verarbeitungslogik
                with self.tracker.result_lock:
                    if self.tracker.current_bayer_frame is None:
                        continue
                    bayer_frame = self.tracker.current_bayer_frame.copy()
                
                frame_count += 1
                
                # Frame verarbeiten
                if self.tracker.use_gpu_processing:
                    try:
                        frame = self.tracker.gpu_preprocessor.process_frame(bayer_frame)
                    except Exception as e:
                        self.log_message.emit(f"GPU-Fehler, wechsle zu CPU: {e}")
                        self.tracker.use_gpu_processing = False
                        frame, _ = self.tracker.cpu_preprocessor.process_frame(bayer_frame)
                else:
                    frame, _ = self.tracker.cpu_preprocessor.process_frame(bayer_frame)
                
                # Frame für Analyse-Threads bereitstellen
                with self.tracker.result_lock:
                    self.tracker.current_frame = frame
                
                if not self.tracker.frame_queue.full():
                    self.tracker.frame_queue.put(frame)
                
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
                    # Visualisierungen hinzufügen
                    display_frame = frame.copy()
                    
                    if self.tracker.visualization_mode in [self.tracker.BALL_ONLY, self.tracker.COMBINED]:
                        self.tracker.draw_ball_visualization(display_frame)
                        
                    if self.tracker.visualization_mode in [self.tracker.FIELD_ONLY, self.tracker.COMBINED]:
                        self.tracker.draw_field_visualization(display_frame)
                    
                    self.tracker.goal_scorer.draw_score_info(display_frame)
                    self.tracker.draw_status_info(display_frame)
                    
                    # Frame an GUI senden
                    self.frame_ready.emit(display_frame)
                    
                    # Score-Update
                    score_text = f"{self.tracker.goal_scorer.score_left}:{self.tracker.goal_scorer.score_right}"
                    self.score_update.emit(score_text)
                
        except Exception as e:
            self.log_message.emit(f"Tracking-Fehler: {e}")
        finally:
            self.tracker.stop_threads()
            self.tracker.camera.stop()
            self.log_message.emit("Tracking beendet")
    
    def stop(self):
        self._running = False
        self.wait(2000)  # Warte max. 2 Sekunden auf Thread-Ende

class KickerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kicker Klaus - Tracking System")
        self.resize(1400, 900)
        
        # Tracker importieren und initialisieren
        from mein_Testbereich.test.main_live_gui import CombinedTracker
        self.tracker = CombinedTracker()
        
        self.tracking_thread = None
        self.setup_ui()
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        
        # Linke Seite: Video
        video_layout = QVBoxLayout()
        self.video_label = QLabel("Kamera nicht gestartet")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        video_layout.addWidget(self.video_label)
        
        # Rechte Seite: Controls
        control_layout = QVBoxLayout()
        
        # Tracking Controls
        tracking_group = QGroupBox("Tracking Controls")
        tracking_layout = QVBoxLayout(tracking_group)
        
        self.start_btn = QPushButton("Start Tracking")
        self.stop_btn = QPushButton("Stop Tracking")
        self.calibrate_btn = QPushButton("Kalibrierung starten")
        self.screenshot_btn = QPushButton("Screenshot")
        
        tracking_layout.addWidget(self.start_btn)
        tracking_layout.addWidget(self.stop_btn)
        tracking_layout.addWidget(self.calibrate_btn)
        tracking_layout.addWidget(self.screenshot_btn)
        
        # Visualization Mode
        viz_group = QGroupBox("Anzeigemodus")
        viz_layout = QVBoxLayout(viz_group)
        
        self.ball_only_btn = QPushButton("Nur Ball")
        self.field_only_btn = QPushButton("Nur Spielfeld")
        self.combined_btn = QPushButton("Kombiniert")
        
        viz_layout.addWidget(self.ball_only_btn)
        viz_layout.addWidget(self.field_only_btn)
        viz_layout.addWidget(self.combined_btn)
        
        # Processing Settings
        settings_group = QGroupBox("Einstellungen")
        settings_layout = QVBoxLayout(settings_group)
        
        self.gpu_checkbox = QCheckBox("GPU-Verarbeitung")
        self.gpu_checkbox.setChecked(True)
        settings_layout.addWidget(self.gpu_checkbox)
        
        # Status und Score
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.fps_label = QLabel("FPS: 0.0")
        self.score_label = QLabel("Score: 0:0")
        self.reset_score_btn = QPushButton("Score zurücksetzen")
        
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.score_label)
        status_layout.addWidget(self.reset_score_btn)
        
        # Log Output
        log_group = QGroupBox("System-Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        # Layouts zusammenfügen
        control_layout.addWidget(tracking_group)
        control_layout.addWidget(viz_group)
        control_layout.addWidget(settings_group)
        control_layout.addWidget(status_group)
        control_layout.addWidget(log_group)
        control_layout.addStretch()
        
        main_layout.addLayout(video_layout, stretch=3)
        main_layout.addLayout(control_layout, stretch=1)
        
        # Signal-Verbindungen
        self.connect_signals()
        
    def connect_signals(self):
        # Button-Signale
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        
        # Visualization Mode
        self.ball_only_btn.clicked.connect(lambda: self.set_visualization_mode(1))
        self.field_only_btn.clicked.connect(lambda: self.set_visualization_mode(2))
        self.combined_btn.clicked.connect(lambda: self.set_visualization_mode(3))
        
        # Settings
        self.gpu_checkbox.toggled.connect(self.toggle_gpu_processing)
        self.reset_score_btn.clicked.connect(self.reset_score)
        
    @Slot()
    def start_tracking(self):
        if self.tracking_thread is None or not self.tracking_thread.isRunning():
            self.tracking_thread = KickerTrackingThread(self.tracker)
            self.tracking_thread.frame_ready.connect(self.update_frame)
            self.tracking_thread.log_message.connect(self.add_log_message)
            self.tracking_thread.fps_update.connect(self.update_fps)
            self.tracking_thread.score_update.connect(self.update_score)
            self.tracking_thread.start()
            self.add_log_message("Tracking gestartet")
        
    @Slot()
    def stop_tracking(self):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            self.add_log_message("Tracking gestoppt")
    
    @Slot()
    def start_calibration(self):
        if self.tracker:
            self.tracker.field_detector.calibrated = False
            self.tracker.field_detector.stable_detection_counter = 0
            self.tracker.calibration_mode = True
            self.tracker.calibration_requested = True
            self.add_log_message("Kalibrierung gestartet")
    
    @Slot()
    def take_screenshot(self):
        if self.tracker and self.tracker.current_frame is not None:
            timestamp = int(time.time())
            cv2.imwrite(f"kicker_screenshot_{timestamp}.jpg", self.tracker.current_frame)
            self.add_log_message(f"Screenshot gespeichert: kicker_screenshot_{timestamp}.jpg")
    
    @Slot(int)
    def set_visualization_mode(self, mode):
        if self.tracker:
            self.tracker.visualization_mode = mode
            mode_names = {1: "Ball", 2: "Spielfeld", 3: "Kombiniert"}
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
        # NumPy-Array zu QImage konvertieren
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Pixmap skalieren und anzeigen
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
    
    @Slot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    @Slot(str)
    def update_score(self, score):
        self.score_label.setText(f"Score: {score}")
    
    def closeEvent(self, event):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = KickerMainWindow()
    window.show()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
