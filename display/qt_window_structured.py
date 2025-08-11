"""
Überarbeitete Hauptfenster-Klasse mit strukturiertem Design
"""

import cv2
import numpy as np
import time
import sys
import os

# Add parent directory to path to find config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor, QMouseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSizePolicy,
    QGroupBox, QCheckBox, QComboBox, QTabWidget, QLineEdit,
    QFormLayout, QSpinBox, QDoubleSpinBox, QSlider
)

# Import neue Komponenten
from .components.tab_widgets import TrackingTab, CalibrationTab, SettingsTab, AboutTab
from .components.ui_components import ScoreSection, ControlSection, VideoSection
from .components.visualization_engine import VisualizationEngine
from .components.event_handlers import EventHandlers


class KickerMainWindow(QMainWindow):
    """Hauptfenster der Kicker Klaus Anwendung für Multiprocessing-Architektur"""
    
    def __init__(self, results_queue, command_queue, running_event):
        super().__init__()
        self.setWindowTitle("K.L.A.U.S. - Kicker Live Analytics Universal System (Structured)")
        self.resize(1280, 720)
        
        # Multi-processing communication
        self.results_queue = results_queue
        self.command_queue = command_queue
        self.running_event = running_event
        
        # Initialize components
        self.visualization_engine = VisualizationEngine()
        self.event_handlers = EventHandlers(self)
        self.score_section = ScoreSection(self)
        self.control_section = ControlSection(self)
        self.video_section = VideoSection(self)
        
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
        
        self.setup_ui()
        self.connect_signals()
        
        # Timer to poll the results queue with optimized rate
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.poll_results_queue)
        self.update_timer.start(16)  # ~60 FPS update rate for UI

        self.add_log_message("GUI initialized with multi-processing architecture")
        
    def setup_ui(self):
        """Erstellt die komplette Benutzeroberfläche mit Tab-System"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs using new modular approach
        self.tracking_tab = TrackingTab(self)
        self.calibration_tab = CalibrationTab(self)
        self.settings_tab = SettingsTab(self)
        self.about_tab = AboutTab(self)
        
        # Add tabs to widget
        self.tab_widget.addTab(self.tracking_tab, "🎯 Tracking")
        self.tab_widget.addTab(self.calibration_tab, "🎨 Calibration")
        self.tab_widget.addTab(self.settings_tab, "⚙️ Settings")
        self.tab_widget.addTab(self.about_tab, "ℹ️ About")

        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
    def create_score_section(self):
        """Delegiert an ScoreSection"""
        return self.score_section.create_score_section()
    
    def create_control_section(self):
        """Delegiert an ControlSection"""
        return self.control_section.create_control_section()
    
    def create_video_section(self):
        """Delegiert an VideoSection"""
        return self.video_section.create_video_section()
        
    def connect_signals(self):
        """Verbindet alle Button-Signale mit ihren Funktionen"""
        
        # Visualization controls
        self.ball_only_btn.clicked.connect(lambda: self.event_handlers.set_visualization_mode(1))
        self.field_only_btn.clicked.connect(lambda: self.event_handlers.set_visualization_mode(2))
        self.combined_btn.clicked.connect(lambda: self.event_handlers.set_visualization_mode(3))
        
        # Score controls
        self.reset_score_btn.clicked.connect(self.event_handlers.reset_score_placeholder)
        self.start_match_btn.clicked.connect(self.event_handlers.start_match_placeholder)
        self.cancel_match_btn.clicked.connect(self.event_handlers.cancel_match_placeholder)
        
        # Manual score control buttons
        self.team1_plus_btn.clicked.connect(self.event_handlers.team1_score_plus_placeholder)
        self.team1_minus_btn.clicked.connect(self.event_handlers.team1_score_minus_placeholder)
        self.team2_plus_btn.clicked.connect(self.event_handlers.team2_score_plus_placeholder)
        self.team2_minus_btn.clicked.connect(self.event_handlers.team2_score_minus_placeholder)
        
        # Goal limit controls
        self.reset_goal_limit_btn.clicked.connect(self.event_handlers.reset_goal_limit_placeholder)
        
        # Settings tab buttons
        if hasattr(self, 'apply_settings_btn'):
            self.apply_settings_btn.clicked.connect(self.event_handlers.apply_settings)
        if hasattr(self, 'reset_settings_btn'):
            self.reset_settings_btn.clicked.connect(self.event_handlers.reset_settings)
        
        # Processing mode toggle
        if hasattr(self, 'processing_mode_checkbox'):
            self.processing_mode_checkbox.toggled.connect(self.event_handlers.toggle_processing_mode)
    
    def poll_results_queue(self):
        """Pollt die Ergebnis-Queue für neue Daten aus dem Verarbeitungsprozess - optimierte Version"""
        try:
            # Begrenze die Anzahl der verarbeiteten Queue-Elemente pro Aufruf
            max_items_per_call = 3
            items_processed = 0
            
            while not self.results_queue.empty() and items_processed < max_items_per_call:
                try:
                    result_package = self.results_queue.get_nowait()
                    items_processed += 1
                    
                    # Update current state
                    self.current_display_frame = result_package.get("display_frame")
                    self.current_ball_data = result_package.get("ball_data")
                    self.current_player_data = result_package.get("player_data")
                    self.current_M_field = result_package.get("M_field")
                    
                    # Update score from processing if goals were detected
                    score_data = result_package.get("score")
                    if score_data:
                        new_score1 = score_data.get('player1', 0)
                        new_score2 = score_data.get('player2', 0)
                        
                        if new_score1 != self.player1_goals or new_score2 != self.player2_goals:
                            self.player1_goals = new_score1
                            self.player2_goals = new_score2
                            self.update_score(f"{self.player1_goals}:{self.player2_goals}")
                            
                            if new_score1 > self.player1_goals or new_score2 > self.player2_goals:
                                self.add_log_message("Goal detected automatically!")
                    
                    # Update FPS data from processing if available (throttled)
                    fps_data = result_package.get("fps_data")
                    if fps_data:
                        current_time = time.time()
                        if not hasattr(self, '_last_fps_update_time'):
                            self._last_fps_update_time = 0
                        
                        if current_time - self._last_fps_update_time > 0.5:
                            self.camera_fps = fps_data.get('camera', 0.0)
                            self.preprocessing_fps = fps_data.get('preprocessing', 0.0)
                            self.ball_detection_fps = fps_data.get('ball_detection', 0.0)
                            self.field_detection_fps = fps_data.get('field_detection', 0.0)
                            self.update_processing_fps()
                            self._last_fps_update_time = current_time
                    
                    # Update field data
                    self.current_field_data = result_package.get("field_data")
                    
                except Exception as e:
                    print(f"Error processing queue item: {e}")
                    break
            
            # Update the display only if we have a frame and processed some data
            if items_processed > 0 and self.current_display_frame is not None:
                self.update_display()
                
                # Update Display FPS
                self.display_frame_count += 1
                current_time = time.time()
                if current_time - self.display_last_fps_time >= 1.0:
                    display_fps = self.display_frame_count / (current_time - self.display_last_fps_time)
                    self.update_display_fps(display_fps)
                    self.display_frame_count = 0
                    self.display_last_fps_time = current_time
                    
        except Exception as e:
            print(f"Error in poll_results_queue: {e}")
    
    def update_display(self):
        """Aktualisiert die Video-Anzeige mit dem neuesten Frame und Visualisierungen"""
        if self.current_display_frame is None:
            return
            
        # Verwende die neue Visualization Engine
        display_frame = self.visualization_engine.apply_visualizations(
            self.current_display_frame,
            self.visualization_mode,
            self.current_ball_data,
            getattr(self, 'current_field_data', None)
        )
        
        # Convert frame to Qt format and display
        if display_frame is not None:
            self.update_frame(display_frame)
    
    # ============= UPDATE METHODS =============
    
    @Slot(np.ndarray)
    def update_frame(self, frame):
        """Aktualisiert die Video-Anzeige mit einem neuen Frame - optimiert für Performance"""
        try:
            if frame is None or frame.size == 0:
                return
                
            current_time = time.time()
            if not hasattr(self, '_last_secondary_update_time'):
                self._last_secondary_update_time = 0

            update_secondary = (current_time - self._last_secondary_update_time) > 0.05

            # Convert to RGB once and reuse
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Always update main tracking tab
            try:
                main_rgb_copy = np.ascontiguousarray(rgb_frame)
                main_qt_image = QImage(main_rgb_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                main_pixmap = QPixmap.fromImage(main_qt_image).scaled(
                    self.video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.video_label.setPixmap(main_pixmap)
            except Exception as e:
                print(f"Error updating main video: {e}")
            
            # Update secondary displays less frequently
            if update_secondary:
                self._update_secondary_displays(rgb_frame, frame, h, w, bytes_per_line)
                self._last_secondary_update_time = current_time
                
        except Exception as e:
            print(f"Critical error in update_frame: {e}")
    
    def _update_secondary_displays(self, rgb_frame, original_frame, h, w, bytes_per_line):
        """Aktualisiert die sekundären Displays (Settings und Calibration)"""
        # Update settings tab video
        if hasattr(self, 'settings_video_label') and self.settings_video_label.isVisible():
            try:
                settings_rgb_copy = np.ascontiguousarray(rgb_frame)
                settings_qt_image = QImage(settings_rgb_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                settings_pixmap = QPixmap.fromImage(settings_qt_image).scaled(
                    self.settings_video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.settings_video_label.setPixmap(settings_pixmap)
            except Exception as e:
                print(f"Error updating settings video: {e}")
        
        # Update calibration tab video
        if hasattr(self, 'calibration_video_label') and self.calibration_video_label.isVisible():
            try:
                calibration_rgb_copy = np.ascontiguousarray(rgb_frame)
                calibration_qt_image = QImage(calibration_rgb_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                calibration_pixmap = QPixmap.fromImage(calibration_qt_image).scaled(
                    self.calibration_video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                frame_copy = np.ascontiguousarray(original_frame)
                self.calibration_video_label.set_frame_data(frame_copy, calibration_pixmap)
            except Exception as e:
                print(f"Error updating calibration video: {e}")
    
    def add_log_message(self, message):
        """Fügt eine Log-Nachricht hinzu"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_processing_fps(self):
        """Aktualisiert die Verarbeitungs-FPS-Anzeigen"""
        self.camera_fps_label.setText(f"Camera: {self.camera_fps:.1f} FPS")
        self.preprocessing_fps_label.setText(f"Preprocessing: {self.preprocessing_fps:.1f} FPS")
        self.ball_detection_fps_label.setText(f"Ball Detection: {self.ball_detection_fps:.1f} FPS")
        self.field_detection_fps_label.setText(f"Field Detection: {self.field_detection_fps:.1f} FPS")
    
    def update_display_fps(self, fps):
        """Aktualisiert die Display-FPS"""
        self.display_fps_label.setText(f"Display: {fps:.1f} FPS")
    
    def update_fps(self, fps):
        """Legacy-Methode - leitet jetzt zur Display-FPS weiter"""
        self.update_display_fps(fps)
    
    def update_score(self, score):
        """Aktualisiert die Punkteanzeige"""
        self.big_score_label.setText(score.replace(":", " : "))
    
    # Event-Handler-Delegationen
    def toggle_processing_mode(self, checked):
        """Delegiert an EventHandlers"""
        self.event_handlers.toggle_processing_mode(checked)
    
    def on_calibration_color_picked(self, r, g, b):
        """Delegiert an EventHandlers"""
        self.event_handlers.on_calibration_color_picked(r, g, b)
    
    def closeEvent(self, event):
        """Aufgerufen, wenn das Fenster geschlossen wird"""
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


if __name__ == "__main__":
    # Change to parent directory so imports work correctly
    import os
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir)
    sys.path.insert(0, parent_dir)
    
    import main
    import multiprocessing as mp
    
    mp.set_start_method('spawn', force=True)
    main.main_gui()
