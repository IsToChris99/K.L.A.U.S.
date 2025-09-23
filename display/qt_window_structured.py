import cv2
import numpy as np
import time
import os
import sys
from datetime import datetime

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget

# Import own ui components
from .components.tab_widgets import TrackingTab, CalibrationTab, SettingsTab, AboutTab
from .components.ui_components import ScoreSection, ControlSection, VideoSection
from .components.visualization_engine import VisualizationEngine
from .components.event_handlers import EventHandlers
from processing.cpu_preprocessor import CPUPreprocessor
from processing.gpu_preprocessor import GPUPreprocessor
from utils.color_picker import ColorPicker
import config

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
        self.cpu_preprocessor = CPUPreprocessor()
        self.gpu_preprocessor = GPUPreprocessor()

        # Current states
        self.current_raw_frame = None
        self.current_display_frame = None
        self.current_ball_data = None
        self.current_player_data = None
        self.current_player_data = None
        self.current_M_persp = None
        self.current_M_field = None
        
        # Visualization modes
        self.BALL_ONLY = 1
        self.FIELD_ONLY = 2  
        self.PLAYER_ONLY = 3
        self.COMBINED = 4
        self.visualization_mode = self.COMBINED
        
        # Statistics and scoring
        self.player1_goals = 0
        self.player2_goals = 0
        self.current_max_goals = 10  # Default value, will be updated from main process
        
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
        
        # Timer to poll the results queue with configurable update rate
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.poll_results_queue)
        self.update_timer.start(config.GUI_UPDATE_INTERVAL_MS)  # Configurable GUI update rate

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
        self.player_only_btn.clicked.connect(lambda: self.event_handlers.set_visualization_mode(3))
        self.combined_btn.clicked.connect(lambda: self.event_handlers.set_visualization_mode(4))
        self.toggle_detections_btn.clicked.connect(self.event_handlers.toggle_detections)

        # Score controls
        self.reset_score_btn.clicked.connect(self.event_handlers.reset_score)

        # Manual score control buttons
        self.team1_plus_btn.clicked.connect(self.event_handlers.team1_score_plus)
        self.team1_minus_btn.clicked.connect(self.event_handlers.team1_score_minus)
        self.team2_plus_btn.clicked.connect(self.event_handlers.team2_score_plus)
        self.team2_minus_btn.clicked.connect(self.event_handlers.team2_score_minus)

        # Goal limit controls
        if hasattr(self, 'default_goal_limit_btn'):
            self.default_goal_limit_btn.clicked.connect(self.event_handlers.set_goal_limit_default)
        if hasattr(self, 'infinity_goal_limit_btn'):
            self.infinity_goal_limit_btn.clicked.connect(self.event_handlers.set_goal_limit_infinity)
        if hasattr(self, 'goal_limit_input'):
            self.goal_limit_input.valueChanged.connect(self.event_handlers.on_goal_limit_changed)

        # Settings tab buttons
        if hasattr(self, 'apply_settings_btn'):
            self.apply_settings_btn.clicked.connect(self.event_handlers.apply_settings)
        if hasattr(self, 'reset_settings_btn'):
            self.reset_settings_btn.clicked.connect(self.event_handlers.reset_settings)
        if hasattr(self, 'wb_one_time_checkbox'):
            self.wb_one_time_checkbox.clicked.connect(self.event_handlers.white_balance_once)

        # Processing mode toggle
        if hasattr(self, 'processing_mode_checkbox'):
            self.processing_mode_checkbox.toggled.connect(self.event_handlers.toggle_processing_mode)
            
        # Calibration tab buttons
        if hasattr(self, 'save_frame_and_colorpicker_btn'):
            self.save_frame_and_colorpicker_btn.clicked.connect(self.save_frame_and_open_colorpicker)
        if hasattr(self, 'reload_player_colors_btn'):
            self.reload_player_colors_btn.clicked.connect(self.reload_player_colors)
      
    def poll_results_queue(self):
        """Pollt die Ergebnis-Queue für neue Daten aus dem Verarbeitungsprozess - optimierte Version"""
        try:
            # Begrenze die Anzahl der verarbeiteten Queue-Elemente pro Aufruf (konfigurierbar)
            max_items_per_call = config.GUI_MAX_ITEMS_PER_UPDATE
            items_processed = 0

            vorher = time.perf_counter_ns()

            while not self.results_queue.empty() and items_processed < max_items_per_call:
                try:
                    result_package = self.results_queue.get_nowait()
                    items_processed += 1
                    
                    # Update current state
                    # self.current_raw_frame = result_package.get("raw_frame")
                    self.current_preprocessed_frame = result_package.get("preprocessed_frame")
                    self.current_ball_data = result_package.get("ball_data")
                    self.current_player_data = result_package.get("player_data")
                    self.current_M_persp = result_package.get("M_persp")
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
                    
                    # Update max_goals from processing
                    max_goals_data = result_package.get("max_goals")
                    if max_goals_data is not None and max_goals_data != self.current_max_goals:
                        self.current_max_goals = max_goals_data
                        # Update UI to reflect the new max_goals if needed
                        if hasattr(self, 'goal_limit_input') and self.goal_limit_input.value() != max_goals_data:
                            self.goal_limit_input.setValue(max_goals_data)

                    # Update FPS data from processing if available (less throttled for smoother updates)
                    fps_data = result_package.get("fps_data")
                    if fps_data:
                        current_time = time.time()
                        if not hasattr(self, '_last_fps_update_time'):
                            self._last_fps_update_time = 0
                        
                        # Update FPS display more frequently for smoother GUI
                        if current_time - self._last_fps_update_time > config.GUI_FPS_UPDATE_INTERVAL:
                            self.camera_fps = fps_data.get('camera', 0.0)
                            self.preprocessing_fps = fps_data.get('preprocessing', 0.0)
                            self.ball_detection_fps = fps_data.get('ball_detection', 0.0)
                            self.field_detection_fps = fps_data.get('field_detection', 0.0)
                            self.player_detection_fps = fps_data.get('player_detection', 0.0)
                            self.update_processing_fps()
                            self._last_fps_update_time = current_time
                    
                    # Update field data
                    self.current_field_data = result_package.get("field_data")
                    
                    # Update player data
                    self.current_player_data = result_package.get("player_data")

                    # Optional debug output (only in debug mode to avoid GUI slowdown)
                    if config.DEBUG_VERBOSE_OUTPUT and items_processed == max_items_per_call:
                        processing_time = (time.perf_counter_ns() - vorher) / 1000000
                        print(f"\rGUI processed {items_processed} items in {processing_time:.1f}ms", end="")

                except Exception as e:
                    print(f"Error processing queue item: {e}")
                    break
            
            # Update the display only if we have a frame and processed some data
            if items_processed > 0 and self.current_preprocessed_frame is not None:
                self.current_display_frame = self.current_preprocessed_frame
                # self.current_display_frame = self.cpu_preprocessor.process_display_frame(self.current_raw_frame, self.current_M_persp)
                # self.current_display_frame = self.gpu_preprocessor.process_display_frame(self.current_raw_frame, self.current_M_persp)
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
            getattr(self, 'current_field_data', None),
            getattr(self, 'current_player_data', None),
            getattr(self, 'current_M_persp', None)
        )
        
        # Convert frame to Qt format and display
        if display_frame is not None:
            self.update_frame(display_frame)
    
    # ============= UPDATE METHODS =============
    
    @Slot(np.ndarray)
    def update_frame(self, frame):
        """Aktualisiert die Video-Anzeige mit einem neuen Frame - optimiert für Performance (zentralisiert)."""
        try:
            if frame is None or frame.size == 0:
                return

            # Einmalige Konvertierung und Pixmap-Erstellung
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            rgb_copy = np.ascontiguousarray(rgb_frame)
            qt_image = QImage(rgb_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Pixmap für Hauptanzeige
            main_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(main_pixmap)

            # Pixmap für Settings-Tab
            if hasattr(self, 'settings_video_label'):
                settings_pixmap = main_pixmap
                if self.settings_video_label.size() != self.video_label.size():
                    settings_pixmap = QPixmap.fromImage(qt_image).scaled(
                        self.settings_video_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                self.settings_video_label.setPixmap(settings_pixmap)

        except Exception as e:
            print(f"Critical error in update_frame: {e}")
    
    def add_log_message(self, message):
        """Fügt eine Log-Nachricht hinzu"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_processing_fps(self):
        """Aktualisiert die Verarbeitungs-FPS-Anzeigen"""
        self.camera_fps_label.setText(f"{self.camera_fps:.1f} FPS")
        self.preprocessing_fps_label.setText(f"{self.preprocessing_fps:.1f} FPS")
        self.ball_detection_fps_label.setText(f"{self.ball_detection_fps:.1f} FPS")
        self.field_detection_fps_label.setText(f"{self.field_detection_fps:.1f} FPS")
        self.player_detection_fps_label.setText(f"{self.player_detection_fps:.1f} FPS")
    
    def update_display_fps(self, fps):
        """Aktualisiert die Display-FPS"""
        self.display_fps_label.setText(f"{fps:.1f} FPS")
    
    def update_fps(self, fps):
        """Legacy-Methode - leitet jetzt zur Display-FPS weiter"""
        self.update_display_fps(fps)
    
    def update_score(self, score):
        """Aktualisiert die Punkteanzeige"""
        self.big_score_label.setText(score.replace(":", " : "))
    
    def get_current_max_goals(self):
        """Gibt die aktuellen max_goals zurück, die vom Hauptprozess empfangen wurden"""
        return self.current_max_goals
    
    # Event-Handler-Delegationen
    def toggle_processing_mode(self, checked):
        """Delegiert an EventHandlers"""
        self.event_handlers.toggle_processing_mode(checked)
    
    def save_frame_and_open_colorpicker(self):
        """Öffnet den ColorPicker mit dem aktuellen Display-Frame"""
        try:
            # Debugging-Info
            self.add_log_message("Button clicked: save_frame_and_open_colorpicker")
            
            if self.current_display_frame is None:
                self.add_log_message("No frame available for ColorPicker")
                return
            
            # Öffne ColorPicker direkt mit dem aktuellen Display-Frame
            try:
                self.add_log_message("Opening ColorPicker with current display frame")
                # Erstelle und zeige ColorPicker direkt mit dem aktuellen Frame
                self.color_picker_window = ColorPicker(frame=self.current_display_frame)
                self.color_picker_window.show()
                self.add_log_message("ColorPicker window created and shown successfully")
                
            except Exception as e:
                self.add_log_message(f"Error opening ColorPicker: {str(e)}")
                print(f"ColorPicker error details: {str(e)}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            self.add_log_message(f"Error in save_frame_and_open_colorpicker: {str(e)}")
            print(f"Error in save_frame_and_open_colorpicker: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
