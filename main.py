import sys
import time
import threading
import multiprocessing as mp
import queue
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication

# Lokale Imports
from detection.ball_detector import BallDetector
from detection.field_detector_markers import FieldDetector  # Verwende markers Version
#from detection.player_detector import PlayerDetector
from analysis.goal_scorer import GoalScorer
from input.ids_camera_sync import IDS_Camera  # Verwende synchrone Version
from processing.cpu_preprocessor import CPUPreprocessor
from processing.gpu_preprocessor import GPUPreprocessor
from display.qt_window_multiprocess import KickerMainWindow
import config

# ================== PROCESSING PROCESS ==================

class ProcessingProcess(mp.Process):
    """
    Ein dedizierter Prozess für die gesamte 250fps-Verarbeitungspipeline.
    Nimmt rohe Frames entgegen und gibt ein komplettes Ergebnispaket aus.
    """
    def __init__(self, raw_frame_queue, results_queue, command_queue, running_event):
        super().__init__()
        self.raw_frame_queue = raw_frame_queue
        self.results_queue = results_queue
        self.command_queue = command_queue
        self.running_event = running_event

        # Diese Objekte werden INNERHALB des neuen Prozesses erstellt
        self.ball_detector = None
        self.field_detector = None
        #self.player_detector = None  # TODO: Implementierung ausstehend
        self.preprocessor = None
        self.cpu_preprocessor = None
        self.gpu_preprocessor = None
        self.goal_scorer = None
        
        # Processing mode control
        self.use_gpu_processing = True  # Start with GPU processing by default

        # Zustand für den "One-Frame-Lag"
        self.latest_M_persp = np.identity(3)
        self.latest_M_field = np.identity(3)
        
        # Field corners für eingeschränkte Ball-Suche
        self.field_corners = None
        self.goals = []
        
        # Calibration state - Always active like in main_live.py
        self.calibration_mode = True
        
        # FPS tracking for different components
        self.fps_trackers = {
            'preprocessing': {'count': 0, 'last_time': time.time()},
            'ball_detection': {'count': 0, 'last_time': time.time()},
            'field_detection': {'count': 0, 'last_time': time.time()}
        }
        self.current_fps = {
            'camera': 0.0,
            'preprocessing': 0.0, 
            'ball_detection': 0.0,
            'field_detection': 0.0
        }

    def run(self):
        """Die Hauptschleife des Verarbeitungsprozesses."""
        print("ProcessingProcess gestartet.")

        # Initialisiere alle Objekte hier, um sicherzustellen, dass sie im richtigen Prozess leben
        self.ball_detector = BallDetector()
        self.field_detector = FieldDetector()
        #self.player_detector = PlayerDetector()
        
        # Initialize both CPU and GPU preprocessors
        try:
            self.gpu_preprocessor = GPUPreprocessor()
            print("GPU preprocessor initialized successfully")
        except Exception as e:
            print(f"GPU preprocessor initialization failed: {e}")
            self.gpu_preprocessor = None
            self.use_gpu_processing = False
            
        self.cpu_preprocessor = CPUPreprocessor()
        print("CPU preprocessor initialized successfully")
        
        # Set active preprocessor based on mode
        self.preprocessor = self.gpu_preprocessor if self.use_gpu_processing and self.gpu_preprocessor else self.cpu_preprocessor
        
        self.goal_scorer = GoalScorer()

        while self.running_event.is_set():
            # Check for commands from UI
            try:
                while not self.command_queue.empty():
                    command = self.command_queue.get_nowait()
                    self.handle_command(command)
            except:
                pass  # No commands or error reading - continue
                
            try:
                # 1. Hole den nächsten rohen Frame
                raw_data = self.raw_frame_queue.get(timeout=1)
                
                # Extract frame and camera FPS if available
                if isinstance(raw_data, tuple):
                    raw_frame, camera_fps = raw_data
                    self.current_fps['camera'] = camera_fps
                else:
                    raw_frame = raw_data
            except queue.Empty:
                continue

            # 2. Preprocessing (Warp) mit der Matrix vom VORHERIGEN Frame
            preprocessing_start = time.time()
            # CPUPreprocessor gibt (resized_frame, undist_frame) zurück - wir nehmen das erste
            preprocessed_result = self.preprocessor.process_frame(raw_frame)
            if isinstance(preprocessed_result, tuple):
                preprocessed_frame = preprocessed_result[0]  # resized_frame für die Verarbeitung
            else:
                preprocessed_frame = preprocessed_result
            self.update_fps_tracker('preprocessing', preprocessing_start)
            
            # 3. Starte alle drei Erkennungen parallel mit Threads
            results = {}
            
            # --- Thread-Funktionen definieren ---
            def field_task():
                field_start = time.time()
                # Optimierte Kalibrierung - jetzt schnell genug für jeden Frame
                if self.calibration_mode:
                    before_calibration = time.perf_counter_ns()
                    self.field_detector.calibrate(preprocessed_frame)
                    after_calibration = time.perf_counter_ns()
                    calibration_time = (after_calibration - before_calibration) / 1e9
                    # print(f'\rCalibration attempt took: {calibration_time:.4f} seconds', end='')
                
                # Store current field data in results
                results['field_calibrated'] = self.field_detector.calibrated
                results['field_corners'] = self.field_detector.field_corners
                results['goals'] = self.field_detector.goals
                results['field_contour'] = getattr(self.field_detector, 'field_contour', None)
                results['field_bounds'] = self.field_detector.field_corners  # Use field_corners as field_bounds
                results['field_rect_points'] = getattr(self.field_detector, 'field_rect_points', None)
                
                # Calculate field width and pixel-to-cm ratio (wie in main_live.py)
                if (results['field_corners'] is not None and len(results['field_corners']) >= 2 
                    and results['field_corners'][0] is not None 
                    and results['field_corners'][1] is not None):
                    field_width = results['field_corners'][1][0] - results['field_corners'][0][0]
                    px_to_cm_ratio = field_width / 118 if field_width != 0 else 0
                    # Store for potential future use
                    results['field_width'] = field_width
                    results['px_to_cm_ratio'] = px_to_cm_ratio
                else:
                    results['field_width'] = 0
                    results['px_to_cm_ratio'] = 0
                
                self.update_fps_tracker('field_detection', field_start)

            def ball_task():
                ball_start = time.time()
                # Field corners für eingeschränkte Ball-Suche (wie in main_live.py)
                field_corners = None
                goals = []
                if results.get('goals'):
                    goals = results['goals']
                
                # Ball detection with field_corners (wie in main_live.py)
                detection_result = self.ball_detector.detect_ball(preprocessed_frame, self.field_corners)
                self.ball_detector.update_tracking(detection_result, self.field_corners)

                # Goal scoring system update (wie in main_live.py)
                ball_position = detection_result[0] if detection_result[0] is not None else None
                self.goal_scorer.update_ball_tracking(
                    ball_position, 
                    goals, 
                    field_corners, 
                    self.ball_detector.missing_counter
                )
                
                # Ball-Ergebnisse in results speichern (wie in main_live.py)
                results['ball_data'] = {
                    'detection': detection_result,
                    'smoothed_pts': list(self.ball_detector.smoothed_pts),
                    'missing_counter': self.ball_detector.missing_counter,
                    'ball_position': ball_position
                }
                
                # Check for goals
                score = self.goal_scorer.get_score()
                results['score'] = score
                
                self.update_fps_tracker('ball_detection', ball_start)

            def player_task():
                # TODO: Player-Erkennung noch nicht implementiert
                # player_data = self.player_detector.detect(preprocessed_frame)
                # results['player_data'] = player_data
                results['player_data'] = None  # Placeholder
                return

            # --- Threads erstellen und starten ---
            thread_field = threading.Thread(target=field_task)
            thread_ball = threading.Thread(target=ball_task)
            #thread_player = threading.Thread(target=player_task)  # TODO: Player-Erkennung auskommentiert
            
            thread_field.start()
            thread_ball.start()
            #thread_player.start()  # TODO: Player-Erkennung auskommentiert
            
            # --- Wait for both threads to complete ---
            thread_field.join()
            thread_ball.join()
            #thread_player.join()  # TODO: Player-Erkennung auskommentiert

            # 4. Aktualisiere die Matrizen und Field-Daten für den NÄCHSTEN Frame
            if results.get('M_persp') is not None:
                self.latest_M_persp = results['M_persp']
            if results.get('M_field') is not None:
                self.latest_M_field = results['M_field']

            # Update field corners and goals for next iteration
            if results.get('field_corners') is not None:
                self.field_corners = results['field_corners']
            if results.get('goals') is not None:
                self.goals = results['goals']
                
            # 5. Stelle ein komplettes Ergebnispaket für die UI zusammen
            final_package = {
                "display_frame": preprocessed_frame,
                "ball_data": results.get('ball_data'),
                "player_data": results.get('player_data'),
                "score": results.get('score', {'player1': 0, 'player2': 0}),
                "M_field": self.latest_M_field,
                "fps_data": self.current_fps.copy(),  # Add FPS data
                "processing_mode": "GPU" if self.use_gpu_processing else "CPU",  # Add processing mode info
                "field_data": {
                    'calibrated': self.field_detector.calibrated if self.field_detector else False,
                    'field_contour': results.get('field_contour'),
                    'field_corners': results.get('field_corners'),
                    'field_bounds': results.get('field_bounds'),
                    'field_rect_points': results.get('field_rect_points'),
                    'goals': results.get('goals', []),
                    'calibration_mode': self.calibration_mode
                }
            }

            # 6. Sende das Paket an die UI
            if not self.results_queue.full():
                self.results_queue.put(final_package)

        print("ProcessingProcess beendet.")
    
    def update_fps_tracker(self, component, start_time):
        """Update FPS tracking for a specific component"""
        if component not in self.fps_trackers:
            return
            
        current_time = time.time()
        tracker = self.fps_trackers[component]
        tracker['count'] += 1
        
        # Calculate FPS every second
        time_diff = current_time - tracker['last_time']
        if time_diff >= 1.0:
            fps = tracker['count'] / time_diff
            self.current_fps[component] = fps
            tracker['count'] = 0
            tracker['last_time'] = current_time
    
    def handle_command(self, command):
        """Handle commands from the UI process"""
        if command.get('type') == 'reset_score':
            print("Resetting score...")
            if self.goal_scorer:
                self.goal_scorer.reset_score()
        elif command.get('type') == 'toggle_processing_mode':
            self.toggle_processing_mode()
        # Field calibration is now automatic - removed manual calibration commands
        
        # Add more command types as needed
        
    def toggle_processing_mode(self):
        """Toggle between CPU and GPU preprocessing"""
        self.use_gpu_processing = not self.use_gpu_processing
        
        # Switch preprocessor
        if self.use_gpu_processing and self.gpu_preprocessor is not None:
            try:
                # Try to reinitialize GPU preprocessor
                self.gpu_preprocessor.force_reinitialize()
                self.preprocessor = self.gpu_preprocessor
                print("Switched to GPU preprocessing")
            except Exception as e:
                print(f"Failed to switch to GPU, staying with CPU: {e}")
                self.use_gpu_processing = False
                self.preprocessor = self.cpu_preprocessor
        else:
            self.preprocessor = self.cpu_preprocessor
            print("Switched to CPU preprocessing")


# ================== CAMERA THREAD ==================

def camera_thread_func(raw_frame_queue, running_event):
    """Synchroner Thread, der Frames von der IDS-Kamera holt."""
    print("Camera-Thread gestartet.")
    camera = None
    
    # FPS tracking for camera
    camera_frame_count = 0
    camera_last_fps_time = time.time()
    camera_fps = 0.0
    
    try:
        camera = IDS_Camera()
        camera.start()  # Startet die synchrone Akquisition
        
        while running_event.is_set():
            try:
                # Blockierender Aufruf mit Timeout - wartet auf nächsten Frame
                bayer_frame, metadata = camera.get_frame()
                
                if bayer_frame is not None:
                    # Update camera FPS tracking
                    camera_frame_count += 1
                    current_time = time.time()
                    if current_time - camera_last_fps_time >= 1.0:
                        camera_fps = camera_frame_count / (current_time - camera_last_fps_time)
                        camera_frame_count = 0
                        camera_last_fps_time = current_time
                        #print(f"Camera FPS: {camera_fps:.1f}")  # Debug output
                    
                    if not raw_frame_queue.full():
                        # Add camera FPS to frame metadata
                        frame_with_fps = (bayer_frame, camera_fps)
                        raw_frame_queue.put(frame_with_fps)
                    # Wenn Queue voll ist, wird der Frame verworfen (neueste Frames sind wichtiger)
                else:
                    # Kein Frame verfügbar - kurz warten
                    time.sleep(0.001)
                    
            except Exception as frame_e:
                if running_event.is_set():  # Nur loggen wenn noch aktiv
                    print(f"Frame acquisition error: {frame_e}")
                time.sleep(0.01)
                
    except Exception as e:
        print(f"Fehler im Kamera-Thread: {e}")
    finally:
        print("Camera-Thread cleanup gestartet...")
        if camera is not None:
            try:
                camera.stop()
                print("Kamera erfolgreich gestoppt.")
            except Exception as e:
                print(f"Fehler beim Stoppen der Kamera: {e}")
        print("Camera-Thread beendet.")


# ================== MAIN PROGRAM ==================

def main_gui():
    """Startet die gesamte Anwendung: UI, Kamera-Thread und Processing-Prozess."""
    app = QApplication(sys.argv)

    # 1. Erstelle Kommunikationsmittel
    # Ein Event, um alle Teile sauber zu beenden
    running_event = mp.Event()
    running_event.set()

    # Prozess-sichere Queues
    raw_frame_queue = mp.Queue(maxsize=5)
    results_queue = mp.Queue(maxsize=5)
    command_queue = mp.Queue(maxsize=10)  # Für UI-Kommandos

    # 2. Erstelle und starte den Verarbeitungs-Prozess
    processing_process = ProcessingProcess(raw_frame_queue, results_queue, command_queue, running_event)
    processing_process.start()

    # 3. Erstelle und starte den Kamera-Thread im Hauptprozess
    cam_thread = threading.Thread(target=camera_thread_func, args=(raw_frame_queue, running_event), daemon=True)
    cam_thread.start()
    
    # 4. Erstelle das Hauptfenster und übergebe die Kommunikationsmittel
    window = KickerMainWindow(results_queue, command_queue, running_event)
    window.show()

    # Log initial messages
    window.add_log_message("Multi-Processing Kicker Klaus gestartet")
    window.add_log_message("Kamera-Thread läuft...")
    window.add_log_message("Processing-Prozess läuft...")
    window.add_log_message("Warten auf Video-Stream...")

    # 5. Starte die Anwendung und warte auf das saubere Herunterfahren
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
        exit_code = 0
    
    print("UI wurde geschlossen. Fahre die Pipeline herunter...")
    
    # 6. Sauberes Herunterfahren
    running_event.clear()  # Signal an Prozess und Thread zum Beenden
    
    # Stoppe Timer in GUI falls vorhanden
    try:
        if hasattr(window, 'update_timer') and window.update_timer:
            window.update_timer.stop()
    except:
        pass
    
    print("Warte auf Processing-Prozess...")
    # Warte auf Processing-Prozess
    processing_process.join(timeout=3)
    if processing_process.is_alive():
        print("Processing-Prozess konnte nicht normal beendet werden, terminiere...")
        processing_process.terminate()
        processing_process.join(timeout=2)
        if processing_process.is_alive():
            print("Processing-Prozess terminiert mit kill...")
            processing_process.kill()
            processing_process.join(timeout=1)
    
    print("Warte auf Kamera-Thread...")
    # Warte auf Kamera-Thread
    if cam_thread.is_alive():
        cam_thread.join(timeout=2)
        if cam_thread.is_alive():
            print("Warnung: Kamera-Thread konnte nicht rechtzeitig beendet werden")
    
    # Queue cleanup - leere alle Queues
    try:
        while not raw_frame_queue.empty():
            raw_frame_queue.get_nowait()
    except:
        pass
    
    try:
        while not results_queue.empty():
            results_queue.get_nowait()
    except:
        pass
    
    try:
        while not command_queue.empty():
            command_queue.get_nowait()
    except:
        pass
    
    # Forciere Garbage Collection für aufräumen
    import gc
    gc.collect()
    
    print("Anwendung sauber heruntergefahren.")
    
    # Wichtig: explizit mit os._exit() beenden um sicherzustellen, 
    # dass alle Threads und Prozesse wirklich terminiert werden
    import os
    os._exit(exit_code)

if __name__ == "__main__":
    # Wichtig für macOS und Windows
    mp.set_start_method('spawn', force=True)
    
    # Starte die Anwendung - os._exit() wird in main_gui() aufgerufen
    main_gui()