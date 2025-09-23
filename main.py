import sys
import time
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import numpy as np
from PySide6.QtWidgets import QApplication
# import qdarkstyle

# Lokale Imports
import config
from detection.ball_detector import BallDetector
from detection.field_detector_markers import FieldDetector  # Verwende markers Version
from detection.player_detector import PlayerDetector
from analysis.goal_scorer import GoalScorer
from input.ids_camera_sync import IDS_Camera  # Verwende synchrone Version
from processing.cpu_preprocessor import CPUPreprocessor
from processing.gpu_preprocessor import GPUPreprocessor
from display.qt_window_structured import KickerMainWindow


# ================== WORKER PROCESSES (Field/Ball) ==================

class FieldWorkerProcess(mp.Process):
    """Own process for field detection working on a shared memory frame."""
    def __init__(self, shm_name, shape, dtype, tick_queue, result_queue, running_event):
        super().__init__()
        self.shm_name = shm_name
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype).str  # ensure picklable
        self.tick_queue = tick_queue
        self.result_queue = result_queue
        self.running_event = running_event

        self.detector = None

    def run(self):
        # Late imports safe for spawn

        shm = shared_memory.SharedMemory(name=self.shm_name)
        frame = np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=shm.buf)
        self.detector = FieldDetector()

        while self.running_event.is_set():
            try:
                seq = self.tick_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Run calibration/detection on current shared frame
            try:
                self.detector.calibrate(frame)
            except Exception:
                # Robustness: swallow per-frame errors
                pass

            result = {
                'seq': seq,
                'field_corners': getattr(self.detector, 'field_corners', None),
                'goals': getattr(self.detector, 'goals', []),
                'calibrated': getattr(self.detector, 'calibrated', False),
                'M_field': getattr(self.detector, 'field_transform_matrix', None),
                'M_persp': getattr(self.detector, 'perspective_transform_matrix', None)
            }

            if not self.result_queue.full():
                self.result_queue.put(result)

        shm.close()

class BallWorkerProcess(mp.Process):
    """Own process for ball detection working on a shared memory frame and optional field state."""
    def __init__(self, shm_name, shape, dtype, tick_queue, field_state_queue, result_queue, running_event):
        super().__init__()
        self.shm_name = shm_name
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype).str
        self.tick_queue = tick_queue
        self.field_state_queue = field_state_queue  # parent broadcasts latest field state
        self.result_queue = result_queue
        self.running_event = running_event

        self.detector = None
        self.latest_field = {'field_corners': None, 'goals': []}

    def run(self):
        # Late imports safe for spawn

        shm = shared_memory.SharedMemory(name=self.shm_name)
        frame = np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=shm.buf)
        self.detector = BallDetector()
        

        while self.running_event.is_set():
            # Drain field state queue (keep only latest)
            try:
                while True:
                    self.latest_field = self.field_state_queue.get_nowait()
            except queue.Empty:
                pass

            # Wait for tick
            try:
                seq = self.tick_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            field_corners = self.latest_field.get('field_corners')
            goals = self.latest_field.get('goals', [])

            # Run detection on shared frame
            try:
                detection_result = self.detector.detect_ball(frame, field_corners)
                self.detector.update_tracking(detection_result, field_corners)
            except Exception:
                detection_result = (None, 0, 0.0, None)

            ball_position = detection_result[0] if isinstance(detection_result, (list, tuple)) else None

            result = {
                'seq': seq,
                'ball_data': {
                    'detection': detection_result,
                    'smoothed_pts': list(getattr(self.detector, 'smoothed_pts', [])),
                    'missing_counter': getattr(self.detector, 'missing_counter', 0),
                    'ball_position': ball_position
                }
            }

            if not self.result_queue.full():
                self.result_queue.put(result)

        shm.close()

class PlayerWorkerProcess(mp.Process):
    """Own process for player detection working on a shared memory frame."""
    def __init__(self, shm_name, shape, dtype, tick_queue, field_state_queue, result_queue, running_event, color_config_path=None):
        super().__init__()
        self.shm_name = shm_name
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype).str
        self.tick_queue = tick_queue
        self.field_state_queue = field_state_queue  # parent broadcasts latest field state
        self.result_queue = result_queue
        self.running_event = running_event
        self.color_config_path = color_config_path

        self.detector = None
        self.latest_field = {'field_corners': None, 'goals': []}

    def run(self):
        # Late imports safe for spawn
        shm = shared_memory.SharedMemory(name=self.shm_name)
        frame = np.ndarray(self.shape, dtype=np.dtype(self.dtype), buffer=shm.buf)
        self.detector = PlayerDetector(self.color_config_path)

        while self.running_event.is_set():
            # Drain field state queue (keep only latest)
            try:
                while True:
                    self.latest_field = self.field_state_queue.get_nowait()
            except queue.Empty:
                pass

            # Wait for tick
            try:
                seq = self.tick_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Run player detection on current shared frame
            try:
                team1_boxes, team2_boxes = self.detector.detect_players(frame, self.latest_field.get('field_corners'))
            except Exception:
                # Robustness: swallow per-frame errors
                team1_boxes, team2_boxes = [], []

            result = {
                'seq': seq,
                'player_data': {
                    'team1_boxes': team1_boxes,
                    'team2_boxes': team2_boxes,
                    'total_players': len(team1_boxes) + len(team2_boxes)
                }
            }

            if not self.result_queue.full():
                self.result_queue.put(result)

        shm.close()


# ================== PROCESSING PROCESS ==================

class ProcessingProcess(mp.Process):
    """
    Ein dedizierter Prozess für die gesamte 250fps-Verarbeitungspipeline.
    Nimmt rohe Frames entgegen und gibt ein komplettes Ergebnispaket aus.
    """
    def __init__(self, raw_frame_queue, results_queue, command_queue, camera_command_queue, running_event):
        super().__init__()
        self.raw_frame_queue = raw_frame_queue
        self.results_queue = results_queue
        self.command_queue = command_queue
        self.camera_command_queue = camera_command_queue  # New queue for camera commands
        self.running_event = running_event

        # Diese Objekte werden INNERHALB des neuen Prozesses erstellt
        self.ball_detector = None
        self.field_detector = None
        # self.player_detector = None  # TODO: Implementierung ausstehend
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
            'field_detection': {'count': 0, 'last_time': time.time()},
            'player_detection': {'count': 0, 'last_time': time.time()},
        }
        self.current_fps = {
            'camera': 0.0,
            'preprocessing': 0.0,
            'ball_detection': 0.0,
            'field_detection': 0.0,
            'player_detection': 0.0,
        }

    def run(self):
        """Die Hauptschleife des Verarbeitungsprozesses."""
        print("ProcessingProcess gestartet.")

        # Initialize preprocessors in this process
        try:
            self.gpu_preprocessor = GPUPreprocessor()
            print("GPU preprocessor initialized successfully")
        except Exception as e:
            print(f"GPU preprocessor initialization failed: {e}")
            self.gpu_preprocessor = None
            self.use_gpu_processing = False

        self.cpu_preprocessor = CPUPreprocessor()
        print("CPU preprocessor initialized successfully")

        self.preprocessor = self.gpu_preprocessor if self.use_gpu_processing and self.gpu_preprocessor else self.cpu_preprocessor
        self.goal_scorer = GoalScorer()

        # Shared memory and worker processes
        shm = None
        shm_name = None
        shm_shape = None
        shm_dtype = None
        seq = 0

        # IPC queues for workers
        field_tick_q = mp.Queue(maxsize=2)
        ball_tick_q = mp.Queue(maxsize=2)
        player_tick_q = mp.Queue(maxsize=2)
        field_result_q = mp.Queue(maxsize=5)
        ball_result_q = mp.Queue(maxsize=5)
        player_result_q = mp.Queue(maxsize=5)
        field_state_broadcast_q = mp.Queue(maxsize=5)

        field_proc = None
        ball_proc = None
        player_proc = None

        try:
            while self.running_event.is_set():
                # Handle UI commands
                try:
                    while not self.command_queue.empty():
                        command = self.command_queue.get_nowait()
                        self.handle_command(command)
                except Exception as e:
                    print(f"ERROR: Exception in command handling: {e}")
                    import traceback
                    traceback.print_exc()

                # Get next raw frame
                try:
                    raw_data = self.raw_frame_queue.get(timeout=1)
                    if isinstance(raw_data, tuple):
                        raw_frame, camera_fps = raw_data
                        self.current_fps['camera'] = camera_fps
                    else:
                        raw_frame = raw_data
                except queue.Empty:
                    continue

                # Preprocess frame once in parent process
                preprocessing_start = time.time()
                preprocessed_frame, _ = self.preprocessor.process_frame(raw_frame)
                self.update_fps_tracker('preprocessing', preprocessing_start)

                # Initialize shared memory and workers on first frame
                if shm is None:
                    shm_shape = preprocessed_frame.shape
                    shm_dtype = preprocessed_frame.dtype
                    shm = shared_memory.SharedMemory(create=True, size=preprocessed_frame.nbytes)
                    shm_name = shm.name

                    field_proc = FieldWorkerProcess(shm_name, shm_shape, shm_dtype, field_tick_q, field_result_q, self.running_event)
                    ball_proc = BallWorkerProcess(shm_name, shm_shape, shm_dtype, ball_tick_q, field_state_broadcast_q, ball_result_q, self.running_event)
                    player_proc = PlayerWorkerProcess(shm_name, shm_shape, shm_dtype, player_tick_q, field_state_broadcast_q, player_result_q, self.running_event)
                    field_proc.start()
                    ball_proc.start()
                    player_proc.start()
                    print(f"Spawned Field/Ball/Player worker processes using shared memory: {shm_name}")

                # Write current frame to shared memory (copy once)
                np_view = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
                np_view[:] = preprocessed_frame

                seq += 1
                # Notify workers with lightweight tick
                for q in (field_tick_q, ball_tick_q, player_tick_q):
                    try:
                        if not q.full():
                            q.put(seq, block=False)
                    except Exception:
                        pass

                # Collect field results (drain queue)
                results = {}
                try:
                    while True:
                        fr = field_result_q.get_nowait()
                        results.update({
                            'field_corners': fr.get('field_corners'),
                            'goals': fr.get('goals'),
                            'field_calibrated': fr.get('calibrated'),
                            'M_field': fr.get('M_field'),
                            'M_persp': fr.get('M_persp'),
                        })
                        # Update cached matrices
                        if results.get('M_field') is not None:
                            self.latest_M_field = results['M_field']
                        if results.get('M_persp') is not None:
                            self.latest_M_persp = results['M_persp']
                        # Broadcast latest field state to ball worker
                        try:
                            if not field_state_broadcast_q.full():
                                field_state_broadcast_q.put({
                                    'field_corners': fr.get('field_corners'),
                                    'goals': fr.get('goals'),
                                }, block=False)
                        except Exception:
                            pass
                        # FPS accounting for field detection
                        self.update_fps_tracker('field_detection', time.time())
                except queue.Empty:
                    pass

                # Collect ball results (drain queue)
                try:
                    while True:
                        br = ball_result_q.get_nowait()
                        results['ball_data'] = br.get('ball_data')
                        # FPS accounting for ball detection
                        self.update_fps_tracker('ball_detection', time.time())
                except queue.Empty:
                    pass

                # Collect player results (drain queue)
                try:
                    while True:
                        pr = player_result_q.get_nowait()
                        results['player_data'] = pr.get('player_data')
                        # FPS accounting for player detection
                        self.update_fps_tracker('player_detection', time.time())
                except queue.Empty:
                    pass

                # Update goal scorer with latest data
                if results.get('ball_data'):
                    ball_pos = results['ball_data'].get('ball_position')
                    velocity = None
                    det = results['ball_data'].get('detection')
                    if isinstance(det, (list, tuple)) and len(det) >= 4:
                        velocity = det[3]
                    self.goal_scorer.update_ball_tracking(
                        ball_pos,
                        results.get('goals', []),
                        results.get('field_corners'),
                        results['ball_data'].get('missing_counter', 0),
                        ball_velocity=velocity,
                    )
                    results['score'] = self.goal_scorer.get_score() if hasattr(self.goal_scorer, 'get_score') else {'player1': 0, 'player2': 0}

                # Pull current score to keep UI synced with manual updates
                current_score = None
                if hasattr(self.goal_scorer, 'get_score'):
                    try:
                        current_score = self.goal_scorer.get_score()
                        goal_infos = {
                            'player1': current_score.get('player1', 0),
                            'player2': current_score.get('player2', 0)
                        }
                        max_goals = current_score.get('max_goals', 1)
                    except Exception:
                        current_score = None

                # Check if we should filter frames based on detection completeness
                should_send_frame = True
                
                if config.GUI_SHOW_ONLY_COMPLETE_DETECTIONS:
                    # Check if both field and player detection have valid data
                    field_data_valid = (results.get('field_corners') is not None and 
                                      results.get('field_calibrated', False))
                    player_data_valid = (results.get('player_data') is not None and 
                                        results.get('player_data', {}).get('total_players', 0) > 0)
                    
                    # Only send to GUI if both field and players are detected
                    should_send_frame = field_data_valid and player_data_valid
                    
                    # Optional debug output
                    if not should_send_frame and config.DEBUG_SHOW_FRAME_FILTERING:
                        field_status = "✓" if field_data_valid else "✗"
                        player_status = "✓" if player_data_valid else "✗"
                        print(f"Frame filtered: Field {field_status}, Player {player_status}")
                
                if should_send_frame:
                    # Package for UI
                    final_package = {
                        'preprocessed_frame': preprocessed_frame,
                        'ball_data': results.get('ball_data'),
                        'player_data': results.get('player_data'),
                        'score': goal_infos if current_score else {'player1': 0, 'player2': 0},
                        'max_goals': max_goals if current_score else 1,
                        'M_persp': results.get('M_persp', self.latest_M_persp),
                        'M_field': results.get('M_field', self.latest_M_field),
                        'fps_data': self.current_fps.copy(),
                        'processing_mode': 'GPU' if self.use_gpu_processing else 'CPU',
                        'field_data': {
                            'calibrated': results.get('field_calibrated', False),
                            'field_contour': results.get('field_contour'),
                            'field_corners': results.get('field_corners'),
                            'field_bounds': results.get('field_corners'),
                            'field_rect_points': results.get('field_rect_points'),
                            'goals': results.get('goals', []),
                            'calibration_mode': self.calibration_mode,
                        },
                    }

                    if not self.results_queue.full():
                        self.results_queue.put(final_package)
                # Else: Frame wird nicht an GUI gesendet, da Erkennungen nicht vollständig sind (falls GUI_SHOW_ONLY_COMPLETE_DETECTIONS=True)

        finally:
            # Cleanup workers and shared memory
            for proc in (field_proc, ball_proc, player_proc):
                if proc is not None:
                    proc.join(timeout=1)
            for proc in (field_proc, ball_proc, player_proc):
                if proc is not None and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
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
        if command.get('type') == 'toggle_processing_mode':
            self.toggle_processing_mode()
            
        elif command.get('type') == 'reset_score':
            print("Resetting score...")
            if self.goal_scorer:
                self.goal_scorer.reset_score()
        
        elif command.get('type') == 'update_score':
            idx = command.get('index')
            amt = command.get('amount')
            self.goal_scorer.update_score(idx, amt)
       
        elif command.get('type') == 'set_max_goals':
            max_goals = command.get('max_goals')
            is_inf = command.get('is_infinity', False)
            self.goal_scorer.set_max_goals(max_goals, is_inf)

        elif command.get('type') == 'set_infinity_goals':
            self.goal_scorer.set_max_goals(1, True)

        elif command.get('type') == 'update_settings':
            # Handle camera settings - forward to camera thread
            camera_settings = command.get('camera_settings', {})
            
            if camera_settings:
                # Forward camera command to camera thread
                camera_command = {
                    'type': 'update_camera_settings',
                    'settings': camera_settings
                }
                try:
                    self.camera_command_queue.put_nowait(camera_command)
                except Exception as e:
                    print(f"ERROR: Failed to forward camera command: {e}")
            
            # Handle processing settings
            processing_settings = command.get('processing_settings', {})
            if processing_settings:
                # Update ball detector settings if available
                if hasattr(self, 'ball_detector') and self.ball_detector:
                    if 'ball_sensitivity' in processing_settings:
                        # Convert percentage to threshold (assuming inverse relationship)
                        threshold = max(1, 101 - processing_settings['ball_sensitivity'])
                        # Note: This would need to be implemented in BallDetector
                        # self.ball_detector.set_sensitivity(threshold)
                    
                    if 'ball_size_min' in processing_settings:
                        # self.ball_detector.set_min_size(processing_settings['ball_size_min'])
                        pass
                    
                    if 'ball_size_max' in processing_settings:
                        # self.ball_detector.set_max_size(processing_settings['ball_size_max'])
                        pass
                
                # Update field detector settings if available
                if hasattr(self, 'field_detector') and self.field_detector:
                    if 'field_threshold' in processing_settings:
                        # self.field_detector.set_threshold(processing_settings['field_threshold'])
                        pass
                
                # Store settings for future reference
                self.processing_settings = processing_settings
            
            print(f"Settings updated - Camera: {camera_settings}, Processing: {processing_settings}")

        elif command.get('type') == 'white_balance_once':
            try:
                self.camera_command_queue.put_nowait({
                    'type': 'white_balance_once'
                })
            except Exception as e:
                print(f"ERROR: Failed to forward white balance command: {e}")

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

def camera_thread_func(raw_frame_queue, camera_command_queue, running_event):
    """Synchroner Thread, der Frames von der IDS-Kamera holt."""
    print("Camera-Thread gestartet.")
    camera = None
    
    # FPS tracking for camera
    camera_frame_count = 0
    camera_last_fps_time = time.time()
    camera_fps = 0.0
    
    def _handle_camera_command(camera, command):
        """Handle camera-specific commands"""
        
        if command.get('type') == 'update_camera_settings':
            settings = command.get('settings', {})
            
            if 'framerate' in settings:
                camera.set_frame_rate_target(settings['framerate'])
            if 'exposure_time' in settings:
                camera.set_exposure(settings['exposure_time'])
            if 'gain' in settings:
                camera.set_gain(settings['gain'])
            if 'black_level' in settings:
                camera.set_black_level(settings['black_level'])
            if 'white_balance_auto' in settings:
                wb_mode = "Continuous" if settings['white_balance_auto'] else "Off"
                camera.set_white_balance_auto(wb_mode)
        
        elif command.get('type') == 'white_balance_once':
            camera.set_white_balance_auto("Once")
    
    try:
        camera = IDS_Camera()
        camera.start()  # Startet die synchrone Akquisition
        
        while running_event.is_set():
            # Process camera commands first
            try:
                while not camera_command_queue.empty():
                    command = camera_command_queue.get_nowait()
                    _handle_camera_command(camera, command)
            except Exception as e:
                print(f"ERROR: Exception in camera command handling: {e}")
            
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
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())  # Dark mode for better visibility

    # 1. Erstelle Kommunikationsmittel
    # Ein Event, um alle Teile sauber zu beenden
    running_event = mp.Event()
    running_event.set()

    # Prozess-sichere Queues
    raw_frame_queue = mp.Queue(maxsize=5)
    results_queue = mp.Queue(maxsize=5)
    command_queue = mp.Queue(maxsize=10)  # Für UI-Kommandos
    camera_command_queue = mp.Queue(maxsize=10)  # Für Kamera-Kommandos

    # 2. Erstelle und starte den Verarbeitungs-Prozess
    processing_process = ProcessingProcess(raw_frame_queue, results_queue, command_queue, camera_command_queue, running_event)
    processing_process.start()

    # 3. Erstelle und starte den Kamera-Thread im Hauptprozess
    cam_thread = threading.Thread(target=camera_thread_func, args=(raw_frame_queue, camera_command_queue, running_event), daemon=True)
    cam_thread.start()
    
    # 4. Erstelle das Hauptfenster und übergebe die Kommunikationsmittel
    time.sleep(0.5)  # Kurze Pause, um sicherzustellen, dass der Thread gestartet ist
    window = KickerMainWindow(results_queue, command_queue, running_event)
    window.show()

    # Log initial messages
    window.add_log_message("Multi-Processing Kicker Klaus gestartet")
    window.add_log_message("Kamera-Thread läuft...")
    window.add_log_message("Processing-Prozess läuft...")
    window.add_log_message("Warten auf Video-Stream...")
    
    # Log frame filtering status
    if config.GUI_SHOW_ONLY_COMPLETE_DETECTIONS:
        window.add_log_message("Frame-Filterung aktiv: Nur vollständige Erkennungen werden angezeigt")
    else:
        window.add_log_message("Frame-Filterung deaktiviert: Alle Frames werden angezeigt")

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