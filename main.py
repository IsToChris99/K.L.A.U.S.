import sys
import time
import threading
import multiprocessing as mp
import queue
import numpy as np
from PySide6.QtWidgets import QApplication

# Lokale Imports bleiben gleich
from detection.ball_detector import BallDetector
from detection.field_detector import FieldDetector
from detection.player_detector import PlayerDetector # Angenommen, Sie haben diese
from analysis.goal_scorer import GoalScorer
from input.ids_camera import IDS_Camera
from processing.cpu_preprocessor import CPUPreprocessor
# from processing.gpu_preprocessor import GPUPreprocessor
from display.qt_window import KickerMainWindow
import config

# ================== PROCESSING PROCESS ==================

class ProcessingProcess(mp.Process):
    """
    Ein dedizierter Prozess für die gesamte 250fps-Verarbeitungspipeline.
    Nimmt rohe Frames entgegen und gibt ein komplettes Ergebnispaket aus.
    """
    def __init__(self, raw_frame_queue, results_queue, running_event):
        super().__init__()
        self.raw_frame_queue = raw_frame_queue
        self.results_queue = results_queue
        self.running_event = running_event

        # Diese Objekte werden INNERHALB des neuen Prozesses erstellt
        self.ball_detector = None
        self.field_detector = None
        self.player_detector = None
        self.preprocessor = None # CPU oder GPU Preprocessor

        # Zustand für den "One-Frame-Lag"
        self.latest_M_persp = np.identity(3)
        self.latest_M_field = np.identity(3)

    def run(self):
        """Die Hauptschleife des Verarbeitungsprozesses."""
        print("ProcessingProcess gestartet.")

        # Initialisiere alle Objekte hier, um sicherzustellen, dass sie im richtigen Prozess leben
        self.ball_detector = BallDetector()
        self.field_detector = FieldDetector()
        self.player_detector = PlayerDetector() # Erstellen Sie Ihre Spieler-Erkennung
        self.preprocessor = CPUPreprocessor(config.CAMERA_CALIBRATION_FILE) # Oder GPU

        while self.running_event.is_set():
            try:
                # 1. Hole den nächsten rohen Frame
                raw_frame = self.raw_frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # 2. Preprocessing (Warp) mit der Matrix vom VORHERIGEN Frame
            # Dieser Schritt ist schnell (besonders auf der GPU) und kann sequentiell erfolgen.
            preprocessed_frame = self.preprocessor.process(raw_frame, self.latest_M_persp)
            
            # 3. Starte alle drei Erkennungen parallel mit Threads
            results = {} # Dictionary zum Sammeln der Thread-Ergebnisse
            
            # --- Thread-Funktionen definieren ---
            def field_task():
                # Arbeitet auf dem rohen Frame
                M_persp, M_field = self.field_detector.detect(raw_frame)
                results['M_persp'] = M_persp
                results['M_field'] = M_field

            def ball_task():
                # Arbeitet auf dem vorverarbeiteten Frame
                ball_data = self.ball_detector.detect_ball(preprocessed_frame)
                results['ball_data'] = ball_data

            def player_task():
                # Arbeitet auf dem vorverarbeiteten Frame
                player_data = self.player_detector.detect(preprocessed_frame)
                results['player_data'] = player_data

            # --- Threads erstellen und starten ---
            thread_field = threading.Thread(target=field_task)
            thread_ball = threading.Thread(target=ball_task)
            thread_player = threading.Thread(target=player_task)
            
            thread_field.start()
            thread_ball.start()
            thread_player.start()
            
            # --- Warten, bis alle Threads für diesen Frame fertig sind ---
            thread_field.join()
            thread_ball.join()
            thread_player.join()

            # 4. Aktualisiere die Matrizen für den NÄCHSTEN Frame
            if results.get('M_persp') is not None:
                # Hier könnte man auch die EMA-Glättung einbauen
                self.latest_M_persp = results['M_persp']
            if results.get('M_field') is not None:
                self.latest_M_field = results['M_field']
                
            # 5. Stelle ein komplettes Ergebnispaket für die UI zusammen
            # Die UI soll ein konsistentes Bild bekommen. Wir senden das vorverarbeitete
            # Bild, auf dem auch die Erkennung lief.
            final_package = {
                "display_frame": preprocessed_frame,
                "ball_data": results.get('ball_data'),
                "player_data": results.get('player_data'),
                # Füge hier weitere Statistiken hinzu, die in diesem Prozess berechnet werden.
                # Wichtig: Die Umrechnung in Feld-Koordinaten passiert jetzt in der UI oder 
                # einem nachgelagerten Thread dort, da die UI M_field braucht.
                "M_field": self.latest_M_field
            }

            # 6. Sende das Paket an die UI
            if not self.results_queue.full():
                 self.results_queue.put(final_package)

        print("ProcessingProcess beendet.")


# ================== CAMERA THREAD ==================

def camera_thread_func(raw_frame_queue, running_event):
    """Ein einfacher Thread im Hauptprozess, der nur die Kamera ausliest."""
    print("Camera-Thread gestartet.")
    try:
        camera = IDS_Camera()
        while running_event.is_set():
            bayer_frame, _ = camera.get_frame()
            if bayer_frame is not None and not raw_frame_queue.full():
                raw_frame_queue.put(bayer_frame)
            else:
                # Verhindert, dass der Thread bei einer vollen Queue hängt
                time.sleep(0.001)
    except Exception as e:
        print(f"Fehler im Kamera-Thread: {e}")
    finally:
        if 'camera' in locals() and camera is not None:
            camera.stop()
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

    # 2. Erstelle und starte den Verarbeitungs-Prozess
    processing_process = ProcessingProcess(raw_frame_queue, results_queue, running_event)
    processing_process.start()

    # 3. Erstelle und starte den Kamera-Thread im Hauptprozess
    cam_thread = threading.Thread(target=camera_thread_func, args=(raw_frame_queue, running_event), daemon=True)
    cam_thread.start()
    
    # 4. Erstelle das Hauptfenster und übergebe die Ergebnis-Queue
    # Das Fenster holt sich die Daten selbstständig über einen Timer.
    window = KickerMainWindow(results_queue, running_event) # KickerMainWindow muss angepasst werden
    window.show()

    # 5. Starte die Anwendung und warte auf das saubere Herunterfahren
    exit_code = app.exec()
    
    print("UI wurde geschlossen. Fahre die Pipeline herunter...")
    running_event.clear() # Signal an Prozess und Thread zum Beenden

    processing_process.join(timeout=5)
    if processing_process.is_alive():
        print("Processing-Prozess konnte nicht beendet werden, terminiere...")
        processing_process.terminate()

    cam_thread.join(timeout=2)
    
    print("Anwendung sauber heruntergefahren.")
    return exit_code

if __name__ == "__main__":
    # Wichtig für macOS und Windows
    mp.set_start_method('spawn', force=True)
    sys.exit(main_gui())