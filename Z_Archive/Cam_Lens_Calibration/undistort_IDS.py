import cv2
import time
import json
import numpy as np

from IDS_readout import IDSReadout

class UndistortVideo:
    def __init__(self):
        self.cam = None
        self.cameraMatrix = None
        self.distCoeffs = None
        self.newCameraMatrix = None
        self.map1 = None
        self.map2 = None
        self.shape = None
        self.compute_time = []
        self.initialized = False

        self.cam = IDSReadout()
        if not self.cam.initialized:
            print("Kamera konnte nicht initialisiert werden.")
            exit()

        # JSON laden
        with open('calibration_data.json', 'r') as f:
            calib_data = json.load(f)

        # Als NumPy-Arrays rekonstruieren
        self.cameraMatrix = np.array(calib_data['cameraMatrix'])
        self.distCoeffs = np.array(calib_data['distCoeffs'])

        print("Geladene Kamera-Matrix:\n", self.cameraMatrix)
        print("Geladene Distortion-Koeffizienten:\n", self.distCoeffs)

        self.shape = self.cam.read_frame().shape[:2]
        (h, w) = self.shape

        # Neue Kameramatrix
        self.newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, (w, h), 1, (w, h))

        # Remap-Tabellen vorberechnen
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.cameraMatrix, self.distCoeffs, None, self.newCameraMatrix, (w, h), cv2.CV_16SC2
        )
        self.initialized = True

    def undistort_frame(self):  
        print("Starte Video-Stream... Drücke 'ESC' zum Beenden.")
        while True:
            frame = self.cam.read_frame()
            if frame is None:
                print("Fehler beim Lesen des Frames.")
                continue

            prev_time = time.time()
            
            # Entzerren mit remap
            undistorted = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            
            current_time = time.time()
            self.compute_time.append(current_time - prev_time)

            # Anzeigen
            undistorted = cv2.resize(undistorted, (720, 540))  # Optional: Framegröße anpassen
            cv2.imshow('Undistorted', undistorted)

            if cv2.waitKey(1) == 27:  # ESC-Taste
                break
        
    def cleanup(self):
        """Aufräumen und Ressourcen freigeben"""
        try:
            # Aufräumen
            self.cam.cleanup()
            cv2.destroyAllWindows()
            print("Kamera und Fenster wurden erfolgreich geschlossen")
            
            time_mean = 0
            for t in self.compute_time:
                time_mean += t
            time_mean /= len(self.compute_time)

            print(f'Durchschnittliche Verarbeitungszeit (Undistort) pro Frame: {time_mean}ms')
        except Exception as e:
            print(f"Fehler beim Aufräumen: {e}")

if __name__ == "__main__":
    undistorter = UndistortVideo()
    if undistorter.initialized:
        undistorter.undistort_frame()
    else:
        print("Undistorter konnte nicht initialisiert werden.")
    undistorter.cleanup()
