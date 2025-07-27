import cv2
import numpy as np
import os
import json
import glob

from IDS_readout import IDSReadout

def main():
    # ─── IDS Kamera initialisieren ─────────────────────────────────────────────
    cam = IDSReadout()
    if not cam.initialized:
        print("Kamera konnte nicht initialisiert werden.")
        return

    # ─── Setup für Kalibrierung ───────────────────────────────────────────────
    CHECKERBOARD = (9, 6)    # innere Ecken (nicht Kacheln!)
    square_size = 0.10        # 10 cm
    SAVE_DIR = "calib_images" # Ordner zum Speichern der Bilder
    os.makedirs(SAVE_DIR, exist_ok=True)

    # === Vorbereitung ===
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    positions = {"left": 0, "right": 0, "top": 0, "bottom": 0, "center": 0}
    img_count = 0

    # ─── Programm Loop ───────────────────────────────────────────────
    try:
        # === Lade Bilder aus dem Ordner ===
        image_files = glob.glob('calib_images/*.jpg')
        print(f"{len(image_files)} Bilder gefunden. Einlesen und verarbeiten...")

        for fname in image_files:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret:
                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                objpoints.append(objp)
                imgpoints.append(refined_corners)
            else:
                print(f"Checkerboard nicht erkannt in {fname}")
        print('Einlesen abgeschlossen. Es wird mit der Aufnahme fortgefahren') 

        print("Kamera gestartet. Drücke 'Esc', um zu beenden.")
        while True:
            frame = cam.read_frame()
            if frame is None:
                print("Fehler beim Lesen des Frames.")
                continue       

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            display_frame = frame.copy()
            display_frame = cv2.resize(display_frame, (720, 540))  # Optional: Größe anpassen

            if ret_cb:
                cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners/2, ret_cb)

                # Eckenzentrum prüfen
                mean = np.mean(corners, axis=0)[0]
                h, w = gray.shape
                pos = []
                if mean[0] < w * 0.33:
                    pos.append("left")
                elif mean[0] > w * 0.66:
                    pos.append("right")
                else:
                    pos.append("center")
                if mean[1] < h * 0.33:
                    pos.append("top")
                elif mean[1] > h * 0.66:
                    pos.append("bottom")

                # Anzeige
                cv2.putText(display_frame, f"Position: {' & '.join(pos)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Checkerboard Capture', display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space → speichern
                if ret_cb:
                    objpoints.append(objp)
                    refined_corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    imgpoints.append(refined_corners)
                    img_filename = os.path.join(SAVE_DIR, f"calib_{img_count:03d}.jpg")
                    cv2.imwrite(img_filename, frame)
                    img_count += 1

                    # Zähler aktualisieren
                    for p in pos:
                        positions[p] += 1

                    print(f"Bild gespeichert: {img_filename}")
                    print("Aktuelle Verteilung:", positions)
                else:
                    print("Checkerboard nicht erkannt! Bitte neu ausrichten.")
                    
            # Warte auf Tastendruck
            if key == 27:    # ESC-Taste zum Beenden
                break
                    
    finally:
        cam.cleanup()
        # === Kalibrierung ===
        if len(objpoints) < 10:
            print("Nicht genug Bilder für Kalibrierung.")
            exit()

        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        # Ergebnisse ausgeben
        print("\n--- Kalibrierungsergebnisse ---")
        print("RMS-Fehler:", ret)
        print("cameraMatrix:\n", cameraMatrix)
        print("distCoeffs:\n", distCoeffs.ravel())

        with open('calibration_data.json', 'w') as f:
            json.dump({
                'cameraMatrix': cameraMatrix.tolist(),
                'distCoeffs': distCoeffs.ravel().tolist()
            }, f, indent=4)

        print("\nKalibrierungsdaten gespeichert als 'calibration_data.json'")
    
if __name__ == "__main__":
    main()
