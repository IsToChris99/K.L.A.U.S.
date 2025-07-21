import cv2
import time

output_path = 'output.avi'

# Codec für das Ausgabevideo
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Erstellen des VideoCapture-Objekts
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)  # FPS des Eingabevideos
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # Breite des Eingabevideos
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Höhe des Eingabevideos

# Überprüfen, ob das Video erfolgreich geöffnet wurde
if not cap.isOpened():
    print("Error opening video file")
    exit()



# Erstellen des VideoWriter-Objekts
out = cv2.VideoWriter(output_path, fourcc, fps, frameSize=(int(width), int(height)))

prev_time = time.time()
fps = 0.0
fps_mean = 0.0
frame_counter = 1

# Lesen und Schreiben des Videos
while(cap.isOpened()):
    ret, frame = cap.read()

    now = time.time()
    dt = now - prev_time
    if dt > 0:
        fps = 1.0 / dt
    prev_time = now
    fps_mean = (fps_mean * frame_counter + fps) / (frame_counter+1)
    frame_counter += 1

    if ret == True:

        out.write(frame) #Frame schreiben

        text = f"FPS: {fps_mean:.1f}"

        cv2.putText(frame, text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2, cv2.LINE_AA) 

        # Anzeige des Frames (optional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == 27:  # 'Esc' Taste
            break
    else:
        print("Fehler beim Lesen des Frames.")
        break

# Freigeben der Ressourcen
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video erfolgreich gespeichert")