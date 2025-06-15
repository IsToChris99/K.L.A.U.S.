import cv2
import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension
from ids_peak_ipl import ids_peak_ipl
import time
import threading
import queue

def main(): 
    # CG: Hier wird die Kamera initialisiert:
    device, nodemap = setup_camera(width=1456, height=1088, fps=500, exposure_time=2000.0, gain=2.0)  
    
    # ─── Rest wie gehabt: DataStream, Puffer, Conversion, Acquisition ────────────

    stream  = device.DataStreams()[0].OpenDataStream()
    payload_size = int(nodemap.FindNode("PayloadSize").Value())
    buf_count    = stream.NumBuffersAnnouncedMinRequired()
    for _ in range(buf_count):
        buf = stream.AllocAndAnnounceBuffer(payload_size)
        stream.QueueBuffer(buf)

    # PixelFormat
    try:
        pf_node = nodemap.FindNode("PixelFormat")
        pf_node.SetValue("BGR8_Packed")
    except Exception:
        pass

    # Converter vorbereiten
    width  = nodemap.FindNode("Width").Value()
    height = nodemap.FindNode("Height").Value()
    inp_pf = ids_peak_ipl.PixelFormat(pf_node.CurrentEntry().Value())
    tgt_pf = ids_peak_ipl.PixelFormatName_BGRa8
    converter = ids_peak_ipl.ImageConverter()
    converter.PreAllocateConversion(inp_pf, tgt_pf, width, height)

    # Acquisition starten
    stream.StartAcquisition()
    nodemap.FindNode("AcquisitionStart").Execute()
    nodemap.FindNode("AcquisitionStart").WaitUntilDone()

    # ─── Live-Loop mit FPS-Overlay ───────────────────────────────────────────────

    cv2.namedWindow("Live IDS Stream", cv2.WINDOW_NORMAL)
    prev_time = time.time()
    fps = 0.0
    fps_mean = 0.0
    frame_counter = 1

    # Frame-Queue zwischen den Threads
    frame_queue = queue.Queue(maxsize=5)  # Begrenze die Anzahl gepufferter Frames
    running = threading.Event()
    running.set()  # Setze Flag auf "True"

    # Video-Aufnahme konfigurieren
    is_recording = False
    video_writer = None
    
    def toggle_recording():                    # CG: Funktion zum Starten/Stoppen der Videoaufnahme
        nonlocal is_recording, video_writer
        
        if is_recording:
            # Aufnahme stoppen
            if video_writer:
                video_writer.release()
            is_recording = False
            print("Aufnahme beendet.")
        else:
            # Aufnahme starten
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"IDS_Camera/video_outputs/ids_aufnahme_{timestamp}.avi"
            fourcc = cv2.VideoWriter.fourcc(*'MJPG')  # Verwende mp4v für MP4-Dateien
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            is_recording = True
            if video_writer.isOpened():
                print(f"Aufnahme gestartet: {output_file}")
            else:
                print("Fehler beim Öffnen des Video Writers. Aufnahme nicht gestartet.")
    
    # Thread für die Bildaufnahme
    def capture_thread():
        while running.is_set():
            buf = stream.WaitForFinishedBuffer(1000)
            if buf is None:
                continue

            # Konvertieren
            ipl_img = ids_peak_ipl_extension.BufferToImage(buf)
            converted = converter.Convert(ipl_img, tgt_pf)
            
            # NumPy-Array formen
            arr1d = converted.get_numpy_1D()
            frame = np.frombuffer(arr1d, dtype=np.uint8)
            frame = frame.reshape((height, width, 4))
            #frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)     # CG: Wirkt sich stark auf die Framerate aus (ca. -140 FPS)!
            
            # Puffer zurückgeben (wichtig: sofort nach Konvertierung!)
            stream.QueueBuffer(buf)
            
            # Frame in die Queue legen, ohne zu blockieren wenn voll
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                # Alten Frame überspringen wenn Queue voll
                continue

    # Thread starten
    capture_t = threading.Thread(target=capture_thread)
    capture_t.daemon = True
    capture_t.start()

    try:
        print("Kamera gestartet. Drücke 'Esc', um zu beenden oder 'r' um eine Aufnahme zu starten.")
        text = "FPS-Anzeige"
        while True:
            # Frame aus der Queue nehmen
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # FPS (mean) berechnen und anzeigen
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now
            fps_mean = (fps_mean * frame_counter + fps) / (frame_counter + 1)
            frame_counter += 1
            
            # Frame aufnehmen wenn Aufnahme aktiv
            if is_recording and video_writer:
                
                video_writer.write(frame)
                # Aufnahme-Indikator anzeigen
                cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)
                            
            # if frame_counter % 100 == 0:
            #     text = f"FPS: {fps_mean:.1f}"
                                           
            display_every_nth_frame = 4  # Nur jeden vierten Frame anzeigen
            
            if frame_counter % display_every_nth_frame != 0:
                # cv2.putText(frame, text, (10, 25),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #     (0, 255, 0), 2, cv2.LINE_AA) 
                cv2.imshow("Live IDS Stream", frame)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('r'):  # 'r' zum Starten/Stoppen der Aufnahme
                toggle_recording()
                
    finally:
        # Threads beenden
        running.clear()
        capture_t.join(timeout=2.0)
        
        print(f"Programm wird beendet. Durchschnittliche FPS waren: {fps_mean:.1f}")
        
        # Aufräumen
        if video_writer:
            video_writer.release()
            
        # Acquisition stoppen und Puffer freigeben
        nodemap.FindNode("AcquisitionStop").Execute()
        stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        for b in stream.AnnouncedBuffers():
            stream.RevokeBuffer(b)
        
        #device.CloseDevice()                 #CG: die Funktion device.CloseDevice() gibt es nicht

        ids_peak.Library.Close()
        cv2.destroyAllWindows()

def setup_camera(width, height, fps=200.0, exposure_time=2000.0, gain=2.0):
    # 1) IDS Peak initialisieren
    ids_peak.Library.Initialize()

    # 2) Erstes Gerät im Control-Modus öffnen
    
    device_manager = ids_peak.DeviceManager.Instance()
    device_manager.Update()
    devices = device_manager.Devices()
    
    if len(devices) == 0:
        print("Keine IDS Kamera gefunden oder nicht openable. Ist das IDS peak Cockpit noch geöffnet?")
        return
    
    # Erste Kamera öffnen
    device = devices[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    print(f"Verbundene Kamera: {device.ModelName}")

    # 3) Nodemap holen
    nodemap = device.RemoteDevice().NodeMaps()[0]

    # ─── Beispiel: Kamera-Parameter setzen ───────────────────────────────────────
    # AcquisitionFrameRate (maximale FPS)
    try:
        fps_node = nodemap.FindNode("AcquisitionFrameRate")
        fps_max = nodemap.FindNode("AcquisitionFrameRateMax").Value()
        if fps > fps_max:
            fps = fps_max
        elif fps < 1:
            fps = 1
        else:
            fps_node.SetValue(fps)   # z.B. auf 120 FPS setzen
    except Exception:
        print("AcquisitionFrameRate nicht verfügbar")

    # Belichtungszeit in µs (ExposureTime)
    try:
        exp_node = nodemap.FindNode("ExposureTime")
        exp_node.SetValue(exposure_time)             # CG: 2000.0 sind 2 ms
    except Exception:
        print("ExposureTime nicht verfügbar")

    # Verstärkung (Gain)
    try:
        gain_node = nodemap.FindNode("Gain")
        gain_node.SetValue(gain)   # z.B. +10 dB
    except Exception:
        print("Gain nicht verfügbar")

    # ROI (AOI) – Breite, Höhe, Offset
    try:
        # CG: Hier kann die ROI (Region of Interest) angepasst werden.
        
        width_max  = int(nodemap.FindNode("WidthMax").Value())
        height_max = int(nodemap.FindNode("HeightMax").Value())        
        
        if width > 0 and width <= width_max and height > 0 and height <= height_max:
            nodemap.FindNode("Width").SetValue(width)
            nodemap.FindNode("Height").SetValue(height)
        else:
            width  = width_max
            height = height_max
    
        nodemap.FindNode("Width").SetValue(width)
        nodemap.FindNode("Height").SetValue(height)
        
        # CG: Hier wird die maximale Auflösung ausgegeben
        print(f"Maximale Auflösung: {width_max}x{height_max}. Aktuelle Auflösung: {nodemap.FindNode("Width").Value()}x{nodemap.FindNode("Height").Value()}")
        
        #nodemap.FindNode("OffsetX").SetValue(100)
        #nodemap.FindNode("OffsetY").SetValue(100)
        
    except Exception:
        print("ROI-Einstellungen nicht verfügbar")
    
    return device, nodemap

if __name__ == "__main__":
    main()
