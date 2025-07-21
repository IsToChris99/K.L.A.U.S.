import cv2
import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension
from ids_peak_ipl import ids_peak_ipl
import time

def main():
    # 1) IDS Peak initialisieren
    ids_peak.Library.Initialize()

    # 2) Erstes Gerät im Control-Modus öffnen
    dm = ids_peak.DeviceManager.Instance()
    dm.Update()
    if dm.Devices().empty():
        print("Keine Kamera gefunden!")
        return

    device = None
    for dev in dm.Devices():
        if dev.IsOpenable():
            device = dev.OpenDevice(ids_peak.DeviceAccessType_Control)
            # CG: Hier wird die Modellbezeichnung der Kamera ausgegeben - Funktioniert aber nicht richtig bei der 3040.
            print(f"Verbundene Kamera: {dev.ModelName}")
            break
    if device is None:
        print("Keine openable device. Ist das IDS peak Cockpit noch geöffnet?")     #CG: angepasst.
        return

    # 3) Nodemap holen
    nodemap = device.RemoteDevice().NodeMaps()[0]

    # ─── Beispiel: Kamera-Parameter setzen ───────────────────────────────────────
    # AcquisitionFrameRate (maximale FPS)
    try:
        fps_node = nodemap.FindNode("AcquisitionFrameRate")
        fps_node.SetValue(250.00)   # z.B. auf 120 FPS setzen
    except Exception:
        print("AcquisitionFrameRate nicht verfügbar")

    # Belichtungszeit in µs (ExposureTime)
    try:
        exp_node = nodemap.FindNode("ExposureTime")
        exp_node.SetValue(2000.0)                           # CG: Sind hier die 2000.0 = 2 ms?
    except Exception:
        print("ExposureTime nicht verfügbar")

    # Verstärkung (Gain)
    try:
        gain_node = nodemap.FindNode("Gain")
        gain_node.SetValue(2.0)   # z.B. +10 dB
    except Exception:
        print("Gain nicht verfügbar")

    # ROI (AOI) – Breite, Höhe, Offset
    try:
        # CG: Hier kann die ROI (Region of Interest) angepasst werden.
        width = 1000
        height = 1000
        width_max  = int(nodemap.FindNode("WidthMax").Value())
        height_max = int(nodemap.FindNode("HeightMax").Value())
        
        nodemap.FindNode("Width").SetValue(width_max)
        nodemap.FindNode("Height").SetValue(height_max)
        
        print(f"Maximale Auflösung: {width_max}x{height_max}")              # CG: Hier wird die maximale Auflösung ausgegeben
        
        #nodemap.FindNode("OffsetX").SetValue(100)
        #nodemap.FindNode("OffsetY").SetValue(100)
    except Exception:
        print("ROI-Einstellungen nicht verfügbar")

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
    width  = int(nodemap.FindNode("Width").Value())
    height = int(nodemap.FindNode("Height").Value())
    inp_pf = ids_peak_ipl.PixelFormat(pf_node.CurrentEntry().Value())
    tgt_pf = ids_peak_ipl.PixelFormatName_BGR8
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
    frame_count = 1

    try:
        print("Kamera gestartet. Drücke 'Esc', um zu beenden.")
        while True:
            buf = stream.WaitForFinishedBuffer(1000)
            if buf is None:
                continue

            # Konvertieren
            ipl_img   = ids_peak_ipl_extension.BufferToImage(buf)
            converted = converter.Convert(ipl_img, tgt_pf)

            # Puffer zurückgeben
            stream.QueueBuffer(buf)

            # NumPy-Array formen
            arr1d = converted.get_numpy_1D()
            frame = np.frombuffer(arr1d, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))        
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) 
            #frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)           # CG: Wirkt sich stark auf die Framerate aus (ca. -140 FPS)!

            # FPS berechnen (auf Instant-Basis)
            now = time.time()
            dt  = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now
            
            # Durchschnittliche FPS berechnen (werden nach Beenden des Loops ausgegeben)       #CG
            fps_mean = (fps_mean * frame_count + fps) / (frame_count + 1)
            frame_count += 1

            """
            CG:      
            Das Overlay der FPS kann hier auskommentiert werden, um die Framerate um ca. 15 FPS zu erhöhen.
            Durchschnittliche FPS werden am Ende des Programms ausgegeben.
            """
            
            # Anzeige des Live-Bildes und Overlay: FPS oben links
            text = f"FPS: {fps:.1f}"
            cv2.putText(frame, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Live IDS Stream", frame)
            
            # Warte auf Tastendruck
            if cv2.waitKey(1) == 27:    # ESC-Taste zum Beenden
                break
            
    finally:
        print(f"Programm wird beendet. Durchschnittliche FPS waren: {fps_mean:.1f}")
        # Aufräumen
        nodemap.FindNode("AcquisitionStop").Execute()
        stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        for b in stream.AnnouncedBuffers():
            stream.RevokeBuffer(b)
        
        #device.CloseDevice()                 #CG: die Funktion device.CloseDevice() gibt es nicht

        ids_peak.Library.Close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
