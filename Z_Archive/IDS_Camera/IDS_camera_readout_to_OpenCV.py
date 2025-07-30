import time
import cv2
import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension
import threading

# --- Globale Variablen für den Datenaustausch ---
# Sperre, um den Zugriff auf die geteilten Daten zu synchronisieren
data_lock = threading.Lock() 
# Hier speichern wir das letzte rohe Bayer-Bild und die zugehörigen Statistiken
latest_raw_frame = None
stats = {
    "delta_t_ms": 0.0,
    "acquisition_fps": 0.0,
    "processing_fps": 0.0, # Misst die Geschwindigkeit des Akquise-Threads
    "dropped_frames": 0,
    "latency_ms": 0.0      # Zeigt den Jitter ohne den Clock Drift
}
# Variable für den sich anpassenden Zeit-Offset zwischen den Uhren
adaptive_offset_ns = 0
# Signal, um die Threads sauber zu beenden
stop_event = threading.Event()

def acquisition_thread_func(device):
    """
    Dieser Thread verwendet die Treiber-interne "NewestOnly"-Strategie und
    einen adaptiven Offset, um den Clock Drift bei der Latenzmessung zu kompensieren.
    """
    global latest_raw_frame, stats, adaptive_offset_ns
    
    stream = None
    nodemap = None
    
    try:
        nodemap = device.RemoteDevice().NodeMaps()[0]
        stream = device.DataStreams()[0].OpenDataStream()
        
        # Holen der NodeMap des Streams, um den Buffer-Modus zu setzen
        stream_nodemap = stream.NodeMaps()[0]
        stream_nodemap.FindNode("StreamBufferHandlingMode").SetCurrentEntry("NewestOnly")
        print("Stream-Puffer-Modus auf 'NewestOnly' (Treiber-intern) gesetzt.")
        
        payload_size = int(nodemap.FindNode("PayloadSize").Value())
        buffer_count = 20
        for _ in range(buffer_count):
            buf = stream.AllocAndAnnounceBuffer(payload_size)
            stream.QueueBuffer(buf)
            
        # Akquise normal starten
        stream.StartAcquisition()
        nodemap.FindNode("AcquisitionStart").Execute()
        nodemap.FindNode("AcquisitionStart").WaitUntilDone()
        
        last_timestamp_ns, last_frame_id, total_dropped_frames = 0, 0, 0
        is_first_frame = True
        
        # Glättungsfaktor für den exponentiellen gleitenden Durchschnitt (EMA)
        # Ein kleinerer Wert bedeutet langsamere Anpassung und mehr Glättung.
        alpha = 0.005
        
        processing_fps_counter, processing_start_time = 0, time.time()
        
        print("Akquise-Thread gestartet mit adaptivem Offset...")

        while not stop_event.is_set():
            try:
                buf = stream.WaitForFinishedBuffer(1000)
                pc_time_arrival_ns = time.time_ns()
            except ids_peak.Exception:
                print("Timeout im Akquise-Thread. Beende...")
                break

            buffer_frame_count = stream_nodemap.FindNode("StreamInputBufferCount").Value()
            print(f"\rFrames in Buffer: {buffer_frame_count}", end='')

            camera_time_capture_ns = buf.Timestamp_ns()
            current_frame_id = buf.FrameID()
            
            delta_t_ms, acquisition_fps, latency_ms = 0.0, 0.0, 0.0
            
            # ADAPTIVE OFFSET LOGIK ZUR KOMPENSATION DES CLOCK DRIFTS
            current_raw_offset = pc_time_arrival_ns - camera_time_capture_ns
            
            if is_first_frame:
                is_first_frame = False
                # Beim ersten Frame den Offset direkt setzen
                adaptive_offset_ns = current_raw_offset
            else:
                # Bei allen weiteren Frames den Offset sanft anpassen (Low-Pass Filter)
                adaptive_offset_ns = int((1 - alpha) * adaptive_offset_ns + alpha * current_raw_offset)
                
                # Die "echte" Latenz ist die Abweichung vom geglätteten Offset
                latency_ns = current_raw_offset - adaptive_offset_ns
                latency_ms = latency_ns / 1_000_000.0

                # Delta-t Berechnung bleibt unberührt und korrekt
                delta_t_ms = (camera_time_capture_ns - last_timestamp_ns) / 1_000_000.0
                if delta_t_ms > 0: acquisition_fps = 1000.0 / delta_t_ms
                
                frame_id_diff = current_frame_id - last_frame_id
                if frame_id_diff > 1: total_dropped_frames += (frame_id_diff - 1)
            
            last_timestamp_ns, last_frame_id = camera_time_capture_ns, current_frame_id
            
            processing_fps_counter += 1
            current_time = time.time()
            if current_time - processing_start_time >= 1.0:
                elapsed = current_time - processing_start_time
                with data_lock: stats["processing_fps"] = processing_fps_counter / elapsed
                processing_fps_counter, processing_start_time = 0, current_time

            ipl_image = ids_peak_ipl_extension.BufferToImage(buf)
            frame_data = ipl_image.get_numpy_3D()
            stream.QueueBuffer(buf)
            
            with data_lock:
                latest_raw_frame = frame_data.copy()
                stats["delta_t_ms"] = delta_t_ms
                stats["acquisition_fps"] = acquisition_fps
                stats["dropped_frames"] = total_dropped_frames
                stats["latency_ms"] = latency_ms

    finally:
        # Robuster Aufräum-Block
        print("Akquise-Thread wird beendet und räumt auf...")
        if stream and nodemap:
            try:
                nodemap.FindNode("AcquisitionStop").Execute()
                stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                for b in stream.AnnouncedBuffers():
                    stream.RevokeBuffer(b)
            except Exception as e:
                print(f"Fehler beim Aufräumen im Thread: {e}")

def main():
    """
    Haupt-Thread: Initialisiert die Kamera, startet den Akquise-Thread
    und kümmert sich um die langsame Anzeige der Bilder.
    """
    global latest_raw_frame, stats

    ids_peak.Library.Initialize()
    
    device = None
    try:
        dm = ids_peak.DeviceManager.Instance()
        dm.Update()
        if dm.Devices().empty(): raise RuntimeError("Keine Kamera gefunden")
        device = dm.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        nodemap = device.RemoteDevice().NodeMaps()[0]

        # --- Kamera-Einstellungen ---
        
        try:
            nodemap.FindNode("PixelFormat").SetCurrentEntry("BayerRG8")
            print("PixelFormat auf BayerRG8 (1 Byte/Pixel) gesetzt.")
        except Exception as e:
            print(f"Konnte PixelFormat nicht auf BayerRG8 setzen: {e}")
            return
        try:
            value = nodemap.FindNode("AcquisitionFrameRateTargetEnable").Value()
            print(value)
            # Set AcquisitionFrameRateTargetEnable to false (bool)
            nodemap.FindNode("AcquisitionFrameRateTargetEnable").SetValue(True)
        except Exception: 
            print("'AcquisitionFrameRateEnable' Node nicht gefunden, wird ignoriert.")
        try:
            fps_node = nodemap.FindNode("AcquisitionFrameRateTarget")
            fps_node.SetValue(250.0)
            print(f"Kamera-Framerate eingestellt auf: {fps_node.Value()} fps")
        except Exception as e: print(f"FPS-Einstellung nicht möglich: {e}")
        try:
            exp_time = nodemap.FindNode("ExposureTime")
            # Belichtungszeit muss < 1/Framerate sein (1/250s = 4000µs)
            exp_time.SetValue(2000.0) 
            print(f"Belichtungszeit: {exp_time.Value()} µs")
        except Exception as e: print(f"ExposureTime nicht verfügbar: {e}")
        try:
            nodemap.FindNode("GainAuto").SetCurrentEntry("Off")
            gain_node = nodemap.FindNode("Gain")
            gain_node.SetValue(10.0) # In der alten Version war hier fälschlicherweise 10 statt 10.0
            print(f"Gain eingestellt auf: {gain_node.Value()} dB")
        except Exception as e: print(f"Gain-Einstellung nicht möglich: {e}")
        try:
            nodemap.FindNode("BalanceWhiteAuto").SetCurrentEntry("Continuous")
            print("Auto-Weißabgleich auf 'Continuous' gesetzt.")
        except Exception as e: print(f"Auto-Weißabgleich nicht möglich: {e}")
        try:
            nodemap.FindNode("BlackLevel").SetValue(10.0)
            print(f"BlackLevel eingestellt auf: {nodemap.FindNode('BlackLevel').Value()}")
        except Exception as e: print(f"BlackLevel-Einstellung nicht möglich: {e}")

        acq_thread = threading.Thread(target=acquisition_thread_func, args=(device,))
        acq_thread.start()

        cv2.namedWindow("Anzeige (ca. 30 FPS)", cv2.WINDOW_NORMAL)
        
        # ... der Rest der Funktion bleibt unverändert ...
        while acq_thread.is_alive():
            raw_frame_to_display, display_stats = None, {}
            with data_lock:
                if latest_raw_frame is not None:
                    raw_frame_to_display = latest_raw_frame.copy()
                    display_stats = stats.copy()
            
            if raw_frame_to_display is not None:
                # Korrektur der Shape, falls es ein 1D-Array ist
                if len(raw_frame_to_display.shape) == 1:
                     # Annahme: Höhe und Breite müssen bekannt sein oder ermittelt werden
                     # Diese Werte sollten idealerweise aus der Kamera-Konfiguration kommen
                     height, width = 1080, 1920 # Beispielwerte, ggf. anpassen!
                     raw_frame_to_display = raw_frame_to_display.reshape((height, width))

                color_frame = cv2.cvtColor(raw_frame_to_display, cv2.COLOR_BAYER_RG2RGB)
                font, font_scale, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                text_lines = [
                    f"Delta t (Aufnahme): {display_stats.get('delta_t_ms', 0):.2f} ms",
                    f"Akquisitions-FPS:   {display_stats.get('acquisition_fps', 0):.1f}",
                    f"Verarbeitungs-FPS:  {display_stats.get('processing_fps', 0):.1f} (im Akquise-Thread)",
                    f"Dropped Frames:     {display_stats.get('dropped_frames', 0)}",
                    f"Latenz (Jitter):    {display_stats.get('latency_ms', 0):.2f} ms"
                ]
                for i, line in enumerate(text_lines):
                    y = 30 + i * 30
                    cv2.putText(color_frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.imshow("Anzeige (ca. 30 FPS)", color_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("'q' gedrückt. Beende Threads...")
                stop_event.set()
                break
        
        acq_thread.join()


    finally:
        ids_peak.Library.Close()
        cv2.destroyAllWindows()
        print("Programm sauber beendet.")

if __name__ == "__main__":
    main()