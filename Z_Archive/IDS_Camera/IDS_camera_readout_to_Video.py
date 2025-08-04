import time
import cv2
import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension
import threading
import queue
 


import os

# --- 1. KONFIGURATION ---
RAM_BUFFER_LIMIT_GB = 32.0
VIDEO_FILENAME = "02.avi"
VIDEO_CODEC = 'MJPG'
CAMERA_FPS = 250
EXPOSURE_TIME_US = 2000.0
GAIN_DB = 15.0
WHITE_BALANCE_MODE = "Continuous"
PIXEL_FORMAT = "BayerRG8"

# --- Globale Objekte ---
frame_queue = queue.Queue(maxsize=100)
stop_event = threading.Event()

def acquisition_thread_func(device, frame_queue_ref):
    # Diese Funktion ist bereits korrekt und bleibt unverändert.
    # Hier verwenden wir die korrigierte Version aus der vorherigen Antwort.
    stream = None
    nodemap = None
    try:
        nodemap = device.RemoteDevice().NodeMaps()[0]
        stream = device.DataStreams()[0].OpenDataStream()
        stream_nodemap = stream.NodeMaps()[0]
        stream_nodemap.FindNode("StreamBufferHandlingMode").SetCurrentEntry("NewestOnly")
        
        payload_size = int(nodemap.FindNode("PayloadSize").Value())
        buffer_count = 50 
        for _ in range(buffer_count):
            buf = stream.AllocAndAnnounceBuffer(payload_size)
            stream.QueueBuffer(buf)

        stream.StartAcquisition()
        nodemap.FindNode("AcquisitionStart").Execute()
        nodemap.FindNode("AcquisitionStart").WaitUntilDone()
        
        print("✅ Akquise-Thread gestartet. Empfange rohe Bilder mit Frame-IDs...")
        while not stop_event.is_set():
            try:
                buf = stream.WaitForFinishedBuffer(1000)
                
                frame_id = buf.FrameID()
                ipl_image = ids_peak_ipl_extension.BufferToImage(buf)
                frame_data = ipl_image.get_numpy_3D()
                
                frame_queue_ref.put((frame_id, frame_data.copy()), block=True, timeout=2)
                
                stream.QueueBuffer(buf)
            except queue.Full:
                print("\nFEHLER: Queue war 2s lang voll. Aufnahme wird abgebrochen.")
                stop_event.set()
                break
            except ids_peak.TimeoutException:
                if stop_event.is_set(): break
                continue
            except ids_peak.Exception as e:
                print(f"\nFehler im Akquise-Thread: {e}. Beende...")
                stop_event.set()
                break
    finally:
        print("\n🛑 Akquise-Thread wird beendet und räumt auf...")
        if stream:
            try:
                if stream.IsGrabbing():
                    stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                stream.revoke_and_free_all_buffers() # Korrigierter Methodenname
            except Exception as e:
                print(f"Fehler beim Aufräumen des Streams: {e}")

def record_to_ram_and_convert(device, width, height):
    ram_frame_buffer = []
    acq_thread = threading.Thread(target=acquisition_thread_func, args=(device, frame_queue))
    acq_thread.start()

    print("\n>>> AUFNAHME IN DEN RAM-PUFFER <<<")
    print(f"Speichere in RAM bis zu {RAM_BUFFER_LIMIT_GB:.1f} GB.")
    print("Drücken Sie im Konsolenfenster ENTER, um die Aufnahme manuell zu beenden.")
    
    def wait_for_quit():
        input() 
        print("\nEingabe erkannt. Beende Aufnahme...")
        stop_event.set()

    input_thread = threading.Thread(target=wait_for_quit)
    input_thread.start()

    # (Aufnahmeteil ist unverändert)
    total_bytes_used = 0
    start_time = time.time()
    while acq_thread.is_alive() or not frame_queue.empty():
        try:
            frame_id, raw_frame = frame_queue.get(timeout=1.0)
            ram_frame_buffer.append((frame_id, raw_frame))
            total_bytes_used += raw_frame.nbytes
            ram_usage_gb = total_bytes_used / (1024**3)
            elapsed_time = time.time() - start_time
            current_fps = len(ram_frame_buffer) / elapsed_time if elapsed_time > 0 else 0
            print(f"\rQueue: {frame_queue.qsize():03d} | Gesp. Frames: {len(ram_frame_buffer):05d} | RAM: {ram_usage_gb:.2f} GB | FPS: {current_fps:.1f}", end="")
            if ram_usage_gb >= RAM_BUFFER_LIMIT_GB:
                print("\nRAM-Limit erreicht! Beende die Aufnahme automatisch.")
                stop_event.set()
                break
        except queue.Empty:
            if not acq_thread.is_alive(): break 
            else: continue
            
    acq_thread.join()
    print(f"\n✅ Aufnahme beendet. {len(ram_frame_buffer)} Frames im RAM-Puffer.")

    if not ram_frame_buffer:
        print("Keine Frames aufgenommen. Es wird kein Video erstellt.")
        return

    # --- SCHRITT 2: Konvertierung mit Zähler für verlorene Frames ---
    print("\n>>> VIDEO-KONVERTIERUNG MIT DROPPED-FRAME-HANDLING <<<")
    
    ram_frame_buffer.sort(key=lambda x: x[0])

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, CAMERA_FPS, (width, height), isColor=True)

    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # NEU: Zählvariablen initialisieren
    total_dropped_frames = 0
    frames_processed_from_buffer = 0
    
    # NEU: Robuste Logik mit "expected_frame_id"
    expected_frame_id = ram_frame_buffer[0][0]
    
    for actual_frame_id, raw_frame in ram_frame_buffer:
        # Berechne die Anzahl der verlorenen Frames
        dropped_count = actual_frame_id - expected_frame_id
        
        if dropped_count > 0:
            print(f"\nWarnung: {dropped_count} Frame(s) zwischen ID {expected_frame_id - 1} und {actual_frame_id} verloren. Füge schwarze Bilder ein.")
            for _ in range(dropped_count):
                video_writer.write(black_frame)
            # NEU: Addiere die verlorenen Frames zum Gesamtzähler
            total_dropped_frames += dropped_count

        # Verarbeite den regulären Frame (mit Ihrer korrekten Farbkonvertierung)
        color_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BAYER_RG2RGB)
        video_writer.write(color_frame)
        
        # Setze die nächste erwartete ID
        expected_frame_id = actual_frame_id + 1
        
        frames_processed_from_buffer += 1
        
        # NEU: Aktualisierte Fortschrittsanzeige
        print(f"\rVerarbeite Puffer-Frame {frames_processed_from_buffer}/{len(ram_frame_buffer)} | Verlorene Frames: {total_dropped_frames}", end="")

    video_writer.release()
    
    # NEU: Detaillierte Zusammenfassung am Ende
    total_frames_in_video = len(ram_frame_buffer) + total_dropped_frames
    print(f"\n\n✅ Video '{VIDEO_FILENAME}' erfolgreich erstellt.")
    print("--- Zusammenfassung ---")
    print(f"  - Aufgenommene Frames:      {len(ram_frame_buffer)}")
    print(f"  - Verlorene Frames (gedroppt): {total_dropped_frames}")
    print(f"  -------------------------------------")
    print(f"  - Gesamt-Frames im Video:   {total_frames_in_video}")

def main():
    ids_peak.Library.Initialize()
    device = None
    try:
        dm = ids_peak.DeviceManager.Instance()
        dm.Update()
        if dm.Devices().empty(): raise RuntimeError("Keine Kamera gefunden!")
        device = dm.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        nodemap = device.RemoteDevice().NodeMaps()[0]
        
        # Optional: Frame-Zähler der Kamera zurücksetzen für konsistente Starts
        try:
            nodemap.FindNode("CounterReset").Execute()
            print("INFO: Frame-Zähler der Kamera zurückgesetzt.")
        except ids_peak.Exception:
            print("WARNUNG: 'CounterReset' nicht verfügbar. Frame-IDs starten möglicherweise nicht bei 0.")

        print("--- Kamera wird konfiguriert ---")
        nodemap.FindNode("PixelFormat").SetCurrentEntry(PIXEL_FORMAT)
        try:
            nodemap.FindNode("AcquisitionFrameRateTargetEnable").SetValue(True)
        except ids_peak.Exception: pass
        
        fps_node = nodemap.FindNode("AcquisitionFrameRateTarget")
        fps_node.SetValue(CAMERA_FPS)
        nodemap.FindNode("ExposureTime").SetValue(EXPOSURE_TIME_US)
        nodemap.FindNode("Gain").SetValue(GAIN_DB)
        nodemap.FindNode("BalanceWhiteAuto").SetCurrentEntry(WHITE_BALANCE_MODE)
        
        width = nodemap.FindNode("Width").Value()
        height = nodemap.FindNode("Height").Value()
        
        print(f"Auflösung: {width}x{height}, FPS: {CAMERA_FPS}, PixelFormat: {PIXEL_FORMAT}")

        record_to_ram_and_convert(device, width, height)

    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {e}")
        stop_event.set()
    finally:
        ids_peak.Library.Close()
        print("\nProgramm beendet.")

if __name__ == "__main__":
    main()