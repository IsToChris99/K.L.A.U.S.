import cv2
import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension
from ids_peak_ipl import ids_peak_ipl

class IDSReadout:
    def __init__(self):
        self.device = None
        self.nodemap = None
        self.stream = None
        self.converter = None
        self.width = None
        self.height = None
        self.tgt_pf = None
        self.initialized = False
        
        try:
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

            self.device = device
            # 3) Nodemap holen
            self.nodemap = device.RemoteDevice().NodeMaps()[0]

            # ─── Beispiel: Kamera-Parameter setzen ───────────────────────────────────────
            # AcquisitionFrameRate (maximale FPS)
            try:
                fps_value = 60.00
                fps_node = self.nodemap.FindNode("AcquisitionFrameRate")
                fps_node.SetValue(fps_value)   # z.B. auf 120 FPS setzen
            except Exception:
                print("AcquisitionFrameRate nicht verfügbar")

            # Belichtungszeit in µs (ExposureTime)
            try:
                exp_node = self.nodemap.FindNode("ExposureTime")
                exp_node.SetValue(2000.0)
            except Exception:
                print("ExposureTime nicht verfügbar")

            # Verstärkung (Gain)
            try:
                gain_node = self.nodemap.FindNode("Gain")
                gain_node.SetValue(10.0)   # z.B. +10 dB
            except Exception:
                print("Gain nicht verfügbar")

            # ROI (AOI) – Breite, Höhe, Offset
            try:
                # CG: Hier kann die ROI (Region of Interest) angepasst werden.
                width = 1440
                height = 1080
                width_max  = int(self.nodemap.FindNode("WidthMax").Value())
                height_max = int(self.nodemap.FindNode("HeightMax").Value())
                
                self.nodemap.FindNode("Width").SetValue(width)
                self.nodemap.FindNode("Height").SetValue(height)
                
                print(f"Maximale Auflösung: {width_max}x{height_max}")              # CG: Hier wird die maximale Auflösung ausgegeben
                
                #self.nodemap.FindNode("OffsetX").SetValue(100)
                #self.nodemap.FindNode("OffsetY").SetValue(100)
            except Exception:
                print("ROI-Einstellungen nicht verfügbar")

            # ─── Rest wie gehabt: DataStream, Puffer, Conversion, Acquisition ────────────

            self.stream = device.DataStreams()[0].OpenDataStream()
            payload_size = int(self.nodemap.FindNode("PayloadSize").Value())
            buf_count = self.stream.NumBuffersAnnouncedMinRequired()
            for _ in range(buf_count):
                buf = self.stream.AllocAndAnnounceBuffer(payload_size)
                self.stream.QueueBuffer(buf)

            # PixelFormat
            try:
                pf_node = self.nodemap.FindNode("PixelFormat")
                pf_node.SetValue("BGR8_Packed")
            except Exception:
                pass

            # Converter vorbereiten
            self.width = int(self.nodemap.FindNode("Width").Value())
            self.height = int(self.nodemap.FindNode("Height").Value())
            inp_pf = ids_peak_ipl.PixelFormat(pf_node.CurrentEntry().Value())
            self.tgt_pf = ids_peak_ipl.PixelFormatName_BGR8
            self.converter = ids_peak_ipl.ImageConverter()
            self.converter.PreAllocateConversion(inp_pf, self.tgt_pf, self.width, self.height)

            # Acquisition starten
            self.stream.StartAcquisition()
            self.nodemap.FindNode("AcquisitionStart").Execute()
            self.nodemap.FindNode("AcquisitionStart").WaitUntilDone()
            
            self.initialized = True
            print("IDS Kamera erfolgreich initialisiert")
            
        except Exception as e:
            print(f"Fehler bei der Kamera-Initialisierung: {e}")
            self.cleanup()

    def read_frame(self):
        if not self.initialized:
            return None
            
        try:
            buf = self.stream.WaitForFinishedBuffer(1000)

            # Konvertieren
            ipl_img = ids_peak_ipl_extension.BufferToImage(buf)
            converted = self.converter.Convert(ipl_img, self.tgt_pf)

            # Puffer zurückgeben
            self.stream.QueueBuffer(buf)

            # NumPy-Array formen
            arr1d = converted.get_numpy_1D()
            frame = np.frombuffer(arr1d, dtype=np.uint8)
            frame = frame.reshape((self.height, self.width, 3))      

            return frame  # Hier wird das Frame zurückgegeben, um es weiter zu verarbeiten
        except Exception as e:
            print(f"Fehler beim Lesen des Frames: {e}")
            return None
                    
    def cleanup(self):
        try:
            # Aufräumen
            if self.nodemap is not None:
                self.nodemap.FindNode("AcquisitionStop").Execute()
            if self.stream is not None:
                self.stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                self.stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                for b in self.stream.AnnouncedBuffers():
                    self.stream.RevokeBuffer(b)
            
            ids_peak.Library.Close()
            cv2.destroyAllWindows()
            print("Kamera wurde erfolgreich geschlossen")
        except Exception as e:
            print(f"Fehler beim Aufräumen: {e}")
    
    def run_live_feed(self):
        """Hauptausführungsfunktion für Live-Feed"""
        if not self.initialized:
            print("Kamera wurde nicht korrekt initialisiert!")
            return
            
        try:
            print("Live-Feed gestartet. Drücken Sie 'q' zum Beenden.")
            while True:
                frame = self.read_frame()
                if frame is None:
                    break
                
                cv2.imshow("Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("\nLive-Feed durch Benutzer beendet")
        finally:
            self.cleanup()

# Beispielaufruf
if __name__ == "__main__":
    ids_readout = IDSReadout()
    
    try:
        print("Starte Live-Feed... Zum Beenden 'ESC' drücken")
        while True:
            frame = ids_readout.read_frame()
            if frame is None:
                print('Kein Frame verhanden')
                continue
            else:
                frame = cv2.resize(frame, (720, 540))  # Optional: Framegröße anpassen
                cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) == 27:
                print("Beende Live-Feed...")
                break
    finally:
        ids_readout.cleanup()