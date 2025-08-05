# Kicker Klaus - Multi-Processing Version

## Überblick der Änderungen

Die `main.py` wurde komplett überarbeitet, um eine effiziente Multi-Processing-Architektur zu implementieren. Die alte Thread-basierte Struktur aus `main_live_gui_new.py` wurde durch echtes Multi-Processing ersetzt.

## Neue Architektur

### 1. ProcessingProcess
- **Eigenständiger Prozess** für alle Computer Vision Aufgaben
- Läuft parallel mit 250fps ohne UI-Blockierung
- Führt Ball-, Field- und Player-Detection parallel in Threads aus
- Integriert Goal Scoring direkt im Processing-Prozess
- Unterstützt Field-Kalibrierung

### 2. Camera Thread
- Läuft im Hauptprozess als Thread
- Liest kontinuierlich Frames von der IDS-Kamera
- Überträgt rohe Bayer-Frames an den Processing-Prozess

### 3. GUI (KickerMainWindow)
- Neue `qt_window_multiprocess.py` für Multi-Processing
- Pollt Ergebnisse vom Processing-Prozess über Timer
- Vollständige GUI mit Score-Anzeige, Match-Modi, Visualisierung
- Kommuniziert über Command-Queue mit Processing-Prozess

## Kommunikation zwischen Prozessen

### Queues:
- **raw_frame_queue**: Kamera → Processing (rohe Frames)
- **results_queue**: Processing → GUI (verarbeitete Ergebnisse + Visualisierung)
- **command_queue**: GUI → Processing (Kommandos wie Kalibrierung, Score-Reset)
- **running_event**: Multiprocessing Event für sauberes Herunterfahren

## Wichtige Features

### Ball Detection & Tracking
- Echte Parallelverarbeitung im Processing-Prozess
- Ball-Trail Visualisierung
- Confidence-basierte Farbanzeige
- Kalman-Filter für Geschwindigkeitsvorhersage

### Field Detection
- Field-Kalibrierung mit Fortschrittsanzeige
- Automatisches Laden gespeicherter Kalibrierungen
- Field-Bounds für optimierte Ball-Suche
- Goal-Erkennung

### Goal Scoring System
- Automatische Tor-Erkennung im Processing-Prozess
- Score-Updates werden automatisch an GUI übertragen
- Manuelle Score-Kontrolle in der GUI
- Match-Modi (Normal, Practice, Tournament)

### Visualisierung
- Ball Only / Field Only / Combined Modi
- Klickbare Video-Ausgabe für Farb-Picking
- Echzeit FPS-Anzeige
- System-Log mit Zeitstempel

### Match Management
- Game Mode Selection (Normal/Practice/Tournament)
- Start/Stop Match Funktionalität
- Score Reset mit Mode-Einschränkungen
- Manual Score +/- Buttons

## Verwendung

```bash
python main.py
```

Die Anwendung startet automatisch:
1. Processing-Prozess für Computer Vision
2. Kamera-Thread für Frame-Akquisition  
3. GUI für Benutzerinteraktion

## Vorteile der neuen Architektur

1. **Echte Parallelisierung**: Processing blockiert nicht die GUI
2. **Skalierbarkeit**: Einfach weitere Processing-Prozesse hinzufügbar
3. **Stabilität**: Prozess-Isolation verhindert GUI-Crashes bei CV-Fehlern
4. **Performance**: Optimale CPU/GPU-Auslastung durch Process-Separation
5. **Wartbarkeit**: Klare Trennung von GUI, Processing und Kamera-Logic

## Technische Details

- **Python Multiprocessing** mit spawn-Methode für Windows-Kompatibilität
- **PySide6** für moderne Qt-GUI
- **OpenCV** für Computer Vision
- **NumPy** für effiziente Datenverarbeitung
- **Queue-basierte** IPC für performante Kommunikation

Die neue Architektur ist bereit für Erweiterungen wie GPU-Processing, mehrere Kameras oder Netzwerk-Streaming.
