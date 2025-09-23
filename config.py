# config.py

# ================== VIDEO SETTINGS ==================
# VIDEO_PATH = "C:\\Users\\Tim\\OneDrive - TH Köln\\03_Hochschule\\6_Semester\\Kicker_Projekt\\Test-Videos\\IDS_Cam2\\08.avi"
VIDEO_PATH = "C:\\Users\\joshu\\OneDrive - TH Köln\\Kicker (Kicker Klaus) - General\\Kicker_Videos\\2025-08-06\\11.avi"
# VIDEO_PATH = "/Users/romanheck/Desktop/Hochschule/achtes Semester/AKAT/kicker_farbkalibrierung/13.avi"
IS_LIVE = False

CAM_WIDTH = 1440
CAM_HEIGHT = 1080

CAM_X_OFFSET = 8
CAM_Y_OFFSET = 4

DETECTION_WIDTH = 720
DETECTION_HEIGHT = 540

DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 540

REFERENCE_WIDTH = 720
REFERENCE_HEIGHT = 540

WIDTH_RATIO = DETECTION_WIDTH / REFERENCE_WIDTH
HEIGHT_RATIO = DETECTION_HEIGHT / REFERENCE_HEIGHT

AREA_RATIO = WIDTH_RATIO * HEIGHT_RATIO

# ================== CAMERA DEFAULT/INITIALIZE SETTINGS ==================
FRAME_RATE_TARGET = 250.0
EXPOSURE_TIME = 2000.0
GAIN = 15.0
BLACK_LEVEL = 10.0
WHITE_BALANCE_AUTO = "Off"


# ================== CALIBRATION FILES ==================
CAMERA_CALIBRATION_FILE = "calibration\\calibration_data.json"
FIELD_CALIBRATION_FILE = "detection\\field_calibration.json"

# ================== BALL DETECTION (HSV VALUES) ==================
# Yellow ball - main color
BALL_LOWER = (20, 100, 100)
BALL_UPPER = (40, 255, 255)

# #Farbeinstellungen HSV (orange)
# BALL_LOWER = (0, 120, 200)
# BALL_UPPER = (30, 255, 255)

# BALL_LOWER_ALT = (10, 120, 200)
# BALL_UPPER_ALT = (40, 255, 255)

# ================== FIELD DETECTION (HSV VALUES) ==================

# # Farbeinstellungen für grünes Spielfeld (HSV)
# FIELD_MARKER_LOWER = ((149/2 - 10), (0.4624*255 - 30), (0.3647*255 - 30))
# FIELD_MARKER_UPPER = ((149/2 + 10), 255, 255)

# # Alternative Farbwerte für verschiedene Beleuchtungen
# FIELD_MARKER_LOWER_ALT = ((149/2 - 10), (0.4624*255 - 30), (0.3647*255 - 30))
# FIELD_MARKER_UPPER_ALT = ((149/2 + 10), 255, 255)

# Farbeinstellungen für grünes Spielfeld (HSV)
FIELD_MARKER_LOWER = (55, 80, 30)
FIELD_MARKER_UPPER = (80, 255, 255)

# Goals (bright/white areas) - unused at the moment
GOAL_LOWER = (0, 0, 70)
GOAL_UPPER = (180, 255, 255)

# ================== TRACKING PARAMETERS ==================
# Ball tracking
BALL_SMOOTHER_WINDOW_SIZE = 500  # Optimized balance between smoothing and responsiveness
BALL_MAX_MISSING_FRAMES = 100
BALL_CONFIDENCE_THRESHOLD = 0.5

# Field tracking
FIELD_MIN_AREA = 50000  # Minimum field size in pixels
FIELD_STABILITY_FRAMES = 1  # Frames for stable detection
GOAL_DETECTION_CONFIDENCE = 0.7
MIN_GOAL_AREA = 800 * AREA_RATIO # Minimum size for goal detection
FIELD_WIDTH_M = 1.18  # 118 cm
FIELD_HEIGHT_M = 0.68 # 68 cm

# Goal scoring system
GOAL_DISAPPEAR_FRAMES = 15  # Frames without detection to count as goal
GOAL_REVERSAL_TIME_WINDOW = 3.0  # Seconds to check for goal reversal
GOAL_DIRECTION_THRESHOLD_DISTANCE = 200  # Distance threshold to goal for direction-based scoring

# ================== PERFORMANCE & DISPLAY ==================
DISPLAY_FPS = 30
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS

# GUI Update Settings
# GUI Performance Profiles:
# - "standard": 30 FPS, 3 items/update - gut für normale Hardware  
# - "smooth": 60 FPS, 5 items/update - empfohlen für flüssige Darstellung
# - "high_performance": 120 FPS, 8 items/update - für leistungsstarke Hardware
GUI_PERFORMANCE_PROFILE = "high_performance"  # "standard", "smooth", "high_performance"

# Automatische Konfiguration basierend auf Profil
if GUI_PERFORMANCE_PROFILE == "standard":
    GUI_UPDATE_FPS = 30
    GUI_MAX_ITEMS_PER_UPDATE = 3
    GUI_FPS_UPDATE_INTERVAL = 0.5
elif GUI_PERFORMANCE_PROFILE == "smooth":
    GUI_UPDATE_FPS = 60
    GUI_MAX_ITEMS_PER_UPDATE = 5
    GUI_FPS_UPDATE_INTERVAL = 0.2
elif GUI_PERFORMANCE_PROFILE == "high_performance":
    GUI_UPDATE_FPS = 120
    GUI_MAX_ITEMS_PER_UPDATE = 8
    GUI_FPS_UPDATE_INTERVAL = 0.1
else:
    # Fallback auf smooth
    GUI_UPDATE_FPS = 60
    GUI_MAX_ITEMS_PER_UPDATE = 5
    GUI_FPS_UPDATE_INTERVAL = 0.2

GUI_UPDATE_INTERVAL_MS = int(1000 / GUI_UPDATE_FPS)  # Millisekunden zwischen GUI-Updates

# Frame filtering - only send frames to GUI when detections are complete
GUI_SHOW_ONLY_COMPLETE_DETECTIONS = True  # True: Nur Frames mit Field und Player anzeigen, False: Alle Frames anzeigen

# Morphological operations - kernel sizes
FIELD_CLOSE_KERNEL_SIZE = (20, 20)
FIELD_OPEN_KERNEL_SIZE = (10, 10)
GOAL_KERNEL_SIZE = (15, 15)
GOAL_THIN_FILTER_KERNEL_SIZE = (10, 10)

# ================== VISUALIZATION ==================
# Colors (BGR format)
COLOR_BALL_HIGH_CONFIDENCE = (0, 255, 0)      # Green
COLOR_BALL_MED_CONFIDENCE = (0, 255, 255)     # Yellow  
COLOR_BALL_LOW_CONFIDENCE = (0, 165, 255)     # Orange
COLOR_BALL_TRAIL = (0, 0, 255)                # Red
COLOR_FIELD_CONTOUR = (0, 255, 0)             # Green
COLOR_FIELD_CORNERS = (255, 0, 0)             # Blue
COLOR_FIELD_BOUNDS = (0, 255, 255)            # Cyan
COLOR_GOALS = (255, 0, 255)                   # Magenta
COLOR_SCORE_TEXT = (255, 255, 255)            # White
COLOR_GOAL_ALERT = (0, 255, 0)                # Green
COLOR_BALL_IN_GOAL = (0, 255, 255)            # Yellow

# Ball trail parameters
BALL_TRAIL_MAX_LENGTH = 64
BALL_TRAIL_THICKNESS_FACTOR = 2.5

# ================== DEBUG SETTINGS ==================
DEBUG_SHOW_MASKS = False
DEBUG_VERBOSE_OUTPUT = True
DEBUG_SHOW_FRAME_FILTERING = False  # True: Zeigt Debug-Nachrichten für gefilterte Frames

# ================== SCREENSHOT & RECORDING ==================
SCREENSHOT_PATH = "screenshots/"
RECORDING_PATH = "recordings/"
