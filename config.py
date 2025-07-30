# config.py

# ================== VIDEO & CAMERA SETTINGS ==================
VIDEO_PATH = "C:/Users/joshu/OneDrive - TH Köln/Python Workspace/Tischkicker/0_Material/2025-07-03_Testvideos_250fps/07.avi"
USE_WEBCAM = False
FRAME_WIDTH = 720
FRAME_HEIGHT = 540

# ================== CALIBRATION FILES ==================
CAMERA_CALIBRATION_FILE = "Z_Archive/Cam_Lens_Calibration/calibration_data.json"
FIELD_CALIBRATION_FILE = "field_calibration.json"

# ================== BALL DETECTION (HSV VALUES) ==================
# Orange ball - main color
BALL_ORANGE_LOWER = (0, 120, 200)
BALL_ORANGE_UPPER = (30, 255, 255)

# Orange ball - alternative values
BALL_ORANGE_LOWER_ALT = (10, 120, 200)
BALL_ORANGE_UPPER_ALT = (40, 255, 255)

# ================== FIELD DETECTION (HSV VALUES) ==================
# Green field - main color
FIELD_GREEN_LOWER = (20, 40, 40)
FIELD_GREEN_UPPER = (80, 255, 100)

# Green field - alternative values
FIELD_GREEN_LOWER_ALT = (35, 30, 30)
FIELD_GREEN_UPPER_ALT = (100, 255, 100)

# Goals (bright/white areas)
GOAL_LOWER = (0, 0, 140)
GOAL_UPPER = (180, 255, 255)

# ================== TRACKING PARAMETERS ==================
# Ball tracking
BALL_SMOOTHER_WINDOW_SIZE = 20
BALL_MAX_MISSING_FRAMES = 100
BALL_CONFIDENCE_THRESHOLD = 0.6

# Field tracking
FIELD_MIN_AREA = 50000  # Minimum field size in pixels
FIELD_STABILITY_FRAMES = 30  # Frames for stable detection
GOAL_DETECTION_CONFIDENCE = 0.7
MIN_GOAL_AREA = 1000  # Minimum size for goal detection

# Goal scoring system
GOAL_DISAPPEAR_FRAMES = 15  # Frames without detection to count as goal
GOAL_REVERSAL_TIME_WINDOW = 3.0  # Seconds to check for goal reversal

# ================== PERFORMANCE & DISPLAY ==================
DISPLAY_FPS = 30
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS

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

# ================== SCREENSHOT & RECORDING ==================
SCREENSHOT_PATH = "screenshots/"
RECORDING_PATH = "recordings/"
