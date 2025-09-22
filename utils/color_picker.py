import sys
import cv2
import numpy as np
import json
import os
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QInputDialog
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PySide6.QtCore import Qt, QRect


class ColorPicker(QWidget):
    def __init__(self, frame, scale_factor=1.0):
        super().__init__()
        self.setWindowTitle("Colorpicker (Live-Picture)")

        self.scale_factor = scale_factor
        self.original_image = frame
        self.display_image = cv2.resize(self.original_image, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

        # Colors per team saved
        self.picked_colors_team1 = []
        self.picked_colors_team2 = []
        self.picked_colors_ball = []
        self.picked_colors_corners = []

        # HSV areas per team (list of (min, max))
        self.hsv_ranges_team1 = []
        self.hsv_ranges_team2 = []
        self.hsv_ranges_ball = []
        self.hsv_ranges_corners = []

        # Current team: 1 or 2
        self.current_calibration = 1

        self.label = QLabel(self)
        self.label.setPixmap(self.get_pixmap(self.display_image))
        self.label.mousePressEvent = self.start_selection
        self.label.mouseMoveEvent = self.update_selection
        self.label.mouseReleaseEvent = self.finish_selection

        self.selection_rect = QRect()
        self.selecting = False

        self.info = QLabel("Click and drag to pick colors (Team 1)", self)

        # Buttons
        self.team1_btn = QPushButton("Calibrate Team 1", self)
        self.team1_btn.clicked.connect(self.set_team1)

        self.team2_btn = QPushButton("Calibrate Team 2", self)
        self.team2_btn.clicked.connect(self.set_team2)

        self.ball_btn = QPushButton("Calibrate Ball", self)
        self.ball_btn.clicked.connect(self.set_ball)

        self.corners_btn = QPushButton("Calibrate Corners", self)
        self.corners_btn.clicked.connect(self.set_corners)        

        self.done_btn = QPushButton("Calculate", self)
        self.done_btn.clicked.connect(self.compute_hsv_ranges)

        self.save_btn = QPushButton("Save", self)
        self.save_btn.clicked.connect(self.save_json)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.info)
        layout.addWidget(self.team1_btn)
        layout.addWidget(self.team2_btn)
        layout.addWidget(self.ball_btn)
        layout.addWidget(self.corners_btn)
        layout.addWidget(self.done_btn)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

    def get_pixmap(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QPixmap.fromImage(QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888))

    def start_selection(self, event):
        self.selecting = True
        self.start_point = event.position().toPoint()
        self.selection_rect = QRect()
        #self.label.setPixmap(self.get_pixmap(self.display_image))

    def update_selection(self, event):
        if self.selecting:
            self.selection_rect = QRect(self.start_point, event.position().toPoint()).normalized()
            self.update()

    def finish_selection(self, event):
        self.selecting = False
        #self.label.setPixmap(self.get_pixmap(self.display_image))
        if self.selection_rect.width() < 5 or self.selection_rect.height() < 5:
            print("Selection too small.")
            return
        self.process_roi()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selecting and not self.selection_rect.isNull():
            # Creates a copy of the pixmap for drawing
            pixmap = self.label.pixmap().copy()
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
            painter.end()  
            self.label.setPixmap(pixmap)

    def process_roi(self):
        x = int(self.selection_rect.x() / self.scale_factor)
        y = int(self.selection_rect.y() / self.scale_factor)
        w = int(self.selection_rect.width() / self.scale_factor)
        h = int(self.selection_rect.height() / self.scale_factor)

        h_img, w_img, _ = self.original_image.shape
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        if w <= 0 or h <= 0:
            print("Invalid selection")
            return

        roi_img = self.original_image[y:y+h, x:x+w]
        roi_img = cv2.GaussianBlur(roi_img, (5, 5), 0)
        roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # CLAHE V-Kanal
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        roi_hsv[:, :, 2] = clahe.apply(roi_hsv[:, :, 2])

        pixels = roi_hsv.reshape(-1, 3)

        saturation_thresh = 50
        value_thresh = 50
        pixels = pixels[(pixels[:, 1] > saturation_thresh) & (pixels[:, 2] > value_thresh)]
        if len(pixels) == 0:
            print("No valid pixels in selection")
            return

        n_clusters = min(5, max(2, len(pixels) // 1000))
        
        # Uses OpenCV K-Means
        pixels_float = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # K-Means with OpenCV
        ret, labels, centers = cv2.kmeans(
            pixels_float, 
            n_clusters, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_RANDOM_CENTERS
        )

        # Finds dominant colors based on cluster size
        unique_labels, counts = np.unique(labels.flatten(), return_counts=True)
        top_n = min(2, len(counts))
        top_clusters = np.argsort(counts)[::-1][:top_n]
        dominant_colors = centers[top_clusters]
        weights = counts[top_clusters]

        dominant_color = np.average(dominant_colors, axis=0, weights=weights).astype(int)

        if self.current_calibration == 1:
            self.picked_colors_team1.append(dominant_color.tolist())
            print(f"Team 1 color picked: {tuple(dominant_color)}")
        elif self.current_calibration == 2:
            self.picked_colors_team2.append(dominant_color.tolist())
            print(f"Team 2 color picked: {tuple(dominant_color)}")
        elif self.current_calibration == 3:
            self.picked_colors_ball.append(dominant_color.tolist())
            print(f"Ball color picked: {tuple(dominant_color)}")
        elif self.current_calibration == 4:
            self.picked_colors_corners.append(dominant_color.tolist())
            print(f"Corners color picked: {tuple(dominant_color)}")

    def set_team1(self):
        self.current_calibration = 1
        self.info.setText("Calibrate Team 1: Select colors")

    def set_team2(self):
        self.current_calibration = 2
        self.info.setText("Calibrate Team 2: Select colors")

    def set_ball(self):
        self.current_calibration = 3
        self.info.setText("Calibrate Ball: Select colors")

    def set_corners(self):
        self.current_calibration = 4
        self.info.setText("Calibrate Corners: Select colors")

    def compute_hsv_range_for_team(self, picked_colors):
        if not picked_colors:
            return []

        picked_array = np.array(picked_colors)
        hues = picked_array[:, 0]

        # Separates Hue into two groups: "small" and "large" (due to cyclic Hue)
        group1 = picked_array[hues < 20]
        group2 = picked_array[hues > 160]

        ranges = []

        for group in [group1, group2]:
            if len(group) == 0:
                continue
            median_color = np.median(group, axis=0)
            std_dev = np.std(group, axis=0)
            tolerance = np.clip(std_dev * 2.5, [5, 30, 30], [30, 70, 70])
            min_hsv = np.maximum(np.min(group, axis=0) - tolerance, [0, 0, 0])
            max_hsv = np.minimum(np.max(group, axis=0) + tolerance, [179, 255, 255])
            ranges.append((min_hsv.astype(int), max_hsv.astype(int)))

        if not ranges:
            # Fallback if no grouping
            median_color = np.median(picked_array, axis=0)
            std_dev = np.std(picked_array, axis=0)
            tolerance = np.clip(std_dev * 2.5, [5, 30, 30], [30, 70, 70])
            min_hsv = np.maximum(np.min(picked_array, axis=0) - tolerance, [0, 0, 0])
            max_hsv = np.minimum(np.max(picked_array, axis=0) + tolerance, [179, 255, 255])
            ranges.append((min_hsv.astype(int), max_hsv.astype(int)))

        return ranges

    def compute_hsv_ranges(self):
        self.hsv_ranges_team1 = self.compute_hsv_range_for_team(self.picked_colors_team1)
        self.hsv_ranges_team2 = self.compute_hsv_range_for_team(self.picked_colors_team2)
        self.hsv_ranges_ball = self.compute_hsv_range_for_team(self.picked_colors_ball)
        self.hsv_ranges_corners = self.compute_hsv_range_for_team(self.picked_colors_corners)

        if self.hsv_ranges_team1:
            for i, (min_hsv, max_hsv) in enumerate(self.hsv_ranges_team1):
                print(f"Team 1 HSV area {i+1}: {min_hsv} - {max_hsv}")
        else:
            print("Team 1 HSV area could not be calculated.")

        if self.hsv_ranges_team2:
            for i, (min_hsv, max_hsv) in enumerate(self.hsv_ranges_team2):
                print(f"Team 2 HSV area {i+1}: {min_hsv} - {max_hsv}")
        else:
            print("Team 2 HSV area could not be calculated.")

        if self.hsv_ranges_ball:
            for i, (min_hsv, max_hsv) in enumerate(self.hsv_ranges_ball):
                print(f"Ball HSV area {i+1}: {min_hsv} - {max_hsv}")
        else:
            print("Ball HSV area could not be calculated.")

        if self.hsv_ranges_corners:
            for i, (min_hsv, max_hsv) in enumerate(self.hsv_ranges_corners):
                print(f"Corners HSV area {i+1}: {min_hsv} - {max_hsv}")
        else:
            print("Corners HSV area could not be calculated.")

    def save_json(self):
        config = {}
        if self.hsv_ranges_team1:
            config["team1"] = {
                "ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_team1]
            }
        if self.hsv_ranges_team2:
            config["team2"] = {
                "ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_team2]
            }
        if self.hsv_ranges_ball:
            config["ball"] = {
                "ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_ball]
            }
        if self.hsv_ranges_corners:
            config["corners"] = {
                "ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_corners]
            }

        project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        filepath = os.path.join(project_folder, "detection/colors.json")
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved as {filepath}")

if __name__ == "__main__":
    frame = os.path.join(os.path.dirname(__file__), "C:\\Users\\guchr\\OneDrive\\Bilder\\Screenshots\\Screenshot 2025-08-29 121615.png")
    app = QApplication(sys.argv)
    picker = ColorPicker(frame=cv2.imread(frame))
    picker.show()
    sys.exit(app.exec())