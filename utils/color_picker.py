import sys
import cv2
import numpy as np
import json
import os
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QInputDialog, QCheckBox, QHBoxLayout
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

        self.mask_visible = False

        self.info = QLabel("Click and drag to pick colors (Team 1)", self)

        # Buttons
        self.mask_checkbox = QCheckBox("Show Mask", self)
        self.mask_checkbox.stateChanged.connect(self.on_mask_checkbox_changed)

        self.team1_btn = QPushButton("Calibrate Team 1", self)
        self.team1_btn.clicked.connect(self.set_team1)

        self.team2_btn = QPushButton("Calibrate Team 2", self)
        self.team2_btn.clicked.connect(self.set_team2)

        self.ball_btn = QPushButton("Calibrate Ball", self)
        self.ball_btn.clicked.connect(self.set_ball)

        self.corners_btn = QPushButton("Calibrate Corners", self)
        self.corners_btn.clicked.connect(self.set_corners)

        self.reset_team1_btn = QPushButton("Reset", self)
        self.reset_team1_btn.clicked.connect(self.reset_team1)

        self.reset_team2_btn = QPushButton("Reset", self)
        self.reset_team2_btn.clicked.connect(self.reset_team2)
        
        self.reset_ball_btn = QPushButton("Reset", self)
        self.reset_ball_btn.clicked.connect(self.reset_ball)
        
        self.reset_corners_btn = QPushButton("Reset", self)
        self.reset_corners_btn.clicked.connect(self.reset_corners)      

        self.save_btn = QPushButton("Save", self)
        self.save_btn.clicked.connect(self.save_json)

        # LAYOUT
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)

        # Info-Text and Checkbox
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.info)
        info_layout.addStretch()
        info_layout.addWidget(self.mask_checkbox)
        main_layout.addLayout(info_layout)

        # Team 1 Buttons
        team1_layout = QHBoxLayout()
        team1_layout.addWidget(self.team1_btn)
        team1_layout.addWidget(self.reset_team1_btn)
        main_layout.addLayout(team1_layout)

        # Team 2 Buttons
        team2_layout = QHBoxLayout()
        team2_layout.addWidget(self.team2_btn)
        team2_layout.addWidget(self.reset_team2_btn)
        main_layout.addLayout(team2_layout)

        # Ball Buttons
        ball_layout = QHBoxLayout()
        ball_layout.addWidget(self.ball_btn)
        ball_layout.addWidget(self.reset_ball_btn)
        main_layout.addLayout(ball_layout)

        # Corners Buttons
        corners_layout = QHBoxLayout()
        corners_layout.addWidget(self.corners_btn)
        corners_layout.addWidget(self.reset_corners_btn)
        main_layout.addLayout(corners_layout)

        main_layout.addWidget(self.save_btn)
        
        self.setLayout(main_layout)

        self.load_existing_colors()

    def get_pixmap(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QPixmap.fromImage(QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888))
    
    def on_mask_checkbox_changed(self, state):
        """Called when the state of the checkbox changes."""
        self.mask_visible = bool(state)
        print(f"Checkbox state: {state}, mask_visible: {self.mask_visible}")
        
        self.update_display()

    def start_selection(self, event):
        self.selecting = True
        self.start_point = event.position().toPoint()
        self.selection_rect = QRect()
        self.label.setPixmap(self.get_pixmap(self.display_image))

    def update_selection(self, event):
        if self.selecting:
            pixmap = self.get_pixmap(self.display_image).copy()
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            
            self.selection_rect = QRect(self.start_point, event.position().toPoint()).normalized()
            painter.drawRect(self.selection_rect)
            painter.end()
            self.label.setPixmap(pixmap)

    def finish_selection(self, event):
        self.selecting = False
        
        click_threshold = 1
        if self.selection_rect.width() < click_threshold and self.selection_rect.height() < click_threshold:
            self.process_single_pixel(event.position().toPoint())
        else:
            self.process_roi()
        self.update_display()

    def paintEvent(self, event):
        super().paintEvent(event)
    
    def get_representative_color(self, picked_colors_hsv):
        """Calculates the median color from a list of HSV colors."""
        if not picked_colors_hsv:
            return None 
        median_hsv = np.median(np.array(picked_colors_hsv), axis=0)
        return median_hsv.astype(int)

    def get_complementary_bgr(self, hsv_color):
        """Calculates the complementary color and returns it as a BGR value for OpenCV."""
        if hsv_color is None:
            return [192, 192, 192] 

        h, s, v = hsv_color
        
        complementary_h = (h + 90) % 180
        
        complementary_v = 255
        complementary_s = 255

        complementary_hsv_np = np.uint8([[[complementary_h, complementary_s, complementary_v]]])
        complementary_bgr = cv2.cvtColor(complementary_hsv_np, cv2.COLOR_HSV2BGR)[0][0]
        
        return complementary_bgr.tolist()

    # def update_display(self):
    #     """Refreshes the display. Each category is displayed in its complementary color."""
    #     if not self.mask_visible:
    #         self.label.setPixmap(self.get_pixmap(self.display_image))
    #         return

    #     hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
    #     final_overlay = np.zeros_like(self.original_image)

    #     categories = [
    #         (self.hsv_ranges_team1, self.picked_colors_team1),
    #         (self.hsv_ranges_team2, self.picked_colors_team2),
    #         (self.hsv_ranges_ball, self.picked_colors_ball),
    #         (self.hsv_ranges_corners, self.picked_colors_corners)
    #     ]

    #     for hsv_ranges, picked_colors in categories:
    #         if not hsv_ranges:
    #             continue 

    #         rep_color_hsv = self.get_representative_color(picked_colors)
            
    #         comp_color_bgr = self.get_complementary_bgr(rep_color_hsv)

    #         # Mask
    #         category_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    #         for min_hsv, max_hsv in hsv_ranges:
    #             lower_bound = np.array(min_hsv)
    #             upper_bound = np.array(max_hsv)
    #             partial_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    #             category_mask = cv2.bitwise_or(category_mask, partial_mask)
            
    #         # Complementary color
    #         final_overlay[category_mask > 0] = comp_color_bgr

    #     # Skale
    #     overlay_resized = cv2.resize(final_overlay, (self.display_image.shape[1], self.display_image.shape[0]))

    #     final_image = self.display_image.copy()
        
    #     mask_pixels = np.any(overlay_resized > 0, axis=2)
    #     final_image[mask_pixels] = overlay_resized[mask_pixels]
        
    #     self.label.setPixmap(self.get_pixmap(final_image))

    def update_display(self):
        """Aktualisiert die Anzeige. Jede Kategorie wird in ihrer Komplementärfarbe angezeigt."""
        if not self.mask_visible:
            self.label.setPixmap(self.get_pixmap(self.display_image))
            return

        hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        final_overlay = np.zeros_like(self.original_image)

        categories = [
            (self.hsv_ranges_team1, self.picked_colors_team1),
            (self.hsv_ranges_team2, self.picked_colors_team2),
            (self.hsv_ranges_ball, self.picked_colors_ball),
            (self.hsv_ranges_corners, self.picked_colors_corners)
        ]

        for hsv_ranges, picked_colors in categories:
            if not hsv_ranges:
                continue

            # --- DIE ENTSCHEIDENDE ÄNDERUNG ---
            rep_color_hsv = None
            if picked_colors:
                # 1. Wenn wir in dieser Sitzung Farben ausgewählt haben, nutze diese.
                rep_color_hsv = self.get_representative_color(picked_colors)
            elif hsv_ranges:
                # 2. ANSONSTEN (z.B. nach Neustart), leite die Farbe aus den gespeicherten Ranges ab.
                min_hsv = hsv_ranges[0][0]
                max_hsv = hsv_ranges[0][1]
                rep_color_hsv = [
                    int((min_hsv[0] + max_hsv[0]) / 2),
                    int((min_hsv[1] + max_hsv[1]) / 2),
                    int((min_hsv[2] + max_hsv[2]) / 2)
                ]
            
            comp_color_bgr = self.get_complementary_bgr(rep_color_hsv)

            category_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            for min_val, max_val in hsv_ranges:
                partial_mask = cv2.inRange(hsv_image, np.array(min_val), np.array(max_val))
                category_mask = cv2.bitwise_or(category_mask, partial_mask)
            
            final_overlay[category_mask > 0] = comp_color_bgr

        # Der Rest der Methode bleibt gleich
        overlay_resized = cv2.resize(final_overlay, (self.display_image.shape[1], self.display_image.shape[0]))
        final_image = self.display_image.copy()
        mask_pixels = np.any(overlay_resized > 0, axis=2)
        final_image[mask_pixels] = overlay_resized[mask_pixels]
        
        self.label.setPixmap(self.get_pixmap(final_image))

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

        self.add_color(dominant_color.tolist())

    def process_single_pixel(self, point):
        """Processes the color of a single clicked pixel."""
        x = int(point.x() / self.scale_factor)
        y = int(point.y() / self.scale_factor)

        h_img, w_img, _ = self.original_image.shape
        if not (0 <= y < h_img and 0 <= x < w_img):
            print("Click was outside the image bounds.")
            return

        pixel_bgr = self.original_image[y, x]

        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        
        self.add_color(pixel_hsv.tolist())

    def add_color(self, hsv_color_list):
        """Adds a picked color to the correct list and the undo stack."""
        print(f"Adding color: {hsv_color_list}, type: {type(hsv_color_list)}, len: {len(hsv_color_list) if hasattr(hsv_color_list, '__len__') else 'N/A'}")
        
        # Ensure it's a proper 3-element list
        if not isinstance(hsv_color_list, (list, tuple)) or len(hsv_color_list) != 3:
            print(f"Warning: Invalid color format: {hsv_color_list}")
            return
            
        # Convert to list of integers to ensure consistency
        hsv_color_list = [int(x) for x in hsv_color_list]
        
        if self.current_calibration == 1:
            self.picked_colors_team1.append(hsv_color_list)
            print(f"Team 1 color picked: {tuple(hsv_color_list)}")
        elif self.current_calibration == 2:
            self.picked_colors_team2.append(hsv_color_list)
            print(f"Team 2 color picked: {tuple(hsv_color_list)}")
        elif self.current_calibration == 3:
            self.picked_colors_ball.append(hsv_color_list)
            print(f"Ball color picked: {tuple(hsv_color_list)}")
        elif self.current_calibration == 4:
            self.picked_colors_corners.append(hsv_color_list)
            print(f"Corners color picked: {tuple(hsv_color_list)}")

        if self.current_calibration == 1:
            self.hsv_ranges_team1 = self.compute_hsv_range_for_team(self.picked_colors_team1)
        elif self.current_calibration == 2:
            self.hsv_ranges_team2 = self.compute_hsv_range_for_team(self.picked_colors_team2)
        elif self.current_calibration == 3:
            self.hsv_ranges_ball = self.compute_hsv_range_for_team(self.picked_colors_ball)
        elif self.current_calibration == 4:
            self.hsv_ranges_corners = self.compute_hsv_range_for_team(self.picked_colors_corners)

        self.update_display()

    def reset_team1(self):
        """Resets all selected colors and areas for Team 1."""
        self.picked_colors_team1.clear()
        self.hsv_ranges_team1.clear()
        print("Team 1 colors have been reset.")
        self.update_display()

    def reset_team2(self):
        """Resets all selected colors and areas for Team 2."""
        self.picked_colors_team2.clear()
        self.hsv_ranges_team2.clear()
        print("Team 2 colors have been reset.")
        self.update_display()

    def reset_ball(self):
        """Resets all selected colors and areas for the ball."""
        self.picked_colors_ball.clear()
        self.hsv_ranges_ball.clear()
        print("Ball colors have been reset.")
        self.update_display()

    def reset_corners(self):
        """Resets all selected colors and areas for the corners."""
        self.picked_colors_corners.clear()
        self.hsv_ranges_corners.clear()
        print("Corners colors have been reset.")
        self.update_display()

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

        print(f"Computing HSV range for colors: {picked_colors}")
        for i, color in enumerate(picked_colors):
            print(f"  Color {i}: {color}, type: {type(color)}, len: {len(color) if hasattr(color, '__len__') else 'N/A'}")

        # Ensures all colors are proper 3-element lists
        cleaned_colors = []
        for color in picked_colors:
            if isinstance(color, (list, tuple)) and len(color) == 3:
                cleaned_colors.append([int(x) for x in color])
            else:
                print(f"Skipping invalid color: {color}")
        
        if not cleaned_colors:
            print("No valid colors found after cleaning")
            return []
            
        try:
            picked_array = np.array(cleaned_colors)
            print(f"Created array shape: {picked_array.shape}")
        except Exception as e:
            print(f"Error creating numpy array: {e}")
            return []
        hues = picked_array[:, 0]

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

    def load_existing_colors(self):
        """Loads the existing color ranges from the colors.json, if available."""
        project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        filepath = os.path.join(project_folder, "detection/colors.json")

        if not os.path.exists(filepath):
            print("No existing colors.json found. Starting fresh.")
            return

        try:
            with open(filepath, "r") as f:
                config = json.load(f)

            # Loads the ranges for each category if they exist in the file
            if "team1" in config and "ranges" in config["team1"]:
                self.hsv_ranges_team1 = [
                    (np.array(r["lower"]), np.array(r["upper"])) for r in config["team1"]["ranges"]
                ]
                print(f"Loaded {len(self.hsv_ranges_team1)} existing ranges for Team 1.")

            if "team2" in config and "ranges" in config["team2"]:
                self.hsv_ranges_team2 = [
                    (np.array(r["lower"]), np.array(r["upper"])) for r in config["team2"]["ranges"]
                ]
                print(f"Loaded {len(self.hsv_ranges_team2)} existing ranges for Team 2.")

            if "ball" in config and "ranges" in config["ball"]:
                self.hsv_ranges_ball = [
                    (np.array(r["lower"]), np.array(r["upper"])) for r in config["ball"]["ranges"]
                ]
                print(f"Loaded {len(self.hsv_ranges_ball)} existing ranges for Ball.")

            if "corners" in config and "ranges" in config["corners"]:
                self.hsv_ranges_corners = [
                    (np.array(r["lower"]), np.array(r["upper"])) for r in config["corners"]["ranges"]
                ]
                print(f"Loaded {len(self.hsv_ranges_corners)} existing ranges for Corners.")

        except Exception as e:
            print(f"Error loading colors.json: {e}. Starting fresh.")

    def save_json(self):
        """Saves the current state of all HSV ranges to the JSON file."""
        config = {}

        if self.hsv_ranges_team1:
            config["team1"] = {"ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_team1]}
        
        if self.hsv_ranges_team2:
            config["team2"] = {"ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_team2]}

        if self.hsv_ranges_ball:
            config["ball"] = {"ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_ball]}

        if self.hsv_ranges_corners:
            config["corners"] = {"ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_corners]}

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