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
    def __init__(self, frame, scale_factor=0.3):
        super().__init__()
        self.setWindowTitle("Farbpicker (Live-Bild)")

        self.scale_factor = scale_factor
        self.original_image = frame
        self.display_image = cv2.resize(self.original_image, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

        # Farben je Team speichern
        self.picked_colors_team1 = []
        self.picked_colors_team2 = []

        # HSV-Bereiche je Team (Liste von (min, max))
        self.hsv_ranges_team1 = []
        self.hsv_ranges_team2 = []

        # Aktuelles Team: 1 oder 2
        self.current_team = 1

        self.label = QLabel(self)
        self.label.setPixmap(self.get_pixmap(self.display_image))
        self.label.mousePressEvent = self.start_selection
        self.label.mouseMoveEvent = self.update_selection
        self.label.mouseReleaseEvent = self.finish_selection

        self.selection_rect = QRect()
        self.selecting = False

        self.info = QLabel("Klicke und ziehe zum Farbpicken (Team 1)", self)

        # Buttons
        self.team1_btn = QPushButton("Team 1 kalibrieren", self)
        self.team1_btn.clicked.connect(self.set_team1)

        self.team2_btn = QPushButton("Team 2 kalibrieren", self)
        self.team2_btn.clicked.connect(self.set_team2)

        self.done_btn = QPushButton("Bereich berechnen", self)
        self.done_btn.clicked.connect(self.compute_hsv_ranges)

        self.save_btn = QPushButton("Speichern", self)
        self.save_btn.clicked.connect(self.save_json)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.info)
        layout.addWidget(self.team1_btn)
        layout.addWidget(self.team2_btn)
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

    def update_selection(self, event):
        if self.selecting:
            self.selection_rect = QRect(self.start_point, event.position().toPoint()).normalized()
            self.update()

    def finish_selection(self, event):
        self.selecting = False
        if self.selection_rect.width() < 5 or self.selection_rect.height() < 5:
            print("Auswahl zu klein.")
            return
        self.process_roi()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selecting and not self.selection_rect.isNull():
            # Erstelle eine Kopie des Pixmaps zum Zeichnen
            pixmap = self.label.pixmap().copy()
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
            painter.end()  # Wichtig: Painter beenden bevor das Pixmap verwendet wird
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
            print("Ungültige Auswahl.")
            return

        roi_img = self.original_image[y:y+h, x:x+w]
        roi_img = cv2.GaussianBlur(roi_img, (5, 5), 0)
        roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # *** CLAHE auf V-Kanal ***
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        roi_hsv[:, :, 2] = clahe.apply(roi_hsv[:, :, 2])

        pixels = roi_hsv.reshape(-1, 3)

        saturation_thresh = 50
        value_thresh = 50
        pixels = pixels[(pixels[:, 1] > saturation_thresh) & (pixels[:, 2] > value_thresh)]
        if len(pixels) == 0:
            print("Keine gültigen Pixel in Auswahl.")
            return

        n_clusters = min(5, max(2, len(pixels) // 1000))
        
        # Verwende OpenCV K-Means statt sklearn
        pixels_float = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # K-Means mit OpenCV
        ret, labels, centers = cv2.kmeans(
            pixels_float, 
            n_clusters, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_RANDOM_CENTERS
        )

        # Finde dominante Farben basierend auf Cluster-Größe
        unique_labels, counts = np.unique(labels.flatten(), return_counts=True)
        top_n = min(2, len(counts))
        top_clusters = np.argsort(counts)[::-1][:top_n]
        dominant_colors = centers[top_clusters]
        weights = counts[top_clusters]

        dominant_color = np.average(dominant_colors, axis=0, weights=weights).astype(int)

        if self.current_team == 1:
            self.picked_colors_team1.append(dominant_color.tolist())
            print(f"Team 1 Farbe gepickt: {tuple(dominant_color)}")
        else:
            self.picked_colors_team2.append(dominant_color.tolist())
            print(f"Team 2 Farbe gepickt: {tuple(dominant_color)}")

    def set_team1(self):
        self.current_team = 1
        self.info.setText("Team 1 kalibrieren: Farben auswählen")

    def set_team2(self):
        self.current_team = 2
        self.info.setText("Team 2 kalibrieren: Farben auswählen")

    def compute_hsv_range_for_team(self, picked_colors):
        if not picked_colors:
            return []

        picked_array = np.array(picked_colors)
        hues = picked_array[:, 0]

        # Trenne Hue in zwei Gruppen: "klein" und "groß" (wegen zyklischem Hue)
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
            # Fallback bei keiner Gruppierung
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

        if self.hsv_ranges_team1:
            for i, (min_hsv, max_hsv) in enumerate(self.hsv_ranges_team1):
                print(f"Team 1 HSV Bereich {i+1}: {min_hsv} - {max_hsv}")
        else:
            print("Team 1 HSV Bereich konnte nicht berechnet werden.")

        if self.hsv_ranges_team2:
            for i, (min_hsv, max_hsv) in enumerate(self.hsv_ranges_team2):
                print(f"Team 2 HSV Bereich {i+1}: {min_hsv} - {max_hsv}")
        else:
            print("Team 2 HSV Bereich konnte nicht berechnet werden.")

    def save_json(self):
        name, ok = QInputDialog.getText(self, "Speichern als", "Dateiname (z.B. colors.json):")
        if not ok or not name.strip():
            return
        config = {}
        if self.hsv_ranges_team1:
            config["team1"] = {
                "ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_team1]
            }
        if self.hsv_ranges_team2:
            config["team2"] = {
                "ranges": [{"lower": r[0].tolist(), "upper": r[1].tolist()} for r in self.hsv_ranges_team2]
            }

        project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        folder = os.path.join(project_folder, "detection")
        filepath = os.path.join(folder, name)
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Gespeichert als {filepath}")