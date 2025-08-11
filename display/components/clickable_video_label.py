"""
Clickable Video Label Component
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QLabel


class ClickableVideoLabel(QLabel):
    """Custom QLabel that can detect mouse clicks and extract pixel colors"""
    color_picked = Signal(int, int, int)  # RGB values
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pixmap = None
        self.original_frame = None
        
    def set_frame_data(self, frame, pixmap):
        """Store both the original frame and the displayed pixmap"""
        self.original_frame = frame
        self.current_pixmap = pixmap
        self.setPixmap(pixmap)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse clicks to extract pixel color"""
        if event.button() == Qt.MouseButton.LeftButton and self.original_frame is not None:
            # Get click position
            click_pos = event.position().toPoint()
            
            # Get the current pixmap and its dimensions
            if self.current_pixmap is not None:
                pixmap_rect = self.current_pixmap.rect()
                label_rect = self.rect()
                
                # Calculate scaling factor and offsets for centered image
                scale_x = pixmap_rect.width() / label_rect.width() if label_rect.width() > 0 else 1
                scale_y = pixmap_rect.height() / label_rect.height() if label_rect.height() > 0 else 1
                scale = max(scale_x, scale_y)
                
                # Calculate actual displayed image size
                displayed_width = int(pixmap_rect.width() / scale)
                displayed_height = int(pixmap_rect.height() / scale)
                
                # Calculate offset for centering
                offset_x = (label_rect.width() - displayed_width) // 2
                offset_y = (label_rect.height() - displayed_height) // 2
                
                # Convert click position to image coordinates
                img_x = int((click_pos.x() - offset_x) * scale)
                img_y = int((click_pos.y() - offset_y) * scale)
                
                # Check if click is within image bounds
                if (0 <= img_x < pixmap_rect.width() and 0 <= img_y < pixmap_rect.height()):
                    # Map to original frame coordinates
                    frame_height, frame_width = self.original_frame.shape[:2]
                    orig_x = int(img_x * frame_width / pixmap_rect.width())
                    orig_y = int(img_y * frame_height / pixmap_rect.height())
                    
                    # Ensure coordinates are within frame bounds
                    orig_x = max(0, min(orig_x, frame_width - 1))
                    orig_y = max(0, min(orig_y, frame_height - 1))
                    
                    # Extract pixel color (BGR format in OpenCV)
                    try:
                        pixel_bgr = self.original_frame[orig_y, orig_x]
                        # Convert BGR to RGB
                        r, g, b = int(pixel_bgr[2]), int(pixel_bgr[1]), int(pixel_bgr[0])
                        
                        # Emit the color values
                        self.color_picked.emit(r, g, b)
                        
                    except IndexError:
                        print(f"Index error: trying to access pixel at ({orig_x}, {orig_y}) in frame of size {frame_width}x{frame_height}")
        
        super().mousePressEvent(event)
