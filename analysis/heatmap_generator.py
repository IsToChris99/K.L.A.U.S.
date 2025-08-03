# analysis/heatmap_generator.py

import cv2
import numpy as np
import os

def create_heatmap_from_points(points, dimensions, output_path="heatmap.png", sigma=15):
    print("\nGenerating heatmap from collected data...")
    if not points:
        print("Warning: No points were provided to generate the heatmap.")
        return

    # Create directory for output if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    width, height = dimensions
    heatmap_array = np.zeros((height, width), dtype=np.float32)

    # Accumulate the points on the heatmap canvas
    for pt in points:
        if pt is not None:
            x, y = pt
            if 0 <= x < width and 0 <= y < height:
                heatmap_array[int(y), int(x)] += 1

    if np.sum(heatmap_array) == 0:
        print("Warning: No valid points to generate heatmap.")
        return

    # Apply smoothing and normalization
    heatmap_array = cv2.GaussianBlur(heatmap_array, (0, 0), sigmaX=sigma, sigmaY=sigma)
    heatmap_norm = cv2.normalize(heatmap_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Save and display the result
    cv2.imwrite(output_path, heatmap_color)
    cv2.imshow("Ball Position Heatmap", heatmap_color)
    print(f"Heatmap successfully saved to: {output_path}")
    print("Press any key in the heatmap window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()