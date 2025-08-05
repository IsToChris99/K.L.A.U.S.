import cv2
import numpy as np
from scipy.signal import find_peaks
import itertools
import time

# ==============================================================================
#  CORE FUNCTION
# ==============================================================================
def find_rods(
    gray_image,
    num_rods=8,
    min_field_width_ratio=0.6,
    scale_factor=0.2,
    peak_prominence_factor=0.15):
    """
    Finds the x-coordinates of foosball table rods in a grayscale image.
    This algorithm is optimized for speed and expects an undistorted input image.

    Args:
        gray_image (np.ndarray): A single-channel (grayscale), undistorted image.
        num_rods (int): The expected number of rods to detect.
        min_field_width_ratio (float): The minimum expected ratio of the rod span to the total image width.
        scale_factor (float): The factor by which to downscale the image for faster analysis.
        peak_prominence_factor (float): Relative factor to determine the peak prominence for adaptive thresholding.

    Returns:
        dict or None: A dictionary containing the detection results or None if detection fails.
                      The dictionary includes:
                      'rod_x_coords': List of the rods' x-coordinates.
                      'avg_spacing': The average spacing in pixels between the detected rods.
                      'confidence_std_dev': The standard deviation of the spacing, a measure of confidence (lower is better).
    """
    
    # --- Step 1: Scale image and create vertical projection profile ---
    h, w = gray_image.shape
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    
    # INTER_AREA is a good choice for downscaling
    scaled_gray = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    projection_profile = np.sum(scaled_gray, axis=0, dtype=np.float32)
    profile_smoothed = cv2.GaussianBlur(projection_profile, (1, 3), 0).flatten()  # Smaller kernel

    # --- Step 2: Robust peak detection ---
    # Prominence is calculated dynamically relative to the max profile height,
    # making the algorithm more robust to lighting changes.
    max_profile_height = np.max(profile_smoothed)
    prominence_threshold = max_profile_height * peak_prominence_factor

    candidate_peaks, _ = find_peaks(
        profile_smoothed, 
        distance=int(20 * scale_factor), # Minimum distance between peaks
        prominence=prominence_threshold
    )
    
    # --- Step 3: Exhaustive search for the best peak combination ---
    if len(candidate_peaks) < num_rods:
        return None # Not enough candidates found to form a set of rods

    best_sequence_scaled = None
    min_std_dev = float('inf')

    # Iterate through all possible combinations of `num_rods` peaks
    for current_combo in itertools.combinations(candidate_peaks, num_rods):
        # Check if the combination spans a plausible width of the table
        span = current_combo[-1] - current_combo[0]
        if span < new_w * min_field_width_ratio:
            continue
        
        # Calculate the standard deviation of the distances between adjacent rods
        deltas = np.diff(current_combo)
        std_dev = np.std(deltas)

        # The combination with the most uniform spacing (lowest std_dev) is the best one
        if std_dev < min_std_dev:
            min_std_dev = std_dev
            best_sequence_scaled = current_combo

    if best_sequence_scaled is None:
        return None # No valid combination found

    # --- Step 4: Format and return results ---
    final_rods_scaled = np.array(best_sequence_scaled)
    # Rescale coordinates back to the original image size
    final_rod_coords = (final_rods_scaled / scale_factor).astype(int)
    
    avg_spacing = np.mean(np.diff(final_rod_coords))

    result = {
        'rod_x_coords': final_rod_coords.tolist(),
        'avg_spacing': avg_spacing,
        'confidence_std_dev': min_std_dev
    }
    
    return result


# ==============================================================================
#  TESTING BLOCK: Only runs when the script is executed directly.
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Running test routine with 'Test-Frame.jpg'...")
    
    image_path = 'Test-Frame.jpg'
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Test image not found at '{image_path}'")
    else:
        # --- Preprocessing for the test run ONLY ---
        start_time_total = time.time()
        
        # 1. Undistort the image
        h, w = original_image.shape[:2]
        # These are estimated parameters for the sample image
        camera_matrix = np.array([
        [
            1011.8273697893925,
            0.0,
            744.8479453817835
        ],
        [
            0.0,
            1011.1891098763836,
            568.9298722017875
        ],
        [
            0.0,
            0.0,
            1.0
        ]
        ], dtype=np.float32)
        distortion_coeffs = np.array([
        -0.21471481292995498,
        0.10719626529193438,
        0.00016044823217297634,
        -0.000343418632488825,
        -0.028175542050742394
        ], dtype=np.float32)
        undistorted_img = cv2.undistort(original_image, camera_matrix, distortion_coeffs, None, None)
        
        resize_factor = 0.5
        resized = cv2.resize(undistorted_img, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

        # 2. Convert to grayscale
        gray_img_for_func = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # --- Call the core function ---s
        start_time_func = time.time()
        rod_detection_result = find_rods(gray_img_for_func)
        end_time_func = time.time()
        
        # --- Print results and performance metrics ---
        total_duration_ms = (time.time() - start_time_total) * 1000
        func_duration_ms = (end_time_func - start_time_func) * 1000

        print("\n--- Performance ---")
        print(f"Core function 'find_rods' runtime: {func_duration_ms:.2f} ms")
        print(f"Total runtime (incl. preprocessing): {total_duration_ms:.2f} ms")
        print(f"Theoretical FPS (core function only): {1000 / func_duration_ms if func_duration_ms > 0 else 'inf':.1f}")
        
        if rod_detection_result:
            print("\n--- Detection Result ---")
            print(f"Rod coordinates: {rod_detection_result['rod_x_coords']}")
            print(f"Average spacing: {rod_detection_result['avg_spacing']:.2f} px")
            print(f"Spacing confidence (std dev): {rod_detection_result['confidence_std_dev']:.2f}")

            # Visualize the detected rods on the image
            output_visualization = resized.copy()
            for x_coord in rod_detection_result['rod_x_coords']:
                cv2.line(output_visualization, (x_coord, 0), (x_coord, h), (0, 0, 255), 3)
            
            cv2.imshow('Final Rod Detection', output_visualization)
        else:
            print("\nERROR: No valid set of rods could be detected.")
            cv2.imshow('Detection Failed', resized)
        
        # Wait for a key press, then close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        {
}