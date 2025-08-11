# main.py
#
# A highly optimized, multi-threaded application for processing a high-speed
# camera stream.
#
# - Acquisition Thread (in IDS_Camera): Runs at full camera speed (~250 FPS), grabbing frames.
# - Analysis Thread: Runs as fast as possible, performing only lightweight calculations
#   (delta_t, dropped frame counting) on every single frame.
# - Display Thread (main): Runs at a slow, user-friendly rate (~30 FPS). It performs
#   the heavy rendering (color conversion, text overlay) only when it needs to
#   update the screen.

import threading
import time
import cv2
from input.ids_camera import IDS_Camera # Adjust the import path if necessary

# --- Shared resources between Analysis and Display threads ---
processing_lock = threading.Lock()
stop_event = threading.Event()

# The shared dictionary now contains the RAW bayer frame and a separate
# dictionary for the latest calculated statistics.
processing_results = {
    "latest_bayer_frame": None,
    "latest_stats": {},
}

def analysis_thread_func(camera: IDS_Camera):
    """
    This thread is responsible for time-critical, lightweight analysis on
    EVERY frame from the camera. It does NOT perform heavy rendering.
    """
    print("Analysis thread started.")
    
    last_timestamp_ns = 0
    last_frame_id = -1
    total_dropped_count = 0
    count = 0
    while not stop_event.is_set():
        # Get the latest raw data from the camera acquisition thread
        bayer_frame, metadata = camera.get_frame()

        if bayer_frame is not None:
            # --- 1. LIGHTWEIGHT ANALYSIS ON EVERY FRAME ---
            
            # Precise delta_t calculation for physics
            current_timestamp_ns = metadata.get("timestamp_ns", 0)
            delta_t_sec = 0.0
            if last_timestamp_ns > 0 and current_timestamp_ns > last_timestamp_ns:
                delta_t_sec = (current_timestamp_ns - last_timestamp_ns) / 1_000_000_000.0
            last_timestamp_ns = current_timestamp_ns

            # Check for dropped frames and update the cumulative counter
            current_frame_id = metadata.get("frame_id", -1)
            if last_frame_id != -1 and current_frame_id > last_frame_id + 1:
                dropped_in_this_gap = current_frame_id - (last_frame_id + 1)
                total_dropped_count += dropped_in_this_gap
            last_frame_id = current_frame_id

            # --- 2. PREPARE STATS FOR THE DISPLAY THREAD ---
            # Create a dictionary with the results of our analysis.
            stats_to_share = {
                "frame_id": current_frame_id,
                "processing_fps": (1 / delta_t_sec) if delta_t_sec > 0 else 0,
                "total_dropped": total_dropped_count,
                "width": metadata.get('width', 0),
                "height": metadata.get('height', 0),
            }

            # --- 3. SHARE RAW FRAME AND STATS WITH THE DISPLAY THREAD ---
            with processing_lock:
                processing_results["latest_bayer_frame"] = bayer_frame
                processing_results["latest_stats"] = stats_to_share
        else:
            # Prevent this loop from spinning at 100% CPU if the queue is empty
            time.sleep(0.001)

    print("Analysis thread finished.")


def main():
    """
    The main function, which initializes everything and runs the GUI display loop.
    Heavy rendering is now done here, at a slower pace.
    """
    camera = None
    analysis_thread = None
    try:
        print("Initializing camera...")
        camera = IDS_Camera()
        
        camera.start()
        
        analysis_thread = threading.Thread(target=analysis_thread_func, args=(camera,))
        analysis_thread.start()
        
        time.sleep(1)

        print("Starting display loop. Press 'q' to exit.")
        cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
        
        while analysis_thread.is_alive():
            # 1. Get the latest raw frame and stats from the analysis thread
            bayer_frame_to_display = None
            stats_to_display = {}
            with processing_lock:
                if processing_results["latest_bayer_frame"] is not None:
                    # Make a copy to work on, releasing the lock quickly
                    bayer_frame_to_display = processing_results["latest_bayer_frame"].copy()
                    stats_to_display = processing_results["latest_stats"].copy()

            # 2. PERFORM HEAVY RENDERING (only if we have a new frame)
            if bayer_frame_to_display is not None:
                # Convert the Bayer format to a color image for display
                color_frame = cv2.cvtColor(bayer_frame_to_display, cv2.COLOR_BAYER_RG2RGB)

                # Prepare overlay text using the latest stats
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (0, 255, 128)
                thickness = 2
                
                fps = stats_to_display.get("processing_fps", 0)
                frame_id = stats_to_display.get("frame_id", "N/A")
                dropped_count = stats_to_display.get("total_dropped", 0)
                
                line1 = f"Frame: {frame_id} | Analysis FPS: {fps:.1f}"
                line2 = f"Size: {stats_to_display.get('width', 0)}x{stats_to_display.get('height', 0)}"
                line3 = f"Total Dropped: {dropped_count}"

                cv2.putText(color_frame, line1, (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
                cv2.putText(color_frame, line2, (10, 60), font, font_scale, font_color, thickness, cv2.LINE_AA)
                cv2.putText(color_frame, line3, (10, 90), font, font_scale, font_color, thickness, cv2.LINE_AA)
                
                # 3. Display the final rendered image
                cv2.imshow("Camera Stream", color_frame)
            else:
                # If there's nothing to display yet, avoid a busy-wait
                time.sleep(0.01)

            # The GUI loop runs at a user-friendly ~30 FPS
            if cv2.waitKey(33) & 0xFF == ord('q'):
                print("'q' pressed, signaling threads to stop.")
                stop_event.set()
                break
                
    except Exception as e:
        print(f"An error occurred in the main application: {e}")
        if not stop_event.is_set():
             stop_event.set()
        
    finally:
        print("Cleaning up resources...")
        
        if analysis_thread:
            analysis_thread.join()
            
        if camera:
            camera.stop()
            
        cv2.destroyAllWindows()
        print("Application finished.")


if __name__ == "__main__":
    main()