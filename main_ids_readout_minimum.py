# minimal_opencv_example.py
#
# A minimal example demonstrating how to use the IDS_Camera class to get
# frames and process them with OpenCV.

import cv2
from input.ids_camera import IDS_Camera # Adjust the import path if necessary

def main():
    """
    The main function that handles camera initialization, the capture loop,
    and cleanup.
    """
    camera = None
    try:
        # 1. Initialize the Camera
        print("Initializing camera...")
        camera = IDS_Camera()
        
        # 2. Start the Acquisition Stream
        camera.start()
        print("Stream started. Press 'q' in the camera window to exit.")

        # 3. The Main Processing Loop
        while True:
            # 3.1. Get the latest frame from the camera class.
            bayer_frame, metadata = camera.get_frame()
            
            # 3.2. Check if a frame was successfully received.
            if bayer_frame is not None:
                
                # a) Convert the raw Bayer image to a color image for display.
                color_frame = cv2.cvtColor(bayer_frame, cv2.COLOR_BAYER_RG2RGB)
                
                # b) Display the resulting color image in a window.
                cv2.imshow("Minimal IDS Camera Stream", color_frame)
                
                # --- YOUR OPENCV PROCESSING ENDS HERE ---

            # 4. Wait for user input to quit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, exiting.")
                break
                
    except Exception as e:
        # Catch any potential errors during initialization or runtime.
        print(f"An error occurred: {e}")
        
    finally:
        # 5. Cleanly stop the camera and release all resources.
        print("Shutting down...")
        if camera:
            camera.stop()
        cv2.destroyAllWindows()
        print("Application finished.")


# Standard entry point for a Python script.
if __name__ == "__main__":
    main()