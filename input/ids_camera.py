# ids_camera.py
#
# Contains the IDS_Camera class for easy handling of IDS peak cameras.
# It encapsulates the camera connection, stream acquisition in a separate
# thread, and provides a clean interface to the main application.

import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from ids_peak import ids_peak
from ids_peak import ids_peak, ids_peak_ipl_extension
import config

class IDS_Camera:
    """
    A wrapper class for IDS peak cameras to simplify acquisition and control.
    
    This class handles the device connection, stream setup, and image
    acquisition in a separate thread. It provides simple methods to start,
    stop, and retrieve frames.
    """

    def __init__(self):
        """
        Initializes the camera. It finds the first available IDS camera,
        opens it, and configures some basic default settings.
        """
        self._device = None
        self._nodemap_remote = None
        self._datastream = None
        
        self._acquisition_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Frame and metadata that are shared between threads
        self._latest_frame: Optional[np.ndarray] = None
        self._metadata: Dict[str, any] = {}

        try:
            # Initialize the library
            ids_peak.Library.Initialize()
            
            # Find and open the first available camera
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            if device_manager.Devices().empty():
                raise RuntimeError("No IDS camera found.")
            
            self._device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            self._nodemap_remote = self._device.RemoteDevice().NodeMaps()[0]
            
            print("Camera connection established.")
            self._configure_chunks() 
            self._apply_initial_settings()

        except Exception as e:
            print(f"ERROR: Failed to initialize camera: {e}")
            self.stop() # Ensure cleanup on initialization failure
            raise

    def _configure_chunks(self):
        """
        Enables Chunk Mode and selects all desired metadata chunks.
        This ensures that metadata is perfectly synchronized with each frame.
        """
        print("Configuring camera chunks...")
        try:
            # Activate Chunk Mode
            self._nodemap_remote.FindNode("ChunkModeActive").SetValue(True)

            # A list of all chunks we want the camera to send
            desired_chunks = [
                # Identification and timing
                "CounterValue",             # Frame ID counter
                "Timestamp",                # Hardware timestamp of the capture
                # Image geometry
                "Width",
                "Height",
                "OffsetX",
                "OffsetY",
            ]
            
            print("Enabling supported chunks...")
            for chunk_name in desired_chunks:
                try:
                    chunk_selector = self._nodemap_remote.FindNode("ChunkSelector")
                    chunk_selector.SetCurrentEntry(chunk_name)
                    self._nodemap_remote.FindNode("ChunkEnable").SetValue(True)
                    print(f"- Enabled: {chunk_name}")
                except ids_peak.Exception:
                    print(f"- WARN: Chunk '{chunk_name}' is not supported, skipping.")

        except Exception as e:
            print(f"ERROR: Failed to configure chunks: {e}")
            # Disable Chunk Mode if it fails
            self._nodemap_remote.FindNode("ChunkModeActive").SetValue(False)

    def _apply_initial_settings(self):
        """Applies a set of default camera settings."""
        print("Applying initial camera settings...")
        try:
            # For optimal performance, we use a Bayer format and convert later
            self._nodemap_remote.FindNode("PixelFormat").SetCurrentEntry("BayerRG8")
            
            # Set a high framerate
            self._nodemap_remote.FindNode("AcquisitionFrameRateTargetEnable").SetValue(True)
            fps_node = self._nodemap_remote.FindNode("AcquisitionFrameRateTarget")
            fps_node.SetValue(250.0)
            print(f"- AcquisitionFrameRate: {fps_node.Value()} fps")

            # Set a reasonable exposure time
            exp_time = self._nodemap_remote.FindNode("ExposureTime")
            exp_time.SetValue(2000.0)
            print(f"- ExposureTime: {exp_time.Value()} µs")

            # Disable auto gain and set a fixed value
            self._nodemap_remote.FindNode("GainAuto").SetCurrentEntry("Off")
            gain_node = self._nodemap_remote.FindNode("Gain")
            gain_node.SetValue(10.0)
            print(f"- Gain: {gain_node.Value()} dB")

            self._nodemap_remote.FindNode("BlackLevel").SetValue(10.0)
            print(f"BlackLevel eingestellt auf: {self._nodemap_remote.FindNode('BlackLevel').Value()}")

        except Exception as e:
            print(f"WARNING: Could not apply all initial settings: {e}")

    def start(self):
        """
        Starts the image acquisition in a separate thread.
        """
        if self._acquisition_thread is not None:
            print("Acquisition already running.")
            return

        print("Starting acquisition...")
        self._stop_event.clear()
        self._acquisition_thread = threading.Thread(target=self._acquisition_loop)
        self._acquisition_thread.start()
        print("Acquisition thread started.")

    def stop(self):
        """
        Stops the image acquisition thread and releases all resources.
        """
        print("Stopping acquisition...")
        if self._acquisition_thread:
            self._stop_event.set()
            self._acquisition_thread.join(timeout=2) # Wait for thread to finish
            self._acquisition_thread = None

        try:
            if self._nodemap_remote:
                self._nodemap_remote.FindNode("ChunkModeActive").SetValue(False)
                print("Chunk mode disabled.")
        except Exception as e:
            print(f"WARN: Could not disable chunk mode on exit: {e}")

        # Release camera resources
        if self._device:
            try:
                # The acquisition thread handles stopping the stream,
                # but we can try to stop it here as a fallback.
                self._nodemap_remote.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass # Might already be stopped
        
        # Close library
        ids_peak.Library.Close()
        print("Camera resources released.")

    def _acquisition_loop(self):
        """
        The main loop for acquiring images from the camera.
        This function runs in a separate thread.
        """
        try:
            # Open data stream
            self._datastream = self._device.DataStreams()[0].OpenDataStream()
            stream_nodemap = self._datastream.NodeMaps()[0]

            # Configure buffer handling for performance
            stream_nodemap.FindNode("StreamBufferHandlingMode").SetCurrentEntry("NewestOnly")
            
            # Allocate and queue buffers
            payload_size = self._nodemap_remote.FindNode("PayloadSize").Value()
            num_buffers = 10
            for _ in range(num_buffers):
                buf = self._datastream.AllocAndAnnounceBuffer(payload_size)
                self._datastream.QueueBuffer(buf)
            
            # Start acquisition
            self._datastream.StartAcquisition()
            self._nodemap_remote.FindNode("AcquisitionStart").Execute()
            self._nodemap_remote.FindNode("AcquisitionStart").WaitUntilDone()
            
            print("Acquisition loop is running.")
            
            # --- Main acquisition loop ---
            while not self._stop_event.is_set():
                buffer = None
                try:
                    buffer = self._datastream.WaitForFinishedBuffer(1000)
                    
                    # --- Extract all data from the buffer first ---
                    
                    # 1. Extract basic metadata that requires chunks (essential data)
                    try:
                        metadata = {
                            "frame_id": buffer.FrameID(),
                            "timestamp_ns": buffer.Timestamp_ns(),
                            "width": buffer.Width(),
                            "height": buffer.Height(),
                        }
                    except Exception as e:
                        # If basic chunk data access fails, this is likely an incomplete buffer
                        print(f"WARN: Incomplete buffer received, dropping frame: {e}")
                        if buffer is not None:
                            self._datastream.QueueBuffer(buffer)
                        continue
                    
                    # 2. Try to get offset data safely (optional data)
                    try:
                        # Check if the buffer has chunk data before accessing
                        if buffer.HasChunkData(ids_peak.ChunkID_OffsetX):
                            metadata["offset_x"] = buffer.XOffset()
                        else:
                            metadata["offset_x"] = config.CAM_X_OFFSET if hasattr(config, 'CAM_X_OFFSET') else 0
                            
                        if buffer.HasChunkData(ids_peak.ChunkID_OffsetY):
                            metadata["offset_y"] = buffer.YOffset()
                        else:
                            metadata["offset_y"] = config.CAM_Y_OFFSET if hasattr(config, 'CAM_Y_OFFSET') else 0
                    except Exception:
                        # Fallback if access fails despite check
                        metadata["offset_x"] = config.CAM_X_OFFSET if hasattr(config, 'CAM_X_OFFSET') else 0
                        metadata["offset_y"] = config.CAM_Y_OFFSET if hasattr(config, 'CAM_Y_OFFSET') else 0

                    # 3. Convert buffer to image using BufferToImage
                    try:
                        ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
                        frame_data = ipl_image.get_numpy_1D().copy()
                        frame_data = frame_data.reshape(metadata["height"], metadata["width"])
                    except Exception as e:
                        # If image conversion fails, this is also likely an incomplete buffer
                        print(f"WARN: Image conversion failed, dropping frame: {e}")
                        if buffer is not None:
                            self._datastream.QueueBuffer(buffer)
                        continue

                    # --- Now that we have everything, re-queue the buffer ---
                    if buffer is not None:
                        self._datastream.QueueBuffer(buffer)
                        buffer = None  # Mark as queued

                    # --- Share data with the main thread ---
                    with self._lock:
                        self._latest_frame = frame_data
                        self._metadata = metadata

                except ids_peak.Exception as e:
                    # Handle different types of IDS exceptions
                    if hasattr(e, 'error_code') and e.error_code == ids_peak.IPL_ERROR_TIMEOUT:
                        continue
                    else:
                        # Don't break on BAD_ACCESS errors - just drop the frame and continue
                        print(f"WARN: Dropped frame due to IDS peak error: {e}")
                        if buffer is not None:
                            try:
                                self._datastream.QueueBuffer(buffer)
                            except Exception:
                                pass
                        continue
                except Exception as e:
                    # Only break on truly critical errors, not buffer access issues
                    if "Buffer is currently not delivered" in str(e) or "BAD_ACCESS" in str(e):
                        print(f"WARN: Buffer access issue, dropping frame: {e}")
                        if buffer is not None:
                            try:
                                self._datastream.QueueBuffer(buffer)
                            except Exception:
                                pass
                        continue
                    else:
                        print(f"CRITICAL: General acquisition error, stopping loop: {e}")
                        break
                finally:
                    # Ensure buffer is always re-queued if still held
                    if buffer is not None:
                        try:
                            self._datastream.QueueBuffer(buffer)
                        except Exception:
                            # Buffer might already be queued, ignore
                            pass
        finally:
            # --- Cleanup in acquisition thread ---
            print("Acquisition loop is cleaning up.")
            if self._datastream:
                try:
                    self._nodemap_remote.FindNode("AcquisitionStop").Execute()
                    self._datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                    self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                    for buf in self._datastream.AnnouncedBuffers():
                        self._datastream.RevokeBuffer(buf)
                except Exception as e:
                    print(f"Error during stream cleanup: {e}")

    def get_frame(self) -> Tuple[Optional[np.ndarray], Dict[str, any]]:
        """
        Retrieves the latest captured frame and its metadata.

        Returns:
            A tuple containing:
            - The latest frame as a NumPy array (or None if not available).
            - A dictionary with metadata about the frame.
        """
        with self._lock:
            frame_copy = self._latest_frame.copy() if self._latest_frame is not None else None
            metadata_copy = self._metadata.copy()
        return frame_copy, metadata_copy

    def set_exposure(self, exposure_us: float):
        """
        Sets the camera's exposure time while the stream is running.
        
        Args:
            exposure_us: The new exposure time in microseconds.
        """
        try:
            exp_node = self._nodemap_remote.FindNode("ExposureTime")
            min_val, max_val = exp_node.Minimum(), exp_node.Maximum()
            
            if min_val <= exposure_us <= max_val:
                exp_node.SetValue(exposure_us)
                print(f"Exposure set to {exposure_us} µs.")
            else:
                print(f"WARN: Exposure value {exposure_us} is out of range ({min_val}-{max_val}).")
        except Exception as e:
            print(f"ERROR: Failed to set exposure: {e}")