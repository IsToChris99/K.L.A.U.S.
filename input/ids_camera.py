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
from ids_peak import ids_peak, ids_peak_ipl_extension
from config import (FRAME_RATE_TARGET, EXPOSURE_TIME, GAIN, BLACK_LEVEL, WHITE_BALANCE_AUTO)


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
        # Core camera objects
        self._device = None
        self._nodemap_remote = None
        self._datastream = None
        
        # Threading
        self._acquisition_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Shared data between threads
        self._latest_frame: Optional[np.ndarray] = None
        self._metadata: Dict[str, any] = {}
        
        # Frame tracking for dropped frame detection
        self._last_frame_id = -1
        self._total_dropped_frames = 0

        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize camera connection and basic setup."""
        try:
            ids_peak.Library.Initialize()
            
            # Find and open camera
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
            self.stop()
            raise

    def _configure_chunks(self):
        """Enable Chunk Mode and configure metadata chunks."""
        print("Configuring camera chunks...")
        
        # Enable chunk mode
        self._nodemap_remote.FindNode("ChunkModeActive").SetValue(True)

        # Configure chunks we need
        chunk_configs = [
            ("CounterValue", "Counter0"),  # Frame ID with counter selector
            ("Timestamp", None),           # Hardware timestamp
            ("Width", None),
            ("Height", None),
            ("OffsetX", None),
            ("OffsetY", None),
            ("Gain", None),
            ("ExposureTime", None),
        ]
        
        for chunk_name, selector_value in chunk_configs:
            try:
                chunk_selector = self._nodemap_remote.FindNode("ChunkSelector")
                chunk_selector.SetCurrentEntry(chunk_name)
                self._nodemap_remote.FindNode("ChunkEnable").SetValue(True)
                
                if selector_value:  # For CounterValue
                    self._nodemap_remote.FindNode("ChunkCounterSelector").SetCurrentEntry(selector_value)
                
                print(f"- Enabled: {chunk_name}")
            except ids_peak.Exception:
                print(f"- WARN: Chunk '{chunk_name}' not supported, skipping.")

        # Configure frame counter
        self._nodemap_remote.FindNode("CounterSelector").SetCurrentEntry("Counter0")
        self._nodemap_remote.FindNode("CounterEventSource").SetCurrentEntry("FrameEnd")
        self._nodemap_remote.FindNode("CounterDuration").SetValue(0)
        self._nodemap_remote.FindNode("CounterReset").Execute()
        self._nodemap_remote.FindNode("CounterTriggerSource").SetCurrentEntry("AcquisitionStart")

    def _apply_initial_settings(self):
        """Apply default camera settings for optimal performance."""
        print("Applying initial camera settings...")
        
        settings = [
            ("PixelFormat", "BayerRG8", "SetCurrentEntry"),
            ("AcquisitionFrameRateTargetEnable", True, "SetValue"),
            ("AcquisitionFrameRateTarget", FRAME_RATE_TARGET, "SetValue"),
            ("ExposureTime", EXPOSURE_TIME, "SetValue"),
            ("GainAuto", "Off", "SetCurrentEntry"),
            ("Gain", GAIN, "SetValue"),
            ("BlackLevel", BLACK_LEVEL, "SetValue"),
        ]
        
        for node_name, value, method in settings:
            try:
                node = self._nodemap_remote.FindNode(node_name)
                getattr(node, method)(value)
                
                # Print current value for verification
                if method == "SetValue":
                    print(f"- {node_name}: {node.Value()}")
                else:
                    print(f"- {node_name}: {value}")
                    
            except Exception as e:
                print(f"WARNING: Could not set {node_name}: {e}")

    def start(self):
        """Start image acquisition in a separate thread."""
        if self._acquisition_thread is not None:
            print("Acquisition already running.")
            return

        print("Starting acquisition...")
        # Reset dropped frame counters
        self._last_frame_id = -1
        self._total_dropped_frames = 0
        
        self._stop_event.clear()
        self._acquisition_thread = threading.Thread(target=self._acquisition_loop)
        self._acquisition_thread.start()
        print("Acquisition thread started.")

    def stop(self):
        """Stop acquisition thread and release all resources."""
        print("Stopping acquisition...")
        
        if self._acquisition_thread:
            self._stop_event.set()
            self._acquisition_thread.join(timeout=2)
            self._acquisition_thread = None

        # Cleanup
        self._cleanup_camera()
        ids_peak.Library.Close()
        print("Camera resources released.")

    def _cleanup_camera(self):
        """Clean up camera resources."""
        try:
            if self._nodemap_remote:
                self._nodemap_remote.FindNode("ChunkModeActive").SetValue(False)
                self._nodemap_remote.FindNode("AcquisitionStop").Execute()
        except Exception as e:
            print(f"WARN: Cleanup warning: {e}")

    def _acquisition_loop(self):
        """Main acquisition loop running in separate thread."""
        try:
            self._setup_datastream()
            self._run_acquisition()
        finally:
            self._cleanup_datastream()

    def _setup_datastream(self):
        """Setup data stream and buffers."""
        self._datastream = self._device.DataStreams()[0].OpenDataStream()
        stream_nodemap = self._datastream.NodeMaps()[0]

        # Increase buffer count for high-speed capture
        num_buffers = 10 
        
        # Configure for even better performance
        stream_nodemap.FindNode("StreamBufferHandlingMode").SetCurrentEntry("OldestFirst")
        # Add buffer alignment if available
        try:
            stream_nodemap.FindNode("StreamBufferAlignment").SetValue(4096)
        except:
            pass
        
        # Allocate buffers
        payload_size = self._nodemap_remote.FindNode("PayloadSize").Value()
        
        for _ in range(num_buffers):
            buf = self._datastream.AllocAndAnnounceBuffer(payload_size)
            self._datastream.QueueBuffer(buf)
        
        # Start acquisition
        self._datastream.StartAcquisition()
        self._nodemap_remote.FindNode("AcquisitionStart").Execute()
        self._nodemap_remote.FindNode("AcquisitionStart").WaitUntilDone()
        
        print("Acquisition loop is running.")

    def _run_acquisition(self):
        """Main acquisition loop."""
        while not self._stop_event.is_set():
            buffer = None
            try:
                buffer = self._datastream.WaitForFinishedBuffer(1000)
                
                # Extract metadata and image data
                metadata = self._extract_metadata(buffer)
                if metadata is None:
                    continue
                
                frame_data = self._extract_image_data(buffer, metadata)
                if frame_data is None:
                    continue
                
                # Check for dropped frames
                current_frame_id = metadata.get("frame_id", -1)
                self._update_dropped_frame_count(current_frame_id)
                
                # Add dropped frame count to metadata
                metadata["dropped_frames"] = self._total_dropped_frames

                # Share data with main thread
                with self._lock:
                    self._latest_frame = frame_data
                    self._metadata = metadata

            except ids_peak.Exception as e:
                if hasattr(e, 'error_code') and e.error_code == ids_peak.IPL_ERROR_TIMEOUT:
                    continue
                else:
                    print(f"WARN: Dropped frame due to IDS peak error: {e}")
                    continue
                    
            except Exception as e:
                if any(err in str(e) for err in ["Buffer is currently not delivered", "BAD_ACCESS"]):
                    print(f"WARN: Buffer access issue, dropping frame: {e}")
                    continue
                else:
                    print(f"CRITICAL: General acquisition error, stopping loop: {e}")
                    break
            finally:
                # Always requeue buffer
                if buffer is not None:
                    try:
                        self._datastream.QueueBuffer(buffer)
                    except Exception:
                        pass

    def _update_dropped_frame_count(self, current_frame_id: int):
        """Update the dropped frame counter based on frame IDs."""
        if self._last_frame_id != -1 and current_frame_id > self._last_frame_id + 1:
            dropped_in_gap = current_frame_id - (self._last_frame_id + 1)
            self._total_dropped_frames += dropped_in_gap
            #if dropped_in_gap > 0:
                #print(f"WARN: {dropped_in_gap} frame(s) dropped between ID {self._last_frame_id} and {current_frame_id}, total dropped: {self._total_dropped_frames}")
        
        self._last_frame_id = current_frame_id

    def _extract_metadata(self, buffer) -> Optional[Dict]:
        """Extract metadata from buffer chunks."""
        try:
            if not buffer.HasChunks():
                return None
            
            self._nodemap_remote.UpdateChunkNodes(buffer)
            self._nodemap_remote.FindNode("ChunkCounterSelector").SetCurrentEntry("Counter0")
            
            return {
                "frame_id": self._nodemap_remote.FindNode("ChunkCounterValue").Value(),
                "timestamp_ns": self._nodemap_remote.FindNode("ChunkTimestamp").Value(),
                "width": self._nodemap_remote.FindNode("ChunkWidth").Value(),
                "height": self._nodemap_remote.FindNode("ChunkHeight").Value(),
                "offset_x": self._nodemap_remote.FindNode("ChunkOffsetX").Value(),
                "offset_y": self._nodemap_remote.FindNode("ChunkOffsetY").Value(),
                "gain": self._nodemap_remote.FindNode("ChunkGain").Value(),
                "exposure_time": self._nodemap_remote.FindNode("ChunkExposureTime").Value()
            }
            
        except Exception as e:
            print(f"WARN: Incomplete buffer received, dropping frame: {e}")
            return None

    def _extract_image_data(self, buffer, metadata) -> Optional[np.ndarray]:
        """Extract and convert image data from buffer."""
        try:
            ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
            frame_data = ipl_image.get_numpy_1D().copy()
            return frame_data.reshape(metadata["height"], metadata["width"])
        except Exception as e:
            print(f"WARN: Image conversion failed, dropping frame: {e}")
            return None

    def _cleanup_datastream(self):
        """Clean up data stream resources."""
        print("Acquisition loop is cleaning up.")
        if not self._datastream:
            return
            
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
        Retrieve the latest captured frame and its metadata.

        Returns:
            Tuple of (frame, metadata) where frame is a NumPy array or None.
            Metadata now includes 'dropped_frames' count.
        """
        with self._lock:
            frame_copy = self._latest_frame.copy() if self._latest_frame is not None else None
            metadata_copy = self._metadata.copy()
        return frame_copy, metadata_copy

    def set_exposure(self, exposure_us: float):
        """
        Set camera exposure time while stream is running.
        
        Args:
            exposure_us: New exposure time in microseconds
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

    def get_dropped_frame_count(self) -> int:
        """
        Get the total number of dropped frames since acquisition started.
        
        Returns:
            Total number of dropped frames
        """
        return self._total_dropped_frames