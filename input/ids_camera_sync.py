# ids_camera_sync.py
# Synchronous version of IDS_Camera for multi-processing architecture

import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension
import config


class IDS_Camera:
    """
    Synchronous wrapper class for IDS peak cameras.
    
    This version does NOT use internal threading - instead it provides
    a simple get_frame() method that blocks until a frame is available.
    Threading is handled externally by the application.
    """

    def __init__(self):
        """Initialize the camera without starting acquisition."""
        # Core camera objects
        self._device = None
        self._nodemap_remote = None
        self._datastream = None
        
        # Frame tracking for dropped frame detection
        self._last_frame_id = -1
        self._total_dropped_frames = 0
        self._acquisition_active = False
        
        # Store default values for runtime adjustment
        self._default_settings = {
            "exposure_time": 2000.0,
            "gain": 10.0,
            "black_level": 10.0,
            "frame_rate_target": 250.0,
            "white_balance_auto": "Off"
        }

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
            ("AcquisitionFrameRateTarget", self._default_settings["frame_rate_target"], "SetValue"),
            ("ExposureTime", self._default_settings["exposure_time"], "SetValue"),
            ("GainAuto", "Off", "SetCurrentEntry"),
            ("Gain", self._default_settings["gain"], "SetValue"),
            ("BlackLevel", self._default_settings["black_level"], "SetValue"),
            ("BalanceWhiteAuto", self._default_settings["white_balance_auto"], "SetCurrentEntry"),
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
        """Start image acquisition (synchronous - no threading)."""
        if self._acquisition_active:
            print("Acquisition already active.")
            return

        print("Starting synchronous acquisition...")
        # Reset dropped frame counters
        self._last_frame_id = -1
        self._total_dropped_frames = 0
        
        self._setup_datastream()
        self._acquisition_active = True
        print("Acquisition ready.")

    def stop(self):
        """Stop acquisition and release all resources."""
        print("Stopping acquisition...")
        
        self._acquisition_active = False
        self._cleanup_datastream()
        self._cleanup_camera()
        ids_peak.Library.Close()
        print("Camera resources released.")

    def _setup_datastream(self):
        """Setup data stream and buffers."""
        self._datastream = self._device.DataStreams()[0].OpenDataStream()
        stream_nodemap = self._datastream.NodeMaps()[0]

        # Increase buffer count for high-speed capture
        num_buffers = 10 
        
        # Configure for performance
        stream_nodemap.FindNode("StreamBufferHandlingMode").SetCurrentEntry("OldestFirst")
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

    def _cleanup_datastream(self):
        """Clean up data stream resources."""
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

    def _cleanup_camera(self):
        """Clean up camera resources."""
        try:
            if self._nodemap_remote:
                self._nodemap_remote.FindNode("ChunkModeActive").SetValue(False)
        except Exception as e:
            print(f"WARN: Cleanup warning: {e}")

    def get_frame(self) -> Tuple[Optional[np.ndarray], Dict[str, any]]:
        """
        Get a single frame synchronously (blocking call).
        
        Returns:
            Tuple of (frame, metadata) where frame is a NumPy array or None.
        """
        if not self._acquisition_active:
            return None, {}
            
        buffer = None
        try:
            # Block until frame is available (with timeout)
            buffer = self._datastream.WaitForFinishedBuffer(1000)  # 1 second timeout
            
            # Extract metadata and image data
            metadata = self._extract_metadata(buffer)
            if metadata is None:
                return None, {}
            
            frame_data = self._extract_image_data(buffer, metadata)
            if frame_data is None:
                return None, {}
            
            # Check for dropped frames
            current_frame_id = metadata.get("frame_id", -1)
            self._update_dropped_frame_count(current_frame_id)
            
            # Add dropped frame count to metadata
            metadata["dropped_frames"] = self._total_dropped_frames

            return frame_data, metadata

        except ids_peak.Exception as e:
            if hasattr(e, 'error_code') and e.error_code == ids_peak.IPL_ERROR_TIMEOUT:
                return None, {}  # Timeout - normal case
            else:
                print(f"WARN: Frame acquisition error: {e}")
                return None, {}
                
        except Exception as e:
            print(f"ERROR: General frame acquisition error: {e}")
            return None, {}
            
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
            print(f"WARN: Incomplete buffer received: {e}")
            return None

    def _extract_image_data(self, buffer, metadata) -> Optional[np.ndarray]:
        """Extract and convert image data from buffer."""
        try:
            ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
            frame_data = ipl_image.get_numpy_1D().copy()
            return frame_data.reshape(metadata["height"], metadata["width"])
        except Exception as e:
            print(f"WARN: Image conversion failed: {e}")
            return None

    def set_exposure(self, exposure_us: float):
        """Set camera exposure time."""
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

    def get_exposure(self) -> float:
        """Get current camera exposure time."""
        try:
            exp_node = self._nodemap_remote.FindNode("ExposureTime")
            return exp_node.Value()
        except Exception as e:
            print(f"ERROR: Failed to get exposure: {e}")
            return self._default_settings["exposure_time"]

    def set_gain(self, gain: float):
        """Set camera gain."""
        try:
            gain_node = self._nodemap_remote.FindNode("Gain")
            min_val, max_val = gain_node.Minimum(), gain_node.Maximum()
            
            if min_val <= gain <= max_val:
                gain_node.SetValue(gain)
                print(f"Gain set to {gain}.")
            else:
                print(f"WARN: Gain value {gain} is out of range ({min_val}-{max_val}).")
        except Exception as e:
            print(f"ERROR: Failed to set gain: {e}")

    def get_gain(self) -> float:
        """Get current camera gain."""
        try:
            gain_node = self._nodemap_remote.FindNode("Gain")
            return gain_node.Value()
        except Exception as e:
            print(f"ERROR: Failed to get gain: {e}")
            return self._default_settings["gain"]

    def set_black_level(self, black_level: float):
        """Set camera black level."""
        try:
            bl_node = self._nodemap_remote.FindNode("BlackLevel")
            min_val, max_val = bl_node.Minimum(), bl_node.Maximum()
            
            if min_val <= black_level <= max_val:
                bl_node.SetValue(black_level)
                print(f"Black level set to {black_level}.")
            else:
                print(f"WARN: Black level value {black_level} is out of range ({min_val}-{max_val}).")
        except Exception as e:
            print(f"ERROR: Failed to set black level: {e}")

    def get_black_level(self) -> float:
        """Get current camera black level."""
        try:
            bl_node = self._nodemap_remote.FindNode("BlackLevel")
            return bl_node.Value()
        except Exception as e:
            print(f"ERROR: Failed to get black level: {e}")
            return self._default_settings["black_level"]

    def set_frame_rate_target(self, frame_rate: float):
        """Set camera frame rate target."""
        try:
            # First check if frame rate control is enabled
            fr_enable_node = self._nodemap_remote.FindNode("AcquisitionFrameRateTargetEnable")
            if not fr_enable_node.Value():
                fr_enable_node.SetValue(True)
                print("Frame rate target control enabled.")
            
            fr_node = self._nodemap_remote.FindNode("AcquisitionFrameRateTarget")
            min_val, max_val = fr_node.Minimum(), fr_node.Maximum()
            
            if min_val <= frame_rate <= max_val:
                fr_node.SetValue(frame_rate)
                print(f"Frame rate target set to {frame_rate} fps.")
            else:
                print(f"WARN: Frame rate value {frame_rate} is out of range ({min_val}-{max_val}).")
        except Exception as e:
            print(f"ERROR: Failed to set frame rate target: {e}")

    def get_frame_rate_target(self) -> float:
        """Get current camera frame rate target."""
        try:
            fr_node = self._nodemap_remote.FindNode("AcquisitionFrameRateTarget")
            return fr_node.Value()
        except Exception as e:
            print(f"ERROR: Failed to get frame rate target: {e}")
            return self._default_settings["frame_rate_target"]

    def set_white_balance_auto(self, mode: str):
        """
        Set white balance auto mode.
        
        Args:
            mode: "Off", "Continuous", or "Once"
                  Note: "Once" will automatically reset to "Off" after execution
        """
        try:
            wb_node = self._nodemap_remote.FindNode("BalanceWhiteAuto")
            
            # Validate mode
            valid_modes = ["Off", "Continuous", "Once"]
            if mode not in valid_modes:
                print(f"WARN: Invalid white balance mode '{mode}'. Valid modes: {valid_modes}")
                return
            
            wb_node.SetCurrentEntry(mode)
            print(f"White balance auto set to '{mode}'.")
            
            # If "Once" was selected, inform user about automatic reset behavior
            if mode == "Once":
                print("Note: White balance mode will automatically reset to 'Off' after execution.")
                
        except Exception as e:
            print(f"ERROR: Failed to set white balance auto: {e}")

    def get_white_balance_auto(self) -> str:
        """Get current white balance auto mode."""
        try:
            wb_node = self._nodemap_remote.FindNode("BalanceWhiteAuto")
            return wb_node.CurrentEntry().StringValue()
        except Exception as e:
            print(f"ERROR: Failed to get white balance auto: {e}")
            return self._default_settings["white_balance_auto"]

    def reset_to_defaults(self):
        """Reset all camera settings to their default values."""
        print("Resetting camera settings to defaults...")
        
        self.set_exposure(self._default_settings["exposure_time"])
        self.set_gain(self._default_settings["gain"])
        self.set_black_level(self._default_settings["black_level"])
        self.set_frame_rate_target(self._default_settings["frame_rate_target"])
        self.set_white_balance_auto(self._default_settings["white_balance_auto"])
        
        print("Camera settings reset to defaults.")

    def get_current_settings(self) -> Dict[str, any]:
        """Get all current camera settings as a dictionary."""
        return {
            "exposure_time": self.get_exposure(),
            "gain": self.get_gain(),
            "black_level": self.get_black_level(),
            "frame_rate_target": self.get_frame_rate_target(),
            "white_balance_auto": self.get_white_balance_auto()
        }

    def get_dropped_frame_count(self) -> int:
        """Get the total number of dropped frames since acquisition started."""
        return self._total_dropped_frames
