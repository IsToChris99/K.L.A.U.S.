import cv2
import numpy as np
import json
import os
import ctypes
import pyglet
from pyglet.gl import *
import time
import config
from .cpu_preprocessor import CPUPreprocessor

# Disable PyOpenGL error checking for maximum speed.
# For debugging, set to True: pyglet.options['debug_gl'] = True
pyglet.options['debug_gl'] = False

class GPUPreprocessor:
    """
    Performs a complete preprocessing pipeline (Debayering, Resize, Undistortion)
    on the GPU using OpenGL.

    Input: Raw Bayer frame (NumPy Array)
    :param bayer_size: Tuple (width, height) of the raw Bayer frame.
    :param target_size: Tuple (width, height) of the final output frame.
    :param calibration_file: Path to the JSON calibration file.
    Output: Processed RGB frame (NumPy Array)
    """

    def __init__(self, bayer_size = (config.CAM_WIDTH, config.CAM_HEIGHT), target_size=(config.DETECTION_WIDTH, config.DETECTION_HEIGHT), calibration_file=config.CAMERA_CALIBRATION_FILE):
        """
        Initializes the GPU processor.

        :param calibration_file: Path to the JSON calibration file.
        :param bayer_size: Tuple (width, height) of the raw Bayer frame.
        :param target_size: Tuple (width, height) of the final output frame.
        """
        self.bayer_width, self.bayer_height = bayer_size
        self.target_width, self.target_height = target_size
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = self._load_calibration(calibration_file)

        if not self.calibrated:
            raise Exception("GPU initialization failed: calibration data could not be loaded.")
        
        # Initialize OpenGL components
        self.window = None
        
        # Multi-pass shader programs
        self.demosaic_shader = None  # Pass 1: Bayer -> RGB
        self.undistort_shader = None # Pass 2: RGB -> Undistorted & Resized
        
        self.vao = None
        self.vbo = None
        
        # Textures and framebuffers for multi-pass rendering
        self.bayer_texture_id = None      # Input: Raw Bayer data
        self.rgb_texture_id = None        # Intermediate: Demosaiced RGB (full size)
        self.output_texture_id = None     # Output: Final undistorted & resized
        
        # Framebuffers for each pass
        self.demosaic_fbo = None          # For Pass 1: Bayer -> RGB
        self.undistort_fbo = None         # For Pass 2: RGB -> Final
        
        # Remap textures for undistortion (like OpenCV's remap)
        self.map1_texture_id = None
        self.map2_texture_id = None
        self.maps_initialized = False
        
        # Pixel Buffer Objects for asynchronous readback
        self.pbo_ids = []  # Will be populated in _initialize_pbos
        self.pbo_index = 0
        self.pbo_initialized = False
        
        # Prepare buffer for reading pixels back from GPU
        self.output_buffer = (GLubyte * (self.target_width * self.target_height * 3))()
        self.output_frame = np.empty((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Initialization flag
        self.initialized = False
        self.use_cpu_fallback = False
        
        # CPU fallback preprocessing instance
        self.cpu_preprocessor = CPUPreprocessor(bayer_size, target_size, calibration_file)
        
        # Initialize OpenGL context immediately
        self._initialize_gl_context()
        
        print(f"GPU Preprocessor initialized: {bayer_size} -> {target_size}")

    def reload_shaders(self):
        """Reloads and recompiles shaders - useful for development"""
        if self.initialized and not self.use_cpu_fallback:
            try:
                self.window.switch_to()
                self._initialize_shaders()
                print("Shaders reloaded successfully")
            except Exception as e:
                print(f"Failed to reload shaders: {e}")

    def force_reinitialize(self):
        """Forces complete reinitialization of OpenGL context and shaders"""
        self.initialized = False
        self.use_cpu_fallback = False
        if self.window:
            self.window.close()
            self.window = None
        self._initialize_gl_context()
        print("GPU preprocessor reinitialized")

    def _load_calibration(self, calibration_file):
        """Loads calibration data from a JSON file."""
        if not os.path.exists(calibration_file):
            print(f"Error: calibration file {calibration_file} not found.")
            return False
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
            self.camera_matrix = np.array(data['cameraMatrix'], dtype=np.float32)
            print(f"Camera matrix loaded: {self.camera_matrix.shape}")
            self.dist_coeffs = np.array(data['distCoeffs'], dtype=np.float32)
            print(f"Distortion coefficients loaded: {self.dist_coeffs.shape}")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

    def _initialize_remap_maps(self):
        """Initialize the undistortion remap maps, same as CPU version"""
        if not self.calibrated:
            return False
            
        # Use bayer size for the maps (undistortion happens before resizing)
        frame_size = (self.bayer_width, self.bayer_height)
        
        # Get optimal camera matrix, same as CPU version
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, frame_size, alpha=1.0)
        
        # Generate the same remap maps as CPU version
        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, 
            new_camera_matrix, frame_size, cv2.CV_16SC2)
        
        print(f"Generated remap maps for size: {frame_size}")
        print(f"Map1 shape: {map1.shape}, dtype: {map1.dtype}")
        print(f"Map2 shape: {map2.shape}, dtype: {map2.dtype}")
        
        # Store maps for GPU upload
        self.map1 = map1
        self.map2 = map2
        self.new_camera_matrix = new_camera_matrix
        
        return True

    def _load_shader_file(self, filename):
        """Loads shader source code from a file."""
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
        shader_path = os.path.join(shader_dir, filename)
        
        if not os.path.exists(shader_path):
            raise FileNotFoundError(f"Shader file not found: {shader_path}")
        
        try:
            with open(shader_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading shader file {shader_path}: {e}")

    def _initialize_shaders(self):
        """Compiles the vertex and fragment shaders for multi-pass rendering."""
        try:
            # Load vertex shader (same for both passes)
            vertex_shader_source = self._load_shader_file('vertex.glsl')
            
            # Load fragment shaders for each pass
            demosaic_fragment_source = self._load_shader_file('demosaic_fragment.glsl')
            undistort_fragment_source = self._load_shader_file('undistort_fragment.glsl')
            
            # Compile Pass 1 shaders (Demosaicing)
            vs1 = pyglet.graphics.shader.Shader(vertex_shader_source, 'vertex')
            fs1 = pyglet.graphics.shader.Shader(demosaic_fragment_source, 'fragment')
            self.demosaic_shader = pyglet.graphics.shader.ShaderProgram(vs1, fs1)
            
            # Compile Pass 2 shaders (Undistortion & Resize)
            vs2 = pyglet.graphics.shader.Shader(vertex_shader_source, 'vertex')
            fs2 = pyglet.graphics.shader.Shader(undistort_fragment_source, 'fragment')
            self.undistort_shader = pyglet.graphics.shader.ShaderProgram(vs2, fs2)
            
            print("Multi-pass shaders loaded and compiled successfully")
            
        except Exception as e:
            print(f"Failed to load/compile multi-pass shaders: {e}")
            raise

    def _initialize_gl_objects(self):
        """Creates all necessary OpenGL objects for multi-pass rendering."""
        # Initialize remap maps first
        if not self._initialize_remap_maps():
            raise Exception("Failed to initialize remap maps")
            
        # --- VAO/VBO for a rectangle that fills the screen ---
        quad_vertices = np.array([0,0, 1,0, 1,1, 0,1], dtype=np.float32)
        self.vao = GLuint()
        glGenVertexArrays(1, self.vao)
        glBindVertexArray(self.vao)
        self.vbo = GLuint()
        glGenBuffers(1, self.vbo)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices.ctypes.data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0)
        glEnableVertexAttribArray(0)

        # --- PASS 1 TEXTURES & FRAMEBUFFER (Bayer -> RGB) ---
        
        # Input texture for Bayer frame
        self.bayer_texture_id = GLuint()
        glGenTextures(1, self.bayer_texture_id)
        glBindTexture(GL_TEXTURE_2D, self.bayer_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, self.bayer_width, self.bayer_height, 0, GL_RED, GL_UNSIGNED_BYTE, None)

        # Intermediate RGB texture (full Bayer size)
        self.rgb_texture_id = GLuint()
        glGenTextures(1, self.rgb_texture_id)
        glBindTexture(GL_TEXTURE_2D, self.rgb_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, self.bayer_width, self.bayer_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        # Framebuffer for Pass 1 (Demosaicing)
        self.demosaic_fbo = GLuint()
        glGenFramebuffers(1, self.demosaic_fbo)
        glBindFramebuffer(GL_FRAMEBUFFER, self.demosaic_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rgb_texture_id, 0)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Demosaic framebuffer is not complete!")

        # --- PASS 2 TEXTURES & FRAMEBUFFER (RGB -> Undistorted & Resized) ---
        
        # Upload remap textures for undistortion
        self._upload_remap_maps_to_gpu()

        # Final output texture (target size)
        self.output_texture_id = GLuint()
        glGenTextures(1, self.output_texture_id)
        glBindTexture(GL_TEXTURE_2D, self.output_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, self.target_width, self.target_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        # Framebuffer for Pass 2 (Undistortion & Resize)
        self.undistort_fbo = GLuint()
        glGenFramebuffers(1, self.undistort_fbo)
        glBindFramebuffer(GL_FRAMEBUFFER, self.undistort_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.output_texture_id, 0)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Undistort framebuffer is not complete!")
        
        # Reset to default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Disable unnecessary OpenGL features for better performance
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_STENCIL_TEST)
        glDisable(GL_DITHER)
        
        # Initialize Pixel Buffer Objects for async readback
        self._initialize_pbos()
        
        print("Multi-pass OpenGL objects initialized successfully")
        glDisable(GL_DITHER)
        
        # Initialize Pixel Buffer Objects for async readback
        self._initialize_pbos()
        
        # Keep framebuffer bound for process_frame calls
        # glBindFramebuffer(GL_FRAMEBUFFER, 0) # Don't reset to default

    def _upload_remap_maps_to_gpu(self):
        """Upload the remap maps to GPU textures"""
        if not hasattr(self, 'map1') or not hasattr(self, 'map2'):
            raise Exception("Remap maps not initialized")
            
        # OpenCV map1 is CV_16SC2 containing x,y coordinates as 16-bit signed integers
        # We need to convert these to normalized float coordinates [0,1] for GPU
        print(f"Map1 shape: {self.map1.shape}, dtype: {self.map1.dtype}")
        
        # Convert from 16-bit signed integer pixel coordinates to normalized float coordinates
        # self.map1 contains absolute pixel coordinates, we need to normalize them
        map1_float = self.map1.astype(np.float32)
        
        # Handle invalid pixels (OpenCV uses negative values for invalid pixels)
        invalid_mask = (map1_float[:,:,0] < 0) | (map1_float[:,:,1] < 0)
        
        # Normalize valid coordinates to [0,1] range
        map1_float[:,:,0] = map1_float[:,:,0] / self.bayer_width
        map1_float[:,:,1] = map1_float[:,:,1] / self.bayer_height
        
        # Set invalid pixels to coordinates outside [0,1] range so they'll be detected as invalid
        map1_float[invalid_mask, 0] = -1.0
        map1_float[invalid_mask, 1] = -1.0
        
        # Clamp valid coordinates to [0,1] range
        map1_float = np.clip(map1_float, 0.0, 1.0)
        
        # Restore invalid pixel markers
        map1_float[invalid_mask, 0] = -1.0
        map1_float[invalid_mask, 1] = -1.0
        
        print(f"Map1 normalized range: x=[{map1_float[:,:,0].min():.3f}, {map1_float[:,:,0].max():.3f}], y=[{map1_float[:,:,1].min():.3f}, {map1_float[:,:,1].max():.3f}]")
        
        # Create map1 texture (RG32F for x,y coordinates)
        self.map1_texture_id = GLuint()
        glGenTextures(1, self.map1_texture_id)
        glBindTexture(GL_TEXTURE_2D, self.map1_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Upload normalized coordinates to GPU
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, self.bayer_width, self.bayer_height, 
                     0, GL_RG, GL_FLOAT, map1_float.ctypes.data)
        
        # Map2 is not used for CV_16SC2 format, but create a dummy texture for compatibility
        self.map2_texture_id = GLuint()
        glGenTextures(1, self.map2_texture_id)
        glBindTexture(GL_TEXTURE_2D, self.map2_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Create a small dummy texture
        dummy_data = np.zeros((1, 1, 1), dtype=np.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 1, 1, 0, GL_RED, GL_UNSIGNED_BYTE, dummy_data.ctypes.data)
        
        self.maps_initialized = True
        print("Remap maps uploaded to GPU textures")

    def _initialize_pbos(self):
        """Initialize Pixel Buffer Objects for asynchronous GPU-CPU transfer"""
        buffer_size = self.target_width * self.target_height * 3
        
        # Create array of GLuint for the PBO IDs
        pbo_array = (GLuint * 2)()
        glGenBuffers(2, pbo_array)
        self.pbo_ids = [pbo_array[0], pbo_array[1]]
        
        for pbo_id in self.pbo_ids:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id)
            glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, None, GL_STREAM_READ)
        
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self.pbo_initialized = True
        print("PBOs initialized for async readback")

    def _initialize_gl_context(self):
        """Initializes OpenGL context and all related objects."""
        if self.use_cpu_fallback:
            return
            
        try:
            print("Initializing GPU preprocessing context...")
            
            # Create a simple OpenGL config for off-screen rendering
            try:
                config = pyglet.gl.Config(
                    double_buffer=False,  # Not needed for off-screen rendering
                    depth_size=0,         # No depth buffer needed
                    stencil_size=0,       # No stencil buffer needed
                    alpha_size=0,         # No alpha channel in window buffer needed
                    sample_buffers=0,     # No anti-aliasing
                    samples=0             # No anti-aliasing
                )
            except pyglet.gl.NoSuchConfigException:
                print("No suitable OpenGL config found, using default.")
                config = pyglet.gl.Config()

            # Create window with explicit, simple configuration
            self.window = pyglet.window.Window(
                width=1,  # Size is irrelevant for invisible window
                height=1,
                caption="GPU Preprocessing Context", 
                visible=False, 
                config=config
            )

            self.window.switch_to()
            
            self._initialize_shaders()
            self._initialize_gl_objects()
            self.initialized = True
            print("GPU preprocessing initialized successfully")
            
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU preprocessing")
            self.use_cpu_fallback = True
            if self.window:
                self.window.close()
                self.window = None

    def process_frame(self, bayer_frame):
        """
        Processes a single raw Bayer frame on the GPU using multi-pass rendering.
        Pass 1: Bayer -> RGB (Demosaicing)  
        Pass 2: RGB -> Undistorted & Resized
        
        :param bayer_frame: NumPy Array (height, width) of the raw Bayer frame.
        :return: Processed NumPy Array (target_height, target_width, 3) in BGR format.
        """
        # Check if we should use CPU fallback
        if self.use_cpu_fallback:
            return self.cpu_preprocessor.process_frame(bayer_frame)

        # Check if initialization is needed
        if not self.initialized:
            return self.cpu_preprocessor.process_frame(bayer_frame)
            
        # Check if input is a color frame (video) - use CPU fallback for now
        if len(bayer_frame.shape) == 3 and bayer_frame.shape[2] == 3:
            return self.cpu_preprocessor.process_frame(bayer_frame)
            
        try:
            # --- PASS 1: BAYER -> RGB (DEMOSAICING) ---
            
            # 1. Upload Bayer frame to GPU texture
            glBindTexture(GL_TEXTURE_2D, self.bayer_texture_id)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.bayer_width, self.bayer_height, GL_RED, GL_UNSIGNED_BYTE, bayer_frame.ctypes.data)

            # 2. Render Pass 1: Demosaicing
            glBindFramebuffer(GL_FRAMEBUFFER, self.demosaic_fbo)
            glViewport(0, 0, self.bayer_width, self.bayer_height)  # Full Bayer size
            
            self.demosaic_shader.use()
            
            # Bind Bayer texture
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.bayer_texture_id)
            
            # Set uniforms for Pass 1
            try:
                program_id = self.demosaic_shader.id if hasattr(self.demosaic_shader, 'id') else self.demosaic_shader._program_id
                
                # Set texture sampler
                bayer_texture_location = glGetUniformLocation(program_id, b"bayerTexture")
                if bayer_texture_location != -1:
                    glUniform1i(bayer_texture_location, 0)  # Texture unit 0
                
                # Set size uniform
                bayer_size_location = glGetUniformLocation(program_id, b"bayerSize")
                if bayer_size_location != -1:
                    glUniform2f(bayer_size_location, float(self.bayer_width), float(self.bayer_height))
                    
            except Exception as e:
                print(f"Warning: Could not set Pass 1 uniform variables: {e}")
            
            # Render to RGB texture
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            # --- PASS 2: RGB -> UNDISTORTED & RESIZED ---
            
            # 3. Render Pass 2: Undistortion and Resize
            glBindFramebuffer(GL_FRAMEBUFFER, self.undistort_fbo)
            glViewport(0, 0, self.target_width, self.target_height)  # Target size
            
            self.undistort_shader.use()
            
            # Bind RGB texture from Pass 1
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.rgb_texture_id)
            
            # Bind remap textures  
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.map1_texture_id)
            
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.map2_texture_id)
            
            # Set uniforms for Pass 2
            try:
                program_id = self.undistort_shader.id if hasattr(self.undistort_shader, 'id') else self.undistort_shader._program_id
                
                # Set texture samplers
                rgb_texture_location = glGetUniformLocation(program_id, b"rgbTexture")
                map1_texture_location = glGetUniformLocation(program_id, b"map1Texture") 
                map2_texture_location = glGetUniformLocation(program_id, b"map2Texture")
                
                if rgb_texture_location != -1:
                    glUniform1i(rgb_texture_location, 0)  # Texture unit 0
                if map1_texture_location != -1:
                    glUniform1i(map1_texture_location, 1)   # Texture unit 1
                if map2_texture_location != -1:
                    glUniform1i(map2_texture_location, 2)   # Texture unit 2
                
                # Set size uniforms
                rgb_size_location = glGetUniformLocation(program_id, b"rgbSize")
                target_size_location = glGetUniformLocation(program_id, b"targetSize")
                
                if rgb_size_location != -1:
                    glUniform2f(rgb_size_location, float(self.bayer_width), float(self.bayer_height))
                if target_size_location != -1:
                    glUniform2f(target_size_location, float(self.target_width), float(self.target_height))
                    
            except Exception as e:
                print(f"Warning: Could not set Pass 2 uniform variables: {e}")
            
            # Render to final output texture
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            # --- READBACK FINAL RESULT ---
            
            # Keep the undistort framebuffer bound for readback
            # 4. Asynchronous readback using PBOs
            if self.pbo_initialized:
                # Use ping-pong PBOs for async transfer
                current_pbo = self.pbo_ids[self.pbo_index]
                next_pbo = self.pbo_ids[1 - self.pbo_index]
                
                # Bind PBO for reading current frame
                glBindBuffer(GL_PIXEL_PACK_BUFFER, current_pbo)
                glReadPixels(0, 0, self.target_width, self.target_height, GL_BGR, GL_UNSIGNED_BYTE, 0)
                
                # Read from the other PBO (previous frame)
                glBindBuffer(GL_PIXEL_PACK_BUFFER, next_pbo)
                buffer_ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
                
                if buffer_ptr:
                    # Copy data from mapped buffer
                    buffer_size = self.target_width * self.target_height * 3
                    buffer_data = np.frombuffer((ctypes.c_ubyte * buffer_size).from_address(buffer_ptr), dtype=np.uint8)
                    frame_data = buffer_data.reshape((self.target_height, self.target_width, 3))
                    self.output_frame[:] = frame_data  # No flipud needed - handled in shader
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
                else:
                    # Fallback to synchronous read
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
                    return self.cpu_preprocessor.process_frame(bayer_frame)
                
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
                self.pbo_index = 1 - self.pbo_index  # Swap PBOs
            else:
                # Fallback to synchronous readback
                glReadPixels(0, 0, self.target_width, self.target_height, GL_BGR, GL_UNSIGNED_BYTE, self.output_buffer)
                buffer_data = np.frombuffer(self.output_buffer, dtype=np.uint8)
                frame_data = buffer_data.reshape((self.target_height, self.target_width, 3))
                self.output_frame[:] = frame_data  # No flipud needed - handled in shader
            
            # The GPU version processes everything in two passes and outputs at target size
            # To match CPU interface, we return (target_size_frame, target_size_frame)
            # Note: GPU doesn't generate separate undistorted full-size frame like CPU
            return self.output_frame, self.output_frame
            
        except Exception as e:
            print(f"GPU multi-pass processing failed, falling back to CPU: {e}")
            import traceback
            traceback.print_exc()
            self.use_cpu_fallback = True
            # Fallback to CPU processing using the dedicated process_frame method
            return self.cpu_preprocessor.process_frame(bayer_frame)

    def close(self):
        """Releases OpenGL resources and window."""
        if self.window:
            self.window.close()
