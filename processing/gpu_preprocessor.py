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
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.bayer_texture_id = None
        self.fbo = None
        self.output_texture_id = None
        
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
        self.cpu_preprocessor = CPUPreprocessor(calibration_file)
        
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

    def _initialize_shaders(self):
        """Compiles the vertex and fragment shaders."""
        vertex_shader_source = """
            #version 330 core
            layout(location = 0) in vec2 position;
            out vec2 TexCoords;
            void main() {
                // Scale positions from [0,1] to [-1,1] for clip space
                gl_Position = vec4(position.x * 2.0 - 1.0, position.y * 2.0 - 1.0, 0.0, 1.0);
                TexCoords = position;
            }
        """

        fragment_shader_source = """
            #version 330 core
            out vec4 FragColor;
            in vec2 TexCoords;

            uniform sampler2D bayerTexture;
            uniform vec2 bayerSize;
            uniform vec2 targetSize;
            uniform mat3 cameraMatrix;
            uniform vec4 distCoeffs; // k1, k2, p1, p2

            vec3 demosaicPixel(vec2 bayerTexCoord) {
                // Boundary check
                if (bayerTexCoord.x < 0.0 || bayerTexCoord.x > 1.0 ||
                    bayerTexCoord.y < 0.0 || bayerTexCoord.y > 1.0) {
                    return vec3(0.0, 0.0, 0.0); // Return black for out-of-bounds
                }
                
                // Get Bayer texture dimensions for pixel coordinate calculations
                vec2 pixelCoord = bayerTexCoord * bayerSize;
                
                // Determine position in Bayer pattern (RG8 format)
                int x = int(floor(pixelCoord.x));
                int y = int(floor(pixelCoord.y));
                
                // Calculate texel size for neighboring pixel sampling
                vec2 texelSize = 1.0 / bayerSize;
                
                // Sample current and neighboring pixels
                float c = texture(bayerTexture, bayerTexCoord).r;
                
                // Determine current pixel type based on position
                bool isEvenRow = (y % 2) == 0;
                bool isEvenCol = (x % 2) == 0;
                
                vec3 rgb;
                
                if (isEvenRow && isEvenCol) {
                    // Red pixel position
                    rgb.r = c;
                    
                    // Green interpolation (horizontal and vertical neighbors)
                    float g1 = texture(bayerTexture, bayerTexCoord + vec2(texelSize.x, 0.0)).r;
                    float g2 = texture(bayerTexture, bayerTexCoord + vec2(-texelSize.x, 0.0)).r;
                    float g3 = texture(bayerTexture, bayerTexCoord + vec2(0.0, texelSize.y)).r;
                    float g4 = texture(bayerTexture, bayerTexCoord + vec2(0.0, -texelSize.y)).r;
                    rgb.g = (g1 + g2 + g3 + g4) * 0.25;
                    
                    // Blue interpolation (diagonal neighbors)
                    float b1 = texture(bayerTexture, bayerTexCoord + vec2(texelSize.x, texelSize.y)).r;
                    float b2 = texture(bayerTexture, bayerTexCoord + vec2(-texelSize.x, texelSize.y)).r;
                    float b3 = texture(bayerTexture, bayerTexCoord + vec2(texelSize.x, -texelSize.y)).r;
                    float b4 = texture(bayerTexture, bayerTexCoord + vec2(-texelSize.x, -texelSize.y)).r;
                    rgb.b = (b1 + b2 + b3 + b4) * 0.25;
                    
                } else if (isEvenRow && !isEvenCol) {
                    // Green pixel position (in red row)
                    rgb.g = c;
                    
                    // Red interpolation (horizontal neighbors)
                    float r1 = texture(bayerTexture, bayerTexCoord + vec2(texelSize.x, 0.0)).r;
                    float r2 = texture(bayerTexture, bayerTexCoord + vec2(-texelSize.x, 0.0)).r;
                    rgb.r = (r1 + r2) * 0.5;
                    
                    // Blue interpolation (vertical neighbors)
                    float b1 = texture(bayerTexture, bayerTexCoord + vec2(0.0, texelSize.y)).r;
                    float b2 = texture(bayerTexture, bayerTexCoord + vec2(0.0, -texelSize.y)).r;
                    rgb.b = (b1 + b2) * 0.5;
                    
                } else if (!isEvenRow && isEvenCol) {
                    // Green pixel position (in blue row)
                    rgb.g = c;
                    
                    // Blue interpolation (horizontal neighbors)
                    float b1 = texture(bayerTexture, bayerTexCoord + vec2(texelSize.x, 0.0)).r;
                    float b2 = texture(bayerTexture, bayerTexCoord + vec2(-texelSize.x, 0.0)).r;
                    rgb.b = (b1 + b2) * 0.5;
                    
                    // Red interpolation (vertical neighbors)
                    float r1 = texture(bayerTexture, bayerTexCoord + vec2(0.0, texelSize.y)).r;
                    float r2 = texture(bayerTexture, bayerTexCoord + vec2(0.0, -texelSize.y)).r;
                    rgb.r = (r1 + r2) * 0.5;
                    
                } else {
                    // Blue pixel position
                    rgb.b = c;
                    
                    // Green interpolation (horizontal and vertical neighbors)
                    float g1 = texture(bayerTexture, bayerTexCoord + vec2(texelSize.x, 0.0)).r;
                    float g2 = texture(bayerTexture, bayerTexCoord + vec2(-texelSize.x, 0.0)).r;
                    float g3 = texture(bayerTexture, bayerTexCoord + vec2(0.0, texelSize.y)).r;
                    float g4 = texture(bayerTexture, bayerTexCoord + vec2(0.0, -texelSize.y)).r;
                    rgb.g = (g1 + g2 + g3 + g4) * 0.25;
                    
                    // Red interpolation (diagonal neighbors)
                    float r1 = texture(bayerTexture, bayerTexCoord + vec2(texelSize.x, texelSize.y)).r;
                    float r2 = texture(bayerTexture, bayerTexCoord + vec2(-texelSize.x, texelSize.y)).r;
                    float r3 = texture(bayerTexture, bayerTexCoord + vec2(texelSize.x, -texelSize.y)).r;
                    float r4 = texture(bayerTexture, bayerTexCoord + vec2(-texelSize.x, -texelSize.y)).r;
                    rgb.r = (r1 + r2 + r3 + r4) * 0.25;
                }
                
                return rgb;
            }

            vec2 undistortCoordinate(vec2 targetPixelCoord) {
                // Convert target pixel coordinates to normalized camera coordinates
                float x = (targetPixelCoord.x - cameraMatrix[0][2]) / cameraMatrix[0][0];
                float y = (targetPixelCoord.y - cameraMatrix[1][2]) / cameraMatrix[1][1];

                // Apply inverse distortion (find distorted coordinates from undistorted)
                float r2 = x*x + y*y;
                float r4 = r2*r2;
                
                // Radial distortion
                float distortion = 1.0 + distCoeffs.x * r2 + distCoeffs.y * r4;
                
                // Tangential distortion
                float dx = 2.0*distCoeffs.z*x*y + distCoeffs.w*(r2+2.0*x*x);
                float dy = distCoeffs.z*(r2+2.0*y*y) + 2.0*distCoeffs.w*x*y;
                
                // Apply distortion
                float x_dist = x * distortion + dx;
                float y_dist = y * distortion + dy;
                
                // Project back to distorted pixel coordinates in bayer image
                vec2 distorted_pixel = vec2(
                    cameraMatrix[0][0] * x_dist + cameraMatrix[0][2],
                    cameraMatrix[1][1] * y_dist + cameraMatrix[1][2]
                );

                // Convert to texture coordinates (normalized [0,1])
                return distorted_pixel / bayerSize;
            }

            vec3 demosaic_undistort_and_resize(vec2 targetTexCoord) {
                // Convert target texture coordinates to pixel coordinates
                vec2 targetPixelCoord = targetTexCoord * targetSize;
                
                // Scale to bayer image size (this is the undistorted coordinate we want)
                vec2 scaledPixelCoord = targetPixelCoord * (bayerSize / targetSize);
                
                // Find corresponding distorted coordinate in source bayer image
                vec2 distortedTexCoord = undistortCoordinate(scaledPixelCoord);
                
                // Check bounds
                if (distortedTexCoord.x < 0.0 || distortedTexCoord.x > 1.0 ||
                    distortedTexCoord.y < 0.0 || distortedTexCoord.y > 1.0) {
                    return vec3(0.0, 0.0, 0.0); // Black for out of bounds
                }
                
                // Demosaic at the distorted coordinate
                return demosaicPixel(distortedTexCoord);
            }

            void main() {
                // Perform combined demosaicing, undistortion, and resizing
                vec3 rgb = demosaic_undistort_and_resize(TexCoords);
                FragColor = vec4(rgb, 1.0);
            }
        """
        vs = pyglet.graphics.shader.Shader(vertex_shader_source, 'vertex')
        fs = pyglet.graphics.shader.Shader(fragment_shader_source, 'fragment')
        self.shader_program = pyglet.graphics.shader.ShaderProgram(vs, fs)

    def _initialize_gl_objects(self):
        """Creates all necessary OpenGL objects (textures, FBO, VAO)."""
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

        # --- Input texture for the Bayer frame ---
        self.bayer_texture_id = GLuint()
        glGenTextures(1, self.bayer_texture_id)
        glBindTexture(GL_TEXTURE_2D, self.bayer_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) # For cv2.INTER_NEAREST resize
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, self.bayer_width, self.bayer_height, 0, GL_RED, GL_UNSIGNED_BYTE, None)

        # --- Framebuffer Object (FBO) for off-screen rendering ---
        self.fbo = GLuint()
        glGenFramebuffers(1, self.fbo)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # --- Output texture to which the result is rendered ---
        self.output_texture_id = GLuint()
        glGenTextures(1, self.output_texture_id)
        glBindTexture(GL_TEXTURE_2D, self.output_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, self.target_width, self.target_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.output_texture_id, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer is not complete!")
        
        # Set up persistent state for better performance
        glViewport(0, 0, self.target_width, self.target_height)
        
        # Disable unnecessary OpenGL features for better performance
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_STENCIL_TEST)
        glDisable(GL_DITHER)
        
        # Initialize Pixel Buffer Objects for async readback
        self._initialize_pbos()
        
        # Keep framebuffer bound for process_frame calls
        # glBindFramebuffer(GL_FRAMEBUFFER, 0) # Don't reset to default

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
        Processes a single raw Bayer frame on the GPU.
        
        :param bayer_frame: NumPy Array (height, width) of the raw Bayer frame.
        :return: Processed NumPy Array (target_height, target_width, 3) in BGR format.
        """
        # Check if we should use CPU fallback
        if self.use_cpu_fallback:
            return self.cpu_preprocessor.process_frame(bayer_frame, target_size=(self.target_width, self.target_height))

        # Check if initialization is needed
        if not self.initialized:
            return self.cpu_preprocessor.process_frame(bayer_frame, target_size=(self.target_width, self.target_height))
            
        try:
            # --- GPU Operations (minimized OpenGL calls) ---
            # 1. Upload Bayer frame to GPU texture
            glBindTexture(GL_TEXTURE_2D, self.bayer_texture_id)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.bayer_width, self.bayer_height, GL_RED, GL_UNSIGNED_BYTE, bayer_frame.ctypes.data)

            # 2. Render to framebuffer (state should be persistent)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            self.shader_program.use()
            
            # Set uniform variables for sizes and calibration
            try:
                # Get the actual OpenGL program ID from pyglet shader program
                program_id = self.shader_program.id if hasattr(self.shader_program, 'id') else self.shader_program._program_id
                
                bayer_size_location = glGetUniformLocation(program_id, b"bayerSize")
                target_size_location = glGetUniformLocation(program_id, b"targetSize")
                camera_matrix_location = glGetUniformLocation(program_id, b"cameraMatrix")
                dist_coeffs_location = glGetUniformLocation(program_id, b"distCoeffs")
                
                if bayer_size_location != -1:
                    glUniform2f(bayer_size_location, float(self.bayer_width), float(self.bayer_height))
                if target_size_location != -1:
                    glUniform2f(target_size_location, float(self.target_width), float(self.target_height))
                
                # Send camera matrix as mat3
                if camera_matrix_location != -1:
                    camera_matrix_flat = self.camera_matrix.flatten().astype(np.float32)
                    # Create proper ctypes pointer for the matrix data
                    matrix_ptr = (GLfloat * len(camera_matrix_flat))(*camera_matrix_flat)
                    glUniformMatrix3fv(camera_matrix_location, 1, GL_FALSE, matrix_ptr)
                
                # Send distortion coefficients as vec4 (k1, k2, p1, p2)
                if dist_coeffs_location != -1:
                    dist_coeffs_vec4 = np.zeros(4, dtype=np.float32)
                    dist_coeffs_vec4[:min(4, len(self.dist_coeffs))] = self.dist_coeffs[:4]
                    # Create proper ctypes pointer for the distortion coefficients
                    dist_ptr = (GLfloat * 4)(*dist_coeffs_vec4)
                    glUniform4fv(dist_coeffs_location, 1, dist_ptr)
                    
            except Exception as e:
                print(f"Warning: Could not set uniform variables: {e}")
            
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            # 3. Asynchronous readback using PBOs
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
                    return self.cpu_preprocessor.process_frame(bayer_frame, target_size=(self.target_width, self.target_height))
                
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
                self.pbo_index = 1 - self.pbo_index  # Swap PBOs
            else:
                # Fallback to synchronous readback
                glReadPixels(0, 0, self.target_width, self.target_height, GL_BGR, GL_UNSIGNED_BYTE, self.output_buffer)
                buffer_data = np.frombuffer(self.output_buffer, dtype=np.uint8)
                frame_data = buffer_data.reshape((self.target_height, self.target_width, 3))
                self.output_frame[:] = frame_data  # No flipud needed - handled in shader
            
            return self.output_frame
            
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            self.use_cpu_fallback = True
            # Fallback to CPU processing using the dedicated process_frame method
            return self.cpu_preprocessor.process_frame(bayer_frame, target_size=(self.target_width, self.target_height))

    def close(self):
        """Releases OpenGL resources and window."""
        if self.window:
            self.window.close()
