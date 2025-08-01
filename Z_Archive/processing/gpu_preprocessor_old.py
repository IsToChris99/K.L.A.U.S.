import cv2
import numpy as np
import json
import os
import pyglet
from pyglet.gl import *
import time
from config import CAMERA_CALIBRATION_FILE, FRAME_WIDTH, FRAME_HEIGHT

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

    def __init__(self, bayer_size = (1440, 1080), target_size=(FRAME_WIDTH, FRAME_HEIGHT), calibration_file=CAMERA_CALIBRATION_FILE):
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
        
        # Prepare buffer for reading pixels back from GPU
        self.output_buffer = (GLubyte * (self.target_width * self.target_height * 3))()
        self.output_frame = np.empty((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Initialization flag
        self.initialized = False
        self.use_cpu_fallback = False
        
        # CPU fallback preprocessing instance
        from ...processing.preprocessor import Preprocessor
        self.cpu_preprocessor = Preprocessor(calibration_file)
        
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
            uniform mat3 cameraMatrix;
            uniform vec4 distCoeffs; // k1, k2, p1, p2
            uniform vec2 bayerSize;

            // Performs simple RGGB debayering
            vec3 demosaic(vec2 coord) {
                vec2 texel = 1.0 / bayerSize;
                vec2 f = fract(coord * bayerSize);

                // Sample the 4 relevant Bayer pixels
                float c1 = texture(bayerTexture, coord - f*texel).r; // R or G
                float c2 = texture(bayerTexture, coord - f*texel + vec2(texel.x, 0)).r; // G or B
                float c3 = texture(bayerTexture, coord - f*texel + vec2(0, texel.y)).r; // G or B
                float c4 = texture(bayerTexture, coord - f*texel + texel).r; // B or G

                vec3 color;
                if (mod(floor(coord.y * bayerSize.y), 2.0) == 0.0) { // Even row RGRG
                    if (mod(floor(coord.x * bayerSize.x), 2.0) == 0.0) { // R position
                        color = vec3(c1, (c2+c3)*0.5, c4);
                    } else { // G position
                        color = vec3((c1+c4)*0.5, c2, (c3+c4)*0.5);
                    }
                } else { // Odd row GBGB
                    if (mod(floor(coord.x * bayerSize.x), 2.0) == 0.0) { // G position
                        color = vec3((c2+c3)*0.5, c1, (c2+c4)*0.5);
                    } else { // B position
                        color = vec3(c3, (c1+c4)*0.5, c2);
                    }
                }
                return color;
            }

            void main() {
                // TexCoords are the normalized coordinates of the target pixel [0,1]
                // We map this to the space of the *original Bayer image*
                vec2 pixel_coords = TexCoords * bayerSize;
                
                // --- Inverse Undistortion ---
                // Convert pixel coordinates to normalized camera coordinates (z=1)
                float x = (pixel_coords.x - cameraMatrix[0][2]) / cameraMatrix[0][0];
                float y = (pixel_coords.y - cameraMatrix[1][2]) / cameraMatrix[1][1];

                float r2 = x*x + y*y;
                float r4 = r2*r2;
                float distortion = 1.0 + distCoeffs.x * r2 + distCoeffs.y * r4;
                float x_dist = x * distortion + (2.0*distCoeffs.z*x*y + distCoeffs.w*(r2+2.0*x*x));
                float y_dist = y * distortion + (distCoeffs.z*(r2+2.0*y*y) + 2.0*distCoeffs.w*x*y);
                
                // Project back to distorted pixel coordinates
                vec2 distorted_pixel = vec2(
                    cameraMatrix[0][0] * x_dist + cameraMatrix[0][2],
                    cameraMatrix[1][1] * y_dist + cameraMatrix[1][2]
                );

                // Normalize to distorted texture coordinate
                vec2 distorted_tex_coord = distorted_pixel / bayerSize;

                if (distorted_tex_coord.x < 0.0 || distorted_tex_coord.x > 1.0 ||
                    distorted_tex_coord.y < 0.0 || distorted_tex_coord.y > 1.0) {
                    FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Outside -> Black
                } else {
                    // --- Debayering ---
                    // Perform debayering at the calculated coordinate
                    FragColor = vec4(demosaic(distorted_tex_coord), 1.0);
                }
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, self.bayer_width, self.bayer_height, 0, GL_RED, GL_UNSIGNED_BYTE, None)

        # --- Framebuffer Object (FBO) for off-screen rendering ---
        self.fbo = GLuint()
        glGenFramebuffers(1, self.fbo)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # --- Output texture to which the result is rendered ---
        self.output_texture_id = GLuint()
        glGenTextures(1, self.output_texture_id)
        glBindTexture(GL_TEXTURE_2D, self.output_texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, self.target_width, self.target_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.output_texture_id, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer is not complete!")
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0) # Back to default framebuffer

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
            # Convert Bayer to RGB using OpenCV
            rgb_frame = cv2.cvtColor(bayer_frame, cv2.COLOR_BayerRG2RGB)
            # Resize to target size
            rgb_frame = cv2.resize(rgb_frame, (self.target_width, self.target_height))
            # Apply undistortion using CPU preprocessor
            return self.cpu_preprocessor.undistort_frame(rgb_frame)
        
        # Check if initialization is needed
        if not self.initialized:
            return self.cpu_preprocessor.undistort_frame(
                cv2.resize(cv2.cvtColor(bayer_frame, cv2.COLOR_BayerRG2RGB), 
                          (self.target_width, self.target_height))
            )
            
        try:
            # Ensure we're in the correct OpenGL context
            self.window.switch_to()
            
            # --- GPU Operations ---
            # 1. Upload Bayer frame to GPU texture
            glBindTexture(GL_TEXTURE_2D, self.bayer_texture_id)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.bayer_width, self.bayer_height, GL_RED, GL_UNSIGNED_BYTE, bayer_frame.ctypes.data)

            # 2. Switch to off-screen framebuffer and activate shader
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            glViewport(0, 0, self.target_width, self.target_height)
            self.shader_program.use()

            # 3. Send uniforms to shader
            # Convert camera matrix to list format for mat3 uniform
            camera_matrix_list = [
                [float(self.camera_matrix[0,0]), float(self.camera_matrix[0,1]), float(self.camera_matrix[0,2])],
                [float(self.camera_matrix[1,0]), float(self.camera_matrix[1,1]), float(self.camera_matrix[1,2])],
                [float(self.camera_matrix[2,0]), float(self.camera_matrix[2,1]), float(self.camera_matrix[2,2])]
            ]
            self.shader_program['cameraMatrix'] = camera_matrix_list
            
            # Send only first 4 distortion coefficients as vec4
            dist_coeffs_list = [float(self.dist_coeffs[0]), float(self.dist_coeffs[1]), 
                               float(self.dist_coeffs[2]), float(self.dist_coeffs[3])]
            #self.shader_program['distCoeffs'] = dist_coeffs_list
            
            #self.shader_program['bayerSize'] = (float(self.bayer_width), float(self.bayer_height))

            # 4. Draw the rectangle -> executes the shader for each pixel
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            # 5. Read pixels back from FBO to CPU (the slowest step)
            glReadPixels(0, 0, self.target_width, self.target_height, GL_BGR, GL_UNSIGNED_BYTE, self.output_buffer)
            
            # --- Cleanup ---
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            self.shader_program.stop()

            # Convert data to NumPy array
            np.frombuffer(self.output_buffer, dtype=np.uint8, count=-1, out=self.output_frame.data)
            return self.output_frame
            
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            self.use_cpu_fallback = True
            # Fallback to CPU processing
            rgb_frame = cv2.cvtColor(bayer_frame, cv2.COLOR_BayerRG2RGB)
            rgb_frame = cv2.resize(rgb_frame, (self.target_width, self.target_height))
            return self.cpu_preprocessor.undistort_frame(rgb_frame)

    def close(self):
        """Releases OpenGL resources and window."""
        if self.window:
            self.window.close()

# --- Example Application ---
if __name__ == '__main__':
    # --- 1. Configuration ---
    BAYER_WIDTH, BAYER_HEIGHT = 1920, 1080
    FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
    CALIBRATION_FILE = "calibration_data.json"

    # --- 2. Create a dummy calibration file for testing ---
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Creating dummy calibration file: {CALIBRATION_FILE}")
        fx, fy = 1600, 1600
        cx, cy = BAYER_WIDTH / 2, BAYER_HEIGHT / 2
        cam_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        dist_coeffs = [-0.1, 0.05, 0.0, 0.0, 0.0]
        calib_data = {"cameraMatrix": cam_matrix, "distCoeffs": dist_coeffs}
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(calib_data, f, indent=4)
    
    # --- 3. Create a dummy Bayer frame ---
    bayer_frame_sim = np.zeros((BAYER_HEIGHT, BAYER_WIDTH), dtype=np.uint8)
    bayer_frame_sim[0::2, 0::2] = 200 # R
    bayer_frame_sim[0::2, 1::2] = 128 # G
    bayer_frame_sim[1::2, 0::2] = 128 # G
    bayer_frame_sim[1::2, 1::2] = 180 # B
    # Add a pattern to see the undistortion
    cv2.circle(bayer_frame_sim, (int(cx), int(cy)), 500, 255, 50)


    # --- 4. Initialize the GPU processor ---
    try:
        gpu_processor = GPUPreprocessor(
            calibration_file=CALIBRATION_FILE,
            target_size=(FRAME_WIDTH, FRAME_HEIGHT)
        )
    except Exception as e:
        print(e)
        exit()

    # --- 5. Processing loop ---
    print("\nStarting processing loop. Press 'q' in the window to exit.")
    frame_count = 0
    start_time = time.time()

    while True:
        # Here you would get the frame from your camera:
        # bayer_frame = get_raw_frame_from_camera()
        
        processed_frame = gpu_processor.process_frame(bayer_frame_sim)
        
        frame_count += 1
        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            print(f"GPU Pipeline FPS: {fps:.2f}")
            frame_count = 0
            start_time = current_time

        cv2.imshow("GPU Processed Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 6. Cleanup ---
    gpu_processor.close()
    cv2.destroyAllWindows()
    print("Program terminated.")