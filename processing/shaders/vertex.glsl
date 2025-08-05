#version 330 core
            layout(location = 0) in vec2 position;
            out vec2 TexCoords;
            void main() {
                // Scale positions from [0,1] to [-1,1] for clip space
                gl_Position = vec4(position.x * 2.0 - 1.0, position.y * 2.0 - 1.0, 0.0, 1.0);
                TexCoords = position;
            }