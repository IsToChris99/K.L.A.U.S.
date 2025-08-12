#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D undistortedTexture;
uniform mat3 perspectiveMatrix;
uniform vec2 undistortedSize;

void main() {
    // Apply perspective transformation
    // Convert normalized texture coordinates [0,1] to pixel coordinates
    vec2 pixelCoords = TexCoords * undistortedSize;
    
    // Apply perspective transformation matrix
    vec3 transformedCoords = perspectiveMatrix * vec3(pixelCoords, 1.0);
    
    // Perspective division
    vec2 perspectiveCoords = transformedCoords.xy / transformedCoords.z;
    
    // Convert back to normalized texture coordinates
    vec2 normalizedCoords = perspectiveCoords / undistortedSize;
    
    // Check if coordinates are within valid range [0,1]
    if (normalizedCoords.x < 0.0 || normalizedCoords.x > 1.0 || 
        normalizedCoords.y < 0.0 || normalizedCoords.y > 1.0) {
        // Outside bounds - use black color
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        // Sample the texture with bilinear interpolation
        FragColor = vec4(texture(undistortedTexture, normalizedCoords).rgb, 1.0);
    }
}
