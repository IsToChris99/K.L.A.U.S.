#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D rgbTexture;
uniform sampler2D map1Texture;
uniform sampler2D map2Texture;
uniform vec2 rgbSize;
uniform vec2 targetSize;

void main() {
    // Second pass: Undistort and resize RGB image using remap maps
    
    // Step 1: Calculate which pixel in the undistorted image we want
    vec2 targetPixelCoord = TexCoords * targetSize;
    vec2 undistortedRgbCoord = targetPixelCoord * (rgbSize / targetSize);
    
    // Step 2: Find corresponding distorted coordinate using remap map
    vec2 mapLookupCoord = undistortedRgbCoord / rgbSize;
    
    // Boundary check for map lookup
    if (mapLookupCoord.x < 0.0 || mapLookupCoord.x >= 1.0 ||
        mapLookupCoord.y < 0.0 || mapLookupCoord.y >= 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black for out of bounds
        return;
    }
    
    // Sample the remap coordinates to get the distorted position
    vec2 distortedCoord = texture(map1Texture, mapLookupCoord).rg;
    
    // Check if the remapped coordinate is valid
    if (distortedCoord.x < 0.0 || distortedCoord.x >= 1.0 ||
        distortedCoord.y < 0.0 || distortedCoord.y >= 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black for invalid remap
        return;
    }
    
    // Step 3: Sample from the demosaiced RGB texture at the distorted coordinate
    vec3 rgb = texture(rgbTexture, distortedCoord).rgb;
    
    FragColor = vec4(rgb, 1.0);
}
