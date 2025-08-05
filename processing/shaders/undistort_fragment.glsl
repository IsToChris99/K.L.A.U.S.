#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D rgbTexture;
uniform sampler2D map1Texture;
uniform sampler2D map2Texture;
uniform vec2 rgbSize;

void main() {
    // Second pass: Undistort RGB image (same size as input)
    // This exactly matches cv2.remap with cv2.INTER_LINEAR
    
    // TexCoords are already in [0,1] range for the full RGB image
    vec2 mapLookupCoord = TexCoords;
    
    // Boundary check for map lookup
    if (mapLookupCoord.x < 0.0 || mapLookupCoord.x >= 1.0 ||
        mapLookupCoord.y < 0.0 || mapLookupCoord.y >= 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black for out of bounds
        return;
    }
    
    // Sample the remap coordinates with bilinear interpolation
    // This exactly matches cv2.INTER_LINEAR behavior
    vec2 mapTexelSize = 1.0 / textureSize(map1Texture, 0);
    vec2 mapPixelCoord = mapLookupCoord * textureSize(map1Texture, 0);
    vec2 mapF = fract(mapPixelCoord);
    vec2 mapBase = (floor(mapPixelCoord)) / textureSize(map1Texture, 0);
    
    // Sample 2x2 neighborhood of remap coordinates
    vec2 mapTL = texture(map1Texture, mapBase).rg;
    vec2 mapTR = texture(map1Texture, mapBase + vec2(mapTexelSize.x, 0.0)).rg;
    vec2 mapBL = texture(map1Texture, mapBase + vec2(0.0, mapTexelSize.y)).rg;
    vec2 mapBR = texture(map1Texture, mapBase + vec2(mapTexelSize.x, mapTexelSize.y)).rg;
    
    // Bilinear interpolation of remap coordinates
    vec2 mapTop = mix(mapTL, mapTR, mapF.x);
    vec2 mapBottom = mix(mapBL, mapBR, mapF.x);
    vec2 distortedCoord = mix(mapTop, mapBottom, mapF.y);
    
    // Check if the interpolated remapped coordinate is valid
    if (distortedCoord.x < 0.0 || distortedCoord.x >= 1.0 ||
        distortedCoord.y < 0.0 || distortedCoord.y >= 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black for invalid remap
        return;
    }
    
    // Sample from the RGB texture at the distorted coordinate
    // OpenGL's texture() function does bilinear interpolation automatically
    vec3 rgb = texture(rgbTexture, distortedCoord).rgb;
    
    FragColor = vec4(rgb, 1.0);
}
