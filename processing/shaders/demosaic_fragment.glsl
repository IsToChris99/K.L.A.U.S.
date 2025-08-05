#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D bayerTexture;
uniform vec2 bayerSize;

vec3 demosaicBayer(vec2 texCoord) {
    vec2 pixelCoord = texCoord * bayerSize;
    ivec2 pixel = ivec2(floor(pixelCoord));
    vec2 texelSize = 1.0 / bayerSize;
    
    bool isEvenRow = (pixel.y % 2) == 0;
    bool isEvenCol = (pixel.x % 2) == 0;
    
    float center = texture(bayerTexture, texCoord).r;
    
    // RGGB Bayer pattern
    // Even row, even col = Red
    // Even row, odd col = Green (Gr)
    // Odd row, even col = Green (Gb) 
    // Odd row, odd col = Blue
    
    float red, green, blue;
    
    if (isEvenRow && isEvenCol) {
        // Red pixel
        red = center;
        
        // Interpolate green from 4 neighbors
        float g_n = texture(bayerTexture, texCoord + vec2(0.0, -texelSize.y)).r;
        float g_s = texture(bayerTexture, texCoord + vec2(0.0, texelSize.y)).r;
        float g_e = texture(bayerTexture, texCoord + vec2(texelSize.x, 0.0)).r;
        float g_w = texture(bayerTexture, texCoord + vec2(-texelSize.x, 0.0)).r;
        green = (g_n + g_s + g_e + g_w) * 0.25;
        
        // Interpolate blue from 4 diagonal neighbors
        float b_ne = texture(bayerTexture, texCoord + vec2(texelSize.x, -texelSize.y)).r;
        float b_nw = texture(bayerTexture, texCoord + vec2(-texelSize.x, -texelSize.y)).r;
        float b_se = texture(bayerTexture, texCoord + vec2(texelSize.x, texelSize.y)).r;
        float b_sw = texture(bayerTexture, texCoord + vec2(-texelSize.x, texelSize.y)).r;
        blue = (b_ne + b_nw + b_se + b_sw) * 0.25;
        
    } else if (isEvenRow && !isEvenCol) {
        // Green pixel in red row (Gr)
        green = center;
        
        // Interpolate red from horizontal neighbors
        float r_e = texture(bayerTexture, texCoord + vec2(texelSize.x, 0.0)).r;
        float r_w = texture(bayerTexture, texCoord + vec2(-texelSize.x, 0.0)).r;
        red = (r_e + r_w) * 0.5;
        
        // Interpolate blue from vertical neighbors
        float b_n = texture(bayerTexture, texCoord + vec2(0.0, -texelSize.y)).r;
        float b_s = texture(bayerTexture, texCoord + vec2(0.0, texelSize.y)).r;
        blue = (b_n + b_s) * 0.5;
        
    } else if (!isEvenRow && isEvenCol) {
        // Green pixel in blue row (Gb)
        green = center;
        
        // Interpolate red from vertical neighbors
        float r_n = texture(bayerTexture, texCoord + vec2(0.0, -texelSize.y)).r;
        float r_s = texture(bayerTexture, texCoord + vec2(0.0, texelSize.y)).r;
        red = (r_n + r_s) * 0.5;
        
        // Interpolate blue from horizontal neighbors
        float b_e = texture(bayerTexture, texCoord + vec2(texelSize.x, 0.0)).r;
        float b_w = texture(bayerTexture, texCoord + vec2(-texelSize.x, 0.0)).r;
        blue = (b_e + b_w) * 0.5;
        
    } else {
        // Blue pixel
        blue = center;
        
        // Interpolate green from 4 neighbors
        float g_n = texture(bayerTexture, texCoord + vec2(0.0, -texelSize.y)).r;
        float g_s = texture(bayerTexture, texCoord + vec2(0.0, texelSize.y)).r;
        float g_e = texture(bayerTexture, texCoord + vec2(texelSize.x, 0.0)).r;
        float g_w = texture(bayerTexture, texCoord + vec2(-texelSize.x, 0.0)).r;
        green = (g_n + g_s + g_e + g_w) * 0.25;
        
        // Interpolate red from 4 diagonal neighbors
        float r_ne = texture(bayerTexture, texCoord + vec2(texelSize.x, -texelSize.y)).r;
        float r_nw = texture(bayerTexture, texCoord + vec2(-texelSize.x, -texelSize.y)).r;
        float r_se = texture(bayerTexture, texCoord + vec2(texelSize.x, texelSize.y)).r;
        float r_sw = texture(bayerTexture, texCoord + vec2(-texelSize.x, texelSize.y)).r;
        red = (r_ne + r_nw + r_se + r_sw) * 0.25;
    }
    
    return clamp(vec3(red, green, blue), 0.0, 1.0);
}

void main() {
    // First pass: Demosaic Bayer pattern to RGB
    vec3 rgb = demosaicBayer(TexCoords);
    FragColor = vec4(rgb, 1.0);
}
