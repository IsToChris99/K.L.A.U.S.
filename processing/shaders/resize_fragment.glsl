#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D undistortedTexture;
uniform vec2 undistortedSize;
uniform vec2 targetSize;

// Bicubic interpolation function (Catmull-Rom spline)
float cubicInterpolate(float p0, float p1, float p2, float p3, float t) {
    float a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    float b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    float c = -0.5 * p0 + 0.5 * p2;
    float d = p1;
    return a * t * t * t + b * t * t + c * t + d;
}

vec3 bicubicSample(sampler2D tex, vec2 coord, vec2 texSize) {
    vec2 texelSize = 1.0 / texSize;
    vec2 f = fract(coord * texSize);
    vec2 base = (floor(coord * texSize) - 1.0) / texSize;
    
    // Sample 4x4 neighborhood
    vec3 samples[16];
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            vec2 sampleCoord = base + vec2(float(x), float(y)) * texelSize;
            sampleCoord = clamp(sampleCoord, vec2(0.0), vec2(1.0));
            samples[y * 4 + x] = texture(tex, sampleCoord).rgb;
        }
    }
    
    // Bicubic interpolation in Y direction for each row
    vec3 rows[4];
    for (int y = 0; y < 4; y++) {
        rows[y] = vec3(
            cubicInterpolate(samples[y*4+0].r, samples[y*4+1].r, samples[y*4+2].r, samples[y*4+3].r, f.x),
            cubicInterpolate(samples[y*4+0].g, samples[y*4+1].g, samples[y*4+2].g, samples[y*4+3].g, f.x),
            cubicInterpolate(samples[y*4+0].b, samples[y*4+1].b, samples[y*4+2].b, samples[y*4+3].b, f.x)
        );
    }
    
    // Bicubic interpolation in X direction for final result
    return vec3(
        cubicInterpolate(rows[0].r, rows[1].r, rows[2].r, rows[3].r, f.y),
        cubicInterpolate(rows[0].g, rows[1].g, rows[2].g, rows[3].g, f.y),
        cubicInterpolate(rows[0].b, rows[1].b, rows[2].b, rows[3].b, f.y)
    );
}

void main() {
    // Third pass: Resize undistorted RGB image to target size
    // Using bicubic interpolation for maximum quality
    
    // Map target coordinates to undistorted coordinates
    vec2 targetPixelCoord = TexCoords * targetSize;
    vec2 undistortedCoord = targetPixelCoord * (undistortedSize / targetSize) / undistortedSize;
    
    // Use bicubic interpolation for superior quality
    vec3 rgb = bicubicSample(undistortedTexture, undistortedCoord, undistortedSize);
    
    // Clamp to avoid artifacts
    rgb = clamp(rgb, 0.0, 1.0);
    
    FragColor = vec4(rgb, 1.0);
}
