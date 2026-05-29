#ifndef GPU_COMMON_CUH
#define GPU_COMMON_CUH

#include <cstdint>

// ============================================================
//  Shared GPU device utilities for all CUDA kernels
// ============================================================

// Device pixel I/O — supports 8-bit and 16-bit interleaved RGB

__device__ __forceinline__ float dev_read_pixel(const uint8_t* data, int x, int y,
                                                  int width, int bit_depth, int channel,
                                                  float maxv) {
    if (bit_depth <= 8) {
        return static_cast<float>(data[(static_cast<size_t>(y) * width + x) * 3 + channel]) / maxv;
    }
    const uint16_t* d16 = reinterpret_cast<const uint16_t*>(data);
    return static_cast<float>(d16[(static_cast<size_t>(y) * width + x) * 3 + channel]) / maxv;
}

__device__ __forceinline__ void dev_write_pixel(uint8_t* data, int x, int y, int width,
                                                  int bit_depth, int channel, float value, int max_val) {
    int iv = static_cast<int>(value * static_cast<float>(max_val) + 0.5f);
    iv = iv < 0 ? 0 : (iv > max_val ? max_val : iv);
    if (bit_depth <= 8) {
        data[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<uint8_t>(iv);
    } else {
        uint16_t* d16 = reinterpret_cast<uint16_t*>(data);
        d16[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<uint16_t>(iv);
    }
}

__device__ __forceinline__ float dev_luma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__device__ __forceinline__ float dev_clamp(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// CCM: apply 3x3 matrix to RGB
__device__ __forceinline__ void dev_ccm_3x3(float& r, float& g, float& b, const float m[9]) {
    float nr = m[0]*r + m[1]*g + m[2]*b;
    float ng = m[3]*r + m[4]*g + m[5]*b;
    float nb = m[6]*r + m[7]*g + m[8]*b;
    r = dev_clamp(nr); g = dev_clamp(ng); b = dev_clamp(nb);
}

// RGB to HSL saturation adjust
__device__ __forceinline__ void dev_saturate(float& r, float& g, float& b, float sat) {
    float l = dev_luma(r, g, b);
    r = l + sat * (r - l);
    g = l + sat * (g - l);
    b = l + sat * (b - l);
    r = dev_clamp(r); g = dev_clamp(g); b = dev_clamp(b);
}

// Color temp: apply RGB multipliers
__device__ __forceinline__ void dev_color_temp(float& r, float& g, float& b,
                                                float rm, float gm, float bm) {
    r = dev_clamp(r * rm); g = dev_clamp(g * gm); b = dev_clamp(b * bm);
}

// Gamma tone curve
__device__ __forceinline__ void dev_apply_gamma(float& r, float& g, float& b, float inv_gamma) {
    r = __powf(dev_clamp(r), inv_gamma);
    g = __powf(dev_clamp(g), inv_gamma);
    b = __powf(dev_clamp(b), inv_gamma);
}

// S-curve contrast
__device__ __forceinline__ float dev_s_curve(float x, float contrast) {
    if (contrast > 0.0f) {
        return 0.5f + (x - 0.5f) / (1.0f - contrast * (1.0f - 2.0f * fabsf(x - 0.5f)) + 1e-6f);
    } else if (contrast < 0.0f) {
        float f = 1.0f + contrast;
        return 0.5f + (x - 0.5f) * f;
    }
    return x;
}

// ACES filmic (GPU version)
__device__ __forceinline__ float dev_filmic(float x) {
    float xs = x;
    float num = xs * (2.51f * xs + 0.03f);
    float den = xs * (2.43f * xs + 0.59f) + 0.14f;
    return num / (den + 1e-10f);
}

// ============================================================
//  Bayer data device utilities (single-channel)
// ============================================================

// Bayer color index: 0=R, 1=Gr, 2=Gb, 3=B
// pattern: 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG
__device__ __forceinline__ int dev_bayer_color(int y, int x, int pattern) {
    int py = y & 1, px = x & 1;
    switch (pattern) {
        case 0: return (py == 0 && px == 0) ? 0 : (py == 1 && px == 1) ? 3 : ((py == 0) ? 1 : 2); // RGGB
        case 1: return (py == 1 && px == 1) ? 0 : (py == 0 && px == 0) ? 3 : ((py == 0) ? 2 : 1); // BGGR
        case 2: return (py == 0 && px == 1) ? 0 : (py == 1 && px == 0) ? 3 : ((py == 0) ? 1 : 2); // GRBG
        case 3: return (py == 1 && px == 0) ? 0 : (py == 0 && px == 1) ? 3 : ((py == 0) ? 2 : 1); // GBRG
        default: return 0;
    }
}

__device__ __forceinline__ float dev_read_bayer(const uint8_t* data, int x, int y,
                                                  int width, int bit_depth, float maxv) {
    if (bit_depth <= 8)
        return static_cast<float>(data[static_cast<size_t>(y) * width + x]) / maxv;
    const uint16_t* d16 = reinterpret_cast<const uint16_t*>(data);
    return static_cast<float>(d16[static_cast<size_t>(y) * width + x]) / maxv;
}

__device__ __forceinline__ void dev_write_bayer(uint8_t* data, int x, int y, int width,
                                                  int bit_depth, float value, int max_val) {
    int iv = static_cast<int>(value * static_cast<float>(max_val) + 0.5f);
    iv = iv < 0 ? 0 : (iv > max_val ? max_val : iv);
    if (bit_depth <= 8)
        data[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(iv);
    else {
        uint16_t* d16 = reinterpret_cast<uint16_t*>(data);
        d16[static_cast<size_t>(y) * width + x] = static_cast<uint16_t>(iv);
    }
}

#endif // GPU_COMMON_CUH
