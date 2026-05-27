#include "hdr_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// ============================================================
//  Device utility functions
// ============================================================

namespace {

__device__ __forceinline__ int dev_read_8bit(const uint8_t* data, int x, int y, int width, int c) {
    return static_cast<int>(data[(static_cast<size_t>(y) * width + x) * 3 + c]);
}

__device__ __forceinline__ int dev_read_16bit(const uint8_t* data, int x, int y, int width, int c) {
    const uint16_t* d16 = reinterpret_cast<const uint16_t*>(data);
    return static_cast<int>(d16[(static_cast<size_t>(y) * width + x) * 3 + c]);
}

__device__ __forceinline__ void dev_write_8bit(uint8_t* data, int x, int y, int width, int c, int val) {
    data[(static_cast<size_t>(y) * width + x) * 3 + c] = static_cast<uint8_t>(val);
}

__device__ __forceinline__ void dev_write_16bit(uint8_t* data, int x, int y, int width, int c, int val) {
    uint16_t* d16 = reinterpret_cast<uint16_t*>(data);
    d16[(static_cast<size_t>(y) * width + x) * 3 + c] = static_cast<uint16_t>(val);
}

__device__ __forceinline__ float dev_luma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

} // anonymous namespace

// ============================================================
//  Global tone mapping kernels (shared structure)
// ============================================================

#define HDR_TM_GRID(width, height) \
    dim3(((width) + 31) / 32, ((height) + 15) / 16)

// ============================================================
//  1. Reinhard: L_out = L / (1 + L)
// ============================================================

__global__ void kern_reinhard(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                               int width, int height, int bit_depth, int max_val,
                               float sat, float inv_gamma, float exposure_mul) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);
    float r, g, b;

    if (bit_depth <= 8) {
        r = dev_read_8bit(input, x, y, width, 0) / mv;
        g = dev_read_8bit(input, x, y, width, 1) / mv;
        b = dev_read_8bit(input, x, y, width, 2) / mv;
    } else {
        r = dev_read_16bit(input, x, y, width, 0) / mv;
        g = dev_read_16bit(input, x, y, width, 1) / mv;
        b = dev_read_16bit(input, x, y, width, 2) / mv;
    }

    r *= exposure_mul; g *= exposure_mul; b *= exposure_mul;
    float L = dev_luma(r, g, b);

    if (L <= 0.0f) {
        if (bit_depth <= 8) {
            dev_write_8bit(output, x, y, width, 0, 0);
            dev_write_8bit(output, x, y, width, 1, 0);
            dev_write_8bit(output, x, y, width, 2, 0);
        } else {
            dev_write_16bit(output, x, y, width, 0, 0);
            dev_write_16bit(output, x, y, width, 1, 0);
            dev_write_16bit(output, x, y, width, 2, 0);
        }
        return;
    }

    float Lout = L / (1.0f + L);
    float scale = __powf(Lout / L, sat);

    float ro = __powf(fmaxf(r * scale, 0.0f), inv_gamma);
    float go = __powf(fmaxf(g * scale, 0.0f), inv_gamma);
    float bo = __powf(fmaxf(b * scale, 0.0f), inv_gamma);

    int iro = static_cast<int>(ro * mv + 0.5f);
    int igo = static_cast<int>(go * mv + 0.5f);
    int ibo = static_cast<int>(bo * mv + 0.5f);
    iro = max(0, min(max_val, iro));
    igo = max(0, min(max_val, igo));
    ibo = max(0, min(max_val, ibo));

    if (bit_depth <= 8) {
        dev_write_8bit(output, x, y, width, 0, iro);
        dev_write_8bit(output, x, y, width, 1, igo);
        dev_write_8bit(output, x, y, width, 2, ibo);
    } else {
        dev_write_16bit(output, x, y, width, 0, iro);
        dev_write_16bit(output, x, y, width, 1, igo);
        dev_write_16bit(output, x, y, width, 2, ibo);
    }
}

// ============================================================
//  2. Reinhard Extended: L_out = Lm*(1+Lm/wp²)/(1+Lm)
// ============================================================

__global__ void kern_reinhard_ext(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                   int width, int height, int bit_depth, int max_val,
                                   float key, float wp2, float sat, float inv_gamma,
                                   float scale_factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);
    float r, g, b;

    if (bit_depth <= 8) {
        r = dev_read_8bit(input, x, y, width, 0) / mv;
        g = dev_read_8bit(input, x, y, width, 1) / mv;
        b = dev_read_8bit(input, x, y, width, 2) / mv;
    } else {
        r = dev_read_16bit(input, x, y, width, 0) / mv;
        g = dev_read_16bit(input, x, y, width, 1) / mv;
        b = dev_read_16bit(input, x, y, width, 2) / mv;
    }

    float L = dev_luma(r, g, b);
    if (L <= 0.0f) {
        if (bit_depth <= 8) {
            dev_write_8bit(output, x, y, width, 0, 0);
            dev_write_8bit(output, x, y, width, 1, 0);
            dev_write_8bit(output, x, y, width, 2, 0);
        } else {
            dev_write_16bit(output, x, y, width, 0, 0);
            dev_write_16bit(output, x, y, width, 1, 0);
            dev_write_16bit(output, x, y, width, 2, 0);
        }
        return;
    }

    float Lm = L * scale_factor;
    float Lout = Lm * (1.0f + Lm / wp2) / (1.0f + Lm);
    float scale = __powf(fmaxf(Lout / L, 0.0f), sat);

    float ro = __powf(fmaxf(r * scale, 0.0f), inv_gamma);
    float go = __powf(fmaxf(g * scale, 0.0f), inv_gamma);
    float bo = __powf(fmaxf(b * scale, 0.0f), inv_gamma);

    int iro = max(0, min(max_val, static_cast<int>(ro * mv + 0.5f)));
    int igo = max(0, min(max_val, static_cast<int>(go * mv + 0.5f)));
    int ibo = max(0, min(max_val, static_cast<int>(bo * mv + 0.5f)));

    if (bit_depth <= 8) {
        dev_write_8bit(output, x, y, width, 0, iro);
        dev_write_8bit(output, x, y, width, 1, igo);
        dev_write_8bit(output, x, y, width, 2, ibo);
    } else {
        dev_write_16bit(output, x, y, width, 0, iro);
        dev_write_16bit(output, x, y, width, 1, igo);
        dev_write_16bit(output, x, y, width, 2, ibo);
    }
}

// ============================================================
//  3. Filmic ACES: f(x) = x*(a*x+b)/(c*x²+d*x+e)
// ============================================================

__global__ void kern_filmic_aces(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                  int width, int height, int bit_depth, int max_val,
                                  float strength, float sat, float inv_gamma, float exposure_mul) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    float mv = static_cast<float>(max_val);
    float r, g, b2;

    if (bit_depth <= 8) {
        r = dev_read_8bit(input, x, y, width, 0) / mv;
        g = dev_read_8bit(input, x, y, width, 1) / mv;
        b2 = dev_read_8bit(input, x, y, width, 2) / mv;
    } else {
        r = dev_read_16bit(input, x, y, width, 0) / mv;
        g = dev_read_16bit(input, x, y, width, 1) / mv;
        b2 = dev_read_16bit(input, x, y, width, 2) / mv;
    }

    r *= exposure_mul; g *= exposure_mul; b2 *= exposure_mul;
    float L = dev_luma(r, g, b2);
    if (L <= 0.0f) {
        if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, 0); dev_write_8bit(output, x, y, width, 1, 0); dev_write_8bit(output, x, y, width, 2, 0); }
        else { dev_write_16bit(output, x, y, width, 0, 0); dev_write_16bit(output, x, y, width, 1, 0); dev_write_16bit(output, x, y, width, 2, 0); }
        return;
    }

    float xs = L * strength;
    float Lout = xs * (a * xs + b) / (xs * (c * xs + d) + e);
    float scale = Lout / L;

    float ro = __powf(fmaxf(r * scale, 0.0f), inv_gamma);
    float go = __powf(fmaxf(g * scale, 0.0f), inv_gamma);
    float bo = __powf(fmaxf(b2 * scale, 0.0f), inv_gamma);

    int iro = max(0, min(max_val, static_cast<int>(ro * mv + 0.5f)));
    int igo = max(0, min(max_val, static_cast<int>(go * mv + 0.5f)));
    int ibo = max(0, min(max_val, static_cast<int>(bo * mv + 0.5f)));

    if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, iro); dev_write_8bit(output, x, y, width, 1, igo); dev_write_8bit(output, x, y, width, 2, ibo); }
    else { dev_write_16bit(output, x, y, width, 0, iro); dev_write_16bit(output, x, y, width, 1, igo); dev_write_16bit(output, x, y, width, 2, ibo); }
}

// ============================================================
//  4. Hable (Uncharted 2): hable(x) normalized to [0,1]
// ============================================================

__global__ void kern_hable(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                            int width, int height, int bit_depth, int max_val,
                            float strength, float sat, float inv_gamma, float exposure_mul) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float A = 0.22f, B = 0.30f, C = 0.10f, D = 0.20f, E = 0.01f, F = 0.30f;
    const float W = 11.2f;
    // Precomputed: hable(W) and normalization
    float hw = (W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F);
    float hw_norm = hw - E / F;
    float inv_range = 1.0f / hw_norm;

    float mv = static_cast<float>(max_val);
    float r, g, b2;

    if (bit_depth <= 8) {
        r = dev_read_8bit(input, x, y, width, 0) / mv;
        g = dev_read_8bit(input, x, y, width, 1) / mv;
        b2 = dev_read_8bit(input, x, y, width, 2) / mv;
    } else {
        r = dev_read_16bit(input, x, y, width, 0) / mv;
        g = dev_read_16bit(input, x, y, width, 1) / mv;
        b2 = dev_read_16bit(input, x, y, width, 2) / mv;
    }

    r *= exposure_mul; g *= exposure_mul; b2 *= exposure_mul;
    float L = dev_luma(r, g, b2);
    if (L <= 0.0f) {
        if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, 0); dev_write_8bit(output, x, y, width, 1, 0); dev_write_8bit(output, x, y, width, 2, 0); }
        else { dev_write_16bit(output, x, y, width, 0, 0); dev_write_16bit(output, x, y, width, 1, 0); dev_write_16bit(output, x, y, width, 2, 0); }
        return;
    }

    float xs = L * strength;
    float hx = (xs * (A * xs + C * B) + D * E) / (xs * (A * xs + B) + D * F);
    hx -= E / F;
    float Lout = max(0.0f, min(1.0f, hx * inv_range));
    float scale = Lout / L;

    float ro = __powf(fmaxf(r * scale, 0.0f), inv_gamma);
    float go = __powf(fmaxf(g * scale, 0.0f), inv_gamma);
    float bo = __powf(fmaxf(b2 * scale, 0.0f), inv_gamma);

    int iro = max(0, min(max_val, static_cast<int>(ro * mv + 0.5f)));
    int igo = max(0, min(max_val, static_cast<int>(go * mv + 0.5f)));
    int ibo = max(0, min(max_val, static_cast<int>(bo * mv + 0.5f)));

    if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, iro); dev_write_8bit(output, x, y, width, 1, igo); dev_write_8bit(output, x, y, width, 2, ibo); }
    else { dev_write_16bit(output, x, y, width, 0, iro); dev_write_16bit(output, x, y, width, 1, igo); dev_write_16bit(output, x, y, width, 2, ibo); }
}

// ============================================================
//  5. Drago adaptive logarithmic
// ============================================================

__global__ void kern_drago(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                            int width, int height, int bit_depth, int max_val,
                            float bias, float sat, float inv_gamma, float exposure_mul,
                            float log_max, float max_L) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);
    float r, g, b2;

    if (bit_depth <= 8) {
        r = dev_read_8bit(input, x, y, width, 0) / mv;
        g = dev_read_8bit(input, x, y, width, 1) / mv;
        b2 = dev_read_8bit(input, x, y, width, 2) / mv;
    } else {
        r = dev_read_16bit(input, x, y, width, 0) / mv;
        g = dev_read_16bit(input, x, y, width, 1) / mv;
        b2 = dev_read_16bit(input, x, y, width, 2) / mv;
    }

    r *= exposure_mul; g *= exposure_mul; b2 *= exposure_mul;
    float L = dev_luma(r, g, b2);
    if (L <= 0.0f) {
        if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, 0); dev_write_8bit(output, x, y, width, 1, 0); dev_write_8bit(output, x, y, width, 2, 0); }
        else { dev_write_16bit(output, x, y, width, 0, 0); dev_write_16bit(output, x, y, width, 1, 0); dev_write_16bit(output, x, y, width, 2, 0); }
        return;
    }

    float local_bias = bias + (1.0f - bias) * (L / max_L);
    float Lout = __log10f(1.0f + L * 100.0f) / log_max;
    Lout = __powf(Lout, local_bias);
    Lout = max(0.0f, min(1.0f, Lout));
    float scale = Lout / L;

    float ro = __powf(fmaxf(r * scale, 0.0f), inv_gamma);
    float go = __powf(fmaxf(g * scale, 0.0f), inv_gamma);
    float bo = __powf(fmaxf(b2 * scale, 0.0f), inv_gamma);

    int iro = max(0, min(max_val, static_cast<int>(ro * mv + 0.5f)));
    int igo = max(0, min(max_val, static_cast<int>(go * mv + 0.5f)));
    int ibo = max(0, min(max_val, static_cast<int>(bo * mv + 0.5f)));

    if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, iro); dev_write_8bit(output, x, y, width, 1, igo); dev_write_8bit(output, x, y, width, 2, ibo); }
    else { dev_write_16bit(output, x, y, width, 0, iro); dev_write_16bit(output, x, y, width, 1, igo); dev_write_16bit(output, x, y, width, 2, ibo); }
}

// ============================================================
//  6. Exponential: L_out = 1 - exp(-k*L)
// ============================================================

__global__ void kern_exponential(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                  int width, int height, int bit_depth, int max_val,
                                  float sat, float inv_gamma, float exposure_mul) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);
    float r, g, b2;

    if (bit_depth <= 8) {
        r = dev_read_8bit(input, x, y, width, 0) / mv;
        g = dev_read_8bit(input, x, y, width, 1) / mv;
        b2 = dev_read_8bit(input, x, y, width, 2) / mv;
    } else {
        r = dev_read_16bit(input, x, y, width, 0) / mv;
        g = dev_read_16bit(input, x, y, width, 1) / mv;
        b2 = dev_read_16bit(input, x, y, width, 2) / mv;
    }

    r *= exposure_mul; g *= exposure_mul; b2 *= exposure_mul;
    float L = dev_luma(r, g, b2);
    if (L <= 0.0f) {
        if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, 0); dev_write_8bit(output, x, y, width, 1, 0); dev_write_8bit(output, x, y, width, 2, 0); }
        else { dev_write_16bit(output, x, y, width, 0, 0); dev_write_16bit(output, x, y, width, 1, 0); dev_write_16bit(output, x, y, width, 2, 0); }
        return;
    }

    float Lout = 1.0f - __expf(-exposure_mul * L);
    float scale = Lout / L;

    float ro = __powf(fmaxf(r * scale, 0.0f), inv_gamma);
    float go = __powf(fmaxf(g * scale, 0.0f), inv_gamma);
    float bo = __powf(fmaxf(b2 * scale, 0.0f), inv_gamma);

    int iro = max(0, min(max_val, static_cast<int>(ro * mv + 0.5f)));
    int igo = max(0, min(max_val, static_cast<int>(go * mv + 0.5f)));
    int ibo = max(0, min(max_val, static_cast<int>(bo * mv + 0.5f)));

    if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, iro); dev_write_8bit(output, x, y, width, 1, igo); dev_write_8bit(output, x, y, width, 2, ibo); }
    else { dev_write_16bit(output, x, y, width, 0, iro); dev_write_16bit(output, x, y, width, 1, igo); dev_write_16bit(output, x, y, width, 2, ibo); }
}

// ============================================================
//  7. Logarithmic: L_out = log(1+k*L)/log(1+k*Lmax)
// ============================================================

__global__ void kern_logarithmic(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                  int width, int height, int bit_depth, int max_val,
                                  float sat, float inv_gamma, float exposure_mul, float denom) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);
    float r, g, b2;

    if (bit_depth <= 8) {
        r = dev_read_8bit(input, x, y, width, 0) / mv;
        g = dev_read_8bit(input, x, y, width, 1) / mv;
        b2 = dev_read_8bit(input, x, y, width, 2) / mv;
    } else {
        r = dev_read_16bit(input, x, y, width, 0) / mv;
        g = dev_read_16bit(input, x, y, width, 1) / mv;
        b2 = dev_read_16bit(input, x, y, width, 2) / mv;
    }

    r *= exposure_mul; g *= exposure_mul; b2 *= exposure_mul;
    float L = dev_luma(r, g, b2);
    if (L <= 0.0f) {
        if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, 0); dev_write_8bit(output, x, y, width, 1, 0); dev_write_8bit(output, x, y, width, 2, 0); }
        else { dev_write_16bit(output, x, y, width, 0, 0); dev_write_16bit(output, x, y, width, 1, 0); dev_write_16bit(output, x, y, width, 2, 0); }
        return;
    }

    float Lout = __logf(1.0f + exposure_mul * L) / denom;
    Lout = max(0.0f, min(1.0f, Lout));
    float scale = Lout / L;

    float ro = __powf(fmaxf(r * scale, 0.0f), inv_gamma);
    float go = __powf(fmaxf(g * scale, 0.0f), inv_gamma);
    float bo = __powf(fmaxf(b2 * scale, 0.0f), inv_gamma);

    int iro = max(0, min(max_val, static_cast<int>(ro * mv + 0.5f)));
    int igo = max(0, min(max_val, static_cast<int>(go * mv + 0.5f)));
    int ibo = max(0, min(max_val, static_cast<int>(bo * mv + 0.5f)));

    if (bit_depth <= 8) { dev_write_8bit(output, x, y, width, 0, iro); dev_write_8bit(output, x, y, width, 1, igo); dev_write_8bit(output, x, y, width, 2, ibo); }
    else { dev_write_16bit(output, x, y, width, 0, iro); dev_write_16bit(output, x, y, width, 1, igo); dev_write_16bit(output, x, y, width, 2, ibo); }
}

// ============================================================
//  8. Linear → PQ (ST.2084)
// ============================================================

__global__ void kern_linear_to_pq(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                   int width, int height, int bit_depth, int max_val,
                                   float exposure_mul) {
    const float m1 = 0.1593017578125f, m2 = 78.84375f;
    const float c1 = 0.8359375f, c2 = 18.8515625f, c3 = 18.6875f;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);

    for (int c = 0; c < 3; c++) {
        float val;
        if (bit_depth <= 8) val = dev_read_8bit(input, x, y, width, c) / mv;
        else val = dev_read_16bit(input, x, y, width, c) / mv;

        val *= exposure_mul;
        float Lp = __powf(fmaxf(val, 0.0f), m1);
        float pq = __powf((c1 + c2 * Lp) / (1.0f + c3 * Lp), m2);
        pq = max(0.0f, min(1.0f, pq));

        int out_val = max(0, min(max_val, static_cast<int>(pq * mv + 0.5f)));
        if (bit_depth <= 8) dev_write_8bit(output, x, y, width, c, out_val);
        else dev_write_16bit(output, x, y, width, c, out_val);
    }
}

// ============================================================
//  9. PQ → Linear (ST.2084)
// ============================================================

__global__ void kern_pq_to_linear(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                   int width, int height, int bit_depth, int max_val,
                                   float exposure_mul) {
    const float m1 = 0.1593017578125f, m2_inv = 1.0f / 78.84375f;
    const float m1_inv = 1.0f / 0.1593017578125f;
    const float c1 = 0.8359375f, c2 = 18.8515625f, c3 = 18.6875f;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);

    for (int c = 0; c < 3; c++) {
        float val;
        if (bit_depth <= 8) val = dev_read_8bit(input, x, y, width, c) / mv;
        else val = dev_read_16bit(input, x, y, width, c) / mv;

        float vp = __powf(val, m2_inv);
        float num = fmaxf(0.0f, vp - c1);
        float den = fmaxf(1e-10f, c2 - c3 * vp);
        float linear = __powf(num / den, m1_inv);
        linear = fmaxf(0.0f, linear);
        linear *= exposure_mul;

        int out_val = max(0, min(max_val, static_cast<int>(linear * mv + 0.5f)));
        if (bit_depth <= 8) dev_write_8bit(output, x, y, width, c, out_val);
        else dev_write_16bit(output, x, y, width, c, out_val);
    }
}

// ============================================================
//  10. Linear → HLG (BT.2100 OETF)
// ============================================================

__global__ void kern_linear_to_hlg(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                    int width, int height, int bit_depth, int max_val,
                                    float exposure_mul) {
    const float a = 0.17883277f, b = 1.0f - 4.0f * a;
    const float cval = 0.5f - a * __logf(4.0f * a);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);

    for (int c = 0; c < 3; c++) {
        float E;
        if (bit_depth <= 8) E = dev_read_8bit(input, x, y, width, c) / mv;
        else E = dev_read_16bit(input, x, y, width, c) / mv;

        E *= exposure_mul;

        float sig;
        if (E <= 1.0f / 12.0f) {
            sig = sqrtf(3.0f * E);
        } else {
            sig = a * __logf(12.0f * E - b) + cval;
        }
        sig = max(0.0f, min(1.0f, sig));

        int out_val = max(0, min(max_val, static_cast<int>(sig * mv + 0.5f)));
        if (bit_depth <= 8) dev_write_8bit(output, x, y, width, c, out_val);
        else dev_write_16bit(output, x, y, width, c, out_val);
    }
}

// ============================================================
//  11. HLG → Linear (BT.2100 EOTF)
// ============================================================

__global__ void kern_hlg_to_linear(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                    int width, int height, int bit_depth, int max_val,
                                    float exposure_mul) {
    const float a = 0.17883277f, b = 1.0f - 4.0f * a;
    const float cval = 0.5f - a * __logf(4.0f * a);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);

    for (int c = 0; c < 3; c++) {
        float sig;
        if (bit_depth <= 8) sig = dev_read_8bit(input, x, y, width, c) / mv;
        else sig = dev_read_16bit(input, x, y, width, c) / mv;

        float E;
        if (sig <= 0.5f) {
            E = sig * sig / 3.0f;
        } else {
            E = (__expf((sig - cval) / a) + b) / 12.0f;
        }
        E = fmaxf(0.0f, E);
        E *= exposure_mul;

        int out_val = max(0, min(max_val, static_cast<int>(E * mv + 0.5f)));
        if (bit_depth <= 8) dev_write_8bit(output, x, y, width, c, out_val);
        else dev_write_16bit(output, x, y, width, c, out_val);
    }
}

// ============================================================
//  Host-side launch wrappers
// ============================================================

namespace hdr {

void cuda_launch_hdr_reinhard(HdrCudaWorkspace* ws, int width, int height,
                               int bit_depth, int max_val,
                               float sat, float inv_gamma, float exposure_mul) {
    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_reinhard<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                    bit_depth, max_val, sat, inv_gamma, exposure_mul);
}

void cuda_launch_hdr_reinhard_ext(HdrCudaWorkspace* ws, int width, int height,
                                   int bit_depth, int max_val,
                                   float key, float wp2, float sat, float inv_gamma) {
    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);

    // Precompute log-average on host, pass scale_factor to kernel
    // For CUDA simplicity, we compute log-average on CPU first
    float* h_luminance = new float[width * height];
    cudaMemcpy(h_luminance, ws->d_fbuf, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    // Re-read from input instead (d_fbuf not initialized)
    // For Reinhard Ext, compute log_avg on CPU from input data
    size_t total = static_cast<size_t>(width) * height;
    float log_sum = 0.0f;
    float eps = 1e-6f;
    size_t rgb_bytes = total * 3 * ((bit_depth <= 8) ? 1 : 2);
    uint8_t* h_input = new uint8_t[rgb_bytes];
    cudaMemcpy(h_input, ws->d_input, rgb_bytes, cudaMemcpyDeviceToHost);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r, g, b;
            float mv = static_cast<float>(max_val);
            if (bit_depth <= 8) {
                r = h_input[(static_cast<size_t>(y) * width + x) * 3 + 0] / mv;
                g = h_input[(static_cast<size_t>(y) * width + x) * 3 + 1] / mv;
                b = h_input[(static_cast<size_t>(y) * width + x) * 3 + 2] / mv;
            } else {
                const uint16_t* d16 = reinterpret_cast<const uint16_t*>(h_input);
                r = d16[(static_cast<size_t>(y) * width + x) * 3 + 0] / mv;
                g = d16[(static_cast<size_t>(y) * width + x) * 3 + 1] / mv;
                b = d16[(static_cast<size_t>(y) * width + x) * 3 + 2] / mv;
            }
            float L = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            log_sum += std::log(eps + L);
        }
    }
    delete[] h_input;
    float log_avg = log_sum / static_cast<float>(total);
    float L_avg = std::exp(log_avg);
    float scale_factor = key / (eps + L_avg);

    delete[] h_luminance;
    kern_reinhard_ext<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                        bit_depth, max_val, key, wp2, sat, inv_gamma, scale_factor);
}

void cuda_launch_hdr_filmic_aces(HdrCudaWorkspace* ws, int width, int height,
                                  int bit_depth, int max_val,
                                  float strength, float sat, float inv_gamma, float exposure_mul) {
    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_filmic_aces<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                       bit_depth, max_val, strength, sat, inv_gamma, exposure_mul);
}

void cuda_launch_hdr_hable(HdrCudaWorkspace* ws, int width, int height,
                            int bit_depth, int max_val,
                            float strength, float sat, float inv_gamma, float exposure_mul) {
    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_hable<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                 bit_depth, max_val, strength, sat, inv_gamma, exposure_mul);
}

void cuda_launch_hdr_drago(HdrCudaWorkspace* ws, int width, int height,
                            int bit_depth, int max_val,
                            float bias, float sat, float inv_gamma, float exposure_mul) {
    // Pre-compute max_L and log_max on CPU
    float max_L = 1e-6f;
    size_t total = static_cast<size_t>(width) * height;
    size_t rgb_bytes = total * 3 * ((bit_depth <= 8) ? 1 : 2);
    uint8_t* h_input = new uint8_t[rgb_bytes];
    cudaMemcpy(h_input, ws->d_input, rgb_bytes, cudaMemcpyDeviceToHost);

    float mv = static_cast<float>(max_val);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r, g, b;
            if (bit_depth <= 8) {
                r = h_input[(static_cast<size_t>(y) * width + x) * 3 + 0] / mv;
                g = h_input[(static_cast<size_t>(y) * width + x) * 3 + 1] / mv;
                b = h_input[(static_cast<size_t>(y) * width + x) * 3 + 2] / mv;
            } else {
                const uint16_t* d16 = reinterpret_cast<const uint16_t*>(h_input);
                r = d16[(static_cast<size_t>(y) * width + x) * 3 + 0] / mv;
                g = d16[(static_cast<size_t>(y) * width + x) * 3 + 1] / mv;
                b = d16[(static_cast<size_t>(y) * width + x) * 3 + 2] / mv;
            }
            float L = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            if (L > max_L) max_L = L;
        }
    }
    delete[] h_input;

    float log_max = std::log10(1.0f + max_L * 100.0f);

    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_drago<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                 bit_depth, max_val, bias, sat, inv_gamma, exposure_mul,
                                 log_max, max_L);
}

void cuda_launch_hdr_exponential(HdrCudaWorkspace* ws, int width, int height,
                                  int bit_depth, int max_val,
                                  float sat, float inv_gamma, float exposure_mul) {
    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_exponential<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                       bit_depth, max_val, sat, inv_gamma, exposure_mul);
}

void cuda_launch_hdr_logarithmic(HdrCudaWorkspace* ws, int width, int height,
                                  int bit_depth, int max_val,
                                  float sat, float inv_gamma, float exposure_mul) {
    // Pre-compute max_L and denom on CPU
    float max_L = 1e-6f;
    size_t total = static_cast<size_t>(width) * height;
    size_t rgb_bytes = total * 3 * ((bit_depth <= 8) ? 1 : 2);
    uint8_t* h_input = new uint8_t[rgb_bytes];
    cudaMemcpy(h_input, ws->d_input, rgb_bytes, cudaMemcpyDeviceToHost);

    float mv = static_cast<float>(max_val);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r, g, b;
            if (bit_depth <= 8) {
                r = h_input[(static_cast<size_t>(y) * width + x) * 3 + 0] / mv;
                g = h_input[(static_cast<size_t>(y) * width + x) * 3 + 1] / mv;
                b = h_input[(static_cast<size_t>(y) * width + x) * 3 + 2] / mv;
            } else {
                const uint16_t* d16 = reinterpret_cast<const uint16_t*>(h_input);
                r = d16[(static_cast<size_t>(y) * width + x) * 3 + 0] / mv;
                g = d16[(static_cast<size_t>(y) * width + x) * 3 + 1] / mv;
                b = d16[(static_cast<size_t>(y) * width + x) * 3 + 2] / mv;
            }
            float L = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            if (L > max_L) max_L = L;
        }
    }
    delete[] h_input;

    float denom = std::log(1.0f + exposure_mul * max_L);

    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_logarithmic<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                       bit_depth, max_val, sat, inv_gamma, exposure_mul, denom);
}

void cuda_launch_hdr_linear_to_pq(HdrCudaWorkspace* ws, int width, int height,
                                   int bit_depth, int max_val, float exposure_mul) {
    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_linear_to_pq<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                        bit_depth, max_val, exposure_mul);
}

void cuda_launch_hdr_pq_to_linear(HdrCudaWorkspace* ws, int width, int height,
                                   int bit_depth, int max_val, float exposure_mul) {
    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_pq_to_linear<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                        bit_depth, max_val, exposure_mul);
}

void cuda_launch_hdr_linear_to_hlg(HdrCudaWorkspace* ws, int width, int height,
                                    int bit_depth, int max_val, float exposure_mul) {
    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_linear_to_hlg<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                         bit_depth, max_val, exposure_mul);
}

void cuda_launch_hdr_hlg_to_linear(HdrCudaWorkspace* ws, int width, int height,
                                    int bit_depth, int max_val, float exposure_mul) {
    dim3 block(32, 16);
    dim3 grid = HDR_TM_GRID(width, height);
    kern_hlg_to_linear<<<grid, block>>>(ws->d_input, ws->d_output, width, height,
                                         bit_depth, max_val, exposure_mul);
}

} // namespace hdr
