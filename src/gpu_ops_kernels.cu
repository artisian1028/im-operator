#include "gpu_ops_kernels.cuh"
#include "gpu_common.cuh"
#include <cuda_runtime.h>
#include <cstring>

// ============================================================
//  Workspace implementation
// ============================================================

bool GpuWorkspace::ensure(int w, int h, int bd, size_t param_bytes) {
    if (width == w && height == h && bit_depth == bd && d_input) return true;
    release();
    width = w; height = h; bit_depth = bd;
    max_val = (bd <= 8) ? 255 : ((1 << bd) - 1);
    if (bd <= 0) max_val = 255;

    size_t rgb = static_cast<size_t>(w) * h * 3 * ((bd <= 8) ? 1 : 2);
    if (cudaMalloc(&d_input, rgb) != cudaSuccess) { release(); return false; }
    if (cudaMalloc(&d_output, rgb) != cudaSuccess) { release(); return false; }
    if (cudaMalloc(&d_scratch, rgb) != cudaSuccess) { release(); return false; }
    if (param_bytes > 0 && cudaMalloc(&d_params, param_bytes) != cudaSuccess) { release(); return false; }
    return true;
}

void GpuWorkspace::release() {
    if (d_input)  { cudaFree(d_input);  d_input = nullptr; }
    if (d_output) { cudaFree(d_output); d_output = nullptr; }
    if (d_scratch) { cudaFree(d_scratch); d_scratch = nullptr; }
    if (d_params) { cudaFree(d_params); d_params = nullptr; }
    width = height = bit_depth = max_val = 0;
}

// ============================================================
//  GPU Pipeline kernel: chains multiple per-pixel ops
// ============================================================

__global__ void kern_gpu_pipeline(const uint8_t* __restrict__ input,
                                   uint8_t* __restrict__ output,
                                   int width, int height, int bit_depth, int max_val,
                                   const float* __restrict__ params,
                                   int op_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float mv = static_cast<float>(max_val);
    float r = dev_read_pixel(input, x, y, width, bit_depth, 0, mv);
    float g = dev_read_pixel(input, x, y, width, bit_depth, 1, mv);
    float b = dev_read_pixel(input, x, y, width, bit_depth, 2, mv);

    int param_offset = 0;
    for (int oi = 0; oi < op_count; oi++) {
        int op_type = static_cast<int>(params[param_offset++]);
        const float* op_p = params + param_offset;

        switch (op_type) {
            case 0: { // TONE_GAMMA: [inv_gamma]
                float inv_g = op_p[0];
                r = __powf(dev_clamp(r), inv_g);
                g = __powf(dev_clamp(g), inv_g);
                b = __powf(dev_clamp(b), inv_g);
                param_offset += 1;
                break;
            }
            case 1: { // TONE_S_CURVE: [contrast, shadows, highlights]
                float contrast = op_p[0], shadows = op_p[1], highlights = op_p[2];
                float l = dev_luma(r, g, b);
                float sw = (1.0f - 2.0f * l); sw = (sw > 0.0f) ? sw * sw : 0.0f;
                float hw = (2.0f * l - 1.0f); hw = (hw > 0.0f) ? hw * hw : 0.0f;
                float adj = 1.0f + shadows * sw * 0.3f - highlights * hw * 0.3f;
                r = dev_s_curve(r * adj, contrast);
                g = dev_s_curve(g * adj, contrast);
                b = dev_s_curve(b * adj, contrast);
                r = dev_clamp(r); g = dev_clamp(g); b = dev_clamp(b);
                param_offset += 3;
                break;
            }
            case 2: { // TONE_LEVELS: [black, white, mid, gamma]
                float blk = op_p[0], wht = op_p[1], mid = op_p[2], gam = op_p[3];
                float range = wht - blk + 1e-6f;
                float mid_norm = (mid - blk) / range;
                float gamma = (mid_norm > 0.0f && mid_norm < 1.0f) ? logf(0.5f)/logf(mid_norm) : 1.0f;
                auto apply = [&](float v) {
                    float s = (v - blk) / range;
                    s = dev_clamp(s);
                    return __powf(s, 1.0f / (gamma + 1e-6f));
                };
                r = apply(r); g = apply(g); b = apply(b);
                param_offset += 4;
                break;
            }
            case 3: { // CCM_3X3: [m0..m8]
                dev_ccm_3x3(r, g, b, op_p);
                param_offset += 9;
                break;
            }
            case 4: { // SATURATION_HSL: [sat]
                dev_saturate(r, g, b, op_p[0]);
                param_offset += 1;
                break;
            }
            case 5: { // SATURATION_VIBRANCE: [vibrance]
                float vib = op_p[0];
                float l = dev_luma(r, g, b);
                float mx = fmaxf(fmaxf(r, g), b);
                float s = (mx > 0.0f) ? (mx - fminf(fminf(r, g), b)) / mx : 0.0f;
                float factor = 1.0f + vib * (1.0f - s) * (1.0f - l);
                r = dev_clamp(l + factor * (r - l));
                g = dev_clamp(l + factor * (g - l));
                b = dev_clamp(l + factor * (b - l));
                param_offset += 1;
                break;
            }
            case 6: { // COLOR_TEMP_MULTIPLY: [rm, gm, bm]
                dev_color_temp(r, g, b, op_p[0], op_p[1], op_p[2]);
                param_offset += 3;
                break;
            }
            case 7: { // WHITE_BALANCE_MANUAL: [rg, gg, bg]
                r = dev_clamp(r * op_p[0]);
                g = dev_clamp(g * op_p[1]);
                b = dev_clamp(b * op_p[2]);
                param_offset += 3;
                break;
            }
            case 8: { // BLACK_LEVEL_GLOBAL: [offset_normalized]
                r = fmaxf(0.0f, r - op_p[0]);
                g = fmaxf(0.0f, g - op_p[0]);
                b = fmaxf(0.0f, b - op_p[0]);
                param_offset += 1;
                break;
            }
            case 9: { // LENS_SHADING_POLY: [cx,cy, a2,a4,a6] × 4 channels = 20 params
                // Simplified: use single gain(r) for all channels
                float cx = op_p[0], cy = op_p[1], a2 = op_p[2], a4 = op_p[3], a6 = op_p[4];
                float dx = (static_cast<float>(x)/static_cast<float>(width)) - cx;
                float dy = (static_cast<float>(y)/static_cast<float>(height)) - cy;
                float r2 = dx*dx + dy*dy;
                float r4 = r2*r2, r6 = r4*r2;
                float gain = 1.0f + a2*r2 + a4*r4 + a6*r6;
                r = dev_clamp(r * gain);
                g = dev_clamp(g * gain);
                b = dev_clamp(b * gain);
                param_offset += 5;
                break;
            }
            case 10: { // HIGHLIGHT_RECON_CHANNEL: [clip_thresh]
                float th = op_p[0];
                bool cr = (r >= th), cg = (g >= th), cb = (b >= th);
                int nc = (cr?1:0)+(cg?1:0)+(cb?1:0);
                if (nc > 0 && nc < 3) {
                    float max_unclip = 0.0f;
                    if (!cr) max_unclip = fmaxf(max_unclip, r);
                    if (!cg) max_unclip = fmaxf(max_unclip, g);
                    if (!cb) max_unclip = fmaxf(max_unclip, b);
                    if (max_unclip > 0.0f) {
                        float scale = 1.0f / max_unclip;
                        if (cr) r = dev_clamp(max_unclip * 1.5f);
                        if (cg) g = dev_clamp(max_unclip * 1.5f);
                        if (cb) b = dev_clamp(max_unclip * 1.5f);
                    }
                }
                param_offset += 1;
                break;
            }
            case 11: { // SHARPEN_UNSHARP_MASK: [amount]
                // Simple pre-sharpening: enhance local contrast
                // Full USM needs blur — done in dedicated kernel
                float amt = op_p[0];
                r = dev_clamp(r + amt * (r - 0.5f));
                g = dev_clamp(g + amt * (g - 0.5f));
                b = dev_clamp(b + amt * (b - 0.5f));
                param_offset += 1;
                break;
            }
            case 13: { // REINHARD_TONE_MAP: [exposure_mul, saturation]
                r *= op_p[0]; g *= op_p[0]; b *= op_p[0];
                float L = dev_luma(r, g, b);
                if (L > 0.0f) {
                    float Lout = L / (1.0f + L);
                    float sc = __powf(Lout / L, op_p[1]);
                    r = dev_clamp(r * sc); g = dev_clamp(g * sc); b = dev_clamp(b * sc);
                }
                param_offset += 2;
                break;
            }
            case 14: { // FILMIC_TONE_MAP: [exposure_mul]
                r *= op_p[0]; g *= op_p[0]; b *= op_p[0];
                r = dev_filmic(r); g = dev_filmic(g); b = dev_filmic(b);
                param_offset += 1;
                break;
            }
            case 15: { // CCM_4X3: [m00..m03, m10..m13, m20..m23]
                float nr = op_p[0]*r + op_p[1]*g + op_p[2]*b + op_p[3];
                float ng = op_p[4]*r + op_p[5]*g + op_p[6]*b + op_p[7];
                float nb = op_p[8]*r + op_p[9]*g + op_p[10]*b + op_p[11];
                r = dev_clamp(nr); g = dev_clamp(ng); b = dev_clamp(nb);
                param_offset += 12;
                break;
            }
            case 16: { // SATURATION_CHANNEL_MIXER: [3x3 matrix, 9 floats]
                float nr = op_p[0]*r + op_p[1]*g + op_p[2]*b;
                float ng = op_p[3]*r + op_p[4]*g + op_p[5]*b;
                float nb = op_p[6]*r + op_p[7]*g + op_p[8]*b;
                r = dev_clamp(nr); g = dev_clamp(ng); b = dev_clamp(nb);
                param_offset += 9;
                break;
            }
            case 17: { // SATURATION_SELECTIVE: [r_sat,g_sat,b_sat, r_th,g_th,b_th]
                auto sel = [&](float v, float sat, float th) {
                    if (v > th) { float s = (v - th) / (1.0f - th + 1e-6f); return v + sat * s * 0.3f; }
                    return v;
                };
                r = dev_clamp(sel(r, op_p[0], op_p[3]));
                g = dev_clamp(sel(g, op_p[1], op_p[4]));
                b = dev_clamp(sel(b, op_p[2], op_p[5]));
                param_offset += 6;
                break;
            }
            default:
                return; // unknown op — skip
        }
    }

    dev_write_pixel(output, x, y, width, bit_depth, 0, r, max_val);
    dev_write_pixel(output, x, y, width, bit_depth, 1, g, max_val);
    dev_write_pixel(output, x, y, width, bit_depth, 2, b, max_val);
}

void cuda_launch_gpu_pipeline(GpuWorkspace* ws,
                              const GpuOpParam* ops, int op_count,
                              int width, int height, int bit_depth, int max_val) {
    size_t param_floats = static_cast<size_t>(op_count) * 16; // 1 type + 15 params per op
    float* h_params = new float[param_floats];
    int off = 0;
    for (int i = 0; i < op_count; i++) {
        h_params[off++] = static_cast<float>(ops[i].op);
        for (int j = 0; j < 15; j++) h_params[off++] = ops[i].params[j];
    }
    cudaMemcpy(ws->d_params, h_params, param_floats * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_params;

    dim3 block(32, 16);
    dim3 grid((width + 31) / 32, (height + 15) / 16);
    kern_gpu_pipeline<<<grid, block>>>(ws->d_input, ws->d_output,
                                        width, height, bit_depth, max_val,
                                        ws->d_params, op_count);
}

// ============================================================
//  Denoise kernels
// ============================================================

__global__ void kern_bilateral(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                int w, int h, int bd, int mv, float sigma_s, float sigma_r, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float fmaxv = static_cast<float>(mv);
    float ss2 = 2.0f * sigma_s * sigma_s;
    float sr2 = 2.0f * sigma_r * sigma_r;

    for (int c = 0; c < 3; c++) {
        float center = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
        float sum = 0.0f, wsum = 0.0f;

        for (int dy = -radius; dy <= radius; dy++) {
            int ny = y + dy;
            if (ny < 0) ny = 0; if (ny >= h) ny = h - 1;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
                float neighbor = dev_read_pixel(in, nx, ny, w, bd, c, fmaxv);
                float ds = static_cast<float>(dx*dx + dy*dy);
                float spatial_w = __expf(-ds / ss2);
                float dr = center - neighbor;
                float range_w = __expf(-dr * dr / sr2);
                float w_val = spatial_w * range_w;
                sum += neighbor * w_val;
                wsum += w_val;
            }
        }
        float result = (wsum > 0.0f) ? sum / wsum : center;
        dev_write_pixel(out, x, y, w, bd, c, result, mv);
    }
}

void cuda_launch_denoise_bilateral(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                    float sigma_s, float sigma_r, int radius) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_bilateral<<<grid, block>>>(ws->d_input, ws->d_output, w, h, bd, mv, sigma_s, sigma_r, radius);
}

__global__ void kern_gaussian_horiz(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                     int w, int h, int bd, int mv, float sigma, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float fmaxv = static_cast<float>(mv);
    float sigma2 = 2.0f * sigma * sigma;

    for (int c = 0; c < 3; c++) {
        float sum = 0.0f, wsum = 0.0f;
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
            float w_val = __expf(-static_cast<float>(dx*dx) / sigma2);
            sum += dev_read_pixel(in, nx, y, w, bd, c, fmaxv) * w_val;
            wsum += w_val;
        }
        dev_write_pixel(out, x, y, w, bd, c, sum / wsum, mv);
    }
}

__global__ void kern_gaussian_vert(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                    int w, int h, int bd, int mv, float sigma, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float fmaxv = static_cast<float>(mv);
    float sigma2 = 2.0f * sigma * sigma;

    for (int c = 0; c < 3; c++) {
        float sum = 0.0f, wsum = 0.0f;
        for (int dy = -radius; dy <= radius; dy++) {
            int ny = y + dy;
            if (ny < 0) ny = 0; if (ny >= h) ny = h - 1;
            float w_val = __expf(-static_cast<float>(dy*dy) / sigma2);
            sum += dev_read_pixel(in, x, ny, w, bd, c, fmaxv) * w_val;
            wsum += w_val;
        }
        dev_write_pixel(out, x, y, w, bd, c, sum / wsum, mv);
    }
}

void cuda_launch_denoise_gaussian(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                   float sigma, int radius) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    // Separable: horizontal → scratch, vertical → output
    kern_gaussian_horiz<<<grid, block>>>(ws->d_input, ws->d_scratch, w, h, bd, mv, sigma, radius);
    cudaDeviceSynchronize();
    kern_gaussian_vert<<<grid, block>>>(ws->d_scratch, ws->d_output, w, h, bd, mv, sigma, radius);
}

__global__ void kern_median(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                             int w, int h, int bd, int mv, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float fmaxv = static_cast<float>(mv);
    int N = (2*radius+1)*(2*radius+1);
    extern __shared__ float shared_vals[];

    for (int c = 0; c < 3; c++) {
        int idx = 0;
        for (int dy = -radius; dy <= radius; dy++) {
            int ny = y + dy;
            if (ny < 0) ny = 0; if (ny >= h) ny = h - 1;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
                float val = dev_read_pixel(in, nx, ny, w, bd, c, fmaxv);
                // Insertion sort into shared memory
                int j = idx;
                while (j > 0 && shared_vals[j-1] > val) {
                    if (j < N) shared_vals[j] = shared_vals[j-1];
                    j--;
                }
                if (j < N) shared_vals[j] = val;
                idx++;
            }
        }
        // Use quick-select position rather than sorted array
        // Simplified: just take middle element from the (unsorted) accumulation
        // Full sort is expensive — use nth_element equivalent
        // For small N (25 for radius=2), brute force sort is fine
        float result = shared_vals[N/2];
        dev_write_pixel(out, x, y, w, bd, c, result, mv);
    }
}

void cuda_launch_denoise_median(GpuWorkspace* ws, int w, int h, int bd, int mv, int radius) {
    int N = (2*radius+1)*(2*radius+1);
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_median<<<grid, block, N * sizeof(float)>>>(ws->d_input, ws->d_output, w, h, bd, mv, radius);
}

// ============================================================
//  LUT kernel: 3D trilinear interpolation
// ============================================================

__global__ void kern_lut(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                          int w, int h, int bd, int mv,
                          const float* __restrict__ lut_data, int lut_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float fmaxv = static_cast<float>(mv);
    float r = dev_read_pixel(in, x, y, w, bd, 0, fmaxv);
    float g = dev_read_pixel(in, x, y, w, bd, 1, fmaxv);
    float b = dev_read_pixel(in, x, y, w, bd, 2, fmaxv);

    // Scale to LUT indices
    int ls = lut_size;
    float ls_f = static_cast<float>(ls - 1);
    float ri = r * ls_f, gi = g * ls_f, bi = b * ls_f;
    int r0 = static_cast<int>(ri), g0 = static_cast<int>(gi), b0 = static_cast<int>(bi);
    int r1 = (r0 < ls-1) ? r0+1 : r0;
    int g1 = (g0 < ls-1) ? g0+1 : g0;
    int b1 = (b0 < ls-1) ? b0+1 : b0;
    float rd = ri - static_cast<float>(r0);
    float gd = gi - static_cast<float>(g0);
    float bdist = bi - static_cast<float>(b0);

    // Trilinear interpolation
    auto fetch = [&](int rii, int gii, int bii, int ch) -> float {
        return lut_data[((static_cast<size_t>(rii) * ls + gii) * ls + bii) * 3 + ch];
    };

    auto lerp = [](float a, float b, float t) { return a + (b-a)*t; };

    for (int c = 0; c < 3; c++) {
        float c000 = fetch(r0,g0,b0,c), c100 = fetch(r1,g0,b0,c);
        float c010 = fetch(r0,g1,b0,c), c110 = fetch(r1,g1,b0,c);
        float c001 = fetch(r0,g0,b1,c), c101 = fetch(r1,g0,b1,c);
        float c011 = fetch(r0,g1,b1,c), c111 = fetch(r1,g1,b1,c);

        float c00 = lerp(c000, c100, rd), c01 = lerp(c001, c101, rd);
        float c10 = lerp(c010, c110, rd), c11 = lerp(c011, c111, rd);
        float c0 = lerp(c00, c10, gd), c1 = lerp(c01, c11, gd);
        float val = lerp(c0, c1, bdist);

        dev_write_pixel(out, x, y, w, bd, c, val, mv);
    }
}

void cuda_launch_lut_apply(GpuWorkspace* ws, int w, int h, int bd, int mv,
                            const float* h_lut_data, int lut_size) {
    size_t lut_bytes = static_cast<size_t>(lut_size) * lut_size * lut_size * 3 * sizeof(float);
    float* d_lut;
    cudaMalloc(&d_lut, lut_bytes);
    cudaMemcpy(d_lut, h_lut_data, lut_bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_lut<<<grid, block>>>(ws->d_input, ws->d_output, w, h, bd, mv, d_lut, lut_size);

    cudaDeviceSynchronize();
    cudaFree(d_lut);
}

// ============================================================
//  Sharpen kernels
// ============================================================

__global__ void kern_sharpen_unsharp(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                      int w, int h, int bd, int mv, float amount, float sigma, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float fmaxv = static_cast<float>(mv);
    float sigma2 = 2.0f * sigma * sigma;

    // Gaussian blur in 5x5 window
    for (int c = 0; c < 3; c++) {
        float center = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
        float blur = 0.0f, wsum = 0.0f;
        for (int dy = -radius; dy <= radius; dy++) {
            int ny = y + dy;
            if (ny < 0) ny = 0; if (ny >= h) ny = h - 1;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
                float w_val = __expf(-static_cast<float>(dx*dx+dy*dy) / sigma2);
                blur += dev_read_pixel(in, nx, ny, w, bd, c, fmaxv) * w_val;
                wsum += w_val;
            }
        }
        blur /= wsum;
        float result = center + amount * (center - blur);
        dev_write_pixel(out, x, y, w, bd, c, result, mv);
    }
}

void cuda_launch_sharpen_unsharp(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                  float amount, float sigma, int radius) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_sharpen_unsharp<<<grid, block>>>(ws->d_input, ws->d_output, w, h, bd, mv, amount, sigma, radius);
}

__global__ void kern_sharpen_laplacian(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                        int w, int h, int bd, int mv, float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float fmaxv = static_cast<float>(mv);
    // 3x3 Laplacian: [0,-1,0; -1,4,-1; 0,-1,0]
    for (int c = 0; c < 3; c++) {
        float center = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
        int xm1 = (x>0)?x-1:0, xp1 = (x<w-1)?x+1:w-1;
        int ym1 = (y>0)?y-1:0, yp1 = (y<h-1)?y+1:h-1;
        float n = dev_read_pixel(in, x, ym1, w, bd, c, fmaxv);
        float s = dev_read_pixel(in, x, yp1, w, bd, c, fmaxv);
        float e = dev_read_pixel(in, xp1, y, w, bd, c, fmaxv);
        float w_val = dev_read_pixel(in, xm1, y, w, bd, c, fmaxv);
        float lap = 4.0f * center - (n + s + e + w_val);
        float result = center - amount * lap / 4.0f;
        dev_write_pixel(out, x, y, w, bd, c, result, mv);
    }
}

void cuda_launch_sharpen_laplacian(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                    float amount, float /*radius*/) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_sharpen_laplacian<<<grid, block>>>(ws->d_input, ws->d_output, w, h, bd, mv, amount);
}

// ============================================================
//  Local Contrast kernel
// ============================================================

__global__ void kern_local_contrast(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                     int w, int h, int bd, int mv, float amount, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float fmaxv = static_cast<float>(mv);
    int radius = static_cast<int>(sigma * 2.0f);
    if (radius < 1) radius = 1;
    float sigma2 = 2.0f * sigma * sigma;

    float r = dev_read_pixel(in, x, y, w, bd, 0, fmaxv);
    float g = dev_read_pixel(in, x, y, w, bd, 1, fmaxv);
    float b = dev_read_pixel(in, x, y, w, bd, 2, fmaxv);
    float l = dev_luma(r, g, b);

    // Gaussian blur of luminance
    float blur = 0.0f, wsum = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        int ny = y + dy;
        if (ny < 0) ny = 0; if (ny >= h) ny = h - 1;
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
            float n_r = dev_read_pixel(in, nx, ny, w, bd, 0, fmaxv);
            float n_g = dev_read_pixel(in, nx, ny, w, bd, 1, fmaxv);
            float n_b = dev_read_pixel(in, nx, ny, w, bd, 2, fmaxv);
            float nl = dev_luma(n_r, n_g, n_b);
            float w_val = __expf(-static_cast<float>(dx*dx+dy*dy) / sigma2);
            blur += nl * w_val;
            wsum += w_val;
        }
    }
    blur /= wsum;

    float detail = l - blur;
    float l_out = blur + amount * detail * 2.0f;
    l_out = dev_clamp(l_out);

    float scale = (l > 0.0f) ? (l_out / l) : 0.0f;
    dev_write_pixel(out, x, y, w, bd, 0, r * scale, mv);
    dev_write_pixel(out, x, y, w, bd, 1, g * scale, mv);
    dev_write_pixel(out, x, y, w, bd, 2, b * scale, mv);
}

void cuda_launch_local_contrast(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                 float amount, float radius) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_local_contrast<<<grid, block>>>(ws->d_input, ws->d_output, w, h, bd, mv, amount, radius);
}

// ============================================================
//  Pipeline kernel: new ops (cases 15-20)
//  Added via editing kern_gpu_pipeline switch to include
//  these after the existing case 14 block
// ============================================================
// Note: The following new pipeline ops are appended after case 14
// in kern_gpu_pipeline. They must be inserted into the switch.

// ============================================================
//  Bayer-domain kernels
// ============================================================

// --- Black level: per-channel ---
__global__ void kern_black_level_per_channel(uint8_t* __restrict__ data,
                                               int w, int h, int bd, int pattern,
                                               float ro, float go0, float go1, float bo) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int bc = dev_bayer_color(y, x, pattern);
    float offset = (bc == 0) ? ro : (bc == 3) ? bo : ((bc == 1) ? go0 : go1);
    int mv = (bd <= 8) ? 255 : ((1 << bd) - 1);
    float maxv = static_cast<float>(mv);
    float v = dev_read_bayer(data, x, y, w, bd, maxv);
    dev_write_bayer(data, x, y, w, bd, fmaxf(0.0f, v - offset), mv);
}

void cuda_launch_black_level_per_channel(uint8_t* d_data, int w, int h, int bd, int pattern,
                                          const float offsets[4]) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_black_level_per_channel<<<grid, block>>>(d_data, w, h, bd, pattern,
                                                    offsets[0], offsets[1], offsets[2], offsets[3]);
}

// --- Black level: global ---
__global__ void kern_black_level_global(uint8_t* __restrict__ data,
                                          int w, int h, int bd, float offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int mv = (bd <= 8) ? 255 : ((1 << bd) - 1);
    float v = dev_read_bayer(data, x, y, w, bd, static_cast<float>(mv));
    dev_write_bayer(data, x, y, w, bd, fmaxf(0.0f, v - offset), mv);
}

void cuda_launch_black_level_global(uint8_t* d_data, int w, int h, int bd, float offset) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_black_level_global<<<grid, block>>>(d_data, w, h, bd, offset);
}

// --- Lens shading: polynomial correction ---
__global__ void kern_lens_shading_poly(uint8_t* __restrict__ data,
                                         int w, int h, int bd, int pattern,
                                         float cx, float cy,
                                         float a2_0, float a2_1, float a2_2, float a2_3,
                                         float a4_0, float a4_1, float a4_2, float a4_3,
                                         float a6_0, float a6_1, float a6_2, float a6_3) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int bc = dev_bayer_color(y, x, pattern);
    float a2 = (bc == 0) ? a2_0 : (bc == 1) ? a2_1 : (bc == 2) ? a2_2 : a2_3;
    float a4 = (bc == 0) ? a4_0 : (bc == 1) ? a4_1 : (bc == 2) ? a4_2 : a4_3;
    float a6 = (bc == 0) ? a6_0 : (bc == 1) ? a6_1 : (bc == 2) ? a6_2 : a6_3;
    float dx = (static_cast<float>(x) / static_cast<float>(w)) - cx;
    float dy = (static_cast<float>(y) / static_cast<float>(h)) - cy;
    float r2 = dx*dx + dy*dy;
    float r4 = r2*r2, r6 = r4*r2;
    float gain = 1.0f + a2*r2 + a4*r4 + a6*r6;
    int mv = (bd <= 8) ? 255 : ((1 << bd) - 1);
    float v = dev_read_bayer(data, x, y, w, bd, static_cast<float>(mv));
    dev_write_bayer(data, x, y, w, bd, v * gain, mv);
}

void cuda_launch_lens_shading_poly(uint8_t* d_data, int w, int h, int bd, int pattern,
                                    float cx, float cy,
                                    const float a2[4], const float a4[4], const float a6[4]) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_lens_shading_poly<<<grid, block>>>(d_data, w, h, bd, pattern,
                                              cx, cy,
                                              a2[0], a2[1], a2[2], a2[3],
                                              a4[0], a4[1], a4[2], a4[3],
                                              a6[0], a6[1], a6[2], a6[3]);
}

// --- Defect correction: adaptive (5x5 same-color median) ---
__global__ void kern_defect_adaptive(uint8_t* __restrict__ data,
                                       int w, int h, int bd, int pattern, float thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 2 || y < 2 || x >= w-2 || y >= h-2) return;
    int bc = dev_bayer_color(y, x, pattern);
    int mv = (bd <= 8) ? 255 : ((1 << bd) - 1);
    float maxv = static_cast<float>(mv);
    float center = dev_read_bayer(data, x, y, w, bd, maxv);
    // Collect same-color neighbors in 5x5 window (step-2)
    float neighbors[12]; int nc = 0;
    for (int dy = -2; dy <= 2; dy += 2) {
        for (int dx = -2; dx <= 2; dx += 2) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                if (dev_bayer_color(ny, nx, pattern) == bc) {
                    float nv = dev_read_bayer(data, nx, ny, w, bd, maxv);
                    // Insert sorted
                    int j = nc;
                    while (j > 0 && neighbors[j-1] > nv) { neighbors[j] = neighbors[j-1]; j--; }
                    neighbors[j] = nv;
                    nc++;
                }
            }
        }
    }
    if (nc > 0) {
        float median = neighbors[nc / 2];
        if (fabsf(center - median) > thresh)
            dev_write_bayer(data, x, y, w, bd, median, mv);
    }
}

void cuda_launch_defect_adaptive(uint8_t* d_data, int w, int h, int bd, int pattern,
                                  float threshold_norm) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_defect_adaptive<<<grid, block>>>(d_data, w, h, bd, pattern, threshold_norm);
}

// --- Defect correction: map-based ---
__global__ void kern_defect_map_based(uint8_t* __restrict__ data,
                                        int w, int h, int bd, int pattern,
                                        const int* __restrict__ dx, const int* __restrict__ dy,
                                        int map_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= map_count) return;
    int px = dx[idx], py = dy[idx];
    if (px < 2 || py < 2 || px >= w-2 || py >= h-2) return;
    int bc = dev_bayer_color(py, px, pattern);
    int mv = (bd <= 8) ? 255 : ((1 << bd) - 1);
    float maxv = static_cast<float>(mv);
    // Collect same-color neighbors
    float neighbors[12]; int nc = 0;
    for (int dy2 = -2; dy2 <= 2; dy2 += 2) {
        for (int dx2 = -2; dx2 <= 2; dx2 += 2) {
            if (dx2 == 0 && dy2 == 0) continue;
            int nx = px + dx2, ny = py + dy2;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                if (dev_bayer_color(ny, nx, pattern) == bc) {
                    float nv = dev_read_bayer(data, nx, ny, w, bd, maxv);
                    int j = nc;
                    while (j > 0 && neighbors[j-1] > nv) { neighbors[j] = neighbors[j-1]; j--; }
                    neighbors[j] = nv;
                    nc++;
                }
            }
        }
    }
    if (nc > 0)
        dev_write_bayer(data, px, py, w, bd, neighbors[nc / 2], mv);
}

void cuda_launch_defect_map_based(uint8_t* d_data, int w, int h, int bd, int pattern,
                                   const int* d_defect_x, const int* d_defect_y, int map_count) {
    dim3 block(256), grid((map_count + 255) / 256);
    kern_defect_map_based<<<grid, block>>>(d_data, w, h, bd, pattern,
                                             d_defect_x, d_defect_y, map_count);
}

// ============================================================
//  Extended RGB-domain dedicated kernels
// ============================================================

// --- Sharpen: high-pass filter (3x3) ---
__global__ void kern_sharpen_high_pass(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                         int w, int h, int bd, int mv, float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float fmaxv = static_cast<float>(mv);
    // 3x3 high-pass: [-1,-1,-1; -1,8,-1; -1,-1,-1] -> detail = sum/8
    for (int c = 0; c < 3; c++) {
        float center = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
        int xm1 = (x>0)?x-1:0, xp1 = (x<w-1)?x+1:w-1;
        int ym1 = (y>0)?y-1:0, yp1 = (y<h-1)?y+1:h-1;
        float nw = dev_read_pixel(in, xm1, ym1, w, bd, c, fmaxv);
        float n  = dev_read_pixel(in, x,   ym1, w, bd, c, fmaxv);
        float ne = dev_read_pixel(in, xp1, ym1, w, bd, c, fmaxv);
        float wv = dev_read_pixel(in, xm1, y,   w, bd, c, fmaxv);
        float ev = dev_read_pixel(in, xp1, y,   w, bd, c, fmaxv);
        float sw = dev_read_pixel(in, xm1, yp1, w, bd, c, fmaxv);
        float s  = dev_read_pixel(in, x,   yp1, w, bd, c, fmaxv);
        float se = dev_read_pixel(in, xp1, yp1, w, bd, c, fmaxv);
        float detail = (8.0f * center - (nw+n+ne+wv+ev+sw+s+se)) / 8.0f;
        dev_write_pixel(out, x, y, w, bd, c, center + amount * detail, mv);
    }
}

void cuda_launch_sharpen_high_pass(GpuWorkspace* ws, int w, int h, int bd, int mv, float amount) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_sharpen_high_pass<<<grid, block>>>(ws->d_input, ws->d_output, w, h, bd, mv, amount);
}

// --- Sharpen: adaptive (edge-aware multi-pass) ---
// Uses d_scratch for intermediate luma plane
__global__ void kern_adaptive_luma(const uint8_t* __restrict__ in, float* __restrict__ luma,
                                     int w, int h, int bd, int mv) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float fmaxv = static_cast<float>(mv);
    float r = dev_read_pixel(in, x, y, w, bd, 0, fmaxv);
    float g = dev_read_pixel(in, x, y, w, bd, 1, fmaxv);
    float b = dev_read_pixel(in, x, y, w, bd, 2, fmaxv);
    luma[y * w + x] = dev_luma(r, g, b);
}

__global__ void kern_adaptive_blur_luma(const float* __restrict__ luma, float* __restrict__ blur,
                                          int w, int h, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int radius = static_cast<int>(sigma);
    if (radius < 1) radius = 1;
    float sigma2 = 2.0f * sigma * sigma;
    float sum = 0.0f, wsum = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        int ny = y + dy;
        if (ny < 0) ny = 0; if (ny >= h) ny = h - 1;
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
            float wv = __expf(-static_cast<float>(dx*dx+dy*dy) / sigma2);
            sum += luma[ny * w + nx] * wv;
            wsum += wv;
        }
    }
    blur[y * w + x] = sum / wsum;
}

__global__ void kern_adaptive_apply(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                      const float* __restrict__ luma, const float* __restrict__ blur,
                                      int w, int h, int bd, int mv, float amount, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float fmaxv = static_cast<float>(mv);
    float l = luma[y * w + x];
    float bl = blur[y * w + x];
    float diff = l - bl;
    float abs_diff = fabsf(diff);
    float weight = (abs_diff > threshold) ? fminf(1.0f, (abs_diff - threshold) / (threshold + 0.05f)) : 0.0f;
    float l_out = l + amount * weight * diff;
    l_out = dev_clamp(l_out);
    float scale = (l > 0.001f) ? (l_out / l) : 1.0f;
    for (int c = 0; c < 3; c++) {
        float v = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
        dev_write_pixel(out, x, y, w, bd, c, v * scale, mv);
    }
}

void cuda_launch_sharpen_adaptive(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                   float amount, float threshold, float radius) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    size_t plane_sz = static_cast<size_t>(w) * h * sizeof(float);
    // Step 1: compute luma into d_scratch
    kern_adaptive_luma<<<grid, block>>>(ws->d_input, reinterpret_cast<float*>(ws->d_scratch),
                                          w, h, bd, mv);
    cudaDeviceSynchronize();
    // Step 2: allocate temp float buffer for blur
    float* d_blur;
    cudaMalloc(&d_blur, plane_sz);
    kern_adaptive_blur_luma<<<grid, block>>>(reinterpret_cast<float*>(ws->d_scratch), d_blur,
                                               w, h, radius);
    cudaDeviceSynchronize();
    // Step 3: apply
    kern_adaptive_apply<<<grid, block>>>(ws->d_input, ws->d_output,
                                           reinterpret_cast<float*>(ws->d_scratch), d_blur,
                                           w, h, bd, mv, amount, threshold);
    cudaFree(d_blur);
}

// --- Denoise: NLM (simplified 3x3 patches in 5x5 search window) ---
__global__ void kern_nlm(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                           int w, int h, int bd, int mv, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= w-1 || y >= h-1) {
        // Border: copy through
        if (x < w && y < h) {
            float fmaxv = static_cast<float>(mv);
            for (int c = 0; c < 3; c++) {
                float v = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
                dev_write_pixel(out, x, y, w, bd, c, v, mv);
            }
        }
        return;
    }
    float fmaxv = static_cast<float>(mv);
    float h2 = strength * strength * 9.0f; // scaling factor for patch size
    const int search_r = 2, patch_r = 1;  // 5x5 search, 3x3 patch
    const int kSearchWindow = (2*search_r+1)*(2*search_r+1); // = 25

    for (int c = 0; c < 3; c++) {
        float sum = 0.0f, wsum = 0.0f;
        float center_v = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
        float max_w = 0.0f;
        // Gather patch weights
        float weights[kSearchWindow]; int wi = 0;
        for (int sy = -search_r; sy <= search_r; sy++) {
            for (int sx = -search_r; sx <= search_r; sx++) {
                int cx = x + sx, cy = y + sy;
                if (cx < patch_r || cy < patch_r || cx >= w-patch_r || cy >= h-patch_r) {
                    weights[wi++] = 0.0f; continue;
                }
                float dist = 0.0f;
                for (int py = -patch_r; py <= patch_r; py++) {
                    for (int px = -patch_r; px <= patch_r; px++) {
                        float v1 = dev_read_pixel(in, x+px, y+py, w, bd, c, fmaxv);
                        float v2 = dev_read_pixel(in, cx+px, cy+py, w, bd, c, fmaxv);
                        float d = v1 - v2;
                        dist += d * d;
                    }
                }
                float wv = __expf(-dist / h2);
                weights[wi++] = wv;
                max_w = fmaxf(max_w, wv);
            }
        }
        // Weighted average
        wi = 0;
        for (int sy = -search_r; sy <= search_r; sy++) {
            for (int sx = -search_r; sx <= search_r; sx++) {
                int cx = x + sx, cy = y + sy;
                if (cx >= 0 && cx < w && cy >= 0 && cy < h && weights[wi] > max_w * 0.1f) {
                    float v = dev_read_pixel(in, cx, cy, w, bd, c, fmaxv);
                    sum += v * weights[wi];
                    wsum += weights[wi];
                }
                wi++;
            }
        }
        float result = (wsum > 0.0f) ? sum / wsum : center_v;
        dev_write_pixel(out, x, y, w, bd, c, result, mv);
    }
}

void cuda_launch_denoise_nlm(GpuWorkspace* ws, int w, int h, int bd, int mv, float strength) {
    dim3 block(16, 16), grid((w+15)/16, (h+15)/16);
    kern_nlm<<<grid, block>>>(ws->d_input, ws->d_output, w, h, bd, mv, strength);
}

// --- Denoise: wavelet (single-level Haar + soft threshold) ---
__global__ void kern_wavelet_haar_horiz(const uint8_t* __restrict__ in, float* __restrict__ tmp,
                                           int w, int h, int bd, int mv) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w/2 || y >= h) return;
    float fmaxv = static_cast<float>(mv);
    for (int c = 0; c < 3; c++) {
        int x0 = x * 2, x1 = x0 + 1;
        if (x1 >= w) x1 = w - 1;
        float v0 = dev_read_pixel(in, x0, y, w, bd, c, fmaxv);
        float v1 = dev_read_pixel(in, x1, y, w, bd, c, fmaxv);
        float avg = (v0 + v1) * 0.5f;
        float diff = (v0 - v1) * 0.5f;
        tmp[(y * w + x0) * 3 + c] = avg;
        tmp[(y * w + x1) * 3 + c] = diff;
    }
}

__global__ void kern_wavelet_haar_vert(const float* __restrict__ tmp, float* __restrict__ out,
                                          int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h/2) return;
    int y0 = y * 2, y1 = y0 + 1;
    if (y1 >= h) y1 = h - 1;
    for (int c = 0; c < 3; c++) {
        float v0 = tmp[(y0 * w + x) * 3 + c];
        float v1 = tmp[(y1 * w + x) * 3 + c];
        out[(y0 * w + x) * 3 + c] = (v0 + v1) * 0.5f;
        out[(y1 * w + x) * 3 + c] = (v0 - v1) * 0.5f;
    }
}

__global__ void kern_wavelet_threshold(float* __restrict__ data, int w, int h, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    // Only threshold detail coefficients (right half and bottom half)
    bool is_detail = (x >= w/2 || y >= h/2);
    if (is_detail) {
        for (int c = 0; c < 3; c++) {
            float v = data[(y * w + x) * 3 + c];
            if (fabsf(v) < threshold)
                data[(y * w + x) * 3 + c] = 0.0f;
            else
                data[(y * w + x) * 3 + c] = (v > 0.0f) ? v - threshold : v + threshold;
        }
    }
}

__global__ void kern_wavelet_ihaar_vert(float* __restrict__ data, float* __restrict__ tmp,
                                           int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h/2) return;
    int y0 = y * 2, y1 = y0 + 1;
    if (y1 >= h) y1 = h - 1;
    for (int c = 0; c < 3; c++) {
        float avg = data[(y0 * w + x) * 3 + c];
        float diff = data[(y1 * w + x) * 3 + c];
        tmp[(y0 * w + x) * 3 + c] = avg + diff;
        tmp[(y1 * w + x) * 3 + c] = avg - diff;
    }
}

__global__ void kern_wavelet_ihaar_horiz(const float* __restrict__ tmp, uint8_t* __restrict__ out,
                                            int w, int h, int bd, int mv) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w/2 || y >= h) return;
    for (int c = 0; c < 3; c++) {
        int x0 = x * 2, x1 = x0 + 1;
        if (x1 >= w) x1 = w - 1;
        float avg = tmp[(y * w + x0) * 3 + c];
        float diff = tmp[(y * w + x1) * 3 + c];
        float v0 = avg + diff, v1 = avg - diff;
        dev_write_pixel(out, x0, y, w, bd, c, v0, mv);
        dev_write_pixel(out, x1, y, w, bd, c, v1, mv);
    }
}

void cuda_launch_denoise_wavelet(GpuWorkspace* ws, int w, int h, int bd, int mv, float strength) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    size_t plane_sz = static_cast<size_t>(w) * h * 3 * sizeof(float);
    float* d_tmp;
    cudaMalloc(&d_tmp, plane_sz);
    float* d_tmp2;
    cudaMalloc(&d_tmp2, plane_sz);
    // Forward wavelet
    kern_wavelet_haar_horiz<<<grid, block>>>(ws->d_input, d_tmp, w, h, bd, mv);
    cudaDeviceSynchronize();
    dim3 grid_half((w+31)/32, (h/2+15)/16);
    kern_wavelet_haar_vert<<<grid_half, block>>>(d_tmp, d_tmp2, w, h);
    cudaDeviceSynchronize();
    // Threshold
    float thresh = strength * 0.15f;
    kern_wavelet_threshold<<<grid, block>>>(d_tmp2, w, h, thresh);
    cudaDeviceSynchronize();
    // Inverse wavelet
    kern_wavelet_ihaar_vert<<<grid_half, block>>>(d_tmp2, d_tmp, w, h);
    cudaDeviceSynchronize();
    kern_wavelet_ihaar_horiz<<<grid, block>>>(d_tmp, ws->d_output, w, h, bd, mv);
    cudaFree(d_tmp);
    cudaFree(d_tmp2);
}

// --- Denoise: Bayer-aware bilateral ---
// Note: This operates on RGB data (post-demosaic) simulating Bayer-domain denoising
// by applying per-pixel luma-guided filtering with reduced cross-talk
__global__ void kern_bayer_denoise(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                     int w, int h, int bd, int mv, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float fmaxv = static_cast<float>(mv);
    float sigma_r = strength * 0.15f;
    float sr2 = 2.0f * sigma_r * sigma_r;
    int radius = static_cast<int>(strength * 1.5f);
    if (radius < 1) radius = 1;
    if (radius > 3) radius = 3;
    for (int c = 0; c < 3; c++) {
        float center = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
        float sum = 0.0f, wsum = 0.0f;
        for (int dy = -radius; dy <= radius; dy++) {
            int ny = y + dy;
            if (ny < 0) ny = 0; if (ny >= h) ny = h - 1;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
                float neighbor = dev_read_pixel(in, nx, ny, w, bd, c, fmaxv);
                float dr = center - neighbor;
                float range_w = __expf(-dr * dr / sr2);
                // Spatial weight: prefer same-color Bayer positions
                int dist = abs(dx) + abs(dy);
                float spatial_w = (dist <= 1) ? 1.0f : (1.0f / static_cast<float>(dist));
                float wv = spatial_w * range_w;
                sum += neighbor * wv;
                wsum += wv;
            }
        }
        float result = (wsum > 0.0f) ? sum / wsum : center;
        dev_write_pixel(out, x, y, w, bd, c, result, mv);
    }
}

void cuda_launch_denoise_bayer(GpuWorkspace* ws, int w, int h, int bd, int mv, float strength) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_bayer_denoise<<<grid, block>>>(ws->d_input, ws->d_output, w, h, bd, mv, strength);
}

// --- Highlight reconstruct: gradient-based (5x5 same-channel interpolation) ---
__global__ void kern_highlight_gradient(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                          int w, int h, int bd, int mv, float clip_thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 2 || y < 2 || x >= w-2 || y >= h-2) {
        // Border: copy
        if (x < w && y < h) {
            float fmaxv = static_cast<float>(mv);
            for (int c = 0; c < 3; c++) {
                float v = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
                dev_write_pixel(out, x, y, w, bd, c, v, mv);
            }
        }
        return;
    }
    float fmaxv = static_cast<float>(mv);
    float thr = clip_thresh;
    for (int c = 0; c < 3; c++) {
        float val = dev_read_pixel(in, x, y, w, bd, c, fmaxv);
        if (val < thr) {
            dev_write_pixel(out, x, y, w, bd, c, val, mv);
            continue;
        }
        // Search 5x5 window for unclipped same-channel neighbors
        float wsum = 0.0f, sum = 0.0f;
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                float nv = dev_read_pixel(in, nx, ny, w, bd, c, fmaxv);
                if (nv < thr) {
                    float dist = sqrtf(static_cast<float>(dx*dx + dy*dy)) + 0.1f;
                    float wv = 1.0f / dist;
                    sum += nv * wv;
                    wsum += wv;
                }
            }
        }
        float est = (wsum > 0.0f) ? sum / wsum : val;
        float result = val * 0.3f + est * 0.7f;
        dev_write_pixel(out, x, y, w, bd, c, result, mv);
    }
}

void cuda_launch_highlight_gradient(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                     float clip_thresh) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_highlight_gradient<<<grid, block>>>(ws->d_input, ws->d_output, w, h, bd, mv, clip_thresh);
}
