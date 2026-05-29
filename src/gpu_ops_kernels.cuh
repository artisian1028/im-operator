#ifndef GPU_OPS_KERNELS_CUH
#define GPU_OPS_KERNELS_CUH

#include <cstdint>

struct GpuWorkspace {
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    uint8_t* d_scratch = nullptr;   // for multi-pass ops
    float* d_params = nullptr;      // parameter buffer for pipeline ops
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    int max_val = 0;

    GpuWorkspace() = default;
    ~GpuWorkspace() { release(); }
    GpuWorkspace(const GpuWorkspace&) = delete;
    GpuWorkspace& operator=(const GpuWorkspace&) = delete;

    bool ensure(int w, int h, int bd, size_t param_bytes);
    void release();
};

// ============================================================
//  GPU Pipeline: chainable per-pixel operations
// ============================================================

// Operation types that can be chained in a single GPU pass
enum class GpuOp : int {
    TONE_GAMMA = 0,
    TONE_S_CURVE,
    TONE_LEVELS,
    CCM_3X3,
    SATURATION_HSL,
    SATURATION_VIBRANCE,
    COLOR_TEMP_MULTIPLY,
    WHITE_BALANCE_MANUAL,
    BLACK_LEVEL_GLOBAL,
    LENS_SHADING_POLY,
    HIGHLIGHT_RECON_CHANNEL,
    SHARPEN_UNSHARP_MASK,
    LOCAL_CONTRAST_CLARITY,
    REINHARD_TONE_MAP,
    FILMIC_TONE_MAP,
    CCM_4X3,
    SATURATION_CHANNEL_MIXER,
    SATURATION_SELECTIVE,
    COLOR_TEMP_AUTO_WB,
    WB_GRAY_WORLD,
    WB_WHITE_PATCH,
    WB_SHADE_OF_GRAY,
    OP_COUNT
};

// Per-operation parameters (max 64 bytes per op for alignment)
struct GpuOpParam {
    GpuOp op;
    float params[15];  // flexible parameter storage
};

// Launch the chained pipeline kernel
void cuda_launch_gpu_pipeline(GpuWorkspace* ws,
                              const GpuOpParam* ops, int op_count,
                              int width, int height, int bit_depth, int max_val);

// ============================================================
//  Dedicated kernels for compute-heavy operations
// ============================================================

// Denoise
void cuda_launch_denoise_bilateral(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                    float sigma_s, float sigma_r, int radius);
void cuda_launch_denoise_gaussian(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                   float sigma, int radius);
void cuda_launch_denoise_median(GpuWorkspace* ws, int w, int h, int bd, int mv, int radius);

// LUT (3D trilinear)
void cuda_launch_lut_apply(GpuWorkspace* ws, int w, int h, int bd, int mv,
                            const float* h_lut_data, int lut_size);

// Sharpen
void cuda_launch_sharpen_unsharp(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                  float amount, float sigma, int radius);
void cuda_launch_sharpen_laplacian(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                    float amount, float radius);

// Local Contrast
void cuda_launch_local_contrast(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                 float amount, float radius);

// ============================================================
//  Bayer-domain kernels (single-channel data)
// ============================================================

// All Bayer kernels take pattern as int: 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG
// and operate on uint8_t* data in-place (single channel per pixel).

// Black level: subtract per-channel offsets
void cuda_launch_black_level_per_channel(uint8_t* d_data, int w, int h, int bd, int pattern,
                                          const float offsets[4]);
void cuda_launch_black_level_global(uint8_t* d_data, int w, int h, int bd,
                                     float offset);

// Lens shading: polynomial gain correction (per-channel a2/a4/a6)
void cuda_launch_lens_shading_poly(uint8_t* d_data, int w, int h, int bd, int pattern,
                                    float cx, float cy,
                                    const float a2[4], const float a4[4], const float a6[4]);

// Defect correction: adaptive (5x5 same-color median) and map-based
void cuda_launch_defect_adaptive(uint8_t* d_data, int w, int h, int bd, int pattern,
                                  float threshold_norm);
void cuda_launch_defect_map_based(uint8_t* d_data, int w, int h, int bd, int pattern,
                                   const int* d_defect_x, const int* d_defect_y, int map_count);

// ============================================================
//  Extended RGB-domain dedicated kernels
// ============================================================

// Sharpen: high-pass filter (3x3 kernel)
void cuda_launch_sharpen_high_pass(GpuWorkspace* ws, int w, int h, int bd, int mv, float amount);

// Sharpen: adaptive (edge-aware multi-pass)
void cuda_launch_sharpen_adaptive(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                   float amount, float threshold, float radius);

// Denoise: NLM (Non-Local Means, simplified 3x3 patches in 5x5 search window)
void cuda_launch_denoise_nlm(GpuWorkspace* ws, int w, int h, int bd, int mv,
                              float strength);

// Denoise: wavelet (single-level Haar with soft thresholding)
void cuda_launch_denoise_wavelet(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                  float strength);

// Denoise: Bayer-aware (bilateral on same-color neighbors only)
void cuda_launch_denoise_bayer(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                float strength);

// Highlight reconstruct: gradient-based (5x5 same-channel interpolation)
void cuda_launch_highlight_gradient(GpuWorkspace* ws, int w, int h, int bd, int mv,
                                     float clip_thresh);

#endif
