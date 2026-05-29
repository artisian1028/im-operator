#include "gpu_ops_kernels.cuh"
#include "tone/algorithms.hpp"
#include "ccm/algorithms.hpp"
#include "saturation/algorithms.hpp"
#include "color_temp/algorithms.hpp"
#include "white_balance/algorithms.hpp"
#include "black_level/algorithms.hpp"
#include "lens_shading/algorithms.hpp"
#include "highlight_reconstruct/algorithms.hpp"
#include "defect_correct/algorithms.hpp"
#include "denoise/algorithms.hpp"
#include "lut/algorithms.hpp"
#include "sharpen/algorithms.hpp"
#include "local_contrast/algorithms.hpp"
#include "hdr/algorithms.hpp"
#include "imop/types.hpp"

#ifdef IM_OPERATOR_HAS_CUDA

#include <cuda_runtime.h>
#include <mutex>
#include <cstring>
#include <vector>

// Check a CUDA API call; return the provided error value on failure.
#define CUDA_CHECK_OR_RETURN(call, err_val) do { \
    if ((call) != cudaSuccess) return (err_val); \
} while(0)

using imop::BayerPattern;

namespace {

static bool s_has_cuda = false;
static std::once_flag s_cuda_once;

bool has_cuda_shared() {
    std::call_once(s_cuda_once, [] {
        int count = 0;
        s_has_cuda = (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
    });
    return s_has_cuda;
}

static thread_local GpuWorkspace s_ws;

size_t rgb_bytes(int w, int h, int bd) {
    size_t n = static_cast<size_t>(w) * h * 3;
    return (bd <= 8) ? n : n * 2;
}

} // anonymous namespace

// ============================================================
//  Externally-linkable has_cuda (shared by all modules)
// ============================================================

// Each module namespace gets a has_cuda() via these forwarders

// ============================================================
//  Module-specific GPU dispatch
// ============================================================

// --- Denoise ---
namespace denoise {

bool has_cuda() { return has_cuda_shared(); }

DenoiseError process_denoise_cuda(const uint8_t* input, uint8_t* output,
                                   int w, int h, int ch, DenoiseAlgorithm algo,
                                   int bd, float strength) {
    (void)ch;
    if (!has_cuda_shared()) return DenoiseError::InternalError;
    int mv = (1 << std::min(bd, 16)) - 1;
    if (!s_ws.ensure(w, h, bd, 0)) return DenoiseError::InternalError;

    size_t sz = rgb_bytes(w, h, bd);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), DenoiseError::InternalError);

    switch (algo) {
        case DenoiseAlgorithm::BILATERAL:
            cuda_launch_denoise_bilateral(&s_ws, w, h, bd, mv,
                                          2.0f * strength, 0.1f * strength, 2);
            break;
        case DenoiseAlgorithm::GAUSSIAN:
            cuda_launch_denoise_gaussian(&s_ws, w, h, bd, mv,
                                          strength * 2.0f, static_cast<int>(strength * 2.0f));
            break;
        case DenoiseAlgorithm::MEDIAN:
            cuda_launch_denoise_median(&s_ws, w, h, bd, mv,
                                        (strength > 1.0f) ? 2 : 1);
            break;
        case DenoiseAlgorithm::NLM:
            cuda_launch_denoise_nlm(&s_ws, w, h, bd, mv, strength);
            break;
        case DenoiseAlgorithm::WAVELET:
            cuda_launch_denoise_wavelet(&s_ws, w, h, bd, mv, strength);
            break;
        case DenoiseAlgorithm::BAYER_DENOISE:
            cuda_launch_denoise_bayer(&s_ws, w, h, bd, mv, strength);
            break;
        default:
            return DenoiseError::InternalError; // fallback to CPU
    }

    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), DenoiseError::InternalError);
    return DenoiseError::Ok;
}

} // namespace denoise

// --- LUT ---
namespace lut {

bool has_cuda() { return has_cuda_shared(); }

LUTError process_lut_cuda(const uint8_t* input, uint8_t* output,
                           int w, int h, int ch, int bd,
                           const LUT3D& lut_data) {
    (void)ch;
    if (!has_cuda_shared()) return LUTError::InternalError;
    if (lut_data.empty()) return LUTError::InternalError;
    int mv = (1 << std::min(bd, 16)) - 1;
    if (!s_ws.ensure(w, h, bd, 0)) return LUTError::InternalError;

    size_t sz = rgb_bytes(w, h, bd);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), LUTError::InternalError);

    cuda_launch_lut_apply(&s_ws, w, h, bd, mv, lut_data.data.data(), lut_data.size);

    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), LUTError::InternalError);
    return LUTError::Ok;
}

} // namespace lut

// --- Sharpen ---
namespace sharpen {

bool has_cuda() { return has_cuda_shared(); }

SharpenError process_sharpen_cuda(const uint8_t* input, uint8_t* output,
                                   int w, int h, int ch, SharpenAlgorithm algo,
                                   int bd, const SharpenParams& params) {
    (void)ch;
    if (!has_cuda_shared()) return SharpenError::InternalError;
    int mv = (1 << std::min(bd, 16)) - 1;
    if (!s_ws.ensure(w, h, bd, 0)) return SharpenError::InternalError;

    size_t sz = rgb_bytes(w, h, bd);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), SharpenError::InternalError);

    switch (algo) {
        case SharpenAlgorithm::UNSHARP_MASK:
            cuda_launch_sharpen_unsharp(&s_ws, w, h, bd, mv,
                                         params.amount, params.radius, 1);
            break;
        case SharpenAlgorithm::LAPLACIAN:
            cuda_launch_sharpen_laplacian(&s_ws, w, h, bd, mv,
                                           params.amount, params.radius);
            break;
        case SharpenAlgorithm::HIGH_PASS:
            cuda_launch_sharpen_high_pass(&s_ws, w, h, bd, mv, params.amount);
            break;
        case SharpenAlgorithm::ADAPTIVE:
            cuda_launch_sharpen_adaptive(&s_ws, w, h, bd, mv,
                                           params.amount, params.threshold, params.radius);
            break;
        default:
            return SharpenError::InternalError; // fallback
    }

    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), SharpenError::InternalError);
    return SharpenError::Ok;
}

} // namespace sharpen

// --- Local Contrast ---
namespace local_contrast {

bool has_cuda() { return has_cuda_shared(); }

LocalContrastError process_local_contrast_cuda(const uint8_t* input, uint8_t* output,
                                                int w, int h, int ch, int bd,
                                                const LocalContrastParams& params) {
    (void)ch;
    if (!has_cuda_shared()) return LocalContrastError::InternalError;
    int mv = (1 << std::min(bd, 16)) - 1;
    if (!s_ws.ensure(w, h, bd, 0)) return LocalContrastError::InternalError;

    size_t sz = rgb_bytes(w, h, bd);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), LocalContrastError::InternalError);

    cuda_launch_local_contrast(&s_ws, w, h, bd, mv, params.amount, params.radius);

    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), LocalContrastError::InternalError);
    return LocalContrastError::Ok;
}

} // namespace local_contrast

// --- GPU Pipeline for Tone, CCM, Saturation, ColorTemp, WhiteBalance ---

namespace {

// Build pipeline ops from algorithm params
void build_tone_ops(tone::ToneAlgorithm algo, const tone::ToneParams& p,
                    GpuOpParam* ops, int& count) {
    switch (algo) {
        case tone::ToneAlgorithm::GAMMA:
            ops[count].op = GpuOp::TONE_GAMMA;
            ops[count].params[0] = 1.0f / p.gamma;
            count++; break;
        case tone::ToneAlgorithm::S_CURVE:
            ops[count].op = GpuOp::TONE_S_CURVE;
            ops[count].params[0] = p.contrast;
            ops[count].params[1] = p.shadows;
            ops[count].params[2] = p.highlights;
            count++; break;
        case tone::ToneAlgorithm::LEVELS:
            ops[count].op = GpuOp::TONE_LEVELS;
            ops[count].params[0] = p.black_point;
            ops[count].params[1] = p.white_point;
            ops[count].params[2] = p.mid_point;
            ops[count].params[3] = p.gamma;
            count++; break;
        case tone::ToneAlgorithm::SHADOWS_HIGHLIGHTS:
            ops[count].op = GpuOp::TONE_S_CURVE;
            ops[count].params[0] = 0.0f;
            ops[count].params[1] = p.shadows;
            ops[count].params[2] = p.highlights;
            count++; break;
        case tone::ToneAlgorithm::CURVES_3POINT:
            // CURVES_3POINT uses Bezier interpolation; approximate via S-curve + gamma
            ops[count].op = GpuOp::TONE_GAMMA;
            ops[count].params[0] = 1.0f / (p.gamma + 0.001f);
            count++;
            ops[count].op = GpuOp::TONE_S_CURVE;
            ops[count].params[0] = p.contrast;
            ops[count].params[1] = p.shadows;
            ops[count].params[2] = p.highlights;
            count++; break;
        default: break;
    }
}

void build_ccm_ops(const float* m, GpuOpParam* ops, int& count) {
    ops[count].op = GpuOp::CCM_3X3;
    for (int i = 0; i < 9; i++) ops[count].params[i] = m[i];
    count++;
}

void build_saturation_ops(float sat, float vib, GpuOpParam* ops, int& count) {
    const float eps = 1e-5f;
    bool vib_active = (vib - 1.0f) > eps || (1.0f - vib) > eps;
    bool sat_active = (sat - 1.0f) > eps || (1.0f - sat) > eps;
    if (vib_active) {
        ops[count].op = GpuOp::SATURATION_VIBRANCE;
        ops[count].params[0] = vib;
        count++;
    }
    if (sat_active) {
        ops[count].op = GpuOp::SATURATION_HSL;
        ops[count].params[0] = sat;
        count++;
    }
}

void build_color_temp_ops(float rm, float gm, float bm, GpuOpParam* ops, int& count) {
    ops[count].op = GpuOp::COLOR_TEMP_MULTIPLY;
    ops[count].params[0] = rm; ops[count].params[1] = gm; ops[count].params[2] = bm;
    count++;
}

void build_wb_ops(float rg, float gg, float bg, GpuOpParam* ops, int& count) {
    ops[count].op = GpuOp::WHITE_BALANCE_MANUAL;
    ops[count].params[0] = rg; ops[count].params[1] = gg; ops[count].params[2] = bg;
    count++;
}

} // anonymous namespace

// Tone GPU dispatch
namespace tone {
bool has_cuda() { return has_cuda_shared(); }

ToneError process_tone_cuda(const uint8_t* input, uint8_t* output,
                             int w, int h, int ch, ToneAlgorithm algo,
                             int bd, const ToneParams& params) {
    (void)ch;
    if (!has_cuda_shared()) return ToneError::InternalError;
    int mv = (1 << std::min(bd, 16)) - 1;
    if (!s_ws.ensure(w, h, bd, 0)) return ToneError::InternalError;

    GpuOpParam ops[4]; int op_count = 0;
    build_tone_ops(algo, params, ops, op_count);
    if (op_count == 0) return ToneError::InternalError;

    size_t sz = rgb_bytes(w, h, bd);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), ToneError::InternalError);

    cuda_launch_gpu_pipeline(&s_ws, ops, op_count, w, h, bd, mv);

    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), ToneError::InternalError);
    return ToneError::Ok;
}
} // namespace tone

// CCM GPU dispatch
namespace ccm {
bool has_cuda() { return has_cuda_shared(); }

CCMError process_ccm_cuda(const uint8_t* input, uint8_t* output,
                           int w, int h, int ch, int bd, const float* matrix) {
    (void)ch;
    if (!has_cuda_shared() || !matrix) return CCMError::InternalError;
    int mv = (1 << std::min(bd, 16)) - 1;
    if (!s_ws.ensure(w, h, bd, 0)) return CCMError::InternalError;

    GpuOpParam ops[1]; int oc = 0;
    if (matrix) { build_ccm_ops(matrix, ops, oc); }
    if (oc == 0) return CCMError::InternalError;

    size_t sz = rgb_bytes(w, h, bd);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), CCMError::InternalError);
    cuda_launch_gpu_pipeline(&s_ws, ops, oc, w, h, bd, mv);
    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), CCMError::InternalError);
    return CCMError::Ok;
}
} // namespace ccm

// White Balance GPU dispatch
namespace white_balance {
bool has_cuda() { return has_cuda_shared(); }

WhiteBalanceError process_white_balance_cuda(const uint8_t* input, uint8_t* output,
                                              int w, int h, int ch, int bd,
                                              float rg, float gg, float bg) {
    (void)ch;
    if (!has_cuda_shared()) return WhiteBalanceError::InternalError;
    int mv = (1 << std::min(bd, 16)) - 1;
    if (!s_ws.ensure(w, h, bd, 0)) return WhiteBalanceError::InternalError;

    GpuOpParam ops[1]; int oc = 0;
    build_wb_ops(rg, gg, bg, ops, oc);

    size_t sz = rgb_bytes(w, h, bd);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), WhiteBalanceError::InternalError);
    cuda_launch_gpu_pipeline(&s_ws, ops, oc, w, h, bd, mv);
    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), WhiteBalanceError::InternalError);
    return WhiteBalanceError::Ok;
}
} // namespace white_balance

// Saturation GPU dispatch
namespace saturation {
bool has_cuda() { return has_cuda_shared(); }

SaturationError process_saturation_cuda(const uint8_t* input, uint8_t* output,
                                         int w, int h, int ch, int bd,
                                         float sat, float vib) {
    (void)ch;
    if (!has_cuda_shared()) return SaturationError::InternalError;
    int mv = (1 << std::min(bd, 16)) - 1;
    if (!s_ws.ensure(w, h, bd, 0)) return SaturationError::InternalError;

    GpuOpParam ops[1]; int oc = 0;
    build_saturation_ops(sat, vib, ops, oc);
    if (oc == 0) return SaturationError::InternalError;

    size_t sz = rgb_bytes(w, h, bd);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), SaturationError::InternalError);
    cuda_launch_gpu_pipeline(&s_ws, ops, oc, w, h, bd, mv);
    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), SaturationError::InternalError);
    return SaturationError::Ok;
}
} // namespace saturation

// Color Temp GPU dispatch
namespace color_temp {
bool has_cuda() { return has_cuda_shared(); }

ColorTempError process_color_temp_cuda(const uint8_t* input, uint8_t* output,
                                        int w, int h, int ch, int bd,
                                        float rm, float gm, float bm) {
    (void)ch;
    if (!has_cuda_shared()) return ColorTempError::InternalError;
    int mv = (1 << std::min(bd, 16)) - 1;
    if (!s_ws.ensure(w, h, bd, 0)) return ColorTempError::InternalError;

    GpuOpParam ops[1]; int oc = 0;
    build_color_temp_ops(rm, gm, bm, ops, oc);

    size_t sz = rgb_bytes(w, h, bd);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), ColorTempError::InternalError);
    cuda_launch_gpu_pipeline(&s_ws, ops, oc, w, h, bd, mv);
    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), ColorTempError::InternalError);
    return ColorTempError::Ok;
}
} // namespace color_temp

// ============================================================
//  New module GPU dispatch functions (Tiers 2-4)
// ============================================================

// --- Black Level ---
namespace black_level {

bool has_cuda() { return has_cuda_shared(); }

BlackLevelError process_black_level_cuda(uint8_t* data,
                                          int width, int height,
                                          BayerPattern pattern,
                                          BlackLevelAlgorithm algorithm,
                                          int bit_depth,
                                          const BlackLevelParams& params) {
    if (!has_cuda_shared()) return BlackLevelError::InternalError;
    int pat = static_cast<int>(pattern);
    int w = width, h = height, bd = bit_depth;
    if (w <= 0 || h <= 0 || bd <= 0) return BlackLevelError::InternalError;
    size_t sz = static_cast<size_t>(w) * h * ((bd <= 8) ? 1 : 2);
    uint8_t* d_data;
    if (cudaMalloc(&d_data, sz) != cudaSuccess) return BlackLevelError::InternalError;
    CUDA_CHECK_OR_RETURN(cudaMemcpy(d_data, data, sz, cudaMemcpyHostToDevice), BlackLevelError::InternalError);

    if (algorithm == BlackLevelAlgorithm::GLOBAL) {
        float offset = params.r_offset;
        cuda_launch_black_level_global(d_data, w, h, bd, offset);
    } else {
        float offsets[4] = { params.r_offset, params.gr_offset,
                             params.gb_offset, params.b_offset };
        cuda_launch_black_level_per_channel(d_data, w, h, bd, pat, offsets);
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(data, d_data, sz, cudaMemcpyDeviceToHost), BlackLevelError::InternalError);
    cudaFree(d_data);
    return BlackLevelError::Ok;
}

} // namespace black_level

// --- Lens Shading ---
namespace lens_shading {

bool has_cuda() { return has_cuda_shared(); }

LensShadingError process_lens_shading_cuda(uint8_t* data,
                                            int width, int height,
                                            BayerPattern pattern,
                                            LensShadingAlgorithm algorithm,
                                            int bit_depth,
                                            const LensShadingParams& params) {
    if (!has_cuda_shared()) return LensShadingError::InternalError;
    if (algorithm != LensShadingAlgorithm::POLYNOMIAL)
        return LensShadingError::InternalError; // FLAT_FIELD needs CPU
    int pat = static_cast<int>(pattern);
    int w = width, h = height, bd = bit_depth;
    size_t sz = static_cast<size_t>(w) * h * ((bd <= 8) ? 1 : 2);
    uint8_t* d_data;
    if (cudaMalloc(&d_data, sz) != cudaSuccess) return LensShadingError::InternalError;
    CUDA_CHECK_OR_RETURN(cudaMemcpy(d_data, data, sz, cudaMemcpyHostToDevice), LensShadingError::InternalError);

    float a2[4] = { params.r_coef.a2, params.gr_coef.a2,
                    params.gb_coef.a2, params.b_coef.a2 };
    float a4[4] = { params.r_coef.a4, params.gr_coef.a4,
                    params.gb_coef.a4, params.b_coef.a4 };
    float a6[4] = { params.r_coef.a6, params.gr_coef.a6,
                    params.gb_coef.a6, params.b_coef.a6 };
    cuda_launch_lens_shading_poly(d_data, w, h, bd, pat,
                                   params.center_x, params.center_y,
                                   a2, a4, a6);
    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(data, d_data, sz, cudaMemcpyDeviceToHost), LensShadingError::InternalError);
    cudaFree(d_data);
    return LensShadingError::Ok;
}

} // namespace lens_shading

// --- Highlight Reconstruct ---
namespace highlight_reconstruct {

bool has_cuda() { return has_cuda_shared(); }

HighlightReconstructError process_highlight_reconstruct_cuda(const uint8_t* input, uint8_t* output,
                                                              int width, int height, int channels,
                                                              HighlightReconstructAlgorithm algorithm,
                                                              int bit_depth,
                                                              const HighlightReconstructParams& params) {
    (void)channels;
    if (!has_cuda_shared()) return HighlightReconstructError::InternalError;
    int mv = (1 << std::min(bit_depth, 16)) - 1;
    if (!s_ws.ensure(width, height, bit_depth, 0)) return HighlightReconstructError::InternalError;

    size_t sz = rgb_bytes(width, height, bit_depth);
    CUDA_CHECK_OR_RETURN(cudaMemcpy(s_ws.d_input, input, sz, cudaMemcpyHostToDevice), HighlightReconstructError::InternalError);

    if (algorithm == HighlightReconstructAlgorithm::CHANNEL_GUIDED) {
        // Use pipeline op
        GpuOpParam ops[1]; ops[0].op = GpuOp::HIGHLIGHT_RECON_CHANNEL;
        ops[0].params[0] = params.threshold;
        cuda_launch_gpu_pipeline(&s_ws, ops, 1, width, height, bit_depth, mv);
    } else {
        // Gradient-based: dedicated kernel
        cuda_launch_highlight_gradient(&s_ws, width, height, bit_depth, mv, params.threshold);
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(output, s_ws.d_output, sz, cudaMemcpyDeviceToHost), HighlightReconstructError::InternalError);
    return HighlightReconstructError::Ok;
}

} // namespace highlight_reconstruct

// --- Defect Correct ---
namespace defect_correct {

bool has_cuda() { return has_cuda_shared(); }

DefectCorrectError process_defect_correct_cuda(uint8_t* data,
                                                int width, int height,
                                                BayerPattern pattern,
                                                DefectCorrectAlgorithm algorithm,
                                                int bit_depth,
                                                const DefectCorrectParams& params) {
    if (!has_cuda_shared()) return DefectCorrectError::InternalError;
    int pat = static_cast<int>(pattern);
    int w = width, h = height, bd = bit_depth;
    size_t sz = static_cast<size_t>(w) * h * ((bd <= 8) ? 1 : 2);
    uint8_t* d_data;
    if (cudaMalloc(&d_data, sz) != cudaSuccess) return DefectCorrectError::InternalError;
    CUDA_CHECK_OR_RETURN(cudaMemcpy(d_data, data, sz, cudaMemcpyHostToDevice), DefectCorrectError::InternalError);

    if (algorithm == DefectCorrectAlgorithm::ADAPTIVE) {
        cuda_launch_defect_adaptive(d_data, w, h, bd, pat, params.threshold);
    } else {
        // Map-based
        if (!params.map || params.map_count <= 0) {
            cudaFree(d_data);
            return DefectCorrectError::InternalError;
        }
        int* d_dx, *d_dy;
        size_t map_sz = static_cast<size_t>(params.map_count) * sizeof(int);
        if (cudaMalloc(&d_dx, map_sz) != cudaSuccess) { cudaFree(d_data); return DefectCorrectError::InternalError; }
        if (cudaMalloc(&d_dy, map_sz) != cudaSuccess) { cudaFree(d_data); cudaFree(d_dx); return DefectCorrectError::InternalError; }
        // Use std::vector for exception-safe host allocation
        std::vector<int> h_dx(params.map_count);
        std::vector<int> h_dy(params.map_count);
        for (int i = 0; i < params.map_count; i++) {
            h_dx[i] = params.map[i].x;
            h_dy[i] = params.map[i].y;
        }
        CUDA_CHECK_OR_RETURN(cudaMemcpy(d_dx, h_dx.data(), map_sz, cudaMemcpyHostToDevice), DefectCorrectError::InternalError);
        CUDA_CHECK_OR_RETURN(cudaMemcpy(d_dy, h_dy.data(), map_sz, cudaMemcpyHostToDevice), DefectCorrectError::InternalError);
        cuda_launch_defect_map_based(d_data, w, h, bd, pat, d_dx, d_dy, params.map_count);
        cudaFree(d_dx); cudaFree(d_dy);
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_OR_RETURN(cudaMemcpy(data, d_data, sz, cudaMemcpyDeviceToHost), DefectCorrectError::InternalError);
    cudaFree(d_data);
    return DefectCorrectError::Ok;
}

} // namespace defect_correct

#else // !IM_OPERATOR_HAS_CUDA

// Stubs for all modules
namespace denoise    { bool has_cuda() { return false; } }
namespace lut        { bool has_cuda() { return false; } }
namespace sharpen    { bool has_cuda() { return false; } }
namespace local_contrast { bool has_cuda() { return false; } }
namespace tone       { bool has_cuda() { return false; } }
namespace ccm        { bool has_cuda() { return false; } }
namespace white_balance { bool has_cuda() { return false; } }
namespace saturation { bool has_cuda() { return false; } }
namespace color_temp { bool has_cuda() { return false; } }
namespace black_level { bool has_cuda() { return false; } }
namespace lens_shading { bool has_cuda() { return false; } }
namespace highlight_reconstruct { bool has_cuda() { return false; } }
namespace defect_correct { bool has_cuda() { return false; } }

#endif
