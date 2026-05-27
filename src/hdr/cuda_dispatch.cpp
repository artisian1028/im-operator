#include "hdr/algorithms.hpp"
#include "hdr_kernels.cuh"

#ifdef IM_OPERATOR_HAS_CUDA

#include <cuda_runtime.h>
#include <mutex>
#include <cstring>
#include <algorithm>

namespace hdr {

// --- CUDA detection (call_once cached) ---

static bool s_has_cuda = false;
static std::once_flag s_cuda_once;

bool has_cuda() {
    std::call_once(s_cuda_once, [] {
        int count = 0;
        cudaError_t e = cudaGetDeviceCount(&count);
        s_has_cuda = (e == cudaSuccess && count > 0);
    });
    return s_has_cuda;
}

// --- Workspace ---

HdrCudaWorkspace::HdrCudaWorkspace(HdrCudaWorkspace&& o) noexcept
    : d_input(o.d_input), d_output(o.d_output), d_fbuf(o.d_fbuf),
      h_input_pinned(o.h_input_pinned), h_output_pinned(o.h_output_pinned),
      h_input_bytes(o.h_input_bytes), h_output_bytes(o.h_output_bytes),
      width(o.width), height(o.height), bit_depth(o.bit_depth), max_val(o.max_val) {
    o.d_input = nullptr; o.d_output = nullptr; o.d_fbuf = nullptr;
    o.h_input_pinned = nullptr; o.h_output_pinned = nullptr;
    o.width = 0; o.height = 0; o.bit_depth = 0; o.max_val = 0;
}

bool HdrCudaWorkspace::ensure(int w, int h, int bd) {
    if (width == w && height == h && bit_depth == bd && d_input) return true;

    release();

    width = w;
    height = h;
    bit_depth = bd;
    max_val = (1u << std::min(bd, 16)) - 1;
    if (bd <= 0) max_val = 255;

    size_t total = static_cast<size_t>(w) * h;
    size_t bytes_per_pixel = (bd <= 8) ? 1 : 2;
    size_t rgb_bytes = total * 3 * bytes_per_pixel;

    cudaError_t e;
    e = cudaMalloc(&d_input, rgb_bytes);
    if (e != cudaSuccess) { release(); return false; }
    e = cudaMalloc(&d_output, rgb_bytes);
    if (e != cudaSuccess) { release(); return false; }
    e = cudaMalloc(&d_fbuf, total * sizeof(float));
    if (e != cudaSuccess) { release(); return false; }

    e = cudaMallocHost(&h_input_pinned, rgb_bytes);
    if (e != cudaSuccess) { release(); return false; }
    h_input_bytes = rgb_bytes;
    e = cudaMallocHost(&h_output_pinned, rgb_bytes);
    if (e != cudaSuccess) { release(); return false; }
    h_output_bytes = rgb_bytes;

    return true;
}

void HdrCudaWorkspace::release() {
    if (d_input)   { cudaFree(d_input);   d_input = nullptr; }
    if (d_output)  { cudaFree(d_output);  d_output = nullptr; }
    if (d_fbuf)    { cudaFree(d_fbuf);    d_fbuf = nullptr; }
    if (h_input_pinned)  { cudaFreeHost(h_input_pinned);  h_input_pinned = nullptr; }
    if (h_output_pinned) { cudaFreeHost(h_output_pinned); h_output_pinned = nullptr; }
    width = 0; height = 0; bit_depth = 0; max_val = 0;
}

// --- Thread-local workspace ---

static thread_local HdrCudaWorkspace s_ws;

// --- Main CUDA dispatch ---

HdrError process_hdr_cuda(const uint8_t* input, uint8_t* output,
                           int width, int height, int channels,
                           HdrAlgorithm algorithm, int bit_depth,
                           const HdrParams& params) {
    (void)channels;

    if (!has_cuda()) return HdrError::CudaNotAvailable;
    if (width < 1 || height < 1) return HdrError::InvalidDimensions;

    // Float32 not supported on GPU; transparent fallback to CPU
    if (bit_depth == 0) return HdrError::InternalError;

    // Ensure workspace
    if (!s_ws.ensure(width, height, bit_depth)) return HdrError::InternalError;

    // Copy input to device
    size_t rgb_bytes = static_cast<size_t>(width) * height * 3;
    if (bit_depth > 8) rgb_bytes *= 2;

    std::memcpy(s_ws.h_input_pinned, input, rgb_bytes);
    cudaMemcpy(s_ws.d_input, s_ws.h_input_pinned, rgb_bytes, cudaMemcpyHostToDevice);

    int max_val = s_ws.max_val;
    float sat = std::max(0.0f, std::min(2.0f, params.saturation));
    float inv_gamma = 1.0f / std::max(0.5f, std::min(4.0f, params.gamma));
    float exposure_mul = std::pow(2.0f, std::max(-8.0f, std::min(8.0f, params.exposure)));
    float strength = std::max(0.0f, std::min(2.0f, params.strength));
    float key = std::max(0.01f, std::min(1.0f, params.key));
    float wp2 = params.white_point * params.white_point;

    // Launch algorithm-specific kernel
    switch (algorithm) {
        case HdrAlgorithm::REINHARD:
            cuda_launch_hdr_reinhard(&s_ws, width, height, bit_depth, max_val,
                                      sat, inv_gamma, exposure_mul);
            break;
        case HdrAlgorithm::REINHARD_EXT:
            cuda_launch_hdr_reinhard_ext(&s_ws, width, height, bit_depth, max_val,
                                          key, wp2, sat, inv_gamma);
            break;
        case HdrAlgorithm::FILMIC_ACES:
            cuda_launch_hdr_filmic_aces(&s_ws, width, height, bit_depth, max_val,
                                         strength, sat, inv_gamma, exposure_mul);
            break;
        case HdrAlgorithm::HABLE:
            cuda_launch_hdr_hable(&s_ws, width, height, bit_depth, max_val,
                                   strength, sat, inv_gamma, exposure_mul);
            break;
        case HdrAlgorithm::DRAGO:
            cuda_launch_hdr_drago(&s_ws, width, height, bit_depth, max_val,
                                   key, sat, inv_gamma, exposure_mul);
            break;
        case HdrAlgorithm::EXPONENTIAL:
            cuda_launch_hdr_exponential(&s_ws, width, height, bit_depth, max_val,
                                         sat, inv_gamma, exposure_mul);
            break;
        case HdrAlgorithm::LOGARITHMIC:
            cuda_launch_hdr_logarithmic(&s_ws, width, height, bit_depth, max_val,
                                         sat, inv_gamma, exposure_mul);
            break;
        case HdrAlgorithm::LINEAR_TO_PQ:
            cuda_launch_hdr_linear_to_pq(&s_ws, width, height, bit_depth, max_val,
                                          exposure_mul);
            break;
        case HdrAlgorithm::PQ_TO_LINEAR:
            cuda_launch_hdr_pq_to_linear(&s_ws, width, height, bit_depth, max_val,
                                          exposure_mul);
            break;
        case HdrAlgorithm::LINEAR_TO_HLG:
            cuda_launch_hdr_linear_to_hlg(&s_ws, width, height, bit_depth, max_val,
                                           exposure_mul);
            break;
        case HdrAlgorithm::HLG_TO_LINEAR:
            cuda_launch_hdr_hlg_to_linear(&s_ws, width, height, bit_depth, max_val,
                                           exposure_mul);
            break;
        default:
            // Adaptive local not supported on GPU (falls back to CPU)
            return HdrError::InternalError;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) return HdrError::InternalError;

    // Copy output back to host
    cudaMemcpy(s_ws.h_output_pinned, s_ws.d_output, rgb_bytes, cudaMemcpyDeviceToHost);
    std::memcpy(output, s_ws.h_output_pinned, rgb_bytes);

    return HdrError::Ok;
}

} // namespace hdr

#else // !IM_OPERATOR_HAS_CUDA

namespace hdr {

bool has_cuda() { return false; }

HdrError process_hdr_cuda(const uint8_t*, uint8_t*,
                           int, int, int,
                           HdrAlgorithm, int,
                           const HdrParams&) {
    return HdrError::CudaNotAvailable;
}

} // namespace hdr

#endif // IM_OPERATOR_HAS_CUDA
