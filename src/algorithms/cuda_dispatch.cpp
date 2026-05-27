#include "imop/algorithms.hpp"
#include "imop/pixel_utils.hpp"
#include <cstdint>
#include <cstring>
#include <mutex>

#ifdef IM_OPERATOR_HAS_CUDA

#include <cuda_runtime.h>
#include "cuda_kernels.cuh"

namespace imop {

static bool s_cuda_available = false;
static std::once_flag s_cuda_once;

static CudaPatternOffsets make_cuda_po(const PatternOffsets& po) {
    CudaPatternOffsets cpo;
    cpo.r_row = po.r_row;
    cpo.r_col = po.r_col;
    cpo.b_row = po.b_row;
    cpo.b_col = po.b_col;
    return cpo;
}

bool has_cuda() {
    std::call_once(s_cuda_once, [] {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        s_cuda_available = (err == cudaSuccess && device_count > 0);
    });
    return s_cuda_available;
}

const char* cuda_device_name() {
    if (!has_cuda()) return "N/A";
    static char name_buf[256];
    static bool cached = false;
    if (!cached) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess) return "Unknown";
        size_t len = 0;
        while (len < 255 && prop.name[len]) { name_buf[len] = prop.name[len]; len++; }
        name_buf[len] = '\0';
        cached = true;
    }
    return name_buf;
}

static thread_local CudaWorkspace t_cuda_ws;

static CudaAlgo to_cuda_algo(DemosaicAlgorithm algo) {
    switch (algo) {
        case DemosaicAlgorithm::SUPER_FAST: return CudaAlgo::SUPER_FAST;
        case DemosaicAlgorithm::HQLI:       return CudaAlgo::HQLI;
        case DemosaicAlgorithm::MG:         return CudaAlgo::MG;
        case DemosaicAlgorithm::L7:         return CudaAlgo::L7;
        case DemosaicAlgorithm::DFPD:       return CudaAlgo::DFPD;
        case DemosaicAlgorithm::AHD:        return CudaAlgo::AHD;
        case DemosaicAlgorithm::AMAZE:      return CudaAlgo::AMAZE;
        case DemosaicAlgorithm::RCD:        return CudaAlgo::RCD;
        case DemosaicAlgorithm::PRISM:      return CudaAlgo::PRISM;
        default: return CudaAlgo::COUNT;
    }
}

DemosaicError demosaic_cuda(const uint8_t* bayer_data, uint8_t* rgb_data,
                                  int width, int height, BayerPattern pattern,
                                  DemosaicAlgorithm algorithm, int bit_depth,
                                  bool is_packed) {
    if (!has_cuda()) return DemosaicError::InternalError;
    // CUDA kernels do not support packed-format data natively;
    // returning non-Ok here lets the caller fall back to CPU path.
    if (is_packed) return DemosaicError::InternalError;

    int min_dim = 2;
    switch (algorithm) {
        case DemosaicAlgorithm::SUPER_FAST: min_dim = 2; break;
        case DemosaicAlgorithm::HQLI:       min_dim = 6; break;
        case DemosaicAlgorithm::MG:         min_dim = 6; break;
        case DemosaicAlgorithm::L7:         min_dim = 8; break;
        case DemosaicAlgorithm::DFPD:       min_dim = 12; break;
        case DemosaicAlgorithm::AHD:        min_dim = 6; break;
        case DemosaicAlgorithm::AMAZE:      min_dim = 6; break;
        case DemosaicAlgorithm::RCD:        min_dim = 10; break;
        case DemosaicAlgorithm::PRISM:      min_dim = 10; break;
    }
    if (width < min_dim || height < min_dim) return DemosaicError::ImageTooSmall;

    if (!t_cuda_ws.ensure(width, height, bit_depth))
        return DemosaicError::InternalError;

    size_t bayer_bytes = pixel::compute_bayer_byte_size(width, height, bit_depth, is_packed);
    size_t rgb_bytes = pixel::compute_rgb_byte_size(width, height, bit_depth);

    cudaError_t err;
    err = cudaMemcpy(t_cuda_ws.d_bayer, bayer_data, bayer_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return DemosaicError::InternalError;
    cudaMemset(t_cuda_ws.d_rgb, 0, rgb_bytes);

    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    CudaPatternOffsets cpo = make_cuda_po(po);

    switch (algorithm) {
        case DemosaicAlgorithm::SUPER_FAST:
            cuda_launch_super_fast(&t_cuda_ws, width, height, cpo, bit_depth, is_packed);
            break;
        case DemosaicAlgorithm::HQLI:
            cuda_launch_hqli(&t_cuda_ws, width, height, cpo, bit_depth, is_packed);
            break;
        case DemosaicAlgorithm::MG:
            cuda_launch_mg(&t_cuda_ws, width, height, cpo, bit_depth, is_packed);
            break;
        case DemosaicAlgorithm::L7:
            cuda_launch_l7(&t_cuda_ws, width, height, cpo, bit_depth, is_packed);
            break;
        case DemosaicAlgorithm::DFPD:
            cuda_launch_dfpd(&t_cuda_ws, width, height, cpo, bit_depth, is_packed);
            break;
        case DemosaicAlgorithm::AHD:
            cuda_launch_ahd(&t_cuda_ws, width, height, cpo, bit_depth, is_packed);
            break;
        case DemosaicAlgorithm::AMAZE:
            cuda_launch_amaze(&t_cuda_ws, width, height, cpo, bit_depth, is_packed);
            break;
        case DemosaicAlgorithm::RCD:
            cuda_launch_rcd(&t_cuda_ws, width, height, cpo, bit_depth, is_packed);
            break;
        case DemosaicAlgorithm::PRISM:
            cuda_launch_prism(&t_cuda_ws, width, height, cpo, bit_depth, is_packed);
            break;
        default:
            return DemosaicError::InternalError;
    }

    err = cudaMemcpy(rgb_data, t_cuda_ws.d_rgb, rgb_bytes, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) return DemosaicError::InternalError;
    return DemosaicError::Ok;
}

struct CudaBatchSlot {
    CudaWorkspace ws;
    CudaGraphCache graph;
    cudaStream_t batch_stream = nullptr;
    bool graph_ready = false;
};

DemosaicError demosaic_cuda_batch(const uint8_t* const* inputs,
                                         uint8_t* const* outputs,
                                         int num_frames,
                                         int width, int height,
                                         BayerPattern pattern,
                                         DemosaicAlgorithm algorithm,
                                         int bit_depth) {
    if (!has_cuda()) return DemosaicError::InternalError;
    if (num_frames <= 0) return DemosaicError::Ok;

    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    CudaPatternOffsets cpo = make_cuda_po(po);
    CudaAlgo calgo = to_cuda_algo(algorithm);

    size_t bayer_bytes = pixel::compute_bayer_byte_size(width, height, bit_depth, false);
    size_t rgb_bytes = pixel::compute_rgb_byte_size(width, height, bit_depth);

    CudaBatchSlot slots[2];
    for (int i = 0; i < 2; i++) {
        if (!slots[i].ws.ensure(width, height, bit_depth))
            return DemosaicError::InternalError;
        cudaError_t stream_err = cudaStreamCreate(&slots[i].batch_stream);
        if (stream_err != cudaSuccess) return DemosaicError::InternalError;
    }

    for (int i = 0; i < 2; i++) {
        slots[i].graph_ready = cuda_graph_capture(&slots[i].graph, &slots[i].ws,
            width, height, cpo, bit_depth, false, calgo);
    }

    int cur = 0;

    std::memcpy(slots[0].ws.h_bayer_pinned, inputs[0], bayer_bytes);
    cudaError_t err = cudaGraphLaunch(slots[0].graph.exec, slots[0].batch_stream);
    if (err != cudaSuccess) return DemosaicError::InternalError;

    for (int frame = 1; frame < num_frames; frame++) {
        int next = 1 - cur;

        std::memcpy(slots[next].ws.h_bayer_pinned, inputs[frame], bayer_bytes);
        err = cudaGraphLaunch(slots[next].graph.exec, slots[next].batch_stream);
        if (err != cudaSuccess) return DemosaicError::InternalError;

        err = cudaStreamSynchronize(slots[cur].batch_stream);
        if (err != cudaSuccess) return DemosaicError::InternalError;

        std::memcpy(outputs[frame - 1], slots[cur].ws.h_rgb_pinned, rgb_bytes);
        cur = next;
    }

    err = cudaStreamSynchronize(slots[cur].batch_stream);
    if (err != cudaSuccess) return DemosaicError::InternalError;
    std::memcpy(outputs[num_frames - 1], slots[cur].ws.h_rgb_pinned, rgb_bytes);

    for (int i = 0; i < 2; i++) {
        cudaStreamDestroy(slots[i].batch_stream);
    }

    return DemosaicError::Ok;
}

void cuda_synchronize() {
    cudaDeviceSynchronize();
}

} // namespace imop

#else

namespace imop {

bool has_cuda() { return false; }

const char* cuda_device_name() { return "N/A"; }

DemosaicError demosaic_cuda(const uint8_t*, uint8_t*,
                                  int, int, BayerPattern,
                                  DemosaicAlgorithm, int, bool) {
    return DemosaicError::InternalError;
}

DemosaicError demosaic_cuda_batch(const uint8_t* const*, uint8_t* const*,
                                         int, int, int, BayerPattern,
                                         DemosaicAlgorithm, int) {
    return DemosaicError::InternalError;
}

void cuda_synchronize() {}

} // namespace imop

#endif // IM_OPERATOR_HAS_CUDA
