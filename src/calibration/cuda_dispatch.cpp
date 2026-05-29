#include "calibration/algorithms.hpp"
#include "calib_kernels.cuh"
#include <cstring>
#include <mutex>

#ifdef IM_OPERATOR_HAS_CUDA

#include <cuda_runtime.h>

namespace calibration {

static bool s_has_cuda = false;
static std::once_flag s_cuda_once;

bool has_cuda() {
    std::call_once(s_cuda_once, [] {
        int count = 0;
        s_has_cuda = (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
    });
    return s_has_cuda;
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

} // namespace calibration

#else

namespace calibration {

bool has_cuda() { return false; }
const char* cuda_device_name() { return "N/A"; }

} // namespace calibration

#endif
