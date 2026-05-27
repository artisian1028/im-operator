#ifndef HDR_KERNELS_CUH
#define HDR_KERNELS_CUH

#include <cstdint>

namespace hdr {

struct HdrCudaWorkspace {
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    float* d_fbuf = nullptr;
    uint8_t* h_input_pinned = nullptr;
    uint8_t* h_output_pinned = nullptr;
    size_t h_input_bytes = 0;
    size_t h_output_bytes = 0;
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    int max_val = 0;

    HdrCudaWorkspace() = default;
    ~HdrCudaWorkspace() { release(); }
    HdrCudaWorkspace(const HdrCudaWorkspace&) = delete;
    HdrCudaWorkspace& operator=(const HdrCudaWorkspace&) = delete;
    HdrCudaWorkspace(HdrCudaWorkspace&& o) noexcept;
    HdrCudaWorkspace& operator=(HdrCudaWorkspace&&) = delete;

    bool ensure(int w, int h, int bd);
    void release();
};

// CUDA kernel launch functions (one per algorithm)
void cuda_launch_hdr_reinhard(HdrCudaWorkspace* ws, int width, int height,
                               int bit_depth, int max_val,
                               float sat, float inv_gamma, float exposure_mul);

void cuda_launch_hdr_reinhard_ext(HdrCudaWorkspace* ws, int width, int height,
                                   int bit_depth, int max_val,
                                   float key, float wp2, float sat, float inv_gamma);

void cuda_launch_hdr_filmic_aces(HdrCudaWorkspace* ws, int width, int height,
                                  int bit_depth, int max_val,
                                  float strength, float sat, float inv_gamma, float exposure_mul);

void cuda_launch_hdr_hable(HdrCudaWorkspace* ws, int width, int height,
                            int bit_depth, int max_val,
                            float strength, float sat, float inv_gamma, float exposure_mul);

void cuda_launch_hdr_drago(HdrCudaWorkspace* ws, int width, int height,
                            int bit_depth, int max_val,
                            float bias, float sat, float inv_gamma, float exposure_mul);

void cuda_launch_hdr_exponential(HdrCudaWorkspace* ws, int width, int height,
                                  int bit_depth, int max_val,
                                  float sat, float inv_gamma, float exposure_mul);

void cuda_launch_hdr_logarithmic(HdrCudaWorkspace* ws, int width, int height,
                                  int bit_depth, int max_val,
                                  float sat, float inv_gamma, float exposure_mul);

void cuda_launch_hdr_linear_to_pq(HdrCudaWorkspace* ws, int width, int height,
                                   int bit_depth, int max_val, float exposure_mul);

void cuda_launch_hdr_pq_to_linear(HdrCudaWorkspace* ws, int width, int height,
                                   int bit_depth, int max_val, float exposure_mul);

void cuda_launch_hdr_linear_to_hlg(HdrCudaWorkspace* ws, int width, int height,
                                    int bit_depth, int max_val, float exposure_mul);

void cuda_launch_hdr_hlg_to_linear(HdrCudaWorkspace* ws, int width, int height,
                                    int bit_depth, int max_val, float exposure_mul);

} // namespace hdr

#endif // HDR_KERNELS_CUH
