#ifndef IMOP_CUDA_KERNELS_CUH
#define IMOP_CUDA_KERNELS_CUH

#include <cstdint>

struct CudaPatternOffsets {
    int r_row, r_col;
    int b_row, b_col;
};

struct CudaWorkspace {
    uint8_t* d_bayer = nullptr;
    uint8_t* d_rgb = nullptr;
    float* d_fbuf[10] = {};
    uint8_t* h_bayer_pinned = nullptr;
    uint8_t* h_rgb_pinned = nullptr;
    size_t h_bayer_bytes = 0;
    size_t h_rgb_bytes = 0;
    cudaStream_t streams[4] = {};
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    int max_val = 0;

    CudaWorkspace() = default;
    ~CudaWorkspace() { release(); }
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;
    CudaWorkspace(CudaWorkspace&& o) noexcept
        : d_bayer(o.d_bayer), d_rgb(o.d_rgb), h_bayer_pinned(o.h_bayer_pinned),
          h_rgb_pinned(o.h_rgb_pinned), h_bayer_bytes(o.h_bayer_bytes),
          h_rgb_bytes(o.h_rgb_bytes), width(o.width), height(o.height),
          bit_depth(o.bit_depth), max_val(o.max_val) {
        for (int i = 0; i < 10; i++) { d_fbuf[i] = o.d_fbuf[i]; o.d_fbuf[i] = nullptr; }
        for (int i = 0; i < 4; i++) { streams[i] = o.streams[i]; o.streams[i] = nullptr; }
        o.d_bayer = nullptr; o.d_rgb = nullptr;
        o.h_bayer_pinned = nullptr; o.h_rgb_pinned = nullptr;
        o.width = 0; o.height = 0; o.bit_depth = 0; o.max_val = 0;
    }
    CudaWorkspace& operator=(CudaWorkspace&&) = delete;

    bool ensure(int w, int h, int bit_depth);
    void release();
};

enum class CudaAlgo : int {
    SUPER_FAST = 0,
    HQLI,
    MG,
    L7,
    DFPD,
    AHD,
    AMAZE,
    RCD,
    PRISM,
    COUNT
};

struct CudaGraphCache {
    cudaGraphExec_t exec = nullptr;
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    CudaPatternOffsets po = {};
    CudaAlgo algo = CudaAlgo::SUPER_FAST;
    bool valid = false;

    void release();
};

#ifdef __cplusplus
extern "C" {
#endif

void cuda_launch_super_fast(CudaWorkspace* ws,
                            int width, int height, CudaPatternOffsets po,
                            int bit_depth, bool is_packed);

void cuda_launch_hqli(CudaWorkspace* ws,
                      int width, int height, CudaPatternOffsets po,
                      int bit_depth, bool is_packed);

void cuda_launch_mg(CudaWorkspace* ws,
                    int width, int height, CudaPatternOffsets po,
                    int bit_depth, bool is_packed);

void cuda_launch_l7(CudaWorkspace* ws,
                    int width, int height, CudaPatternOffsets po,
                    int bit_depth, bool is_packed);

void cuda_launch_dfpd(CudaWorkspace* ws,
                      int width, int height, CudaPatternOffsets po,
                      int bit_depth, bool is_packed);

void cuda_launch_ahd(CudaWorkspace* ws,
                     int width, int height, CudaPatternOffsets po,
                     int bit_depth, bool is_packed);

void cuda_launch_amaze(CudaWorkspace* ws,
                       int width, int height, CudaPatternOffsets po,
                       int bit_depth, bool is_packed);

void cuda_launch_rcd(CudaWorkspace* ws,
                     int width, int height, CudaPatternOffsets po,
                     int bit_depth, bool is_packed);

void cuda_launch_prism(CudaWorkspace* ws,
                       int width, int height, CudaPatternOffsets po,
                       int bit_depth, bool is_packed);

bool cuda_graph_capture(CudaGraphCache* cache, CudaWorkspace* ws,
                        int width, int height, CudaPatternOffsets po,
                        int bit_depth, bool is_packed, CudaAlgo algo);

void cuda_graph_launch(CudaGraphCache* cache);

#ifdef __cplusplus
}
#endif

#endif