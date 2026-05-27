#include "cuda_kernels.cuh"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__device__ __forceinline__
bool cuda_is_r_at(const CudaPatternOffsets& po, int row, int col) {
    return ((row & 1) == po.r_row) && ((col & 1) == po.r_col);
}

__device__ __forceinline__
bool cuda_is_b_at(const CudaPatternOffsets& po, int row, int col) {
    return ((row & 1) == po.b_row) && ((col & 1) == po.b_col);
}

__device__ __forceinline__
bool cuda_is_g_at(const CudaPatternOffsets& po, int row, int col) {
    return !cuda_is_r_at(po, row, col) && !cuda_is_b_at(po, row, col);
}

__device__ __forceinline__
int cuda_safe_max_val(int bit_depth) {
    if (bit_depth <= 0) return 0;
    if (bit_depth > 16) return 65535;
    return (1 << bit_depth) - 1;
}

__device__ __forceinline__
int cuda_get_raw_8bit(const uint8_t* data, int x, int y, int width) {
    return static_cast<int>(__ldg(data + static_cast<size_t>(y) * width + x));
}

__device__ __forceinline__
int cuda_get_raw_16bit(const uint8_t* data, int x, int y, int width, int bit_depth) {
    const uint16_t* u16 = reinterpret_cast<const uint16_t*>(data);
    int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(y) * width + x));
    if (bit_depth > 8 && bit_depth < 16) {
        int max_val = (1 << bit_depth) - 1;
        if (v > max_val) v >>= (16 - bit_depth);
    }
    return v;
}

__device__ __forceinline__
int cuda_get_clamped_8bit(const uint8_t* data, int x, int y, int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return static_cast<int>(__ldg(data + static_cast<size_t>(y) * width + x));
}

__device__ __forceinline__
int cuda_get_clamped_16bit(const uint8_t* data, int x, int y, int width, int height, int bit_depth) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    const uint16_t* u16 = reinterpret_cast<const uint16_t*>(data);
    int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(y) * width + x));
    if (bit_depth > 8 && bit_depth < 16) {
        int max_val = (1 << bit_depth) - 1;
        if (v > max_val) v >>= (16 - bit_depth);
    }
    return v;
}

__device__ __forceinline__
void cuda_set_rgb_8bit(uint8_t* rgb, int x, int y, int width, int r, int g, int b) {
    size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    rgb[idx + 0] = static_cast<uint8_t>(r);
    rgb[idx + 1] = static_cast<uint8_t>(g);
    rgb[idx + 2] = static_cast<uint8_t>(b);
}

__device__ __forceinline__
void cuda_set_rgb_16bit(uint8_t* rgb, int x, int y, int width, int r, int g, int b) {
    size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    uint16_t* rgb16 = reinterpret_cast<uint16_t*>(rgb);
    rgb16[idx + 0] = static_cast<uint16_t>(r);
    rgb16[idx + 1] = static_cast<uint16_t>(g);
    rgb16[idx + 2] = static_cast<uint16_t>(b);
}

__device__ __forceinline__
void cuda_set_rgb_clamped_8bit(uint8_t* rgb, int x, int y, int width, int r, int g, int b, int max_val) {
    r = max(0, min(r, max_val));
    g = max(0, min(g, max_val));
    b = max(0, min(b, max_val));
    cuda_set_rgb_8bit(rgb, x, y, width, r, g, b);
}

__device__ __forceinline__
void cuda_set_rgb_clamped_16bit(uint8_t* rgb, int x, int y, int width, int r, int g, int b, int max_val) {
    r = max(0, min(r, max_val));
    g = max(0, min(g, max_val));
    b = max(0, min(b, max_val));
    cuda_set_rgb_16bit(rgb, x, y, width, r, g, b);
}

__device__ __forceinline__
int cuda_fc(int r, int c, CudaPatternOffsets po) {
    if ((r & 1) == po.r_row && (c & 1) == po.r_col) return 0;
    if ((r & 1) == po.b_row && (c & 1) == po.b_col) return 2;
    return 1;
}

__device__ __forceinline__
float cuda_median_3x3(float* w) {
#define CSWAP(a, b) do { float _t = w[a]; if (_t > w[b]) { w[a] = w[b]; w[b] = _t; } } while(0)
    CSWAP(0,1); CSWAP(3,4); CSWAP(6,7);
    CSWAP(1,2); CSWAP(4,5); CSWAP(7,8);
    CSWAP(0,1); CSWAP(3,4); CSWAP(6,7);
    CSWAP(0,3); CSWAP(4,7); CSWAP(1,4);
    CSWAP(3,6); CSWAP(2,5); CSWAP(5,8);
    CSWAP(1,3); CSWAP(4,6); CSWAP(2,5);
    CSWAP(2,3); CSWAP(5,6); CSWAP(2,4);
    CSWAP(3,5); CSWAP(3,4);
    return w[4];
#undef CSWAP
}

bool CudaWorkspace::ensure(int w, int h, int bd) {
    if (w == width && h == height && bit_depth == bd && d_bayer) return true;

    release();

    width = w;
    height = h;
    bit_depth = bd;
    if (bd <= 0) max_val = 0;
    else if (bd > 16) max_val = 65535;
    else max_val = (1 << bd) - 1;
    size_t total = static_cast<size_t>(w) * h;
    size_t bayer_bytes = total * ((bd <= 8) ? 1 : 2);
    size_t rgb_bytes = total * 3 * ((bd <= 8) ? 1 : 2);
    size_t float_bytes = total * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&d_bayer, bayer_bytes);
    if (err != cudaSuccess) { release(); return false; }
    err = cudaMalloc(&d_rgb, rgb_bytes);
    if (err != cudaSuccess) { release(); return false; }
    for (int i = 0; i < 10; i++) {
        err = cudaMalloc(&d_fbuf[i], float_bytes);
        if (err != cudaSuccess) { release(); return false; }
    }

    h_bayer_bytes = bayer_bytes;
    h_rgb_bytes = rgb_bytes;
    err = cudaMallocHost(&h_bayer_pinned, bayer_bytes);
    if (err != cudaSuccess) { release(); return false; }
    err = cudaMallocHost(&h_rgb_pinned, rgb_bytes);
    if (err != cudaSuccess) { release(); return false; }

    for (int i = 0; i < 4; i++) {
        err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) { release(); return false; }
    }
    return true;
}

void CudaWorkspace::release() {
    if (d_bayer) { cudaFree(d_bayer); d_bayer = nullptr; }
    if (d_rgb) { cudaFree(d_rgb); d_rgb = nullptr; }
    for (int i = 0; i < 10; i++) {
        if (d_fbuf[i]) { cudaFree(d_fbuf[i]); d_fbuf[i] = nullptr; }
    }
    if (h_bayer_pinned) { cudaFreeHost(h_bayer_pinned); h_bayer_pinned = nullptr; }
    if (h_rgb_pinned) { cudaFreeHost(h_rgb_pinned); h_rgb_pinned = nullptr; }
    h_bayer_bytes = 0;
    h_rgb_bytes = 0;
    for (int i = 0; i < 4; i++) {
        if (streams[i]) { cudaStreamDestroy(streams[i]); streams[i] = nullptr; }
    }
    width = 0;
    height = 0;
    bit_depth = 0;
    max_val = 0;
}

__global__
void super_fast_kernel(const uint8_t* bayer, uint8_t* rgb,
                       int width, int height, CudaPatternOffsets po,
                       int bit_depth, bool is_packed) {
    constexpr int BW = 32, BH = 16;
    constexpr int border = 1;
    constexpr int TILE_W = BW + 2 * border;
    constexpr int TILE_H = BH + 2 * border;

    __shared__ int tile[TILE_H][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;

    for (int row = ty; row < TILE_H; row += BH) {
        int gy = static_cast<int>(blockIdx.y) * BH + row - border;
        gy = max(0, min(gy, height - 1));
        for (int col = tx; col < TILE_W; col += BW) {
            int gx = static_cast<int>(blockIdx.x) * BW + col - border;
            gx = max(0, min(gx, width - 1));
            if (bit_depth <= 8)
                tile[row][col] = static_cast<int>(__ldg(bayer + static_cast<size_t>(gy) * width + gx));
            else {
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(bayer);
                int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(gy) * width + gx));
                if (bit_depth > 8 && bit_depth < 16) {
                    int mv = (1 << bit_depth) - 1;
                    if (v > mv) v >>= (16 - bit_depth);
                }
                tile[row][col] = v;
            }
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;

    int max_val = cuda_safe_max_val(bit_depth);
    int r = 0, g = 0, b = 0;

    auto sm = [&](int lx, int ly) -> int {
        return tile[ty + border + ly][tx + border + lx];
    };

    if (cuda_is_r_at(po, y, x)) {
        r = sm(0, 0);
        g = (sm(-1, 0) + sm(1, 0) + sm(0, -1) + sm(0, 1) + 2) >> 2;
        b = sm((x & 1) ? -1 : 1, (y & 1) ? -1 : 1);
    } else if (cuda_is_b_at(po, y, x)) {
        b = sm(0, 0);
        g = (sm(-1, 0) + sm(1, 0) + sm(0, -1) + sm(0, 1) + 2) >> 2;
        r = sm((x & 1) ? -1 : 1, (y & 1) ? -1 : 1);
    } else {
        g = sm(0, 0);
        bool is_gr_row = ((y & 1) == po.r_row);
        if (is_gr_row) {
            r = (sm(-1, 0) + sm(1, 0) + 1) >> 1;
            b = (sm(0, -1) + sm(0, 1) + 1) >> 1;
        } else {
            b = (sm(-1, 0) + sm(1, 0) + 1) >> 1;
            r = (sm(0, -1) + sm(0, 1) + 1) >> 1;
        }
    }

    if (bit_depth <= 8)
        cuda_set_rgb_clamped_8bit(rgb, x, y, width, r, g, b, max_val);
    else
        cuda_set_rgb_clamped_16bit(rgb, x, y, width, r, g, b, max_val);
}

__global__
void hqli_kernel(const uint8_t* bayer, uint8_t* rgb,
                 int width, int height, CudaPatternOffsets po,
                 int bit_depth, bool is_packed) {
    constexpr int BW = 32, BH = 16;
    constexpr int border = 2;
    constexpr int TILE_W = BW + 2 * border;
    constexpr int TILE_H = BH + 2 * border;

    __shared__ int tile[TILE_H][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;

    for (int row = ty; row < TILE_H; row += BH) {
        int gy = static_cast<int>(blockIdx.y) * BH + row - border;
        gy = max(0, min(gy, height - 1));
        for (int col = tx; col < TILE_W; col += BW) {
            int gx = static_cast<int>(blockIdx.x) * BW + col - border;
            gx = max(0, min(gx, width - 1));
            if (bit_depth <= 8)
                tile[row][col] = static_cast<int>(__ldg(bayer + static_cast<size_t>(gy) * width + gx));
            else {
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(bayer);
                int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(gy) * width + gx));
                if (bit_depth > 8 && bit_depth < 16) {
                    int mv = (1 << bit_depth) - 1;
                    if (v > mv) v >>= (16 - bit_depth);
                }
                tile[row][col] = v;
            }
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;

    if (x < border || x >= width - border || y < border || y >= height - border) {
        int ref_y = max(border, min(y, height - 1 - border));
        int ref_x = max(border, min(x, width - 1 - border));
        if (ref_y != y || ref_x != x) {
            if (bit_depth <= 8) {
                size_t src = (static_cast<size_t>(ref_y) * width + ref_x) * 3;
                size_t dst = (static_cast<size_t>(y) * width + x) * 3;
                rgb[dst + 0] = rgb[src + 0];
                rgb[dst + 1] = rgb[src + 1];
                rgb[dst + 2] = rgb[src + 2];
            } else {
                size_t src = (static_cast<size_t>(ref_y) * width + ref_x) * 3;
                size_t dst = (static_cast<size_t>(y) * width + x) * 3;
                for (int c = 0; c < 3; c++) {
                    uint16_t val;
                    memcpy(&val, rgb + (src + c) * 2, sizeof(val));
                    memcpy(rgb + (dst + c) * 2, &val, sizeof(val));
                }
            }
        }
        return;
    }

    int max_val = cuda_safe_max_val(bit_depth);

    auto sm = [&](int lx, int ly) -> int {
        return tile[ty + border + ly][tx + border + lx];
    };

    bool at_r = cuda_is_r_at(po, y, x);
    bool at_b = cuda_is_b_at(po, y, x);
    bool is_gr_row = ((y & 1) == po.r_row);

    int r_val, g_val, b_val;

    if (at_r) {
        r_val = sm(0, 0);
    } else if (at_b) {
        r_val = (sm(-1, -1) + sm(1, -1) + sm(-1, 1) + sm(1, 1) + 2) >> 2;
    } else {
        if (is_gr_row)
            r_val = (sm(-1, 0) + sm(1, 0) + 1) >> 1;
        else
            r_val = (sm(0, -1) + sm(0, 1) + 1) >> 1;
    }

    if (!at_r && !at_b) {
        g_val = sm(0, 0);
    } else {
        g_val = (sm(-1, 0) + sm(1, 0) + sm(0, -1) + sm(0, 1) + 2) >> 2;
    }

    if (at_b) {
        b_val = sm(0, 0);
    } else if (at_r) {
        b_val = (sm(-1, -1) + sm(1, -1) + sm(-1, 1) + sm(1, 1) + 2) >> 2;
    } else {
        if (is_gr_row)
            b_val = (sm(0, -1) + sm(0, 1) + 1) >> 1;
        else
            b_val = (sm(-1, 0) + sm(1, 0) + 1) >> 1;
    }

    if (r_val < 0) r_val = 0; else if (r_val > max_val) r_val = max_val;
    if (g_val < 0) g_val = 0; else if (g_val > max_val) g_val = max_val;
    if (b_val < 0) b_val = 0; else if (b_val > max_val) b_val = max_val;

    if (bit_depth <= 8)
        cuda_set_rgb_8bit(rgb, x, y, width, r_val, g_val, b_val);
    else
        cuda_set_rgb_16bit(rgb, x, y, width, r_val, g_val, b_val);
}

__global__
void mg_green_kernel(const uint8_t* bayer, float* G,
                     int width, int height, CudaPatternOffsets po,
                     int bit_depth, bool is_packed) {
    constexpr int BW = 32, BH = 16;
    constexpr int border = 2;
    constexpr int TILE_W = BW + 2 * border;
    constexpr int TILE_H = BH + 2 * border;

    __shared__ float tile[TILE_H][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;

    for (int row = ty; row < TILE_H; row += BH) {
        int gy = static_cast<int>(blockIdx.y) * BH + row - border;
        gy = max(0, min(gy, height - 1));
        for (int col = tx; col < TILE_W; col += BW) {
            int gx = static_cast<int>(blockIdx.x) * BW + col - border;
            gx = max(0, min(gx, width - 1));
            if (bit_depth <= 8)
                tile[row][col] = static_cast<float>(__ldg(bayer + static_cast<size_t>(gy) * width + gx));
            else {
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(bayer);
                int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(gy) * width + gx));
                if (bit_depth > 8 && bit_depth < 16) {
                    int mv = (1 << bit_depth) - 1;
                    if (v > mv) v >>= (16 - bit_depth);
                }
                tile[row][col] = static_cast<float>(v);
            }
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;

    size_t idx = static_cast<size_t>(y) * width + x;

    auto sm = [&](int lx, int ly) -> float {
        return tile[ty + border + ly][tx + border + lx];
    };

    if (!cuda_is_r_at(po, y, x) && !cuda_is_b_at(po, y, x)) {
        G[idx] = sm(0, 0);
    } else if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        float n  = sm(0, -1);
        float s  = sm(0, 1);
        float e  = sm(1, 0);
        float w  = sm(-1, 0);
        float n2 = sm(0, -2);
        float s2 = sm(0, 2);
        float e2 = sm(2, 0);
        float w2 = sm(-2, 0);
        float c  = sm(0, 0);
        G[idx] = (4.0f * c + 2.0f * (n + s + e + w) - (n2 + s2 + e2 + w2)) / 8.0f;
    }
}

__global__
void mg_rb_kernel(const uint8_t* bayer, float* R, float* B, const float* G,
                  int width, int height, CudaPatternOffsets po,
                  int bit_depth, bool is_packed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    size_t idx = static_cast<size_t>(y) * width + x;
    float g = G[idx];

    auto raw = [&](int px, int py) -> float {
        if (bit_depth <= 8)
            return static_cast<float>(cuda_get_clamped_8bit(bayer, px, py, width, height));
        else
            return static_cast<float>(cuda_get_clamped_16bit(bayer, px, py, width, height, bit_depth));
    };

    if (cuda_is_r_at(po, y, x)) {
        R[idx] = raw(x, y);
        float cd_sum = 0.0f; int cd_cnt = 0;
        int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
        for (int k = 0; k < 4; k++) {
            int nx = x + diag[k][0], ny = y + diag[k][1];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height && cuda_is_b_at(po, ny, nx)) {
                cd_sum += raw(nx, ny) - G[static_cast<size_t>(ny)*width+nx]; cd_cnt++;
            }
        }
        B[idx] = g + (cd_cnt > 0 ? cd_sum / static_cast<float>(cd_cnt) : 0.0f);
    } else if (cuda_is_b_at(po, y, x)) {
        B[idx] = raw(x, y);
        float cd_sum = 0.0f; int cd_cnt = 0;
        int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
        for (int k = 0; k < 4; k++) {
            int nx = x + diag[k][0], ny = y + diag[k][1];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height && cuda_is_r_at(po, ny, nx)) {
                cd_sum += raw(nx, ny) - G[static_cast<size_t>(ny)*width+nx]; cd_cnt++;
            }
        }
        R[idx] = g + (cd_cnt > 0 ? cd_sum / static_cast<float>(cd_cnt) : 0.0f);
    } else {
        float r_cd = 0.0f, b_cd = 0.0f; int r_cnt = 0, b_cnt = 0;
        if (cuda_is_r_at(po, y-1, x)) { r_cd += raw(x, y-1) - G[static_cast<size_t>(y-1)*width+x]; r_cnt++; }
        if (cuda_is_r_at(po, y+1, x)) { r_cd += raw(x, y+1) - G[static_cast<size_t>(y+1)*width+x]; r_cnt++; }
        if (cuda_is_r_at(po, y, x-1)) { r_cd += raw(x-1, y) - G[static_cast<size_t>(y)*width+(x-1)]; r_cnt++; }
        if (cuda_is_r_at(po, y, x+1)) { r_cd += raw(x+1, y) - G[static_cast<size_t>(y)*width+(x+1)]; r_cnt++; }
        if (cuda_is_b_at(po, y-1, x)) { b_cd += raw(x, y-1) - G[static_cast<size_t>(y-1)*width+x]; b_cnt++; }
        if (cuda_is_b_at(po, y+1, x)) { b_cd += raw(x, y+1) - G[static_cast<size_t>(y+1)*width+x]; b_cnt++; }
        if (cuda_is_b_at(po, y, x-1)) { b_cd += raw(x-1, y) - G[static_cast<size_t>(y)*width+(x-1)]; b_cnt++; }
        if (cuda_is_b_at(po, y, x+1)) { b_cd += raw(x+1, y) - G[static_cast<size_t>(y)*width+(x+1)]; b_cnt++; }
        R[idx] = g + (r_cnt > 0 ? r_cd / static_cast<float>(r_cnt) : 0.0f);
        B[idx] = g + (b_cnt > 0 ? b_cd / static_cast<float>(b_cnt) : 0.0f);
    }
}

__global__
void mg_fill_borders_kernel(float* R, float* G, float* B,
                            int width, int height, int border) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int ref_y = max(border, min(y, height - 1 - border));
    int ref_x = max(border, min(x, width - 1 - border));
    if (ref_y != y || ref_x != x) {
        size_t ri = static_cast<size_t>(ref_y) * width + ref_x;
        size_t di = static_cast<size_t>(y) * width + x;
        R[di] = R[ri]; G[di] = G[ri]; B[di] = B[ri];
    }
}

__global__
void planes_to_rgb_kernel(const float* R, const float* G, const float* B,
                          uint8_t* rgb, int width, int height, int bit_depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int max_val = cuda_safe_max_val(bit_depth);
    size_t idx = static_cast<size_t>(y) * width + x;
    int r_out = max(0, min(static_cast<int>(R[idx] + 0.5f), max_val));
    int g_out = max(0, min(static_cast<int>(G[idx] + 0.5f), max_val));
    int b_out = max(0, min(static_cast<int>(B[idx] + 0.5f), max_val));

    if (bit_depth <= 8)
        cuda_set_rgb_8bit(rgb, x, y, width, r_out, g_out, b_out);
    else
        cuda_set_rgb_16bit(rgb, x, y, width, r_out, g_out, b_out);
}

__global__
void l7_kernel(const uint8_t* bayer, uint8_t* rgb,
               int width, int height, CudaPatternOffsets po,
               int bit_depth, bool is_packed) {
    constexpr int BW = 32, BH = 16;
    constexpr int border = 3;
    constexpr int TILE_W = BW + 2 * border;
    constexpr int TILE_H = BH + 2 * border;

    __shared__ int tile[TILE_H][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;

    // Collaborative tile load
    for (int row = ty; row < TILE_H; row += BH) {
        int gy = static_cast<int>(blockIdx.y) * BH + row - border;
        gy = max(0, min(gy, height - 1));
        for (int col = tx; col < TILE_W; col += BW) {
            int gx = static_cast<int>(blockIdx.x) * BW + col - border;
            gx = max(0, min(gx, width - 1));
            if (bit_depth <= 8)
                tile[row][col] = static_cast<int>(__ldg(bayer + static_cast<size_t>(gy) * width + gx));
            else {
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(bayer);
                int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(gy) * width + gx));
                if (bit_depth > 8 && bit_depth < 16) {
                    int mv = (1 << bit_depth) - 1;
                    if (v > mv) v >>= (16 - bit_depth);
                }
                tile[row][col] = v;
            }
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;

    int max_val = cuda_safe_max_val(bit_depth);

    auto sm = [&](int lx, int ly) -> int {
        return tile[ty + border + ly][tx + border + lx];
    };

    if (x < border || x >= width - border || y < border || y >= height - border) {
        int ref_y = max(border, min(y, height - 1 - border));
        int ref_x = max(border, min(x, width - 1 - border));
        if (ref_y != y || ref_x != x) {
            if (bit_depth <= 8) {
                size_t src = (static_cast<size_t>(ref_y) * width + ref_x) * 3;
                size_t dst = (static_cast<size_t>(y) * width + x) * 3;
                rgb[dst + 0] = rgb[src + 0];
                rgb[dst + 1] = rgb[src + 1];
                rgb[dst + 2] = rgb[src + 2];
            } else {
                uint16_t* rgb16 = reinterpret_cast<uint16_t*>(rgb);
                size_t src = (static_cast<size_t>(ref_y) * width + ref_x) * 3;
                size_t dst = (static_cast<size_t>(y) * width + x) * 3;
                rgb16[dst + 0] = rgb16[src + 0];
                rgb16[dst + 1] = rgb16[src + 1];
                rgb16[dst + 2] = rgb16[src + 2];
            }
        }
        return;
    }

    int r_sum = 0, r_wsum = 0, g_sum = 0, g_wsum = 0, b_sum = 0, b_wsum = 0;
    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            if (dx == 0 && dy == 0) continue;
            int ny = y + dy, nx = x + dx;
            bool match_r = cuda_is_r_at(po, ny, nx);
            bool match_b = cuda_is_b_at(po, ny, nx);
            int dist = abs(dx) + abs(dy);
            int w = (7 - dist) * (7 - dist);
            int pv = sm(dx, dy);
            if (match_r) { r_sum += pv * w; r_wsum += w; }
            else if (match_b) { b_sum += pv * w; b_wsum += w; }
            else { g_sum += pv * w; g_wsum += w; }
        }
    }
    bool at_r = cuda_is_r_at(po, y, x);
    bool at_b = cuda_is_b_at(po, y, x);
    bool at_g = !at_r && !at_b;

    int r_val = at_r ? sm(0, 0) : (r_wsum > 0 ? (r_sum + r_wsum / 2) / r_wsum : 0);
    int g_val = at_g ? sm(0, 0) : (g_wsum > 0 ? (g_sum + g_wsum / 2) / g_wsum : 0);
    int b_val = at_b ? sm(0, 0) : (b_wsum > 0 ? (b_sum + b_wsum / 2) / b_wsum : 0);

    if (r_val < 0) r_val = 0; else if (r_val > max_val) r_val = max_val;
    if (g_val < 0) g_val = 0; else if (g_val > max_val) g_val = max_val;
    if (b_val < 0) b_val = 0; else if (b_val > max_val) b_val = max_val;

    if (bit_depth <= 8)
        cuda_set_rgb_8bit(rgb, x, y, width, r_val, g_val, b_val);
    else
        cuda_set_rgb_16bit(rgb, x, y, width, r_val, g_val, b_val);
}

__global__
void dfpd_green_kernel(const uint8_t* bayer, float* G,
                       int width, int height, CudaPatternOffsets po,
                       int bit_depth, bool is_packed) {
    constexpr int BW = 32, BH = 16;
    constexpr int border = 2;
    constexpr int TILE_W = BW + 2 * border;
    constexpr int TILE_H = BH + 2 * border;

    __shared__ float tile[TILE_H][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;

    for (int row = ty; row < TILE_H; row += BH) {
        int gy = static_cast<int>(blockIdx.y) * BH + row - border;
        gy = max(0, min(gy, height - 1));
        for (int col = tx; col < TILE_W; col += BW) {
            int gx = static_cast<int>(blockIdx.x) * BW + col - border;
            gx = max(0, min(gx, width - 1));
            if (bit_depth <= 8)
                tile[row][col] = static_cast<float>(__ldg(bayer + static_cast<size_t>(gy) * width + gx));
            else {
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(bayer);
                int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(gy) * width + gx));
                if (bit_depth > 8 && bit_depth < 16) {
                    int mv = (1 << bit_depth) - 1;
                    if (v > mv) v >>= (16 - bit_depth);
                }
                tile[row][col] = static_cast<float>(v);
            }
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;
    if (x < border || x >= width - border || y < border || y >= height - border) return;

    size_t idx = static_cast<size_t>(y) * width + x;

    auto sm = [&](int lx, int ly) -> float {
        return tile[ty + border + ly][tx + border + lx];
    };

    if (!cuda_is_r_at(po, y, x) && !cuda_is_b_at(po, y, x)) {
        G[idx] = sm(0, 0);
    } else {
        float gh = fabsf(sm(-1, 0) - sm(1, 0)) +
                   fabsf(2.0f * sm(0, 0) - sm(-2, 0) - sm(2, 0));
        float gv = fabsf(sm(0, -1) - sm(0, 1)) +
                   fabsf(2.0f * sm(0, 0) - sm(0, -2) - sm(0, 2));
        float g_h = (sm(-1, 0) + sm(1, 0)) * 0.5f;
        float g_v = (sm(0, -1) + sm(0, 1)) * 0.5f;
        if (gh < gv) {
            float correction = sm(0, 0) - (sm(-2, 0) + sm(2, 0)) * 0.5f;
            G[idx] = g_h + correction * 0.5f;
        } else if (gv < gh) {
            float correction = sm(0, 0) - (sm(0, -2) + sm(0, 2)) * 0.5f;
            G[idx] = g_v + correction * 0.5f;
        } else {
            G[idx] = (g_h + g_v) * 0.5f;
        }
    }
}

__global__
void dfpd_rb_kernel(const uint8_t* bayer, uint8_t* rgb, const float* G,
                    int width, int height, CudaPatternOffsets po,
                    int bit_depth, bool is_packed) {
    constexpr int BW = 32, BH = 16;
    constexpr int border = 3;
    constexpr int TILE_W = BW + 2 * border;
    constexpr int TILE_H = BH + 2 * border;

    __shared__ float tile[TILE_H][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;

    for (int row = ty; row < TILE_H; row += BH) {
        int gy = static_cast<int>(blockIdx.y) * BH + row - border;
        gy = max(0, min(gy, height - 1));
        for (int col = tx; col < TILE_W; col += BW) {
            int gx = static_cast<int>(blockIdx.x) * BW + col - border;
            gx = max(0, min(gx, width - 1));
            if (bit_depth <= 8)
                tile[row][col] = static_cast<float>(__ldg(bayer + static_cast<size_t>(gy) * width + gx));
            else {
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(bayer);
                int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(gy) * width + gx));
                if (bit_depth > 8 && bit_depth < 16) {
                    int mv = (1 << bit_depth) - 1;
                    if (v > mv) v >>= (16 - bit_depth);
                }
                tile[row][col] = static_cast<float>(v);
            }
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;
    if (x < border || x >= width - border || y < border || y >= height - border) {
        int ref_border = 5;
        int ref_y = max(ref_border, min(y, height - 1 - ref_border));
        int ref_x = max(ref_border, min(x, width - 1 - ref_border));
        if (ref_y != y || ref_x != x) {
            if (bit_depth <= 8) {
                size_t src = (static_cast<size_t>(ref_y) * width + ref_x) * 3;
                size_t dst = (static_cast<size_t>(y) * width + x) * 3;
                rgb[dst + 0] = rgb[src + 0];
                rgb[dst + 1] = rgb[src + 1];
                rgb[dst + 2] = rgb[src + 2];
            } else {
                size_t src = (static_cast<size_t>(ref_y) * width + ref_x) * 3;
                size_t dst = (static_cast<size_t>(y) * width + x) * 3;
                for (int c = 0; c < 3; c++) {
                    uint16_t val;
                    memcpy(&val, rgb + (src + c) * 2, sizeof(val));
                    memcpy(rgb + (dst + c) * 2, &val, sizeof(val));
                }
            }
        }
        return;
    }

    auto sm = [&](int lx, int ly) -> float {
        return tile[ty + border + ly][tx + border + lx];
    };

    auto gv = [&](int px, int py) -> float {
        size_t i = static_cast<size_t>(py) * width + px;
        return G[i];
    };

    float g_val = gv(x, y);
    int r_val = 0, b_val_out = 0;
    int max_val = cuda_safe_max_val(bit_depth);

    if (cuda_is_r_at(po, y, x)) {
        r_val = static_cast<int>(sm(0, 0) + 0.5f);
        float cd_sum = 0.0f; int cd_cnt = 0;
        int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
        for (int k = 0; k < 4; k++) {
            int nx = x + diag[k][0], ny = y + diag[k][1];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height && cuda_is_b_at(po, ny, nx)) {
                cd_sum += sm(diag[k][0], diag[k][1]) - gv(nx, ny); cd_cnt++;
            }
        }
        b_val_out = static_cast<int>(g_val + (cd_cnt > 0 ? cd_sum / cd_cnt : 0.0f) + 0.5f);
    } else if (cuda_is_b_at(po, y, x)) {
        b_val_out = static_cast<int>(sm(0, 0) + 0.5f);
        float cd_sum = 0.0f; int cd_cnt = 0;
        int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
        for (int k = 0; k < 4; k++) {
            int nx = x + diag[k][0], ny = y + diag[k][1];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height && cuda_is_r_at(po, ny, nx)) {
                cd_sum += sm(diag[k][0], diag[k][1]) - gv(nx, ny); cd_cnt++;
            }
        }
        r_val = static_cast<int>(g_val + (cd_cnt > 0 ? cd_sum / cd_cnt : 0.0f) + 0.5f);
    } else {
        float r_cd = 0.0f, b_cd = 0.0f; int r_cnt = 0, b_cnt = 0;
        if (cuda_is_r_at(po, y-1, x)) { r_cd += sm(0, -1) - gv(x, y-1); r_cnt++; }
        if (cuda_is_r_at(po, y+1, x)) { r_cd += sm(0, 1) - gv(x, y+1); r_cnt++; }
        if (cuda_is_r_at(po, y, x-1)) { r_cd += sm(-1, 0) - gv(x-1, y); r_cnt++; }
        if (cuda_is_r_at(po, y, x+1)) { r_cd += sm(1, 0) - gv(x+1, y); r_cnt++; }
        if (cuda_is_b_at(po, y-1, x)) { b_cd += sm(0, -1) - gv(x, y-1); b_cnt++; }
        if (cuda_is_b_at(po, y+1, x)) { b_cd += sm(0, 1) - gv(x, y+1); b_cnt++; }
        if (cuda_is_b_at(po, y, x-1)) { b_cd += sm(-1, 0) - gv(x-1, y); b_cnt++; }
        if (cuda_is_b_at(po, y, x+1)) { b_cd += sm(1, 0) - gv(x+1, y); b_cnt++; }
        r_val = static_cast<int>(g_val + (r_cnt > 0 ? r_cd / r_cnt : 0.0f) + 0.5f);
        b_val_out = static_cast<int>(g_val + (b_cnt > 0 ? b_cd / b_cnt : 0.0f) + 0.5f);
    }

    int g_out = static_cast<int>(g_val + 0.5f);
    if (bit_depth <= 8)
        cuda_set_rgb_clamped_8bit(rgb, x, y, width, r_val, g_out, b_val_out, max_val);
    else
        cuda_set_rgb_clamped_16bit(rgb, x, y, width, r_val, g_out, b_val_out, max_val);
}

__global__
void ahd_hv_interp_kernel(const uint8_t* bayer,
                          float* Rh, float* Gh, float* Bh,
                          float* Rv, float* Gv, float* Bv,
                          int width, int height, CudaPatternOffsets po,
                          int bit_depth, bool is_packed) {
    constexpr int BW = 32, BH = 16;
    constexpr int border = 1;
    constexpr int TILE_W = BW + 2 * border;
    constexpr int TILE_H = BH + 2 * border;

    __shared__ float tile[TILE_H][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;

    for (int row = ty; row < TILE_H; row += BH) {
        int gy = static_cast<int>(blockIdx.y) * BH + row - border;
        gy = max(0, min(gy, height - 1));
        for (int col = tx; col < TILE_W; col += BW) {
            int gx = static_cast<int>(blockIdx.x) * BW + col - border;
            gx = max(0, min(gx, width - 1));
            if (bit_depth <= 8)
                tile[row][col] = static_cast<float>(__ldg(bayer + static_cast<size_t>(gy) * width + gx));
            else {
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(bayer);
                int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(gy) * width + gx));
                if (bit_depth > 8 && bit_depth < 16) {
                    int mv = (1 << bit_depth) - 1;
                    if (v > mv) v >>= (16 - bit_depth);
                }
                tile[row][col] = static_cast<float>(v);
            }
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;

    size_t idx = static_cast<size_t>(y) * width + x;

    auto sm = [&](int lx, int ly) -> float {
        return tile[ty + border + ly][tx + border + lx];
    };

    float r_h, g_h, b_h;
    if (!cuda_is_r_at(po, y, x) && !cuda_is_b_at(po, y, x)) {
        g_h = sm(0, 0);
        if (cuda_is_r_at(po, y, x-1) || cuda_is_r_at(po, y, x+1)) {
            r_h = (sm(-1, 0) + sm(1, 0)) * 0.5f;
            b_h = (sm(0, -1) + sm(0, 1)) * 0.5f;
        } else {
            b_h = (sm(-1, 0) + sm(1, 0)) * 0.5f;
            r_h = (sm(0, -1) + sm(0, 1)) * 0.5f;
        }
    } else if (cuda_is_r_at(po, y, x)) {
        r_h = sm(0, 0);
        g_h = (sm(-1, 0) + sm(1, 0)) * 0.5f;
        b_h = (sm(-1, -1) + sm(1, -1) + sm(-1, 1) + sm(1, 1)) * 0.25f;
    } else {
        b_h = sm(0, 0);
        g_h = (sm(-1, 0) + sm(1, 0)) * 0.5f;
        r_h = (sm(-1, -1) + sm(1, -1) + sm(-1, 1) + sm(1, 1)) * 0.25f;
    }
    Rh[idx] = r_h; Gh[idx] = g_h; Bh[idx] = b_h;

    float r_v, g_v, b_v;
    if (!cuda_is_r_at(po, y, x) && !cuda_is_b_at(po, y, x)) {
        g_v = sm(0, 0);
        if (cuda_is_r_at(po, y-1, x) || cuda_is_r_at(po, y+1, x)) {
            r_v = (sm(0, -1) + sm(0, 1)) * 0.5f;
            b_v = (sm(-1, 0) + sm(1, 0)) * 0.5f;
        } else {
            b_v = (sm(0, -1) + sm(0, 1)) * 0.5f;
            r_v = (sm(-1, 0) + sm(1, 0)) * 0.5f;
        }
    } else if (cuda_is_r_at(po, y, x)) {
        r_v = sm(0, 0);
        g_v = (sm(0, -1) + sm(0, 1)) * 0.5f;
        b_v = (sm(-1, -1) + sm(-1, 1) + sm(1, -1) + sm(1, 1)) * 0.25f;
    } else {
        b_v = sm(0, 0);
        g_v = (sm(0, -1) + sm(0, 1)) * 0.5f;
        r_v = (sm(-1, -1) + sm(-1, 1) + sm(1, -1) + sm(1, 1)) * 0.25f;
    }
    Rv[idx] = r_v; Gv[idx] = g_v; Bv[idx] = b_v;
}

__global__
void ahd_chroma_kernel(const float* Rh, const float* Gh, const float* Bh,
                       const float* Rv, const float* Gv, const float* Bv,
                       float* Lh, float* Mh, float* Lv, float* Mv,
                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t idx = static_cast<size_t>(y) * width + x;
    Lh[idx] = Rh[idx] - Gh[idx]; Mh[idx] = Bh[idx] - Gh[idx];
    Lv[idx] = Rv[idx] - Gv[idx]; Mv[idx] = Bv[idx] - Gv[idx];
}

__global__
void ahd_median_kernel(float* plane, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    float w[9];
    int vi = 0;
    for (int dy = -1; dy <= 1; dy++) {
        size_t row_off = static_cast<size_t>(y + dy) * width;
        for (int dx = -1; dx <= 1; dx++) {
            w[vi++] = plane[row_off + (x + dx)];
        }
    }
    plane[static_cast<size_t>(y) * width + x] = cuda_median_3x3(w);
}

__global__
void ahd_select_kernel(const float* Rh, const float* Gh, const float* Bh,
                       const float* Rv, const float* Gv, const float* Bv,
                       const float* Lh, const float* Mh,
                       const float* Lv, const float* Mv,
                       uint8_t* rgb, int width, int height, int bit_depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int max_val = cuda_safe_max_val(bit_depth);

    auto compute_variance = [&](const float* plane, int cx, int cy) -> float {
        float mean = 0.0f; int cnt = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = max(0, min(cx + dx, width - 1));
                int ny = max(0, min(cy + dy, height - 1));
                mean += plane[static_cast<size_t>(ny) * width + nx]; cnt++;
            }
        }
        mean /= cnt;
        float var = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = max(0, min(cx + dx, width - 1));
                int ny = max(0, min(cy + dy, height - 1));
                float diff = plane[static_cast<size_t>(ny) * width + nx] - mean;
                var += diff * diff;
            }
        }
        return var / cnt;
    };

    float varLh = compute_variance(Lh, x, y);
    float varMh = compute_variance(Mh, x, y);
    float varLv = compute_variance(Lv, x, y);
    float varMv = compute_variance(Mv, x, y);

    size_t idx = static_cast<size_t>(y) * width + x;
    float finalR, finalG, finalB;
    if ((varLh + varMh) < (varLv + varMv)) {
        finalR = Rh[idx]; finalG = Gh[idx]; finalB = Bh[idx];
    } else {
        finalR = Rv[idx]; finalG = Gv[idx]; finalB = Bv[idx];
    }

    int r_out = max(0, min(static_cast<int>(finalR + 0.5f), max_val));
    int g_out = max(0, min(static_cast<int>(finalG + 0.5f), max_val));
    int b_out = max(0, min(static_cast<int>(finalB + 0.5f), max_val));

    if (bit_depth <= 8)
        cuda_set_rgb_8bit(rgb, x, y, width, r_out, g_out, b_out);
    else
        cuda_set_rgb_16bit(rgb, x, y, width, r_out, g_out, b_out);
}

__global__
void amaze_green_kernel(const uint8_t* bayer, float* G,
                        int width, int height, CudaPatternOffsets po,
                        int bit_depth, bool is_packed) {
    constexpr int BW = 32, BH = 16;
    constexpr int border = 2;
    constexpr int TILE_W = BW + 2 * border;
    constexpr int TILE_H = BH + 2 * border;

    __shared__ float tile[TILE_H][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;

    for (int row = ty; row < TILE_H; row += BH) {
        int gy = static_cast<int>(blockIdx.y) * BH + row - border;
        gy = max(0, min(gy, height - 1));
        for (int col = tx; col < TILE_W; col += BW) {
            int gx = static_cast<int>(blockIdx.x) * BW + col - border;
            gx = max(0, min(gx, width - 1));
            if (bit_depth <= 8)
                tile[row][col] = static_cast<float>(__ldg(bayer + static_cast<size_t>(gy) * width + gx));
            else {
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(bayer);
                int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(gy) * width + gx));
                if (bit_depth > 8 && bit_depth < 16) {
                    int mv = (1 << bit_depth) - 1;
                    if (v > mv) v >>= (16 - bit_depth);
                }
                tile[row][col] = static_cast<float>(v);
            }
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;
    if (x < border || x >= width - border || y < border || y >= height - border) return;

    size_t idx = static_cast<size_t>(y) * width + x;

    auto sm = [&](int lx, int ly) -> float {
        return tile[ty + border + ly][tx + border + lx];
    };

    if (!cuda_is_r_at(po, y, x) && !cuda_is_b_at(po, y, x)) {
        G[idx] = sm(0, 0);
    } else {
        float gradH = fabsf(sm(-1, 0) - sm(1, 0)) +
                      fabsf(2.0f * sm(0, 0) - sm(-2, 0) - sm(2, 0));
        float gradV = fabsf(sm(0, -1) - sm(0, 1)) +
                      fabsf(2.0f * sm(0, 0) - sm(0, -2) - sm(0, 2));
        float wh = 1.0f / (1.0f + gradH);
        float wv = 1.0f / (1.0f + gradV);
        float sum_w = wh + wv;
        wh /= sum_w; wv /= sum_w;
        float gh = (sm(-1, 0) + sm(1, 0)) * 0.5f;
        float gv = (sm(0, -1) + sm(0, 1)) * 0.5f;
        float laplacianH = 2.0f * sm(0, 0) - sm(-2, 0) - sm(2, 0);
        float laplacianV = 2.0f * sm(0, 0) - sm(0, -2) - sm(0, 2);
        gh += laplacianH * 0.25f;
        gv += laplacianV * 0.25f;
        G[idx] = wh * gh + wv * gv;
    }
}

__global__
void amaze_rb_kernel(const uint8_t* bayer, float* R, float* B, const float* G,
                     int width, int height, CudaPatternOffsets po,
                     int bit_depth, bool is_packed) {
    constexpr int BW = 32, BH = 16;
    constexpr int border = 1;
    constexpr int TILE_W = BW + 2 * border;
    constexpr int TILE_H = BH + 2 * border;

    __shared__ float tile[TILE_H][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;

    for (int row = ty; row < TILE_H; row += BH) {
        int gy = static_cast<int>(blockIdx.y) * BH + row - border;
        gy = max(0, min(gy, height - 1));
        for (int col = tx; col < TILE_W; col += BW) {
            int gx = static_cast<int>(blockIdx.x) * BW + col - border;
            gx = max(0, min(gx, width - 1));
            if (bit_depth <= 8)
                tile[row][col] = static_cast<float>(__ldg(bayer + static_cast<size_t>(gy) * width + gx));
            else {
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(bayer);
                int v = static_cast<int>(__ldg(u16 + static_cast<size_t>(gy) * width + gx));
                if (bit_depth > 8 && bit_depth < 16) {
                    int mv = (1 << bit_depth) - 1;
                    if (v > mv) v >>= (16 - bit_depth);
                }
                tile[row][col] = static_cast<float>(v);
            }
        }
    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;
    if (x < border || x >= width - border || y < border || y >= height - border) return;

    size_t idx = static_cast<size_t>(y) * width + x;
    float g = G[idx];

    auto sm = [&](int lx, int ly) -> float {
        return tile[ty + border + ly][tx + border + lx];
    };

    if (cuda_is_r_at(po, y, x)) {
        R[idx] = sm(0, 0);
        float cd_sum = 0.0f; int cd_cnt = 0;
        int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
        for (int k = 0; k < 4; k++) {
            int nx = x + diag[k][0], ny = y + diag[k][1];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height && cuda_is_b_at(po, ny, nx)) {
                cd_sum += sm(diag[k][0], diag[k][1]) - G[static_cast<size_t>(ny)*width+nx]; cd_cnt++;
            }
        }
        B[idx] = g + (cd_cnt > 0 ? cd_sum / static_cast<float>(cd_cnt) : 0.0f);
    } else if (cuda_is_b_at(po, y, x)) {
        B[idx] = sm(0, 0);
        float cd_sum = 0.0f; int cd_cnt = 0;
        int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
        for (int k = 0; k < 4; k++) {
            int nx = x + diag[k][0], ny = y + diag[k][1];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height && cuda_is_r_at(po, ny, nx)) {
                cd_sum += sm(diag[k][0], diag[k][1]) - G[static_cast<size_t>(ny)*width+nx]; cd_cnt++;
            }
        }
        R[idx] = g + (cd_cnt > 0 ? cd_sum / static_cast<float>(cd_cnt) : 0.0f);
    } else {
        float r_cd = 0.0f, b_cd = 0.0f; int r_cnt = 0, b_cnt = 0;
        if (cuda_is_r_at(po, y-1, x)) { r_cd += sm(0, -1) - G[static_cast<size_t>(y-1)*width+x]; r_cnt++; }
        if (cuda_is_r_at(po, y+1, x)) { r_cd += sm(0, 1) - G[static_cast<size_t>(y+1)*width+x]; r_cnt++; }
        if (cuda_is_r_at(po, y, x-1)) { r_cd += sm(-1, 0) - G[static_cast<size_t>(y)*width+(x-1)]; r_cnt++; }
        if (cuda_is_r_at(po, y, x+1)) { r_cd += sm(1, 0) - G[static_cast<size_t>(y)*width+(x+1)]; r_cnt++; }
        if (cuda_is_b_at(po, y-1, x)) { b_cd += sm(0, -1) - G[static_cast<size_t>(y-1)*width+x]; b_cnt++; }
        if (cuda_is_b_at(po, y+1, x)) { b_cd += sm(0, 1) - G[static_cast<size_t>(y+1)*width+x]; b_cnt++; }
        if (cuda_is_b_at(po, y, x-1)) { b_cd += sm(-1, 0) - G[static_cast<size_t>(y)*width+(x-1)]; b_cnt++; }
        if (cuda_is_b_at(po, y, x+1)) { b_cd += sm(1, 0) - G[static_cast<size_t>(y)*width+(x+1)]; b_cnt++; }
        R[idx] = g + (r_cnt > 0 ? r_cd / static_cast<float>(r_cnt) : 0.0f);
        B[idx] = g + (b_cnt > 0 ? b_cd / static_cast<float>(b_cnt) : 0.0f);
    }
}

__global__
void rcd_cfa_init_kernel(const uint8_t* bayer, float* cfa, float* rgb0, float* rgb1, float* rgb2,
                         int width, int height, CudaPatternOffsets po,
                         int bit_depth, bool is_packed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t idx = static_cast<size_t>(y) * width + x;
    int max_val = cuda_safe_max_val(bit_depth);
    float scale = static_cast<float>(max_val);
    float val;
    if (bit_depth <= 8)
        val = static_cast<float>(cuda_get_clamped_8bit(bayer, x, y, width, height)) / scale;
    else
        val = static_cast<float>(cuda_get_clamped_16bit(bayer, x, y, width, height, bit_depth)) / scale;

    cfa[idx] = val;
    int ch = cuda_fc(y, x, po);
    if (ch == 0) rgb0[idx] = val;
    else if (ch == 1) rgb1[idx] = val;
    else rgb2[idx] = val;
}

__global__
void rcd_vh_dir_kernel(const float* cfa, float* VH_Dir,
                       int width, int height, CudaPatternOffsets po) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x < 4 || x >= width - 4 || y < 4 || y >= height - 4) return;

    const float epssq = 1e-10f;
    size_t indx = static_cast<size_t>(y) * width + x;
    float V_Stat = 0.0f, H_Stat = 0.0f;
    for (int k = -4; k <= 4; k++) {
        float cv = cfa[static_cast<size_t>(y + k) * width + x];
        float ch = cfa[static_cast<size_t>(y) * width + (x + k)];
        V_Stat += fabsf(cfa[indx] - cv);
        H_Stat += fabsf(cfa[indx] - ch);
    }
    V_Stat = fmaxf(V_Stat, epssq);
    H_Stat = fmaxf(H_Stat, epssq);
    VH_Dir[indx] = V_Stat / (V_Stat + H_Stat);
}

__global__
void rcd_lpf_kernel(const float* cfa, float* lpf,
                    int width, int height, CudaPatternOffsets po) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return;

    int start_col = 2 + (cuda_fc(y, 0, po) & 1);
    if (((x - start_col) & 1) != 0) return;

    size_t indx = static_cast<size_t>(y) * width + x;
    lpf[indx] = 0.25f * cfa[indx]
        + 0.125f * (cfa[static_cast<size_t>(y-1)*width+x] + cfa[static_cast<size_t>(y+1)*width+x]
                  + cfa[static_cast<size_t>(y)*width+(x-1)] + cfa[static_cast<size_t>(y)*width+(x+1)])
        + 0.0625f * (cfa[static_cast<size_t>(y-1)*width+(x-1)] + cfa[static_cast<size_t>(y-1)*width+(x+1)]
                   + cfa[static_cast<size_t>(y+1)*width+(x-1)] + cfa[static_cast<size_t>(y+1)*width+(x+1)]);
}

__global__
void rcd_green_kernel(const float* cfa, const float* VH_Dir, const float* lpf,
                      float* rgb1,
                      int width, int height, CudaPatternOffsets po) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x < 4 || x >= width - 4 || y < 4 || y >= height - 4) return;

    int start_col = 4 + (cuda_fc(y, 0, po) & 1);
    if (((x - start_col) & 1) != 0) return;

    const float eps = 1e-5f;
    size_t indx = static_cast<size_t>(y) * width + x;

    float VH_Central = VH_Dir[indx];
    float VH_Neighbour = 0.25f * (
        VH_Dir[static_cast<size_t>(y-1)*width+(x-1)] + VH_Dir[static_cast<size_t>(y-1)*width+(x+1)] +
        VH_Dir[static_cast<size_t>(y+1)*width+(x-1)] + VH_Dir[static_cast<size_t>(y+1)*width+(x+1)]);
    float VH_Disc = (fabsf(0.5f - VH_Central) < fabsf(0.5f - VH_Neighbour)) ? VH_Neighbour : VH_Central;

    float N_Grad = eps + fabsf(cfa[static_cast<size_t>(y-1)*width+x] - cfa[static_cast<size_t>(y+1)*width+x])
        + fabsf(cfa[indx] - cfa[static_cast<size_t>(y-2)*width+x])
        + fabsf(cfa[static_cast<size_t>(y-1)*width+x] - cfa[static_cast<size_t>(y-3)*width+x]);
    float S_Grad = eps + fabsf(cfa[static_cast<size_t>(y+1)*width+x] - cfa[static_cast<size_t>(y-1)*width+x])
        + fabsf(cfa[indx] - cfa[static_cast<size_t>(y+2)*width+x])
        + fabsf(cfa[static_cast<size_t>(y+1)*width+x] - cfa[static_cast<size_t>(y+3)*width+x]);
    float W_Grad = eps + fabsf(cfa[static_cast<size_t>(y)*width+(x-1)] - cfa[static_cast<size_t>(y)*width+(x+1)])
        + fabsf(cfa[indx] - cfa[static_cast<size_t>(y)*width+(x-2)])
        + fabsf(cfa[static_cast<size_t>(y)*width+(x-1)] - cfa[static_cast<size_t>(y)*width+(x-3)]);
    float E_Grad = eps + fabsf(cfa[static_cast<size_t>(y)*width+(x+1)] - cfa[static_cast<size_t>(y)*width+(x-1)])
        + fabsf(cfa[indx] - cfa[static_cast<size_t>(y)*width+(x+2)])
        + fabsf(cfa[static_cast<size_t>(y)*width+(x+1)] - cfa[static_cast<size_t>(y)*width+(x+3)]);

    auto safe_lpf = [&](int r, int c) -> float {
        int sr = max(2, min(r, height - 3));
        int sc = max(2, min(c, width - 3));
        return lpf[static_cast<size_t>(sr) * width + sc];
    };

    float N_Est = cfa[static_cast<size_t>(y-1)*width+x] * (1.0f + (safe_lpf(y,x) - safe_lpf(y-2,x))
        / (eps + safe_lpf(y,x) + safe_lpf(y-2,x)));
    float S_Est = cfa[static_cast<size_t>(y+1)*width+x] * (1.0f + (safe_lpf(y,x) - safe_lpf(y+2,x))
        / (eps + safe_lpf(y,x) + safe_lpf(y+2,x)));
    float W_Est = cfa[static_cast<size_t>(y)*width+(x-1)] * (1.0f + (safe_lpf(y,x) - safe_lpf(y,x-2))
        / (eps + safe_lpf(y,x) + safe_lpf(y,x-2)));
    float E_Est = cfa[static_cast<size_t>(y)*width+(x+1)] * (1.0f + (safe_lpf(y,x) - safe_lpf(y,x+2))
        / (eps + safe_lpf(y,x) + safe_lpf(y,x+2)));

    float V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad);
    float H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad);
    rgb1[indx] = fmaxf(0.0f, fminf(1.0f, VH_Disc * H_Est + (1.0f - VH_Disc) * V_Est));
}

__global__
void rcd_pq_dir_kernel(const float* cfa, float* PQ_Dir,
                       int width, int height, CudaPatternOffsets po) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x < 4 || x >= width - 4 || y < 4 || y >= height - 4) return;

    int start_col = 4 + (cuda_fc(y, 0, po) & 1);
    if (((x - start_col) & 1) != 0) return;

    const float epssq = 1e-10f;
    size_t indx = static_cast<size_t>(y) * width + x;
    float P_Stat = 0.0f, Q_Stat = 0.0f;
    for (int k = -4; k <= 4; k++) {
        float cp = cfa[static_cast<size_t>(y+k)*width+(x+k)];
        float cq = cfa[static_cast<size_t>(y+k)*width+(x-k)];
        P_Stat += fabsf(cfa[indx] - cp); Q_Stat += fabsf(cfa[indx] - cq);
    }
    P_Stat = fmaxf(P_Stat, epssq); Q_Stat = fmaxf(Q_Stat, epssq);
    PQ_Dir[indx] = P_Stat / (P_Stat + Q_Stat);
}

__global__
void rcd_rb_at_green_kernel(const float* cfa, const float* rgb1, const float* PQ_Dir,
                            float* rgb0, float* rgb2,
                            int width, int height, CudaPatternOffsets po) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x < 4 || x >= width - 4 || y < 4 || y >= height - 4) return;

    int start_col = 4 + (cuda_fc(y, 0, po) & 1);
    if (((x - start_col) & 1) != 0) return;

    const float eps = 1e-5f;
    size_t indx = static_cast<size_t>(y) * width + x;
    int c = 2 - cuda_fc(y, x, po);

    float PQ_Central = PQ_Dir[indx];
    float PQ_Neighbour = 0.25f * (
        PQ_Dir[static_cast<size_t>(y-1)*width+(x-1)] + PQ_Dir[static_cast<size_t>(y-1)*width+(x+1)] +
        PQ_Dir[static_cast<size_t>(y+1)*width+(x-1)] + PQ_Dir[static_cast<size_t>(y+1)*width+(x+1)]);
    float PQ_Disc = (fabsf(0.5f - PQ_Central) < fabsf(0.5f - PQ_Neighbour)) ? PQ_Neighbour : PQ_Central;

    auto get_rgb_c = [&](int r, int cc, int channel) -> float {
        if (r < 0 || r >= height || cc < 0 || cc >= width) return 0.0f;
        size_t i = static_cast<size_t>(r) * width + cc;
        if (channel == 0) return rgb0[i];
        if (channel == 1) return rgb1[i];
        return rgb2[i];
    };

    float NW_Grad = eps + fabsf(get_rgb_c(y-1,x-1,c) - get_rgb_c(y+1,x+1,c))
        + fabsf(rgb1[indx] - rgb1[static_cast<size_t>(y-2)*width+(x-2)])
        + fabsf(get_rgb_c(y-1,x-1,c) - get_rgb_c(y-3,x-3,c));
    float NE_Grad = eps + fabsf(get_rgb_c(y-1,x+1,c) - get_rgb_c(y+1,x-1,c))
        + fabsf(rgb1[indx] - rgb1[static_cast<size_t>(y-2)*width+(x+2)])
        + fabsf(get_rgb_c(y-1,x+1,c) - get_rgb_c(y-3,x+3,c));
    float SW_Grad = eps + fabsf(get_rgb_c(y+1,x-1,c) - get_rgb_c(y-1,x+1,c))
        + fabsf(rgb1[indx] - rgb1[static_cast<size_t>(y+2)*width+(x-2)])
        + fabsf(get_rgb_c(y+1,x-1,c) - get_rgb_c(y+3,x-3,c));
    float SE_Grad = eps + fabsf(get_rgb_c(y+1,x+1,c) - get_rgb_c(y-1,x-1,c))
        + fabsf(rgb1[indx] - rgb1[static_cast<size_t>(y+2)*width+(x+2)])
        + fabsf(get_rgb_c(y+1,x+1,c) - get_rgb_c(y+3,x+3,c));

    float NW_Est = get_rgb_c(y-1,x-1,c) - rgb1[static_cast<size_t>(y-1)*width+(x-1)];
    float NE_Est = get_rgb_c(y-1,x+1,c) - rgb1[static_cast<size_t>(y-1)*width+(x+1)];
    float SW_Est = get_rgb_c(y+1,x-1,c) - rgb1[static_cast<size_t>(y+1)*width+(x-1)];
    float SE_Est = get_rgb_c(y+1,x+1,c) - rgb1[static_cast<size_t>(y+1)*width+(x+1)];

    float P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est) / (NW_Grad + SE_Grad);
    float Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est) / (NE_Grad + SW_Grad);

    float val = rgb1[indx] + (1.0f - PQ_Disc) * P_Est + PQ_Disc * Q_Est;
    val = fmaxf(0.0f, fminf(1.0f, val));

    if (c == 0) rgb0[indx] = val;
    else rgb2[indx] = val;
}

__global__
void rcd_rb_at_rb_kernel(const float* rgb0, const float* rgb1, const float* rgb2,
                         const float* VH_Dir,
                         float* rgb0_out, float* rgb2_out,
                         int width, int height, CudaPatternOffsets po) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x < 4 || x >= width - 4 || y < 4 || y >= height - 4) return;

    int start_col = 4 + (cuda_fc(y, 1, po) & 1);
    if (((x - start_col) & 1) != 0) return;

    const float eps = 1e-5f;
    size_t indx = static_cast<size_t>(y) * width + x;

    float VH_Central = VH_Dir[indx];
    float VH_Neighbour = 0.25f * (
        VH_Dir[static_cast<size_t>(y-1)*width+(x-1)] + VH_Dir[static_cast<size_t>(y-1)*width+(x+1)] +
        VH_Dir[static_cast<size_t>(y+1)*width+(x-1)] + VH_Dir[static_cast<size_t>(y+1)*width+(x+1)]);
    float VH_Disc = (fabsf(0.5f - VH_Central) < fabsf(0.5f - VH_Neighbour)) ? VH_Neighbour : VH_Central;

    for (int c = 0; c <= 2; c += 2) {
        const float* plane = (c == 0) ? rgb0 : rgb2;

        float N_Grad = eps + fabsf(rgb1[indx] - rgb1[static_cast<size_t>(y-2)*width+x])
            + fabsf(plane[static_cast<size_t>(y-1)*width+x] - plane[static_cast<size_t>(y+1)*width+x])
            + fabsf(plane[static_cast<size_t>(y-1)*width+x] - plane[static_cast<size_t>(y-3)*width+x]);
        float S_Grad = eps + fabsf(rgb1[indx] - rgb1[static_cast<size_t>(y+2)*width+x])
            + fabsf(plane[static_cast<size_t>(y+1)*width+x] - plane[static_cast<size_t>(y-1)*width+x])
            + fabsf(plane[static_cast<size_t>(y+1)*width+x] - plane[static_cast<size_t>(y+3)*width+x]);
        float W_Grad = eps + fabsf(rgb1[indx] - rgb1[static_cast<size_t>(y)*width+(x-2)])
            + fabsf(plane[static_cast<size_t>(y)*width+(x-1)] - plane[static_cast<size_t>(y)*width+(x+1)])
            + fabsf(plane[static_cast<size_t>(y)*width+(x-1)] - plane[static_cast<size_t>(y)*width+(x-3)]);
        float E_Grad = eps + fabsf(rgb1[indx] - rgb1[static_cast<size_t>(y)*width+(x+2)])
            + fabsf(plane[static_cast<size_t>(y)*width+(x+1)] - plane[static_cast<size_t>(y)*width+(x-1)])
            + fabsf(plane[static_cast<size_t>(y)*width+(x+1)] - plane[static_cast<size_t>(y)*width+(x+3)]);

        float N_CDiff = plane[static_cast<size_t>(y-1)*width+x] - rgb1[static_cast<size_t>(y-1)*width+x];
        float S_CDiff = plane[static_cast<size_t>(y+1)*width+x] - rgb1[static_cast<size_t>(y+1)*width+x];
        float W_CDiff = plane[static_cast<size_t>(y)*width+(x-1)] - rgb1[static_cast<size_t>(y)*width+(x-1)];
        float E_CDiff = plane[static_cast<size_t>(y)*width+(x+1)] - rgb1[static_cast<size_t>(y)*width+(x+1)];

        float V_Est = (S_Grad * N_CDiff + N_Grad * S_CDiff) / (N_Grad + S_Grad);
        float H_Est = (W_Grad * E_CDiff + E_Grad * W_CDiff) / (E_Grad + W_Grad);

        float val = rgb1[indx] + VH_Disc * H_Est + (1.0f - VH_Disc) * V_Est;
        val = fmaxf(0.0f, fminf(1.0f, val));

        if (c == 0) rgb0_out[indx] = val;
        else rgb2_out[indx] = val;
    }
}

__global__
void rcd_fill_borders_kernel(float* rgb0, float* rgb1, float* rgb2,
                             int width, int height, int border) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int ref_y = max(border, min(y, height - 1 - border));
    int ref_x = max(border, min(x, width - 1 - border));
    if (ref_y != y || ref_x != x) {
        size_t ri = static_cast<size_t>(ref_y) * width + ref_x;
        size_t di = static_cast<size_t>(y) * width + x;
        rgb0[di] = rgb0[ri]; rgb1[di] = rgb1[ri]; rgb2[di] = rgb2[ri];
    }
}

__global__
void rcd_to_rgb_kernel(const float* rgb0, const float* rgb1, const float* rgb2,
                       uint8_t* rgb, int width, int height, int bit_depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int max_val = cuda_safe_max_val(bit_depth);
    float scale = static_cast<float>(max_val);
    size_t idx = static_cast<size_t>(y) * width + x;
    int r_out = max(0, min(static_cast<int>(rgb0[idx] * scale + 0.5f), max_val));
    int g_out = max(0, min(static_cast<int>(rgb1[idx] * scale + 0.5f), max_val));
    int b_out = max(0, min(static_cast<int>(rgb2[idx] * scale + 0.5f), max_val));

    if (bit_depth <= 8)
        cuda_set_rgb_8bit(rgb, x, y, width, r_out, g_out, b_out);
    else
        cuda_set_rgb_16bit(rgb, x, y, width, r_out, g_out, b_out);
}

__global__
void prism_green_kernel(const float* cfa, const float* VH_Dir, const float* lpf,
                        float* rgb1,
                        int width, int height, CudaPatternOffsets po) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x < 4 || x >= width - 4 || y < 4 || y >= height - 4) return;

    int start_col = 4 + (cuda_fc(y, 0, po) & 1);
    if (((x - start_col) & 1) != 0) return;

    const float eps = 1e-5f;
    size_t indx = static_cast<size_t>(y) * width + x;

    float VH_Central = VH_Dir[indx];
    float VH_Neighbour = 0.25f * (
        VH_Dir[static_cast<size_t>(y-1)*width+(x-1)] + VH_Dir[static_cast<size_t>(y-1)*width+(x+1)] +
        VH_Dir[static_cast<size_t>(y+1)*width+(x-1)] + VH_Dir[static_cast<size_t>(y+1)*width+(x+1)]);
    float VH_Disc = (fabsf(0.5f - VH_Central) < fabsf(0.5f - VH_Neighbour)) ? VH_Neighbour : VH_Central;

    float N_Grad = eps + fabsf(cfa[static_cast<size_t>(y-1)*width+x] - cfa[static_cast<size_t>(y+1)*width+x])
        + fabsf(cfa[indx] - cfa[static_cast<size_t>(y-2)*width+x])
        + fabsf(cfa[static_cast<size_t>(y-1)*width+x] - cfa[static_cast<size_t>(y-3)*width+x]);
    float S_Grad = eps + fabsf(cfa[static_cast<size_t>(y+1)*width+x] - cfa[static_cast<size_t>(y-1)*width+x])
        + fabsf(cfa[indx] - cfa[static_cast<size_t>(y+2)*width+x])
        + fabsf(cfa[static_cast<size_t>(y+1)*width+x] - cfa[static_cast<size_t>(y+3)*width+x]);
    float W_Grad = eps + fabsf(cfa[static_cast<size_t>(y)*width+(x-1)] - cfa[static_cast<size_t>(y)*width+(x+1)])
        + fabsf(cfa[indx] - cfa[static_cast<size_t>(y)*width+(x-2)])
        + fabsf(cfa[static_cast<size_t>(y)*width+(x-1)] - cfa[static_cast<size_t>(y)*width+(x-3)]);
    float E_Grad = eps + fabsf(cfa[static_cast<size_t>(y)*width+(x+1)] - cfa[static_cast<size_t>(y)*width+(x-1)])
        + fabsf(cfa[indx] - cfa[static_cast<size_t>(y)*width+(x+2)])
        + fabsf(cfa[static_cast<size_t>(y)*width+(x+1)] - cfa[static_cast<size_t>(y)*width+(x+3)]);

    auto safe_lpf = [&](int r, int c) -> float {
        int sr = max(2, min(r, height - 3));
        int sc = max(2, min(c, width - 3));
        return lpf[static_cast<size_t>(sr) * width + sc];
    };

    float N_Est = cfa[static_cast<size_t>(y-1)*width+x] * (1.0f + (safe_lpf(y,x) - safe_lpf(y-2,x))
        / (eps + safe_lpf(y,x) + safe_lpf(y-2,x)));
    float S_Est = cfa[static_cast<size_t>(y+1)*width+x] * (1.0f + (safe_lpf(y,x) - safe_lpf(y+2,x))
        / (eps + safe_lpf(y,x) + safe_lpf(y+2,x)));
    float W_Est = cfa[static_cast<size_t>(y)*width+(x-1)] * (1.0f + (safe_lpf(y,x) - safe_lpf(y,x-2))
        / (eps + safe_lpf(y,x) + safe_lpf(y,x-2)));
    float E_Est = cfa[static_cast<size_t>(y)*width+(x+1)] * (1.0f + (safe_lpf(y,x) - safe_lpf(y,x+2))
        / (eps + safe_lpf(y,x) + safe_lpf(y,x+2)));

    float V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad);
    float H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad);

    float laplacianV = (2.0f * cfa[indx] - cfa[static_cast<size_t>(y-2)*width+x] - cfa[static_cast<size_t>(y+2)*width+x]) * 0.25f;
    float laplacianH = (2.0f * cfa[indx] - cfa[static_cast<size_t>(y)*width+(x-2)] - cfa[static_cast<size_t>(y)*width+(x+2)]) * 0.25f;
    V_Est += laplacianV;
    H_Est += laplacianH;

    rgb1[indx] = fmaxf(0.0f, fminf(1.0f, VH_Disc * H_Est + (1.0f - VH_Disc) * V_Est));
}

__global__
void prism_chroma_compute_kernel(const float* rgb0, const float* rgb1, const float* rgb2,
                                 float* L, float* M,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t idx = static_cast<size_t>(y) * width + x;
    L[idx] = rgb0[idx] - rgb1[idx];
    M[idx] = rgb2[idx] - rgb1[idx];
}

__global__
void prism_chroma_reconstruct_kernel(float* rgb0, float* rgb1, float* rgb2,
                                     const float* L, const float* M,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t idx = static_cast<size_t>(y) * width + x;
    rgb0[idx] = fmaxf(0.0f, fminf(1.0f, rgb1[idx] + L[idx]));
    rgb2[idx] = fmaxf(0.0f, fminf(1.0f, rgb1[idx] + M[idx]));
}

static const dim3 BLOCK_OPT(32, 16);

static dim3 make_grid(int width, int height, dim3 block) {
    dim3 grid;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (height + block.y - 1) / block.y;
    grid.z = 1;
    return grid;
}

void CudaGraphCache::release() {
    if (exec) {
        cudaGraphExecDestroy(exec);
        exec = nullptr;
    }
    valid = false;
}

void cuda_graph_launch(CudaGraphCache* cache) {
    if (cache && cache->valid && cache->exec) {
        cudaGraphLaunch(cache->exec, 0);
    }
}

bool cuda_graph_capture(CudaGraphCache* cache, CudaWorkspace* ws,
                        int width, int height, CudaPatternOffsets po,
                        int bit_depth, bool is_packed, CudaAlgo algo) {
    cache->release();

    size_t bayer_bytes = static_cast<size_t>(width) * height * ((bit_depth <= 8) ? 1 : 2);

    cudaStream_t capStream = ws->streams[2];
    cudaError_t err = cudaStreamBeginCapture(capStream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) return false;

    cudaMemcpyAsync(ws->d_bayer, ws->h_bayer_pinned, bayer_bytes, cudaMemcpyHostToDevice, capStream);

     cudaMemsetAsync(ws->d_rgb, 0, static_cast<size_t>(width) * height * 3 * ((bit_depth <= 8) ? 1 : 2), capStream);

     dim3 grid = make_grid(width, height, BLOCK_OPT);
    cudaStream_t s0 = ws->streams[0];
    cudaStream_t s1 = ws->streams[1];
    cudaStream_t s3 = ws->streams[3];

    switch (algo) {
    case CudaAlgo::SUPER_FAST:
        super_fast_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, ws->d_rgb,
            width, height, po, bit_depth, is_packed);
        break;
    case CudaAlgo::HQLI:
        hqli_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, ws->d_rgb,
            width, height, po, bit_depth, is_packed);
        break;
    case CudaAlgo::MG: {
        float* d_G = ws->d_fbuf[0];
        float* d_R = ws->d_fbuf[1];
        float* d_B = ws->d_fbuf[2];
        mg_green_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, d_G, width, height, po, bit_depth, is_packed);
        mg_rb_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, d_R, d_B, d_G, width, height, po, bit_depth, is_packed);
        mg_fill_borders_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_R, d_G, d_B, width, height, 2);
        planes_to_rgb_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_R, d_G, d_B, ws->d_rgb, width, height, bit_depth);
        break;
    }
    case CudaAlgo::L7:
        l7_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, ws->d_rgb,
            width, height, po, bit_depth, is_packed);
        break;
    case CudaAlgo::DFPD: {
        float* d_G = ws->d_fbuf[0];
        dfpd_green_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, d_G, width, height, po, bit_depth, is_packed);
        dfpd_rb_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, ws->d_rgb, d_G, width, height, po, bit_depth, is_packed);
        break;
    }
    case CudaAlgo::AHD: {
         float* d_Rh = ws->d_fbuf[0];
         float* d_Gh = ws->d_fbuf[1];
         float* d_Bh = ws->d_fbuf[2];
         float* d_Rv = ws->d_fbuf[3];
         float* d_Gv = ws->d_fbuf[4];
         float* d_Bv = ws->d_fbuf[5];
         float* d_Lh = ws->d_fbuf[6];
         float* d_Mh = ws->d_fbuf[7];
         float* d_Lv = ws->d_fbuf[8];
         float* d_Mv = ws->d_fbuf[9];
         ahd_hv_interp_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, d_Rh, d_Gh, d_Bh, d_Rv, d_Gv, d_Bv,
             width, height, po, bit_depth, is_packed);
         ahd_chroma_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_Rh, d_Gh, d_Bh, d_Rv, d_Gv, d_Bv,
             d_Lh, d_Mh, d_Lv, d_Mv, width, height);
         ahd_median_kernel<<<grid, BLOCK_OPT, 0, s0>>>(d_Lh, width, height);
         ahd_median_kernel<<<grid, BLOCK_OPT, 0, s1>>>(d_Mh, width, height);
         ahd_median_kernel<<<grid, BLOCK_OPT, 0, s3>>>(d_Lv, width, height);
         cudaEvent_t ev_mh, ev_lv, ev_mv;
         cudaEventCreate(&ev_mh); cudaEventCreate(&ev_lv); cudaEventCreate(&ev_mv);
         cudaEventRecord(ev_mh, s1);
         cudaEventRecord(ev_lv, s3);
         cudaStreamWaitEvent(capStream, ev_mh);
         cudaStreamWaitEvent(capStream, ev_lv);
         cudaEventDestroy(ev_mh); cudaEventDestroy(ev_lv);
         ahd_median_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_Mv, width, height);
         ahd_select_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_Rh, d_Gh, d_Bh, d_Rv, d_Gv, d_Bv,
             d_Lh, d_Mh, d_Lv, d_Mv, ws->d_rgb, width, height, bit_depth);
         cudaEventRecord(ev_mv, capStream);
         cudaStreamWaitEvent(s0, ev_mv);
         cudaEventDestroy(ev_mv);
         break;
     }
    case CudaAlgo::AMAZE: {
        float* d_G = ws->d_fbuf[0];
        float* d_R = ws->d_fbuf[1];
        float* d_B = ws->d_fbuf[2];
        amaze_green_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, d_G, width, height, po, bit_depth, is_packed);
        amaze_rb_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, d_R, d_B, d_G, width, height, po, bit_depth, is_packed);
        mg_fill_borders_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_R, d_G, d_B, width, height, 2);
        planes_to_rgb_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_R, d_G, d_B, ws->d_rgb, width, height, bit_depth);
        break;
    }
    case CudaAlgo::RCD: {
        float* d_cfa  = ws->d_fbuf[0];
        float* d_rgb0 = ws->d_fbuf[1];
        float* d_rgb1 = ws->d_fbuf[2];
        float* d_rgb2 = ws->d_fbuf[3];
        float* d_VH   = ws->d_fbuf[4];
        float* d_lpf  = ws->d_fbuf[5];
        float* d_PQ   = ws->d_fbuf[6];
        rcd_cfa_init_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, d_cfa, d_rgb0, d_rgb1, d_rgb2,
            width, height, po, bit_depth, is_packed);
        rcd_vh_dir_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_VH, width, height, po);
        rcd_lpf_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_lpf, width, height, po);
        rcd_green_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_VH, d_lpf, d_rgb1, width, height, po);
        rcd_pq_dir_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_PQ, width, height, po);
        rcd_rb_at_green_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_rgb1, d_PQ, d_rgb0, d_rgb2, width, height, po);
        rcd_rb_at_rb_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_rgb0, d_rgb1, d_rgb2, d_VH, d_rgb0, d_rgb2, width, height, po);
        rcd_fill_borders_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_rgb0, d_rgb1, d_rgb2, width, height, 4);
        rcd_to_rgb_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_rgb0, d_rgb1, d_rgb2, ws->d_rgb, width, height, bit_depth);
        break;
    }
    case CudaAlgo::PRISM: {
        float* d_cfa  = ws->d_fbuf[0];
        float* d_rgb0 = ws->d_fbuf[1];
        float* d_rgb1 = ws->d_fbuf[2];
        float* d_rgb2 = ws->d_fbuf[3];
        float* d_VH   = ws->d_fbuf[4];
        float* d_lpf  = ws->d_fbuf[5];
        float* d_PQ   = ws->d_fbuf[6];
        float* d_L    = ws->d_fbuf[7];
        float* d_M    = ws->d_fbuf[8];
        rcd_cfa_init_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(ws->d_bayer, d_cfa, d_rgb0, d_rgb1, d_rgb2,
            width, height, po, bit_depth, is_packed);
        rcd_vh_dir_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_VH, width, height, po);
        rcd_lpf_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_lpf, width, height, po);
        prism_green_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_VH, d_lpf, d_rgb1, width, height, po);
        rcd_pq_dir_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_PQ, width, height, po);
        rcd_rb_at_green_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_cfa, d_rgb1, d_PQ, d_rgb0, d_rgb2, width, height, po);
        rcd_rb_at_rb_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_rgb0, d_rgb1, d_rgb2, d_VH, d_rgb0, d_rgb2, width, height, po);
        prism_chroma_compute_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_rgb0, d_rgb1, d_rgb2, d_L, d_M, width, height);
        ahd_median_kernel<<<grid, BLOCK_OPT, 0, s0>>>(d_L, width, height);
        ahd_median_kernel<<<grid, BLOCK_OPT, 0, s1>>>(d_M, width, height);
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, s0); cudaEventRecord(ev1, s1);
        cudaStreamWaitEvent(capStream, ev0); cudaStreamWaitEvent(capStream, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        prism_chroma_reconstruct_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_rgb0, d_rgb1, d_rgb2, d_L, d_M, width, height);
        rcd_fill_borders_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_rgb0, d_rgb1, d_rgb2, width, height, 4);
        rcd_to_rgb_kernel<<<grid, BLOCK_OPT, 0, capStream>>>(d_rgb0, d_rgb1, d_rgb2, ws->d_rgb, width, height, bit_depth);
        break;
    }
    default:
        cudaStreamEndCapture(capStream, nullptr);
        return false;
    }

    cudaMemcpyAsync(ws->h_rgb_pinned, ws->d_rgb,
        static_cast<size_t>(width) * height * 3 * ((bit_depth <= 8) ? 1 : 2),
        cudaMemcpyDeviceToHost, capStream);

    cudaGraph_t graph;
     err = cudaStreamEndCapture(capStream, &graph);
     if (err != cudaSuccess) return false;

     cudaGraphInstantiate(&cache->exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);

    if (!cache->exec) return false;

    cache->width = width;
    cache->height = height;
    cache->bit_depth = bit_depth;
    cache->po = po;
    cache->algo = algo;
    cache->valid = true;
    return true;
}

void cuda_launch_super_fast(CudaWorkspace* ws,
                            int width, int height, CudaPatternOffsets po,
                            int bit_depth, bool is_packed) {
    dim3 grid = make_grid(width, height, BLOCK_OPT);
    super_fast_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, ws->d_rgb,
                                           width, height, po, bit_depth, is_packed);
}

void cuda_launch_hqli(CudaWorkspace* ws,
                      int width, int height, CudaPatternOffsets po,
                      int bit_depth, bool is_packed) {
    dim3 grid = make_grid(width, height, BLOCK_OPT);
    hqli_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, ws->d_rgb,
                                     width, height, po, bit_depth, is_packed);
}

void cuda_launch_mg(CudaWorkspace* ws,
                    int width, int height, CudaPatternOffsets po,
                    int bit_depth, bool is_packed) {
    float* d_G = ws->d_fbuf[0];
    float* d_R = ws->d_fbuf[1];
    float* d_B = ws->d_fbuf[2];

    dim3 grid = make_grid(width, height, BLOCK_OPT);

    mg_green_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, d_G, width, height, po, bit_depth, is_packed);
    mg_rb_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, d_R, d_B, d_G, width, height, po, bit_depth, is_packed);
    mg_fill_borders_kernel<<<grid, BLOCK_OPT>>>(d_R, d_G, d_B, width, height, 2);
    planes_to_rgb_kernel<<<grid, BLOCK_OPT>>>(d_R, d_G, d_B, ws->d_rgb, width, height, bit_depth);

    cudaDeviceSynchronize();
}

void cuda_launch_l7(CudaWorkspace* ws,
                    int width, int height, CudaPatternOffsets po,
                    int bit_depth, bool is_packed) {
    dim3 grid = make_grid(width, height, BLOCK_OPT);
    l7_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, ws->d_rgb,
                                   width, height, po, bit_depth, is_packed);
}

void cuda_launch_dfpd(CudaWorkspace* ws,
                      int width, int height, CudaPatternOffsets po,
                      int bit_depth, bool is_packed) {
    float* d_G = ws->d_fbuf[0];

    dim3 grid = make_grid(width, height, BLOCK_OPT);

    dfpd_green_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, d_G, width, height, po, bit_depth, is_packed);
    dfpd_rb_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, ws->d_rgb, d_G, width, height, po, bit_depth, is_packed);

    cudaDeviceSynchronize();
}

void cuda_launch_ahd(CudaWorkspace* ws,
                     int width, int height, CudaPatternOffsets po,
                     int bit_depth, bool is_packed) {
    float* d_Rh = ws->d_fbuf[0];
    float* d_Gh = ws->d_fbuf[1];
    float* d_Bh = ws->d_fbuf[2];
    float* d_Rv = ws->d_fbuf[3];
    float* d_Gv = ws->d_fbuf[4];
    float* d_Bv = ws->d_fbuf[5];
    float* d_Lh = ws->d_fbuf[6];
    float* d_Mh = ws->d_fbuf[7];
    float* d_Lv = ws->d_fbuf[8];
    float* d_Mv = ws->d_fbuf[9];

    dim3 grid = make_grid(width, height, BLOCK_OPT);

    ahd_hv_interp_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, d_Rh, d_Gh, d_Bh, d_Rv, d_Gv, d_Bv,
                                              width, height, po, bit_depth, is_packed);

    ahd_chroma_kernel<<<grid, BLOCK_OPT>>>(d_Rh, d_Gh, d_Bh, d_Rv, d_Gv, d_Bv,
                                           d_Lh, d_Mh, d_Lv, d_Mv, width, height);

    cudaStream_t s0 = ws->streams[0];
    cudaStream_t s1 = ws->streams[1];
    cudaStream_t s2 = ws->streams[2];
    cudaStream_t s3 = ws->streams[3];
    ahd_median_kernel<<<grid, BLOCK_OPT, 0, s0>>>(d_Lh, width, height);
    ahd_median_kernel<<<grid, BLOCK_OPT, 0, s1>>>(d_Mh, width, height);
    ahd_median_kernel<<<grid, BLOCK_OPT, 0, s2>>>(d_Lv, width, height);
    ahd_median_kernel<<<grid, BLOCK_OPT, 0, s3>>>(d_Mv, width, height);

    cudaDeviceSynchronize();

    ahd_select_kernel<<<grid, BLOCK_OPT>>>(d_Rh, d_Gh, d_Bh, d_Rv, d_Gv, d_Bv,
                                           d_Lh, d_Mh, d_Lv, d_Mv,
                                           ws->d_rgb, width, height, bit_depth);

    cudaDeviceSynchronize();
}

void cuda_launch_amaze(CudaWorkspace* ws,
                       int width, int height, CudaPatternOffsets po,
                       int bit_depth, bool is_packed) {
    float* d_G = ws->d_fbuf[0];
    float* d_R = ws->d_fbuf[1];
    float* d_B = ws->d_fbuf[2];

    dim3 grid = make_grid(width, height, BLOCK_OPT);

    amaze_green_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, d_G, width, height, po, bit_depth, is_packed);
    amaze_rb_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, d_R, d_B, d_G, width, height, po, bit_depth, is_packed);
    mg_fill_borders_kernel<<<grid, BLOCK_OPT>>>(d_R, d_G, d_B, width, height, 2);
    planes_to_rgb_kernel<<<grid, BLOCK_OPT>>>(d_R, d_G, d_B, ws->d_rgb, width, height, bit_depth);

    cudaDeviceSynchronize();
}

void cuda_launch_rcd(CudaWorkspace* ws,
                     int width, int height, CudaPatternOffsets po,
                     int bit_depth, bool is_packed) {
    float* d_cfa  = ws->d_fbuf[0];
    float* d_rgb0 = ws->d_fbuf[1];
    float* d_rgb1 = ws->d_fbuf[2];
    float* d_rgb2 = ws->d_fbuf[3];
    float* d_VH   = ws->d_fbuf[4];
    float* d_lpf  = ws->d_fbuf[5];
    float* d_PQ   = ws->d_fbuf[6];

    dim3 grid = make_grid(width, height, BLOCK_OPT);

    rcd_cfa_init_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, d_cfa, d_rgb0, d_rgb1, d_rgb2,
                                             width, height, po, bit_depth, is_packed);
    rcd_vh_dir_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_VH, width, height, po);
    rcd_lpf_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_lpf, width, height, po);
    rcd_green_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_VH, d_lpf, d_rgb1, width, height, po);
    rcd_pq_dir_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_PQ, width, height, po);
    rcd_rb_at_green_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_rgb1, d_PQ, d_rgb0, d_rgb2, width, height, po);
    rcd_rb_at_rb_kernel<<<grid, BLOCK_OPT>>>(d_rgb0, d_rgb1, d_rgb2, d_VH, d_rgb0, d_rgb2, width, height, po);
    rcd_fill_borders_kernel<<<grid, BLOCK_OPT>>>(d_rgb0, d_rgb1, d_rgb2, width, height, 4);
    rcd_to_rgb_kernel<<<grid, BLOCK_OPT>>>(d_rgb0, d_rgb1, d_rgb2, ws->d_rgb, width, height, bit_depth);

    cudaDeviceSynchronize();
}

void cuda_launch_prism(CudaWorkspace* ws,
                       int width, int height, CudaPatternOffsets po,
                       int bit_depth, bool is_packed) {
    float* d_cfa  = ws->d_fbuf[0];
    float* d_rgb0 = ws->d_fbuf[1];
    float* d_rgb1 = ws->d_fbuf[2];
    float* d_rgb2 = ws->d_fbuf[3];
    float* d_VH   = ws->d_fbuf[4];
    float* d_lpf  = ws->d_fbuf[5];
    float* d_PQ   = ws->d_fbuf[6];
    float* d_L    = ws->d_fbuf[7];
    float* d_M    = ws->d_fbuf[8];

    dim3 grid = make_grid(width, height, BLOCK_OPT);

    rcd_cfa_init_kernel<<<grid, BLOCK_OPT>>>(ws->d_bayer, d_cfa, d_rgb0, d_rgb1, d_rgb2,
                                             width, height, po, bit_depth, is_packed);
    rcd_vh_dir_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_VH, width, height, po);
    rcd_lpf_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_lpf, width, height, po);
    prism_green_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_VH, d_lpf, d_rgb1, width, height, po);
    rcd_pq_dir_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_PQ, width, height, po);
    rcd_rb_at_green_kernel<<<grid, BLOCK_OPT>>>(d_cfa, d_rgb1, d_PQ, d_rgb0, d_rgb2, width, height, po);
    rcd_rb_at_rb_kernel<<<grid, BLOCK_OPT>>>(d_rgb0, d_rgb1, d_rgb2, d_VH, d_rgb0, d_rgb2, width, height, po);
    prism_chroma_compute_kernel<<<grid, BLOCK_OPT>>>(d_rgb0, d_rgb1, d_rgb2, d_L, d_M, width, height);

    cudaStream_t s0 = ws->streams[0];
    cudaStream_t s1 = ws->streams[1];
    ahd_median_kernel<<<grid, BLOCK_OPT, 0, s0>>>(d_L, width, height);
    ahd_median_kernel<<<grid, BLOCK_OPT, 0, s1>>>(d_M, width, height);

    cudaDeviceSynchronize();

    prism_chroma_reconstruct_kernel<<<grid, BLOCK_OPT>>>(d_rgb0, d_rgb1, d_rgb2, d_L, d_M, width, height);

    rcd_fill_borders_kernel<<<grid, BLOCK_OPT>>>(d_rgb0, d_rgb1, d_rgb2, width, height, 4);
    rcd_to_rgb_kernel<<<grid, BLOCK_OPT>>>(d_rgb0, d_rgb1, d_rgb2, ws->d_rgb, width, height, bit_depth);

    cudaDeviceSynchronize();
}
