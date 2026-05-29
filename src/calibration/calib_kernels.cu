#include "calib_kernels.cuh"
#include "../gpu_common.cuh"
#include <cuda_runtime.h>

// ============================================================
//  Checkerboard detection GPU kernels
// ============================================================

// --- Grayscale conversion ---
__global__ void kern_checkerboard_gray(const uint8_t* __restrict__ rgb,
                                         float* __restrict__ gray,
                                         int w, int h, int ch, int bd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int mv = (bd <= 8) ? 255 : ((1 << bd) - 1);
    float maxv = static_cast<float>(mv);
    if (ch >= 3) {
        float r = dev_read_pixel(rgb, x, y, w, bd, 0, maxv);
        float g = dev_read_pixel(rgb, x, y, w, bd, 1, maxv);
        float b = dev_read_pixel(rgb, x, y, w, bd, 2, maxv);
        gray[y * w + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    } else {
        gray[y * w + x] = dev_read_pixel(rgb, x, y, w, bd, 0, maxv);
    }
}

void cuda_launch_checkerboard_gray(const uint8_t* d_rgb, float* d_gray,
                                     int w, int h, int channels, int bit_depth) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_checkerboard_gray<<<grid, block>>>(d_rgb, d_gray, w, h, channels, bit_depth);
}

// --- Corner score computation (matches CPU checkerboard_detect.cpp) ---
__global__ void kern_checkerboard_corner_scores(const float* __restrict__ gray,
                                                  float* __restrict__ scores,
                                                  int w, int h, float k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 3 || y < 3 || x >= w-3 || y >= h-3) {
        if (x < w && y < h) scores[y * w + x] = 0.0f;
        return;
    }
    // Accumulate structure tensor over 5x5 window using central-difference gradients
    float ixx = 0.0f, iyy = 0.0f, ixy = 0.0f;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int nx = x + dx, ny = y + dy;
            int xp1 = (nx + 1 < w) ? nx + 1 : nx;
            int xm1 = (nx - 1 >= 0) ? nx - 1 : nx;
            int yp1 = (ny + 1 < h) ? ny + 1 : ny;
            int ym1 = (ny - 1 >= 0) ? ny - 1 : ny;
            float gx = gray[yp1 * w + nx] - gray[ym1 * w + nx];
            float gy = gray[ny * w + xp1] - gray[ny * w + xm1];
            ixx += gx * gx;
            iyy += gy * gy;
            ixy += gx * gy;
        }
    }
    float det = ixx * iyy - ixy * ixy;
    float trace = ixx + iyy + 1e-6f;
    scores[y * w + x] = det / (trace * trace + 1e-10f);
}

void cuda_launch_checkerboard_corner_scores(const float* d_gray, float* d_scores,
                                              int w, int h, float k) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_checkerboard_corner_scores<<<grid, block>>>(d_gray, d_scores, w, h, k);
}

// --- Non-maximum suppression ---
__global__ void kern_checkerboard_nms(const float* __restrict__ scores,
                                        float* __restrict__ mask,
                                        int w, int h, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 2 || y < 2 || x >= w-2 || y >= h-2) {
        if (x < w && y < h) mask[y * w + x] = 0.0f;
        return;
    }
    float center = scores[y * w + x];
    mask[y * w + x] = 0.0f;
    if (center < threshold) return;
    // Check if center is local maximum in 5x5 window
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (scores[(y+dy) * w + (x+dx)] >= center) return;
        }
    }
    mask[y * w + x] = 1.0f;
}

void cuda_launch_checkerboard_nms(const float* d_scores, float* d_mask,
                                    int w, int h, float threshold) {
    dim3 block(32, 16), grid((w+31)/32, (h+15)/16);
    kern_checkerboard_nms<<<grid, block>>>(d_scores, d_mask, w, h, threshold);
}
