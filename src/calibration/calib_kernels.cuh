#ifndef CALIB_KERNELS_CUH
#define CALIB_KERNELS_CUH

#include <cstdint>

// ============================================================
//  Calibration GPU kernel declarations
// ============================================================

// RGB to grayscale: input RGB interleaved, output single-channel float gray
void cuda_launch_checkerboard_gray(const uint8_t* d_rgb, float* d_gray,
                                     int w, int h, int channels, int bit_depth);

// Corner response: 5x5 stencil on gray image, output float scores
// Uses Harris-like eigenvalue approximation: det(M) - k*trace(M)^2
void cuda_launch_checkerboard_corner_scores(const float* d_gray, float* d_scores,
                                              int w, int h, float k);

// Non-maximum suppression: find local maxima in 5x5 window,
// output binary mask (1.0f = candidate corner)
void cuda_launch_checkerboard_nms(const float* d_scores, float* d_mask,
                                    int w, int h, float threshold);

#endif // CALIB_KERNELS_CUH
