#include "common.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

namespace denoise {

// ── Haar 2D Discrete Wavelet Transform with soft thresholding ─────────
//
// Decomposition (1 level per pass, iterated for multi-level):
//   Each level: rows → columns → yields LL, LH, HL, HH subbands.
//   LL is recursively decomposed for the next level.
//
// Thresholding:
//   VisuShrink: T = sigma * sqrt(2 * log(N))
//   sigma estimated from HH1 detail coefficients via MAD:
//     sigma = median(|HH1|) / 0.6745
//   Soft threshold: sign(x) * max(|x| - T, 0)
//   strength scales T linearly (1.0 → full threshold, 0.5 → half, etc.)
//
// Reconstruction: inverse of decomposition.
//
// Padding: image is truncated to largest power-of-2 dimensions to avoid
// edge artifacts. The wavelet transform operates on the power-of-2
// sub-region only. Border pixels (outside the power-of-2 region) are
// copied unchanged.

namespace {

// ── 1D Haar transform (in-place on a row/column) ──────────────────────

// Orthonormal Haar: forward = [a+b, a-b]/sqrt(2), inverse = [c+d, c-d]/sqrt(2)
// Layout: first half = approximation, second half = detail.

void haar_1d_decompose(float* data, int n, int stride) {
    std::vector<float> tmp(n);
    int half = n / 2;
    float inv_sqrt2 = 1.0f / 1.414213562f;
    for (int i = 0; i < half; i++) {
        float a = data[(2 * i) * stride];
        float b = data[(2 * i + 1) * stride];
        tmp[i]         = (a + b) * inv_sqrt2;
        tmp[half + i]  = (a - b) * inv_sqrt2;
    }
    for (int i = 0; i < n; i++) {
        data[i * stride] = tmp[i];
    }
}

void haar_1d_reconstruct(float* data, int n, int stride) {
    std::vector<float> tmp(n);
    int half = n / 2;
    float inv_sqrt2 = 1.0f / 1.414213562f;
    for (int i = 0; i < half; i++) {
        float c = data[i * stride];
        float d = data[(half + i) * stride];
        tmp[2 * i]     = (c + d) * inv_sqrt2;
        tmp[2 * i + 1] = (c - d) * inv_sqrt2;
    }
    for (int i = 0; i < n; i++) {
        data[i * stride] = tmp[i];
    }
}

// ── 2D DWT on a square power-of-2 plane ───────────────────────────────
//
// plane is `full_stride × full_stride` (row-major).
// Each level processes the top-left `sub_size × sub_size` block.
// Column stride is always `full_stride`.

void dwt_2d_decompose_level(float* plane, int sub_size, int stride) {
    for (int y = 0; y < sub_size; y++) {
        haar_1d_decompose(plane + static_cast<size_t>(y) * stride, sub_size, 1);
    }
    for (int x = 0; x < sub_size; x++) {
        haar_1d_decompose(plane + x, sub_size, stride);
    }
}

void dwt_2d_reconstruct_level(float* plane, int sub_size, int stride) {
    // Vertical inverse first
    for (int x = 0; x < sub_size; x++) {
        haar_1d_reconstruct(plane + x, sub_size, stride);
    }
    // Horizontal inverse
    for (int y = 0; y < sub_size; y++) {
        haar_1d_reconstruct(plane + static_cast<size_t>(y) * stride, sub_size, 1);
    }
}

void dwt_2d_decompose(float* plane, int full_stride, int levels) {
    int cur_size = full_stride;
    for (int level = 0; level < levels; level++) {
        dwt_2d_decompose_level(plane, cur_size, full_stride);
        cur_size /= 2;
    }
}

void dwt_2d_reconstruct(float* plane, int full_stride, int levels) {
    // Reconstruct must replay decompositions in exact reverse order.
    // Decompose sizes: full_stride, full_stride/2, ..., full_stride>>(levels-1)
    // The LAST decompose called dwt_2d_decompose_level(plane, full_stride>>(levels-1), ...).
    // So the FIRST reconstruct must call dwt_2d_reconstruct_level with the SAME size:
    //   cur_size = full_stride >> (levels - 1)
    // After that, cur_size doubles each iteration back to full_stride.
    int cur_size = full_stride >> (levels - 1);
    for (int level = 0; level < levels; level++) {
        dwt_2d_reconstruct_level(plane, cur_size, full_stride);
        cur_size *= 2;
    }
}

// ── Thresholding ──────────────────────────────────────────────────────

float estimate_sigma(float* plane, int stride) {
    int half = stride / 2;
    std::vector<float> hh1;
    hh1.reserve(static_cast<size_t>(half) * half);
    for (int y = half; y < stride; y++) {
        for (int x = half; x < stride; x++) {
            hh1.push_back(std::abs(plane[static_cast<size_t>(y) * stride + x]));
        }
    }
    if (hh1.empty()) return 0.0f;
    size_t mid = hh1.size() / 2;
    std::nth_element(hh1.begin(), hh1.begin() + mid, hh1.end());
    return hh1[mid] / 0.6745f;
}

void soft_threshold_details(float* plane, int stride, int levels, float threshold) {
    if (threshold <= 0.0f) return;

    int cur_size = stride;
    for (int level = 0; level < levels; level++) {
        int half = cur_size / 2;
        // LH: (0..half-1, half..cur_size-1)
        for (int y = 0; y < half; y++) {
            for (int x = half; x < cur_size; x++) {
                float& v = plane[static_cast<size_t>(y) * stride + x];
                if (v > threshold) v -= threshold;
                else if (v < -threshold) v += threshold;
                else v = 0.0f;
            }
        }
        // HL: (half..cur_size-1, 0..half-1)
        for (int y = half; y < cur_size; y++) {
            for (int x = 0; x < half; x++) {
                float& v = plane[static_cast<size_t>(y) * stride + x];
                if (v > threshold) v -= threshold;
                else if (v < -threshold) v += threshold;
                else v = 0.0f;
            }
        }
        // HH: (half..cur_size-1, half..cur_size-1)
        for (int y = half; y < cur_size; y++) {
            for (int x = half; x < cur_size; x++) {
                float& v = plane[static_cast<size_t>(y) * stride + x];
                if (v > threshold) v -= threshold;
                else if (v < -threshold) v += threshold;
                else v = 0.0f;
            }
        }
        cur_size = half;
    }
}

// ── Power-of-2 helpers ────────────────────────────────────────────────

int largest_pow2_le(int n) {
    int p = 1;
    while ((p << 1) <= n) p <<= 1;
    return p;
}

} // anonymous namespace

// ── Public API ─────────────────────────────────────────────────────────

DenoiseError process_wavelet(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth, float strength) {
    DenoiseError err = validate_denoise_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != DenoiseError::Ok) return err;

    // Need at least 4x4 for meaningful wavelet decomposition
    if (width < 4 || height < 4) return DenoiseError::ImageTooSmall;

    // Use largest power-of-2 subregion ≤ min(width, height) to avoid
    // padding artifacts. Non-power-of-2 borders are left unchanged.
    int dwt_size = largest_pow2_le(std::min(width, height));
    if (dwt_size < 4) dwt_size = 4;

    // Determine levels: log2(dwt_size) - 1, capped at 4
    int levels = 0;
    int s = dwt_size;
    while (s >= 8) { s /= 2; levels++; }
    if (levels < 1) levels = 1;
    if (levels > 4) levels = 4;

    size_t plane_pixels = static_cast<size_t>(dwt_size) * dwt_size;
    std::vector<float> work_plane(plane_pixels);

    int max_val = detail::safe_max_val(bit_depth);

    for (int c = 0; c < channels; c++) {
        // Extract power-of-2 region into work plane
        for (int y = 0; y < dwt_size; y++) {
            for (int x = 0; x < dwt_size; x++) {
                int val = detail::read_pixel(input, x, y, width, channels, bit_depth, c);
                work_plane[static_cast<size_t>(y) * dwt_size + x] = static_cast<float>(val);
            }
        }

        // Forward DWT
        dwt_2d_decompose(work_plane.data(), dwt_size, levels);

        // Estimate sigma from HH1
        float sigma = estimate_sigma(work_plane.data(), dwt_size);

        // Threshold proportional to estimated noise sigma.
        // k ≈ 3 gives moderate denoising; scaled by strength.
        float threshold = sigma * 3.0f * strength;

        // Soft threshold detail coefficients
        soft_threshold_details(work_plane.data(), dwt_size, levels, threshold);

        // Inverse DWT
        dwt_2d_reconstruct(work_plane.data(), dwt_size, levels);

        // Write back power-of-2 region
        for (int y = 0; y < dwt_size; y++) {
            for (int x = 0; x < dwt_size; x++) {
                int val = detail::clamp_val(
                    static_cast<int>(work_plane[static_cast<size_t>(y) * dwt_size + x] + 0.5f),
                    max_val);
                detail::write_pixel(output, x, y, width, channels, bit_depth, c, val);
            }
        }

        // Copy border pixels (outside power-of-2 region) unchanged
        for (int y = 0; y < height; y++) {
            if (y < dwt_size) {
                for (int x = dwt_size; x < width; x++) {
                    int val = detail::read_pixel(input, x, y, width, channels, bit_depth, c);
                    detail::write_pixel(output, x, y, width, channels, bit_depth, c, val);
                }
            } else {
                for (int x = 0; x < width; x++) {
                    int val = detail::read_pixel(input, x, y, width, channels, bit_depth, c);
                    detail::write_pixel(output, x, y, width, channels, bit_depth, c, val);
                }
            }
        }
    }

    return DenoiseError::Ok;
}

} // namespace denoise
