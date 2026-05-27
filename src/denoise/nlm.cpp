#include "common.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace denoise {

// Non-Local Means denoising.
//
// For each pixel p, compute weighted average over a search window S:
//   I'(p) = sum_{q in S} w(p,q) * I(q) / sum w(p,q)
//
// Weight based on patch similarity:
//   w(p,q) = exp( -||patch(p) - patch(q)||^2 / h^2 )
//
// patch(p): small window (patch_radius) around p
// search window S: larger window (search_radius) around p
//
// strength controls h (filtering parameter).

namespace {

template<int BPC>
void nlm_core(const uint8_t* input, uint8_t* output,
              int width, int height, int channels, int bit_depth,
              int patch_radius, int search_radius, float h) {
    int max_val = detail::safe_max_val(bit_depth);
    int patch_w = 2 * patch_radius + 1;
    int patch_size = patch_w * patch_w * channels;
    float h2 = h * h;

    // Pre-extract patch for every pixel
    // For memory efficiency, we extract patches lazily (compute on the fly)

    std::vector<float> center_patch;
    center_patch.reserve(patch_size);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                // Build center patch (reuse allocated vector)
                center_patch.clear();
                for (int pdy = -patch_radius; pdy <= patch_radius; pdy++) {
                    int pny = y + pdy;
                    if (pny < 0) pny = 0;
                    if (pny >= height) pny = height - 1;
                    for (int pdx = -patch_radius; pdx <= patch_radius; pdx++) {
                        int pnx = x + pdx;
                        if (pnx < 0) pnx = 0;
                        if (pnx >= width) pnx = width - 1;
                        if (channels == 1) {
                            center_patch.push_back(static_cast<float>(
                                detail::read_pixel(input, pnx, pny, width,
                                                    channels, bit_depth, 0)));
                        } else {
                            // For multi-channel, include all channels in patch distance
                            for (int cc = 0; cc < channels; cc++) {
                                center_patch.push_back(static_cast<float>(
                                    detail::read_pixel(input, pnx, pny, width,
                                                        channels, bit_depth, cc)));
                            }
                        }
                    }
                }

                // Search for similar patches in search window
                float weighted_sum = 0.0f;
                float weight_sum = 0.0f;
                float max_weight = 1.0f;  // ensure non-zero for center pixel weight

                int step = 1; // Step size for search (every pixel)
                for (int sdy = -search_radius; sdy <= search_radius; sdy += step) {
                    for (int sdx = -search_radius; sdx <= search_radius; sdx += step) {
                        int qx = x + sdx;
                        int qy = y + sdy;
                        if (qx < 0) qx = 0;
                        if (qx >= width) qx = width - 1;
                        if (qy < 0) qy = 0;
                        if (qy >= height) qy = height - 1;

                        // Build neighbor patch and compute SSD
                        float dist = 0.0f;
                        int pi = 0;
                        for (int pdy = -patch_radius; pdy <= patch_radius; pdy++) {
                            int pny = y + pdy;
                            if (pny < 0) pny = 0;
                            if (pny >= height) pny = height - 1;
                            int qny = qy + pdy;
                            if (qny < 0) qny = 0;
                            if (qny >= height) qny = height - 1;

                            for (int pdx = -patch_radius; pdx <= patch_radius; pdx++) {
                                int pnx = x + pdx;
                                if (pnx < 0) pnx = 0;
                                if (pnx >= width) pnx = width - 1;
                                int qnx = qx + pdx;
                                if (qnx < 0) qnx = 0;
                                if (qnx >= width) qnx = width - 1;

                                float diff;
                                if (channels == 1) {
                                    diff = static_cast<float>(
                                        detail::read_pixel(input, pnx, pny, width,
                                                            channels, bit_depth, 0) -
                                        detail::read_pixel(input, qnx, qny, width,
                                                            channels, bit_depth, 0));
                                } else {
                                    // Combine all channels for multi-channel
                                    diff = 0.0f;
                                    for (int cc = 0; cc < channels; cc++) {
                                        float d = static_cast<float>(
                                            detail::read_pixel(input, pnx, pny, width,
                                                                channels, bit_depth, cc) -
                                            detail::read_pixel(input, qnx, qny, width,
                                                                channels, bit_depth, cc));
                                        diff += d * d;
                                    }
                                    diff = std::sqrt(diff);
                                }
                                dist += diff * diff;
                                pi++;
                            }
                        }

                        float w = std::exp(-dist / h2);
                        if (w > max_weight) max_weight = w;

                        int q_val;
                        if (channels == 1) {
                            q_val = detail::read_pixel(input, qx, qy, width,
                                                       1, bit_depth, 0);
                        } else {
                            q_val = detail::read_pixel(input, qx, qy, width,
                                                       channels, bit_depth, c);
                        }

                        weighted_sum += w * static_cast<float>(q_val);
                        weight_sum += w;
                    }
                }

                // Include center pixel with max weight to avoid over-smoothing
                // (standard trick: set self-weight to max neighbor weight)
                int center_val;
                if (channels == 1) {
                    center_val = detail::read_pixel(input, x, y, width, 1, bit_depth, 0);
                } else {
                    center_val = detail::read_pixel(input, x, y, width,
                                                     channels, bit_depth, c);
                }
                weighted_sum += max_weight * static_cast<float>(center_val);
                weight_sum += max_weight;

                int result = detail::clamp_val(
                    static_cast<int>((weighted_sum / weight_sum) + 0.5f), max_val);
                detail::write_pixel(output, x, y, width, channels, bit_depth,
                                    c, result);
            }
        }
    }
}

} // anonymous namespace

DenoiseError process_nlm(const uint8_t* input, uint8_t* output,
                          int width, int height, int channels,
                          int bit_depth, float strength) {
    DenoiseError err = validate_denoise_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != DenoiseError::Ok) return err;

    // Need enough room for patch + search windows
    int min_size = 7;
    if (width < min_size || height < min_size) return DenoiseError::ImageTooSmall;

    // strength -> h (filtering parameter)
    // For 8-bit: h ~ 10 * strength. For higher bits, scale proportionally.
    float max_val = static_cast<float>(detail::safe_max_val(bit_depth));
    float h = strength * 10.0f * (max_val / 255.0f);

    // Patch radius 1 (3x3 patches), search radius 3 (7x7 search window)
    int patch_radius = 1;
    int search_radius = 3;

    if (bit_depth <= 8) {
        nlm_core<1>(input, output, width, height, channels, bit_depth,
                     patch_radius, search_radius, h);
    } else {
        nlm_core<2>(input, output, width, height, channels, bit_depth,
                     patch_radius, search_radius, h);
    }

    return DenoiseError::Ok;
}

} // namespace denoise
