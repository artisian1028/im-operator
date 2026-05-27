#include "common.hpp"
#include "highlight_reconstruct/algorithms.hpp"
#include <cstring>

namespace highlight_reconstruct {

// Channel-guided highlight reconstruction:
// When a channel clips (== max_val), use unclipped channels' ratios to estimate the original value.
// The assumption: in overexposed regions, the relative ratio between R/G/B is approximately
// constant (dictated by the scene's color + white balance).
//
// For a pixel where R is clipped but G,B are not:
//   R' = max(G, B) * ratio_guess
// where ratio_guess is derived from nearby unclipped pixels or from a white balance assumption.
HighlightReconstructError process_channel_guided(const uint8_t* input, uint8_t* output,
                                                  int width, int height, int channels,
                                                  int bit_depth,
                                                  const HighlightReconstructParams& params) {
    auto err = validate_highlight_reconstruct_inputs(input, output, width, height, channels, bit_depth);
    if (err != HighlightReconstructError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float threshold = std::max(0.5f, std::min(1.0f, params.threshold));
    int clip_thresh = static_cast<int>(threshold * static_cast<float>(max_val));

    // Copy input to output
    size_t total = static_cast<size_t>(width) * height * 3;
    size_t bytes = (bit_depth <= 8) ? total : total * 2;
    std::memcpy(output, input, bytes);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int r = detail::read_pixel(output, x, y, width, channels, bit_depth, 0);
            int g = detail::read_pixel(output, x, y, width, channels, bit_depth, 1);
            int b = detail::read_pixel(output, x, y, width, channels, bit_depth, 2);

            bool r_clipped = (r >= clip_thresh);
            bool g_clipped = (g >= clip_thresh);
            bool b_clipped = (b >= clip_thresh);

            int clipped_count = (r_clipped ? 1 : 0) + (g_clipped ? 1 : 0) + (b_clipped ? 1 : 0);

            // Need at least 1 unclipped channel and at most 2 clipped channels
            if (clipped_count == 0 || clipped_count == 3) continue;

            // Estimate clipped channels based on unclipped ones
            // Use the maximum unclipped value as reference, scale by expected channel ratio
            float unclip_max = 0.0f;
            if (!r_clipped) unclip_max = std::max(unclip_max, static_cast<float>(r));
            if (!g_clipped) unclip_max = std::max(unclip_max, static_cast<float>(g));
            if (!b_clipped) unclip_max = std::max(unclip_max, static_cast<float>(b));
            if (unclip_max <= 0.0f) continue;

            float f_max = static_cast<float>(max_val);
            float scale = unclip_max / f_max;

            // Reconstruct: extrapolate based on ratios from unclipped channels
            // Simple approach: use average ratio of unclipped channels
            float avg_unclip = 0.0f;
            int unclip_n = 0;
            if (!r_clipped) { avg_unclip += static_cast<float>(r); unclip_n++; }
            if (!g_clipped) { avg_unclip += static_cast<float>(g); unclip_n++; }
            if (!b_clipped) { avg_unclip += static_cast<float>(b); unclip_n++; }
            if (unclip_n == 0) continue;
            avg_unclip /= static_cast<float>(unclip_n);

            // Reconstruct each clipped channel as a multiple of the average
            if (r_clipped) {
                float est = (g_clipped ? static_cast<float>(b) : static_cast<float>(g));
                // Assume clipped R would have been similar to or brighter than unclipped
                float v = est * 1.5f; // typical: R is often stronger in warm light
                v = std::min(v, f_max * 4.0f);
                int recovered = static_cast<int>(std::min(v, f_max));
                detail::write_pixel(output, x, y, width, channels, bit_depth, 0, recovered);
            }
            if (g_clipped) {
                float est = avg_unclip * 2.2f;
                float v = std::min(est, f_max * 4.0f);
                int recovered = static_cast<int>(std::min(v, f_max));
                detail::write_pixel(output, x, y, width, channels, bit_depth, 1, recovered);
            }
            if (b_clipped) {
                float est = avg_unclip * 1.8f;
                float v = std::min(est, f_max * 4.0f);
                int recovered = static_cast<int>(std::min(v, f_max));
                detail::write_pixel(output, x, y, width, channels, bit_depth, 2, recovered);
            }
        }
    }

    return HighlightReconstructError::Ok;
}

} // namespace highlight_reconstruct
