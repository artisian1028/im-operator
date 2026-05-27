#include "common.hpp"
#include "highlight_reconstruct/algorithms.hpp"
#include <cstring>

namespace highlight_reconstruct {

// Gradient-based highlight reconstruction:
// For clipped pixels, propagate color from nearby unclipped pixels using gradient direction.
// This handles the case where pixels are fully clipped in all 3 channels (pure white).
//
// Approach: for each clipped pixel, search in 5x5 window for unclipped neighbors.
// Use the average of unclipped neighbors weighted by inverse gradient magnitude.
HighlightReconstructError process_gradient_based(const uint8_t* input, uint8_t* output,
                                                  int width, int height, int channels,
                                                  int bit_depth,
                                                  const HighlightReconstructParams& params) {
    auto err = validate_highlight_reconstruct_inputs(input, output, width, height, channels, bit_depth);
    if (err != HighlightReconstructError::Ok) return err;

    if (width < 5 || height < 5) return HighlightReconstructError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);
    float threshold = std::max(0.5f, std::min(1.0f, params.threshold));
    int clip_thresh = static_cast<int>(threshold * static_cast<float>(max_val));
    float f_max = static_cast<float>(max_val);

    // Copy input to output first
    size_t total = static_cast<size_t>(width) * height * 3;
    size_t bytes = (bit_depth <= 8) ? total : total * 2;
    std::memcpy(output, input, bytes);

    // For each pixel, if any channel is clipped, attempt reconstruction
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            int r = detail::read_pixel(output, x, y, width, channels, bit_depth, 0);
            int g = detail::read_pixel(output, x, y, width, channels, bit_depth, 1);
            int b = detail::read_pixel(output, x, y, width, channels, bit_depth, 2);

            bool cr = (r >= clip_thresh);
            bool cg = (g >= clip_thresh);
            bool cb = (b >= clip_thresh);

            if (!cr && !cg && !cb) continue; // nothing clipped

            // For each clipped channel, collect weighted estimate from unclipped neighbors
            for (int ch = 0; ch < 3; ch++) {
                bool* clip = (ch == 0) ? &cr : (ch == 1) ? &cg : &cb;
                if (!*clip) continue;

                float sum = 0.0f;
                float wsum = 0.0f;

                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

                        int nv = detail::read_pixel(output, nx, ny, width, channels, bit_depth, ch);
                        if (nv >= clip_thresh) continue; // skip clipped neighbors

                        float dist = std::sqrt(static_cast<float>(dx*dx + dy*dy));
                        float w = 1.0f / (dist + 0.1f);

                        sum += static_cast<float>(nv) * w;
                        wsum += w;
                    }
                }

                if (wsum > 0.0f) {
                    float estimated = sum / wsum;
                    // Blend: keep some original, add estimated
                    int orig = detail::read_pixel(input, x, y, width, channels, bit_depth, ch);
                    float blended = static_cast<float>(orig) * 0.3f + estimated * 0.7f;
                    blended = std::min(blended, f_max);
                    detail::write_pixel(output, x, y, width, channels, bit_depth, ch,
                                        static_cast<int>(blended + 0.5f));
                }
            }
        }
    }

    return HighlightReconstructError::Ok;
}

} // namespace highlight_reconstruct
