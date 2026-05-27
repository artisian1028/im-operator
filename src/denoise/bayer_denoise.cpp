#include "common.hpp"
#include <algorithm>
#include <vector>

namespace denoise {

// Bayer-domain denoise.
// Input: single-channel Bayer raw data (channels must be 1).
// Since neighboring Bayer pixels alternate colors, we cannot apply a naive
// spatial filter (it would mix R, G, B values). Instead, for each pixel we
// collect only the same-color neighbors within a 5x5 window and take the
// median of those values.
//
// In a standard 2x2 Bayer cell [R, Gr; Gb, B], same-color pixels repeat
// every 2 rows and 2 columns. A 5x5 window gives up to 9 same-color samples.

static int median_same_color(const uint8_t* bayer, int x, int y,
                              int width, int height, int bit_depth) {
    int vals[13];  // up to 13 same-color pixels in 5x5
    int idx = 0;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            if ((dx & 1) != 0 || (dy & 1) != 0) continue;  // same parity = same color
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0) nx = -nx;              // mirror
            if (nx >= width) nx = 2 * (width - 1) - nx;
            if (ny < 0) ny = -ny;
            if (ny >= height) ny = 2 * (height - 1) - ny;
            // Re-clamp after mirror to be safe
            if (nx < 0) nx = 0;
            if (nx >= width) nx = width - 1;
            if (ny < 0) ny = 0;
            if (ny >= height) ny = height - 1;
            vals[idx++] = detail::read_bayer(bayer, nx, ny, width, bit_depth);
        }
    }
    std::nth_element(vals, vals + idx / 2, vals + idx);
    return vals[idx / 2];
}

DenoiseError process_bayer_denoise(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    int bit_depth, float /*strength*/) {
    DenoiseError err = validate_denoise_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != DenoiseError::Ok) return err;

    // Bayer denoise only makes sense on single-channel CFA data
    if (channels != 1) return DenoiseError::InvalidChannels;

    if (width < 5 || height < 5) return DenoiseError::ImageTooSmall;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int val = median_same_color(input, x, y, width, height, bit_depth);
            detail::write_bayer(output, x, y, width, bit_depth, val);
        }
    }

    return DenoiseError::Ok;
}

} // namespace denoise
