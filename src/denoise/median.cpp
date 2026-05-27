#include "common.hpp"
#include <algorithm>
#include <vector>

namespace denoise {

// 3x3 median filter.
// Picks up the 9 values in the 3x3 neighborhood, sorts, takes the middle.

static int median_3x3_channel(const uint8_t* input, int x, int y,
                               int width, int height, int channels,
                               int bit_depth, int channel) {
    int vals[9];
    int idx = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            // Mirror edge handling
            if (nx < 0) nx = 0;
            if (nx >= width) nx = width - 1;
            if (ny < 0) ny = 0;
            if (ny >= height) ny = height - 1;
            vals[idx++] = detail::read_pixel(input, nx, ny, width,
                                              channels, bit_depth, channel);
        }
    }
    std::nth_element(vals, vals + 4, vals + 9);
    return vals[4];
}

DenoiseError process_median(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, float /*strength*/) {
    DenoiseError err = validate_denoise_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != DenoiseError::Ok) return err;

    if (width < 3 || height < 3) return DenoiseError::ImageTooSmall;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int val = median_3x3_channel(input, x, y, width, height,
                                              channels, bit_depth, c);
                detail::write_pixel(output, x, y, width, channels, bit_depth, c, val);
            }
        }
    }

    return DenoiseError::Ok;
}

} // namespace denoise
