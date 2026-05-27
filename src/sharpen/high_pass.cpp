#include "common.hpp"

namespace sharpen {

// High-pass filter overlay: extract high-frequency detail via a high-pass kernel
// and add it back to the original.
//
// HP kernel (3x3):
//   [-1 -1 -1]
//   [-1  8 -1]
//   [-1 -1 -1]
//
// result = original + amount * HP_filter(original)
SharpenError process_high_pass(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, const SharpenParams& params) {
    SharpenError err = validate_sharpen_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != SharpenError::Ok) return err;

    if (width < 3 || height < 3) return SharpenError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);
    float amount = std::max(0.0f, std::min(3.0f, params.amount));

    // Predefine 3x3 high-pass kernel: center=8, neighbors=-1
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int nx = std::clamp(x, 0, width - 1);
            int ny = std::clamp(y, 0, height - 1);
            int nx_m1 = std::clamp(x - 1, 0, width - 1);
            int nx_p1 = std::clamp(x + 1, 0, width - 1);
            int ny_m1 = std::clamp(y - 1, 0, height - 1);
            int ny_p1 = std::clamp(y + 1, 0, height - 1);
            for (int c = 0; c < 3; c++) {
                int c00 = detail::read_pixel(input, nx_m1, ny_m1, width, channels, bit_depth, c);
                int c01 = detail::read_pixel(input, nx,    ny_m1, width, channels, bit_depth, c);
                int c02 = detail::read_pixel(input, nx_p1, ny_m1, width, channels, bit_depth, c);
                int c10 = detail::read_pixel(input, nx_m1, ny,    width, channels, bit_depth, c);
                int c11 = detail::read_pixel(input, nx,    ny,    width, channels, bit_depth, c);
                int c12 = detail::read_pixel(input, nx_p1, ny,    width, channels, bit_depth, c);
                int c20 = detail::read_pixel(input, nx_m1, ny_p1, width, channels, bit_depth, c);
                int c21 = detail::read_pixel(input, nx,    ny_p1, width, channels, bit_depth, c);
                int c22 = detail::read_pixel(input, nx_p1, ny_p1, width, channels, bit_depth, c);

                float hp = -1.0f * c00 - 1.0f * c01 - 1.0f * c02
                           - 1.0f * c10 + 8.0f * c11 - 1.0f * c12
                           - 1.0f * c20 - 1.0f * c21 - 1.0f * c22;

                // Scale HP by 1/8 to normalize and mix with original
                float detail = hp / 8.0f;
                float result = c11 + amount * detail;

                detail::write_pixel(output, x, y, width, channels, bit_depth, c,
                                    detail::clamp_val(static_cast<int>(result + 0.5f), max_val));
            }
        }
    }

    return SharpenError::Ok;
}

} // namespace sharpen
