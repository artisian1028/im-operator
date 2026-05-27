#include "common.hpp"

namespace sharpen {

// Laplacian sharpening: enhance edges by subtracting the Laplacian
// sharpen = original - amount * Laplacian(original)
// Uses a 3x3 Laplacian kernel:
//   [ 0 -1  0]
//   [-1  4 -1]
//   [ 0 -1  0]
// or 5x5 if radius >= 2.0:
//   [-1 -1 -1 -1 -1]
//   [-1 -1 -1 -1 -1]
//   [-1 -1 24 -1 -1]
//   [-1 -1 -1 -1 -1]
//   [-1 -1 -1 -1 -1]
SharpenError process_laplacian(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, const SharpenParams& params) {
    SharpenError err = validate_sharpen_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != SharpenError::Ok) return err;

    int min_size = (params.radius >= 2.0f) ? 5 : 3;
    if (width < min_size || height < min_size) return SharpenError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);
    float amount = std::max(0.0f, std::min(3.0f, params.amount));

    bool large_kernel = (params.radius >= 2.0f);
    int radius = large_kernel ? 2 : 1;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                int center = detail::read_pixel(input, x, y, width, channels, bit_depth, c);

                float lap = 0.0f;
                if (!large_kernel) {
                    // 3x3 Laplacian (clamp edge access for safety)
                    int nx = std::clamp(x, 0, width - 1);
                    int ny_m1 = std::clamp(y - 1, 0, height - 1);
                    int ny_p1 = std::clamp(y + 1, 0, height - 1);
                    int nx_m1 = std::clamp(x - 1, 0, width - 1);
                    int nx_p1 = std::clamp(x + 1, 0, width - 1);
                    int n  = detail::read_pixel(input, nx, ny_m1, width, channels, bit_depth, c);
                    int s  = detail::read_pixel(input, nx, ny_p1, width, channels, bit_depth, c);
                    int e  = detail::read_pixel(input, nx_p1, y, width, channels, bit_depth, c);
                    int w  = detail::read_pixel(input, nx_m1, y, width, channels, bit_depth, c);
                    lap = static_cast<float>(4 * center - (n + s + e + w));
                } else {
                    // 5x5 Laplacian
                    float sum_neighbors = 0.0f;
                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            if (dx == 0 && dy == 0) continue;
                            int nx = std::clamp(x + dx, 0, width - 1);
                            int ny = std::clamp(y + dy, 0, height - 1);
                            sum_neighbors += detail::read_pixel(input, nx, ny, width, channels, bit_depth, c);
                        }
                    }
                    lap = 24.0f * center - sum_neighbors;
                }

                float result = center - amount * lap / (large_kernel ? 24.0f : 4.0f);
                detail::write_pixel(output, x, y, width, channels, bit_depth, c,
                                    detail::clamp_val(static_cast<int>(result + 0.5f), max_val));
            }
        }
    }

    return SharpenError::Ok;
}

} // namespace sharpen
