#include "common.hpp"

namespace sharpen {

// Unsharp Mask: sharpen = original + amount * (original - blurred)
//  1. Gaussian blur the input
//  2. Compute detail = original - blurred
//  3. Add amount * detail back to original
SharpenError process_unsharp_mask(const uint8_t* input, uint8_t* output,
                                   int width, int height, int channels,
                                   int bit_depth, const SharpenParams& params) {
    SharpenError err = validate_sharpen_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != SharpenError::Ok) return err;

    if (width < 3 || height < 3) return SharpenError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);
    float amount = std::max(0.0f, std::min(3.0f, params.amount));
    float sigma = std::max(0.3f, std::min(5.0f, params.radius));

    size_t total = static_cast<size_t>(width) * height;

    // Process per channel through float planes
    for (int c = 0; c < 3; c++) {
        std::vector<float> src_plane(total);
        std::vector<float> blurred(total);

        for (size_t i = 0; i < total; i++) {
            src_plane[i] = static_cast<float>(detail::read_pixel(input, static_cast<int>(i % width),
                            static_cast<int>(i / width), width, channels, bit_depth, c));
        }

        detail::gaussian_blur_plane(src_plane.data(), blurred.data(), width, height, sigma);

        for (size_t i = 0; i < total; i++) {
            float detail = src_plane[i] - blurred[i];
            float result = src_plane[i] + amount * detail;
            int x = static_cast<int>(i % width);
            int y = static_cast<int>(i / width);
            detail::write_pixel(output, x, y, width, channels, bit_depth, c,
                                detail::clamp_val(static_cast<int>(result + 0.5f), max_val));
        }
    }

    return SharpenError::Ok;
}

} // namespace sharpen
