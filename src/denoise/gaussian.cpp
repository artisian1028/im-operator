#include "common.hpp"
#include <cmath>
#include <vector>
#include <cstring>

namespace denoise {

static void make_gaussian_kernel_1d(float sigma, int kernel_size,
                                     std::vector<float>& kernel) {
    kernel.resize(kernel_size);
    float sum = 0.0f;
    int half = kernel_size / 2;
    float s2 = 2.0f * sigma * sigma;
    for (int i = 0; i < kernel_size; i++) {
        float x = static_cast<float>(i - half);
        float w = std::exp(-(x * x) / s2);
        kernel[i] = w;
        sum += w;
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }
}

// Separable Gaussian: horizontal pass then vertical pass
// Uses a temp buffer for the intermediate result
template<int BPC>  // bytes per component: 1 or 2
static void gaussian_separable(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, const std::vector<float>& kernel,
                                int kernel_size) {
    int max_val = detail::safe_max_val(bit_depth);
    int half = kernel_size / 2;
    int pixel_stride = channels * BPC;
    int row_stride = width * pixel_stride;

    // Temporary buffer for horizontal pass result
    std::vector<float> temp(static_cast<size_t>(width) * height * channels, 0.0f);

    // Horizontal pass: input -> temp
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                for (int k = -half; k <= half; k++) {
                    int nx = x + k;
                    if (nx < 0) nx = 0;
                    if (nx >= width) nx = width - 1;
                    if (BPC == 1) {
                        sum += kernel[k + half] *
                               static_cast<float>(input[y * row_stride + nx * pixel_stride + c]);
                    } else {
                        const uint16_t* in16 = reinterpret_cast<const uint16_t*>(input);
                        sum += kernel[k + half] *
                               static_cast<float>(in16[y * width * channels + nx * channels + c]);
                    }
                }
                temp[(static_cast<size_t>(y) * width + x) * channels + c] = sum;
            }
        }
    }

    // Vertical pass: temp -> output
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                for (int k = -half; k <= half; k++) {
                    int ny = y + k;
                    if (ny < 0) ny = 0;
                    if (ny >= height) ny = height - 1;
                    sum += kernel[k + half] *
                           temp[(static_cast<size_t>(ny) * width + x) * channels + c];
                }
                int val = detail::clamp_val(static_cast<int>(sum + 0.5f), max_val);
                detail::write_pixel(output, x, y, width, channels, bit_depth, c, val);
            }
        }
    }
}

DenoiseError process_gaussian(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, float strength) {
    DenoiseError err = validate_denoise_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != DenoiseError::Ok) return err;

    if (width < 3 || height < 3) return DenoiseError::ImageTooSmall;

    // Sigma derived from strength (strength 1.0 -> sigma 1.0, strength 2.0 -> sigma 2.0)
    float sigma = strength;
    // Kernel size: 2 * ceil(3*sigma) + 1, minimum 3
    int kernel_size = std::max(3, 2 * static_cast<int>(std::ceil(3.0f * sigma)) + 1);
    // Cap kernel size for performance
    if (kernel_size > 15) kernel_size = 15;
    // Ensure odd
    if (kernel_size % 2 == 0) kernel_size++;

    std::vector<float> kernel;
    make_gaussian_kernel_1d(sigma, kernel_size, kernel);

    if (bit_depth <= 8) {
        gaussian_separable<1>(input, output, width, height, channels,
                               bit_depth, kernel, kernel_size);
    } else {
        gaussian_separable<2>(input, output, width, height, channels,
                               bit_depth, kernel, kernel_size);
    }

    return DenoiseError::Ok;
}

} // namespace denoise
