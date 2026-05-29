#include "denoise/algorithms.hpp"
#include "denoise/types.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

int main() {
    printf("=== Denoise Example ===\n");

    const int width = 1920, height = 1080, channels = 3;
    const int img_size = width * height * channels;

    // Load RGB data
    std::vector<uint8_t> input(img_size);
    FILE* f = fopen("test_data/rgb_1920x1080_8bit.raw", "rb");
    if (!f) {
        printf("ERROR: Cannot open test_data/rgb_1920x1080_8bit.raw\n");
        return 1;
    }
    if (fread(input.data(), 1, img_size, f) != (size_t)img_size) {
        printf("ERROR: Failed to read full image data (%d bytes)\n", img_size);
        fclose(f);
        return 1;
    }
    fclose(f);
    printf("Loaded RGB data: %dx%d, %d channels, %d bytes\n", width, height, channels, img_size);

    std::vector<uint8_t> output(img_size);

    // Helper: compute mean absolute difference (pseudo-PSNR indicator)
    auto compute_mad = [&](const std::vector<uint8_t>& out) -> double {
        double sum = 0.0;
        for (int i = 0; i < img_size; ++i) {
            sum += std::abs((int)out[i] - (int)input[i]);
        }
        return sum / img_size;
    };

    // Demo BILATERAL
    {
        auto start = std::chrono::steady_clock::now();
        auto err = denoise::process_denoise(input.data(), output.data(),
                                            width, height, channels,
                                            denoise::DenoiseAlgorithm::BILATERAL,
                                            8, 1.0f);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("BILATERAL denoise: ERROR - %s\n", denoise::denoise_error_message(err));
        } else {
            printf("BILATERAL denoise: OK (%lld ms)\n  MAD = %.4f\n", (long long)ms, compute_mad(output));
            FILE* out = fopen("denoise_bilateral.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved denoise_bilateral.rgb\n");
        }
    }

    // Demo GAUSSIAN
    {
        auto start = std::chrono::steady_clock::now();
        auto err = denoise::process_denoise(input.data(), output.data(),
                                            width, height, channels,
                                            denoise::DenoiseAlgorithm::GAUSSIAN,
                                            8, 1.0f);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("GAUSSIAN denoise: ERROR - %s\n", denoise::denoise_error_message(err));
        } else {
            printf("GAUSSIAN denoise: OK (%lld ms)\n  MAD = %.4f\n", (long long)ms, compute_mad(output));
            FILE* out = fopen("denoise_gaussian.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved denoise_gaussian.rgb\n");
        }
    }

    // Demo MEDIAN
    {
        auto start = std::chrono::steady_clock::now();
        auto err = denoise::process_denoise(input.data(), output.data(),
                                            width, height, channels,
                                            denoise::DenoiseAlgorithm::MEDIAN,
                                            8, 1.0f);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("MEDIAN denoise: ERROR - %s\n", denoise::denoise_error_message(err));
        } else {
            printf("MEDIAN denoise: OK (%lld ms)\n  MAD = %.4f\n", (long long)ms, compute_mad(output));
            FILE* out = fopen("denoise_median.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved denoise_median.rgb\n");
        }
    }

    // Demo NLM
    {
        auto start = std::chrono::steady_clock::now();
        auto err = denoise::process_denoise(input.data(), output.data(),
                                            width, height, channels,
                                            denoise::DenoiseAlgorithm::NLM,
                                            8, 1.0f);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("NLM denoise: ERROR - %s\n", denoise::denoise_error_message(err));
        } else {
            printf("NLM denoise: OK (%lld ms)\n  MAD = %.4f\n", (long long)ms, compute_mad(output));
            FILE* out = fopen("denoise_nlm.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved denoise_nlm.rgb\n");
        }
    }

    // Demo WAVELET
    {
        auto start = std::chrono::steady_clock::now();
        auto err = denoise::process_denoise(input.data(), output.data(),
                                            width, height, channels,
                                            denoise::DenoiseAlgorithm::WAVELET,
                                            8, 1.0f);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("WAVELET denoise: ERROR - %s\n", denoise::denoise_error_message(err));
        } else {
            printf("WAVELET denoise: OK (%lld ms)\n  MAD = %.4f\n", (long long)ms, compute_mad(output));
            FILE* out = fopen("denoise_wavelet.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved denoise_wavelet.rgb\n");
        }
    }

    // Demo BAYER_DENOISE
    {
        auto start = std::chrono::steady_clock::now();
        auto err = denoise::process_denoise(input.data(), output.data(),
                                            width, height, channels,
                                            denoise::DenoiseAlgorithm::BAYER_DENOISE,
                                            8, 1.0f);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("BAYER_DENOISE denoise: ERROR - %s\n", denoise::denoise_error_message(err));
        } else {
            printf("BAYER_DENOISE denoise: OK (%lld ms)\n  MAD = %.4f\n", (long long)ms, compute_mad(output));
            FILE* out = fopen("denoise_bayer_denoise.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved denoise_bayer_denoise.rgb\n");
        }
    }

    printf("Done.\n");
    return 0;
}
