#include "sharpen/algorithms.hpp"
#include "sharpen/types.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

int main() {
    printf("=== Sharpen Example ===\n");

    const int width = 640, height = 480, channels = 3;
    const int img_size = width * height * channels;

    // Load RGB data
    std::vector<uint8_t> input(img_size);
    FILE* f = fopen("test_data/rgb_640x480_8bit.raw", "rb");
    if (!f) {
        printf("ERROR: Cannot open test_data/rgb_640x480_8bit.raw\n");
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

    // Demo UNSHARP_MASK (amount=1.5, sigma=1.0)
    {
        sharpen::SharpenParams params;
        params.amount = 1.5f;
        params.radius = 1.0f;

        auto start = std::chrono::steady_clock::now();
        auto err = sharpen::process_sharpen(input.data(), output.data(),
                                            width, height, channels,
                                            sharpen::SharpenAlgorithm::UNSHARP_MASK,
                                            8, params);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("UNSHARP_MASK: ERROR - %s\n", sharpen::sharpen_error_message(err));
        } else {
            printf("UNSHARP_MASK (amount=1.5, sigma=1.0): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("sharpen_usm.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved sharpen_usm.rgb\n");
        }
    }

    // Demo LAPLACIAN (amount=0.8)
    {
        sharpen::SharpenParams params;
        params.amount = 0.8f;

        auto start = std::chrono::steady_clock::now();
        auto err = sharpen::process_sharpen(input.data(), output.data(),
                                            width, height, channels,
                                            sharpen::SharpenAlgorithm::LAPLACIAN,
                                            8, params);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("LAPLACIAN: ERROR - %s\n", sharpen::sharpen_error_message(err));
        } else {
            printf("LAPLACIAN (amount=0.8): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("sharpen_lap.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved sharpen_lap.rgb\n");
        }
    }

    // Demo HIGH_PASS (amount=0.5)
    {
        sharpen::SharpenParams params;
        params.amount = 0.5f;

        auto start = std::chrono::steady_clock::now();
        auto err = sharpen::process_sharpen(input.data(), output.data(),
                                            width, height, channels,
                                            sharpen::SharpenAlgorithm::HIGH_PASS,
                                            8, params);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("HIGH_PASS: ERROR - %s\n", sharpen::sharpen_error_message(err));
        } else {
            printf("HIGH_PASS (amount=0.5): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("sharpen_hp.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved sharpen_hp.rgb\n");
        }
    }

    // Demo ADAPTIVE (amount=1.0, radius=1.5)
    {
        sharpen::SharpenParams params;
        params.amount = 1.0f;
        params.radius = 1.5f;

        auto start = std::chrono::steady_clock::now();
        auto err = sharpen::process_sharpen(input.data(), output.data(),
                                            width, height, channels,
                                            sharpen::SharpenAlgorithm::ADAPTIVE,
                                            8, params);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("ADAPTIVE: ERROR - %s\n", sharpen::sharpen_error_message(err));
        } else {
            printf("ADAPTIVE (amount=1.0, radius=1.5): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("sharpen_ada.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved sharpen_ada.rgb\n");
        }
    }

    printf("Done.\n");
    return 0;
}
