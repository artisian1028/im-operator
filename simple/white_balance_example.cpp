#include "white_balance/algorithms.hpp"
#include "white_balance/types.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

int main() {
    printf("=== White Balance Example ===\n");

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

    // Demo GRAY_WORLD (auto)
    {
        auto start = std::chrono::steady_clock::now();
        auto err = white_balance::process_white_balance(
            input.data(), output.data(), width, height, channels,
            white_balance::WhiteBalanceAlgorithm::GRAY_WORLD, 8);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("GRAY_WORLD: ERROR - %s\n", white_balance::white_balance_error_message(err));
        } else {
            printf("GRAY_WORLD: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("wb_gray_world.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved wb_gray_world.rgb\n");
        }
    }

    // Demo WHITE_PATCH
    {
        auto start = std::chrono::steady_clock::now();
        auto err = white_balance::process_white_balance(
            input.data(), output.data(), width, height, channels,
            white_balance::WhiteBalanceAlgorithm::WHITE_PATCH, 8);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("WHITE_PATCH: ERROR - %s\n", white_balance::white_balance_error_message(err));
        } else {
            printf("WHITE_PATCH: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("wb_white_patch.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved wb_white_patch.rgb\n");
        }
    }

    // Demo SHADE_OF_GRAY
    {
        auto start = std::chrono::steady_clock::now();
        auto err = white_balance::process_white_balance(
            input.data(), output.data(), width, height, channels,
            white_balance::WhiteBalanceAlgorithm::SHADE_OF_GRAY, 8, 6.0f);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("SHADE_OF_GRAY: ERROR - %s\n", white_balance::white_balance_error_message(err));
        } else {
            printf("SHADE_OF_GRAY: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("wb_shade_of_gray.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved wb_shade_of_gray.rgb\n");
        }
    }

    // Demo MANUAL (custom gains)
    {
        white_balance::WBCoefficients gains;
        gains.r_gain = 1.2f;
        gains.g_gain = 1.0f;
        gains.b_gain = 1.5f;

        auto start = std::chrono::steady_clock::now();
        auto err = white_balance::process_white_balance(
            input.data(), output.data(), width, height, channels,
            white_balance::WhiteBalanceAlgorithm::MANUAL, 8, 0.0f, gains);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("MANUAL: ERROR - %s\n", white_balance::white_balance_error_message(err));
        } else {
            printf("MANUAL (r=1.2, g=1.0, b=1.5): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("wb_manual.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved wb_manual.rgb\n");
        }
    }

    printf("Done.\n");
    return 0;
}
