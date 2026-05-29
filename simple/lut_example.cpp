#include "lut/algorithms.hpp"
#include "lut/types.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

int main() {
    printf("=== LUT Example ===\n");

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

    // Demo WARM preset
    {
        auto start = std::chrono::steady_clock::now();
        auto err = lut::process_lut(input.data(), output.data(),
                                    width, height, channels,
                                    lut::LUTAlgorithm::WARM, 8);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("WARM preset: ERROR - %s\n", lut::lut_error_message(err));
        } else {
            printf("WARM preset: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("lut_warm.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_warm.rgb\n");
        }
    }

    // Demo VINTAGE_FADE preset
    {
        auto start = std::chrono::steady_clock::now();
        auto err = lut::process_lut(input.data(), output.data(),
                                    width, height, channels,
                                    lut::LUTAlgorithm::VINTAGE_FADE, 8);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("VINTAGE_FADE preset: ERROR - %s\n", lut::lut_error_message(err));
        } else {
            printf("VINTAGE_FADE preset: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("lut_vintage.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_vintage.rgb\n");
        }
    }

    // Demo CUSTOM_3D with identity LUT (size 17)
    {
        lut::LUT3D identity_lut = lut::build_identity_lut(17);

        auto start = std::chrono::steady_clock::now();
        auto err = lut::process_lut(input.data(), output.data(),
                                    width, height, channels,
                                    lut::LUTAlgorithm::CUSTOM_3D, 8,
                                    &identity_lut, 17);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("CUSTOM_3D (identity, size 17): ERROR - %s\n", lut::lut_error_message(err));
        } else {
            printf("CUSTOM_3D (identity, size 17): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("lut_custom.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_custom.rgb\n");
        }
    }

    // Demo SEPIA preset
    {
        auto start = std::chrono::steady_clock::now();
        auto err = lut::process_lut(input.data(), output.data(),
                                    width, height, channels,
                                    lut::LUTAlgorithm::SEPIA, 8);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("SEPIA preset: ERROR - %s\n", lut::lut_error_message(err));
        } else {
            printf("SEPIA preset: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("lut_sepia.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_sepia.rgb\n");
        }
    }

    // Demo COOL preset
    {
        auto start = std::chrono::steady_clock::now();
        auto err = lut::process_lut(input.data(), output.data(),
                                    width, height, channels,
                                    lut::LUTAlgorithm::COOL, 8);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("COOL preset: ERROR - %s\n", lut::lut_error_message(err));
        } else {
            printf("COOL preset: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("lut_cool.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_cool.rgb\n");
        }
    }

    // Demo HIGH_CONTRAST preset
    {
        auto start = std::chrono::steady_clock::now();
        auto err = lut::process_lut(input.data(), output.data(),
                                    width, height, channels,
                                    lut::LUTAlgorithm::HIGH_CONTRAST, 8);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("HIGH_CONTRAST preset: ERROR - %s\n", lut::lut_error_message(err));
        } else {
            printf("HIGH_CONTRAST preset: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("lut_high_contrast.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_high_contrast.rgb\n");
        }
    }

    // Demo LOW_CONTRAST preset
    {
        auto start = std::chrono::steady_clock::now();
        auto err = lut::process_lut(input.data(), output.data(),
                                    width, height, channels,
                                    lut::LUTAlgorithm::LOW_CONTRAST, 8);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("LOW_CONTRAST preset: ERROR - %s\n", lut::lut_error_message(err));
        } else {
            printf("LOW_CONTRAST preset: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("lut_low_contrast.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_low_contrast.rgb\n");
        }
    }

    // Demo INVERT preset
    {
        auto start = std::chrono::steady_clock::now();
        auto err = lut::process_lut(input.data(), output.data(),
                                    width, height, channels,
                                    lut::LUTAlgorithm::INVERT, 8);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("INVERT preset: ERROR - %s\n", lut::lut_error_message(err));
        } else {
            printf("INVERT preset: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("lut_invert.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_invert.rgb\n");
        }
    }

    // Demo CUBE_FILE (graceful error if file does not exist)
    {
        const char* cube_path = "test_data/sample_lut.cube";
        auto start = std::chrono::steady_clock::now();
        auto err = lut::process_lut(input.data(), output.data(),
                                    width, height, channels,
                                    lut::LUTAlgorithm::CUBE_FILE, 8,
                                    (const void*)cube_path, 33);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("CUBE_FILE (%s): ERROR - %s\n", cube_path, lut::lut_error_message(err));
        } else {
            printf("CUBE_FILE (%s): OK (%lld ms)\n", cube_path, (long long)ms);
            FILE* out = fopen("lut_cube_file.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_cube_file.rgb\n");
        }
    }

    // Demo building a sepia LUT and applying it manually via apply_lut
    {
        auto build_start = std::chrono::steady_clock::now();
        lut::LUT3D sepia_lut = lut::build_sepia_lut(33);
        auto build_end = std::chrono::steady_clock::now();
        auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
        printf("build_sepia_lut(33): %lld ms\n", (long long)build_ms);

        auto apply_start = std::chrono::steady_clock::now();
        auto err = lut::apply_lut(sepia_lut, input.data(), output.data(),
                                  width, height, 8);
        auto apply_end = std::chrono::steady_clock::now();
        auto apply_ms = std::chrono::duration_cast<std::chrono::milliseconds>(apply_end - apply_start).count();

        if (!err) {
            printf("apply_lut (sepia): ERROR - %s\n", lut::lut_error_message(err));
        } else {
            printf("apply_lut (sepia): OK (%lld ms)\n", (long long)apply_ms);
            FILE* out = fopen("lut_sepia_manual.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved lut_sepia_manual.rgb\n");
        }
    }

    printf("Done.\n");
    return 0;
}
