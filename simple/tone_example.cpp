#include "tone/algorithms.hpp"
#include "tone/types.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

int main() {
    printf("=== Tone Curve Example ===\n");

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

    // Demo GAMMA (gamma=2.2)
    {
        tone::ToneParams params;
        params.gamma = 2.2f;

        auto start = std::chrono::steady_clock::now();
        auto err = tone::process_tone(input.data(), output.data(),
                                      width, height, channels,
                                      tone::ToneAlgorithm::GAMMA, 8, params);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("GAMMA (2.2): ERROR - %s\n", tone::tone_error_message(err));
        } else {
            printf("GAMMA (2.2): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("tone_gamma.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved tone_gamma.rgb\n");
        }
    }

    // Demo S_CURVE (shadow=0.1, highlight=0.9)
    {
        tone::ToneParams params;
        params.shadows = 0.1f;
        params.highlights = 0.9f;

        auto start = std::chrono::steady_clock::now();
        auto err = tone::process_tone(input.data(), output.data(),
                                      width, height, channels,
                                      tone::ToneAlgorithm::S_CURVE, 8, params);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("S_CURVE: ERROR - %s\n", tone::tone_error_message(err));
        } else {
            printf("S_CURVE (shadow=0.1, highlight=0.9): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("tone_s_curve.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved tone_s_curve.rgb\n");
        }
    }

    // Demo LEVELS (black=0.1, white=0.9)
    {
        tone::ToneParams params;
        params.black_point = 0.1f;
        params.white_point = 0.9f;

        auto start = std::chrono::steady_clock::now();
        auto err = tone::process_tone(input.data(), output.data(),
                                      width, height, channels,
                                      tone::ToneAlgorithm::LEVELS, 8, params);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("LEVELS: ERROR - %s\n", tone::tone_error_message(err));
        } else {
            printf("LEVELS (black=0.1, white=0.9): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("tone_levels.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved tone_levels.rgb\n");
        }
    }

    // Demo SHADOWS_HIGHLIGHTS (shadow=0.2, highlight=0.2)
    {
        tone::ToneParams params;
        params.shadows = 0.2f;
        params.highlights = 0.2f;

        auto start = std::chrono::steady_clock::now();
        auto err = tone::process_tone(input.data(), output.data(),
                                      width, height, channels,
                                      tone::ToneAlgorithm::SHADOWS_HIGHLIGHTS, 8, params);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("SHADOWS_HIGHLIGHTS: ERROR - %s\n", tone::tone_error_message(err));
        } else {
            printf("SHADOWS_HIGHLIGHTS (shadow=0.2, highlight=0.2): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("tone_shadows_highlights.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved tone_shadows_highlights.rgb\n");
        }
    }

    // Demo CURVES_3POINT (three-point cubic Bezier tone curve)
    {
        tone::ToneParams params;
        params.shadows = 0.15f;
        params.highlights = 0.85f;

        auto start = std::chrono::steady_clock::now();
        auto err = tone::process_tone(input.data(), output.data(),
                                      width, height, channels,
                                      tone::ToneAlgorithm::CURVES_3POINT, 8, params);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("CURVES_3POINT: ERROR - %s\n", tone::tone_error_message(err));
        } else {
            printf("CURVES_3POINT (shadow=0.15, highlight=0.85): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("tone_curves_3point.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved tone_curves_3point.rgb\n");
        }
    }

    printf("Done.\n");
    return 0;
}
