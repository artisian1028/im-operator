#include "ccm/algorithms.hpp"
#include "ccm/types.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

int main() {
    printf("=== Color Correction Matrix Example ===\n");

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

    // Demo LINEAR_3X3 (warm-tone matrix)
    {
        ccm::CCMatrix3x3 warm;
        warm.m[0] = 1.2f; warm.m[1] = 0.0f; warm.m[2] = 0.0f;
        warm.m[3] = 0.0f; warm.m[4] = 1.0f; warm.m[5] = 0.0f;
        warm.m[6] = 0.0f; warm.m[7] = 0.0f; warm.m[8] = 0.9f;

        auto start = std::chrono::steady_clock::now();
        auto err = ccm::process_ccm(input.data(), output.data(),
                                    width, height, channels,
                                    ccm::CCMAlgorithm::LINEAR_3X3, 8, &warm);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("LINEAR_3X3: ERROR - %s\n", ccm::ccm_error_message(err));
        } else {
            printf("LINEAR_3X3 (warm-tone): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("ccm_3x3.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved ccm_3x3.rgb\n");
        }
    }

    // Demo LINEAR_4X3 (with offset)
    {
        ccm::CCMatrix3x4 offset;
        // Row 0: 1.1*R + 0.0*G + 0.0*B + 0.05
        offset.m[0] = 1.1f; offset.m[1] = 0.0f; offset.m[2] = 0.0f; offset.m[3] = 0.05f;
        // Row 1: 0.0*R + 1.0*G + 0.0*B + 0.0
        offset.m[4] = 0.0f; offset.m[5] = 1.0f; offset.m[6] = 0.0f; offset.m[7] = 0.0f;
        // Row 2: 0.0*R + 0.0*G + 1.0*B + 0.0
        offset.m[8] = 0.0f; offset.m[9] = 0.0f; offset.m[10] = 1.0f; offset.m[11] = 0.0f;

        auto start = std::chrono::steady_clock::now();
        auto err = ccm::process_ccm(input.data(), output.data(),
                                    width, height, channels,
                                    ccm::CCMAlgorithm::LINEAR_4X3, 8, &offset);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("LINEAR_4X3: ERROR - %s\n", ccm::ccm_error_message(err));
        } else {
            printf("LINEAR_4X3 (with offset): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("ccm_4x3.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved ccm_4x3.rgb\n");
        }
    }

    // Demo POLYNOMIAL_3X9 (cross-channel + quadratic terms)
    {
        ccm::CCMatrix3x9 poly;
        // Set up a matrix with some cross-channel and quadratic coefficients.
        // Row 0 (R output): R + 0.1*G + 0.05*B + 0.02*RG + 0.0*RB + 0.01*GB + 0.0*R^2 + 0.0*G^2 + 0.0*B^2
        poly.m[0] = 1.0f; poly.m[1] = 0.1f; poly.m[2] = 0.05f;
        poly.m[3] = 0.02f; poly.m[4] = 0.0f; poly.m[5] = 0.01f;
        poly.m[6] = 0.0f; poly.m[7] = 0.0f; poly.m[8] = 0.0f;
        // Row 1 (G output): 0.05*R + 1.0*G + 0.05*B
        poly.m[9] = 0.05f; poly.m[10] = 1.0f; poly.m[11] = 0.05f;
        poly.m[12] = 0.0f; poly.m[13] = 0.0f; poly.m[14] = 0.0f;
        poly.m[15] = 0.0f; poly.m[16] = 0.0f; poly.m[17] = 0.0f;
        // Row 2 (B output): 0.05*R + 0.1*G + 1.0*B
        poly.m[18] = 0.05f; poly.m[19] = 0.1f; poly.m[20] = 1.0f;
        poly.m[21] = 0.0f; poly.m[22] = 0.0f; poly.m[23] = 0.0f;
        poly.m[24] = 0.0f; poly.m[25] = 0.0f; poly.m[26] = 0.0f;

        auto start = std::chrono::steady_clock::now();
        auto err = ccm::process_ccm(input.data(), output.data(),
                                    width, height, channels,
                                    ccm::CCMAlgorithm::POLYNOMIAL_3X9, 8, &poly);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("POLYNOMIAL_3X9: ERROR - %s\n", ccm::ccm_error_message(err));
        } else {
            printf("POLYNOMIAL_3X9 (cross-channel): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("ccm_polynomial.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved ccm_polynomial.rgb\n");
        }
    }

    // Demo MANUAL with a CCMatrix3x3
    {
        ccm::CCMatrix3x3 manual;
        // Red channel: boost R slightly, add a touch of G
        manual.m[0] = 1.3f; manual.m[1] = 0.1f; manual.m[2] = 0.0f;
        // Green channel: identity
        manual.m[3] = 0.0f; manual.m[4] = 1.0f; manual.m[5] = 0.0f;
        // Blue channel: reduce B slightly
        manual.m[6] = 0.0f; manual.m[7] = 0.0f; manual.m[8] = 0.8f;

        auto start = std::chrono::steady_clock::now();
        auto err = ccm::process_ccm(input.data(), output.data(),
                                    width, height, channels,
                                    ccm::CCMAlgorithm::MANUAL, 8, &manual);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("MANUAL: ERROR - %s\n", ccm::ccm_error_message(err));
        } else {
            printf("MANUAL (r=1.3, g=1.0, b=0.8): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("ccm_manual.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved ccm_manual.rgb\n");
        }
    }

    // Demo predefined matrix: sRGB to XYZ (D65) conversion
    {
        ccm::CCMatrix3x3 srgb_to_xyz = ccm::srgb_to_xyz_d65();

        auto start = std::chrono::steady_clock::now();
        auto err = ccm::process_ccm(input.data(), output.data(),
                                    width, height, channels,
                                    ccm::CCMAlgorithm::LINEAR_3X3, 8, &srgb_to_xyz);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("sRGB_to_XYZ (D65): ERROR - %s\n", ccm::ccm_error_message(err));
        } else {
            printf("sRGB_to_XYZ (D65): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("ccm_srgb_to_xyz.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved ccm_srgb_to_xyz.rgb\n");
        }
    }

    // Demo predefined matrix: saturation_matrix(1.5f)
    {
        ccm::CCMatrix3x3 sat = ccm::saturation_matrix(1.5f);

        auto start = std::chrono::steady_clock::now();
        auto err = ccm::process_ccm(input.data(), output.data(),
                                    width, height, channels,
                                    ccm::CCMAlgorithm::LINEAR_3X3, 8, &sat);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("saturation_matrix(1.5): ERROR - %s\n", ccm::ccm_error_message(err));
        } else {
            printf("saturation_matrix(1.5): OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("ccm_saturation.rgb", "wb");
            if (out) { fwrite(output.data(), 1, img_size, out); fclose(out); }
            printf("  Saved ccm_saturation.rgb\n");
        }
    }

    printf("Done.\n");
    return 0;
}
