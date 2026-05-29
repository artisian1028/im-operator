// lens_shading_example.cpp
// Demonstrates lens shading (vignetting) correction on Bayer RGGB data
// Compile: linked with im_operator library
// Requires: test_data/lens_shading_test.raw (1918080 bytes)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include "lens_shading.h"

static bool load_raw(const char* filename, std::vector<uint8_t>& buf) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", filename); return false; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize(sz);
    size_t rd = fread(buf.data(), 1, sz, f);
    fclose(f);
    return rd == (size_t)sz;
}

static bool save_raw(const char* filename, const std::vector<uint8_t>& buf) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", filename); return false; }
    size_t wr = fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
    return wr == buf.size();
}

int main() {
    printf("=== Lens Shading Correction Example ===\n\n");

    // Load Bayer RGGB 1920x1080 8-bit test image
    std::vector<uint8_t> src;
    if (!load_raw("test_data/lens_shading_test.raw", src)) return 1;
    printf("Loaded lens_shading_test.raw (%zu bytes)\n", src.size());

    // Copy to work buffer (lens shading operates in-place on Bayer data)
    std::vector<uint8_t> buf = src;
    printf("Copied to work buffer\n");

    // Configure polynomial shading correction parameters
    // gain(r) = 1 + a2*r^2 + a4*r^4 + a6*r^6
    using namespace lens_shading;

    LensShadingParams params;
    // Optical center in pixel coordinates, normalized to 0-1
    params.center_x = 960.0f / 1920.0f;  // 960 / 1920 = 0.5
    params.center_y = 540.0f / 1080.0f;  // 540 / 1080 = 0.5

    // Per-channel quadratic (a2) coefficients: R, Gr, Gb, B
    params.r_coef.a2  = 0.10f;
    params.gr_coef.a2 = 0.10f;
    params.gb_coef.a2 = 0.08f;
    params.b_coef.a2  = 0.12f;

    // Per-channel quartic (a4) coefficients
    params.r_coef.a4  = 0.05f;
    params.gr_coef.a4 = 0.05f;
    params.gb_coef.a4 = 0.04f;
    params.b_coef.a4  = 0.06f;

    // a6 coefficients left at default (0.0)

    printf("Polynomial params: center=(%.3f,%.3f), a2={%.2f,%.2f,%.2f,%.2f}, a4={%.2f,%.2f,%.2f,%.2f}\n",
           params.center_x, params.center_y,
           params.r_coef.a2, params.gr_coef.a2, params.gb_coef.a2, params.b_coef.a2,
           params.r_coef.a4, params.gr_coef.a4, params.gb_coef.a4, params.b_coef.a4);

    // Apply lens shading correction
    LensShadingError err = process_lens_shading(
        buf.data(),
        1920, 1080,                     // width, height
        imop::BayerPattern::RGGB,        // Bayer pattern
        LensShadingAlgorithm::POLYNOMIAL, // algorithm
        8,                               // bit depth
        params                           // shading parameters
    );

    if (err == LensShadingError::Ok) {
        printf("SUCCESS: Lens shading correction applied.\n");
        printf("  Algorithm: %s\n", algorithm_name(LensShadingAlgorithm::POLYNOMIAL).c_str());
        if (save_raw("lens_shading_corrected.raw", buf))
            printf("  Saved result to lens_shading_corrected.raw\n");
        else
            printf("  WARNING: Failed to save output file\n");
    } else {
        printf("FAIL: %s\n", lens_shading_error_message(err));
        return 1;
    }

    // --- FLAT_FIELD: gain correction using a flat-field reference image ---
    {
        // Make a fresh copy of the original data
        std::vector<uint8_t> ff_buf = src;
        printf("\n");

        LensShadingParams ff_params;
        // Use the original (uncorrected) image as a demonstration flat-field reference.
        // In practice this would be an image of a uniformly-lit white target.
        ff_params.flat_field = src.data();
        ff_params.flat_field_width = 1920;
        ff_params.flat_field_height = 1080;

        LensShadingError ff_err = process_lens_shading(
            ff_buf.data(),
            1920, 1080,
            imop::BayerPattern::RGGB,
            LensShadingAlgorithm::FLAT_FIELD,
            8,
            ff_params
        );

        if (ff_err == LensShadingError::Ok) {
            printf("SUCCESS: Flat-field correction applied.\n");
            printf("  Algorithm: %s\n", algorithm_name(LensShadingAlgorithm::FLAT_FIELD).c_str());
            if (save_raw("lens_shading_flatfield.raw", ff_buf))
                printf("  Saved result to lens_shading_flatfield.raw\n");
            else
                printf("  WARNING: Failed to save output file\n");
        } else {
            printf("FAIL (flat-field): %s\n", lens_shading_error_message(ff_err));
        }
    }

    // Verify: corner pixels should be brighter (corrected) after processing
    // We just check the first pixel value changed (vignette darkens corners -> correction brightens them)
    printf("\nSample pixel values (before -> after):\n");
    printf("  Top-left (0,0):       %3d -> %3d\n", src[0], buf[0]);
    printf("  Center (960,540):     %3d -> %3d\n",
           src[960 + 540 * 1920], buf[960 + 540 * 1920]);
    printf("  Bottom-right (1919,1079): %3d -> %3d\n",
           src[1919 + 1079 * 1920], buf[1919 + 1079 * 1920]);

    printf("\nDone.\n");
    return 0;
}
