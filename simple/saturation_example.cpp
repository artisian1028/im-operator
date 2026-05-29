// saturation_example.cpp
// Demonstrates saturation adjustment: HSL, vibrance, channel mixer, selective.
// Links against im_operator library.

#include "saturation.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

static std::vector<uint8_t> load_raw(const char* path, size_t expected_bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return {}; }
    std::vector<uint8_t> buf(expected_bytes);
    size_t n = fread(buf.data(), 1, expected_bytes, f);
    fclose(f);
    if (n != expected_bytes) { fprintf(stderr, "ERROR: short read %s\n", path); return {}; }
    return buf;
}

static bool save_raw(const char* path, const std::vector<uint8_t>& buf) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path); return false; }
    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
    return true;
}

#define TEST_DATA(path) "test_data/" path

int main() {
    printf("=== Saturation Example ===\n");

    const int W = 640, H = 480, C = 3, BD = 8;
    size_t bytes = (size_t)W * H * C;
    auto src = load_raw(TEST_DATA("rgb_640x480_8bit.raw"), bytes);
    if (src.empty()) return 1;
    printf("Loaded rgb_640x480_8bit.raw (%zux%zux%d, %d-bit)\n\n", (size_t)W, (size_t)H, C, BD);

    std::vector<uint8_t> dst(bytes);
    using namespace saturation;

    // --- HSL: boost saturation to 1.5 ---
    SaturationParams hsl_params;
    hsl_params.saturation = 1.5f;
    auto err = process_saturation(src.data(), dst.data(), W, H, C,
                                  SaturationAlgorithm::HSL, BD, hsl_params);
    printf("  HSL (saturation=1.5):        %s\n", saturation_error_message(err));
    if (ok(err)) save_raw("saturation_hsl_boost.rgb", dst);

    // --- HSL: desaturate to 0.5 (partial B&W) ---
    hsl_params.saturation = 0.5f;
    err = process_saturation(src.data(), dst.data(), W, H, C,
                             SaturationAlgorithm::HSL, BD, hsl_params);
    printf("  HSL (desaturate=0.5):        %s\n", saturation_error_message(err));
    if (ok(err)) save_raw("saturation_hsl_desaturate.rgb", dst);

    // --- VIBRANCE: intelligent saturation boost ---
    SaturationParams vib_params;
    vib_params.vibrance = 0.8f;
    err = process_saturation(src.data(), dst.data(), W, H, C,
                             SaturationAlgorithm::VIBRANCE, BD, vib_params);
    printf("  VIBRANCE (vibrance=0.8):     %s\n", saturation_error_message(err));
    if (ok(err)) save_raw("saturation_vibrance.rgb", dst);

    // --- CHANNEL_MIXER: strong cross-channel mixing ---
    SaturationParams mixer_params;
    mixer_params.saturation = 1.5f;
    err = process_saturation(src.data(), dst.data(), W, H, C,
                             SaturationAlgorithm::CHANNEL_MIXER, BD, mixer_params);
    printf("  CHANNEL_MIXER (sat=1.5):     %s\n", saturation_error_message(err));
    if (ok(err)) save_raw("saturation_channel_mixer.rgb", dst);

    // --- SELECTIVE: boost red channel ---
    SaturationParams sel_params;
    sel_params.r_sat = 2.0f;   // boost reds
    sel_params.g_sat = 1.0f;
    sel_params.b_sat = 1.0f;
    err = process_saturation(src.data(), dst.data(), W, H, C,
                             SaturationAlgorithm::SELECTIVE, BD, sel_params);
    printf("  SELECTIVE (boost reds):      %s\n", saturation_error_message(err));
    if (ok(err)) save_raw("saturation_selective_reds.rgb", dst);

    printf("\nAll saturation demos complete.\n");
    return 0;
}
