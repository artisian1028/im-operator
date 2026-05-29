// hdr_example.cpp
// Demonstrates HDR tone mapping operators on 16-bit input, outputting 8-bit.
// Links against im_operator library.

#include "hdr.h"
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
    printf("=== HDR Tone Mapping Example ===\n");

    const int W = 1920, H = 1080, C = 3;
    const int BD_IN = 16, BD_OUT = 8;
    size_t bytes_in = (size_t)W * H * C * (BD_IN / 8);   // 2 bytes per channel
    size_t bytes_out = (size_t)W * H * C;                 // 1 byte per channel

    auto src = load_raw(TEST_DATA("rgb_1920x1080_16bit.raw"), bytes_in);
    if (src.empty()) return 1;
    printf("Loaded rgb_1920x1080_16bit.raw (%zux%zux%d, %d-bit)\n", (size_t)W, (size_t)H, C, BD_IN);

    // For 8-bit output, allocate a separate 8-bit buffer.
    // The HDR function takes uint8_t* for both input and output; the bit_depth
    // parameter tells it how to interpret the pointer.
    std::vector<uint8_t> out8(bytes_out);
    using namespace hdr;

    HdrParams params;
    params.exposure = 0.0f;
    params.gamma = 2.2f;
    params.saturation = 1.0f;

    // --- REINHARD: classic photographic tone mapping ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::REINHARD, BD_OUT, params);
        printf("  REINHARD:                %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_reinhard.rgb", out8);
    }

    // --- FILMIC_ACES: cinematic look ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::FILMIC_ACES, BD_OUT, params);
        printf("  FILMIC_ACES:             %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_aces.rgb", out8);
    }

    // --- HABLE: Uncharted 2 style ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::HABLE, BD_OUT, params);
        printf("  HABLE:                   %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_hable.rgb", out8);
    }

    // --- DRAGO: adaptive log-based tone mapping ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::DRAGO, BD_OUT, params);
        printf("  DRAGO:                   %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_drago.rgb", out8);
    }

    // --- REINHARD_EXT: Reinhard extended with key + white point ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::REINHARD_EXT, BD_OUT, params);
        printf("  REINHARD_EXT:            %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_reinhard_ext.rgb", out8);
    }

    // --- ADAPTIVE_LOCAL: bilateral decomposition local tone mapping ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::ADAPTIVE_LOCAL, BD_OUT, params);
        printf("  ADAPTIVE_LOCAL:          %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_adaptive_local.rgb", out8);
    }

    // --- EXPONENTIAL: 1 - exp(-k * L) ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::EXPONENTIAL, BD_OUT, params);
        printf("  EXPONENTIAL:             %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_exponential.rgb", out8);
    }

    // --- LOGARITHMIC: log(1 + k*L) / log(1 + k) ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::LOGARITHMIC, BD_OUT, params);
        printf("  LOGARITHMIC:             %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_logarithmic.rgb", out8);
    }

    // --- LINEAR_TO_PQ: Linear to ST.2084 PQ ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::LINEAR_TO_PQ, BD_OUT, params);
        printf("  LINEAR_TO_PQ:            %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_linear_to_pq.rgb", out8);
    }

    // --- LINEAR_TO_HLG: Linear to BT.2100 HLG ---
    {
        auto err = process_hdr(src.data(), out8.data(), W, H, C,
                               HdrAlgorithm::LINEAR_TO_HLG, BD_OUT, params);
        printf("  LINEAR_TO_HLG:           %s\n", hdr_error_message(err));
        if (ok(err)) save_raw("hdr_linear_to_hlg.rgb", out8);
    }

    printf("\nAll HDR tone mapping demos complete.\n");
    return 0;
}
