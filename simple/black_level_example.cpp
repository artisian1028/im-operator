// black_level_example.cpp
// Demonstrates black level subtraction on Bayer pattern data (in-place).
// Links against im_operator library.

#include "black_level.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>

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
    printf("=== Black Level Example ===\n");

    const int W = 1920, H = 1080, C = 1, BD = 8;  // Bayer = 1 channel
    size_t bytes = (size_t)W * H;
    auto src = load_raw(TEST_DATA("bayer_rggb_1920x1080_8bit.raw"), bytes);
    if (src.empty()) return 1;
    printf("Loaded bayer_rggb_1920x1080_8bit.raw (%zux%zu, Bayer, %d-bit)\n\n",
           (size_t)W, (size_t)H, BD);

    using namespace black_level;
    using namespace imop;

    // --- PER_CHANNEL: different offsets per Bayer channel ---
    {
        // Copy data since black_level modifies in-place
        std::vector<uint8_t> buf = src;
        BlackLevelParams params;
        params.r_offset  = 8.0f;
        params.gr_offset = 6.0f;
        params.gb_offset = 6.0f;
        params.b_offset  = 10.0f;
        auto err = process_black_level(buf.data(), W, H, BayerPattern::RGGB,
                                        BlackLevelAlgorithm::PER_CHANNEL, BD, params);
        printf("  PER_CHANNEL (r=8,gr=6,gb=6,b=10): %s\n", black_level_error_message(err));
        if (ok(err)) save_raw("bl_per_channel.raw", buf);
    }

    // --- GLOBAL: single offset for all pixels ---
    {
        std::vector<uint8_t> buf = src;
        BlackLevelParams params;
        params.r_offset = params.gr_offset = params.gb_offset = params.b_offset = 10.0f;
        auto err = process_black_level(buf.data(), W, H, BayerPattern::RGGB,
                                        BlackLevelAlgorithm::GLOBAL, BD, params);
        printf("  GLOBAL (offset=10):               %s\n", black_level_error_message(err));
        if (ok(err)) save_raw("bl_global.raw", buf);
    }

    printf("\nBlack level demos complete.\n");
    return 0;
}
