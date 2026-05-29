// defect_correct_example.cpp
// Demonstrates defect pixel correction on Bayer data (in-place, ADAPTIVE mode).
// Links against im_operator library.

#include "defect_correct.h"
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
    printf("=== Defect Pixel Correction Example ===\n");

    const int W = 1920, H = 1080, C = 1, BD = 8;  // Bayer = 1 channel
    size_t bytes = (size_t)W * H;
    auto src = load_raw(TEST_DATA("defect_pixels.raw"), bytes);
    if (src.empty()) return 1;
    printf("Loaded defect_pixels.raw (%zux%zu, Bayer RGGB, %d-bit)\n\n", (size_t)W, (size_t)H, BD);

    using namespace defect_correct;
    using namespace imop;

    // --- ADAPTIVE: auto-detect defects by comparing to same-color neighbors ---
    {
        // Copy data since correction modifies in-place
        std::vector<uint8_t> buf = src;

        DefectCorrectParams params;
        params.threshold = 0.3f;  // relative deviation threshold

        auto err = process_defect_correct(buf.data(), W, H, BayerPattern::RGGB,
                                           DefectCorrectAlgorithm::ADAPTIVE, BD, params);
        printf("  ADAPTIVE (threshold=0.3):  %s\n", defect_correct_error_message(err));
        if (ok(err)) save_raw("defect_corrected.raw", buf);
    }

    // --- MAP_BASED: fix known defect positions from a pre-defined defect map ---
    {
        std::vector<uint8_t> buf = src;

        // Define a defect map with a few known bad pixels
        std::vector<DefectPoint> defect_map = {
            {100, 50},   // hot pixel at (100, 50)
            {200, 150},  // hot pixel at (200, 150)
            {300, 250},  // hot pixel at (300, 250)
            {500, 400},  // hot pixel at (500, 400)
            {800, 600},  // hot pixel at (800, 600)
            {1200, 800}, // hot pixel at (1200, 800)
            {1500, 900}, // hot pixel at (1500, 900)
            {1800, 1000} // hot pixel at (1800, 1000)
        };

        DefectCorrectParams params;
        params.threshold = 0.0f;             // not used by MAP_BASED
        params.map = defect_map.data();
        params.map_count = (int)defect_map.size();

        auto err = process_defect_correct(buf.data(), W, H, BayerPattern::RGGB,
                                           DefectCorrectAlgorithm::MAP_BASED, BD, params);
        printf("  MAP_BASED (%d defect points):  %s\n", params.map_count, defect_correct_error_message(err));
        if (ok(err)) save_raw("defect_corrected_map.raw", buf);
    }

    printf("\nDefect correction demo complete.\n");
    return 0;
}
