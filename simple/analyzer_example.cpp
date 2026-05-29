// analyzer_example.cpp
// Demonstrates the data analyzer for raw Bayer data inspection
// Purely in-memory demo with a synthetic gradient pattern

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include "im_operator.h"

int main() {
    printf("=== Data Analyzer Example ===\n\n");
    using namespace imop;

    // Create a 100x100 Bayer RGGB 8-bit buffer with a gradient pattern
    const int W = 100, H = 100;
    std::vector<uint8_t> bayer(W * H);

    // Fill with a smooth gradient: left-to-right increasing, with per-channel variation
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // RGGB: R at (even, even), Gr at (even, odd), Gb at (odd, even), B at (odd, odd)
            int chan = (y & 1) * 2 + (x & 1);  // 0=R, 1=Gr, 2=Gb, 3=B
            double base = (double)x / (W - 1) * 200.0;  // 0..200 range
            double row_factor = (double)y / (H - 1) * 50.0; // 0..50 vertical variation
            double chan_offset[4] = {10.0, 30.0, 50.0, 70.0};
            int val = (int)(base + row_factor + chan_offset[chan]);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            bayer[y * W + x] = (uint8_t)val;
        }
    }
    printf("Created 100x100 Bayer RGGB 8-bit gradient pattern.\n");
    printf("  File size: %zu bytes\n\n", bayer.size());

    // --- detect_bit_depth ---
    printf("--- detect_bit_depth ---\n");
    int detected_bd = detect_bit_depth(bayer.data(), bayer.size());
    printf("  Detected bit depth: %d\n", detected_bd);
    if (detected_bd == 8)
        printf("  RESULT: Correct - detected 8-bit\n");
    else if (detected_bd == 0)
        printf("  RESULT: Detection returned 0 (inconclusive)\n");
    else
        printf("  RESULT: Unexpected - got %d, expected 8\n", detected_bd);

    // --- guess_pattern ---
    printf("\n--- guess_pattern ---\n");
    BayerPattern guessed = guess_pattern(bayer.data(), W, H, 8, false);
    const char* pattern_names[] = { "RGGB", "BGGR", "GRBG", "GBRG" };
    printf("  Guessed pattern: %s\n", pattern_names[(int)guessed]);
    printf("  Expected pattern: RGGB\n");
    if (guessed == BayerPattern::RGGB)
        printf("  RESULT: Correct\n");
    else
        printf("  RESULT: Mismatch (may depend on input data statistics)\n");

    // --- analyze_data (comprehensive) ---
    printf("\n--- analyze_data (full analysis) ---\n");
    DataInfo info = analyze_data(bayer.data(), bayer.size());
    printf("  detected_bit_depth: %d\n", info.detected_bit_depth);
    printf("  pixel_count:        %d\n", info.pixel_count);
    printf("  max_value:          %d\n", info.max_value);
    printf("  min_value:          %d\n", info.min_value);
    printf("  is_likely_16bit:    %s\n", info.is_likely_16bit ? "true" : "false");
    printf("  is_packed:          %s\n", info.is_packed ? "true" : "false");
    printf("  suggested_width:    %d\n", info.suggested_width);
    printf("  suggested_height:   %d\n", info.suggested_height);

    // --- suggest_dimensions ---
    printf("\n--- suggest_dimensions ---\n");
    size_t pixel_count = (size_t)W * H;  // 10000 pixels
    auto dims = suggest_dimensions(pixel_count);
    printf("  Pixel count: %zu\n", pixel_count);
    printf("  Suggested dimension pairs:\n");
    int shown = 0;
    for (auto& d : dims) {
        printf("    %4d x %-4d  (= %d pixels)\n", d.first, d.second, d.first * d.second);
        if (++shown >= 10) break;
    }
    if (dims.size() > 10)
        printf("    ... and %zu more\n", dims.size() - 10);

    // Verify that the known dimensions are in the list
    bool found = false;
    for (auto& d : dims) {
        if (d.first == W && d.second == H) { found = true; break; }
        if (d.first == H && d.second == W) { found = true; break; }
    }
    if (found)
        printf("  RESULT: (100,100) found in suggested dimensions\n");
    else
        printf("  NOTE: (100,100) not in top suggestions (other factors may come first)\n");

    printf("\nDone.\n");
    return 0;
}
