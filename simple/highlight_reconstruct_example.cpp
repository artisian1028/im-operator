// highlight_reconstruct_example.cpp
// Demonstrates highlight (clipped pixel) reconstruction on RGB data.
// Links against im_operator library.

#include "highlight_reconstruct.h"
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
    printf("=== Highlight Reconstruction Example ===\n");

    const int W = 1920, H = 1080, C = 3, BD = 8;
    size_t bytes = (size_t)W * H * C;
    auto src = load_raw(TEST_DATA("highlight_clipped.raw"), bytes);
    if (src.empty()) return 1;
    printf("Loaded highlight_clipped.raw (%zux%zux%d, %d-bit)\n\n", (size_t)W, (size_t)H, C, BD);

    std::vector<uint8_t> dst(bytes);
    using namespace highlight_reconstruct;

    HighlightReconstructParams params;
    params.threshold = 0.95f;  // clip detection threshold

    // --- CHANNEL_GUIDED: uses unclipped channel data to reconstruct ---
    {
        auto err = process_highlight_reconstruct(src.data(), dst.data(), W, H, C,
                                                  HighlightReconstructAlgorithm::CHANNEL_GUIDED,
                                                  BD, params);
        printf("  CHANNEL_GUIDED:  %s\n", highlight_reconstruct_error_message(err));
        if (ok(err)) save_raw("highlight_channel_guided.rgb", dst);
    }

    // --- GRADIENT_BASED: uses gradient direction for interpolation ---
    {
        auto err = process_highlight_reconstruct(src.data(), dst.data(), W, H, C,
                                                  HighlightReconstructAlgorithm::GRADIENT_BASED,
                                                  BD, params);
        printf("  GRADIENT_BASED:  %s\n", highlight_reconstruct_error_message(err));
        if (ok(err)) save_raw("highlight_gradient_based.rgb", dst);
    }

    printf("\nHighlight reconstruction demos complete.\n");
    return 0;
}
