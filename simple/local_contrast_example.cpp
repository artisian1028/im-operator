// local_contrast_example.cpp
// Demonstrates local contrast enhancement (clarity/texture) on RGB data.
// Links against im_operator library.

#include "local_contrast.h"
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
    printf("=== Local Contrast Example ===\n");

    const int W = 640, H = 480, C = 3, BD = 8;
    size_t bytes = (size_t)W * H * C;
    auto src = load_raw(TEST_DATA("rgb_640x480_8bit.raw"), bytes);
    if (src.empty()) return 1;
    printf("Loaded rgb_640x480_8bit.raw (%zux%zux%d, %d-bit)\n\n", (size_t)W, (size_t)H, C, BD);

    std::vector<uint8_t> dst(bytes);
    using namespace local_contrast;

    // --- UNSHARP: large-radius unsharp mask (clarity) ---
    {
        LocalContrastParams params;
        params.amount = 0.3f;   // moderate strength
        params.radius = 10.0f;  // sigma = 10 pixels

        auto err = process_local_contrast(src.data(), dst.data(), W, H, C,
                                           LocalContrastAlgorithm::UNSHARP, BD, params);
        printf("  UNSHARP (amount=0.3, sigma=10): %s\n", local_contrast_error_message(err));
        if (ok(err)) save_raw("local_contrast_unsharp.rgb", dst);
    }

    // --- BILATERAL: bilateral decomposition based local contrast ---
    {
        LocalContrastParams params;
        params.amount = 0.4f;   // moderate strength
        params.radius = 5.0f;   // radius = 5 pixels

        auto err = process_local_contrast(src.data(), dst.data(), W, H, C,
                                           LocalContrastAlgorithm::BILATERAL, BD, params);
        printf("  BILATERAL (amount=0.4, radius=5): %s\n", local_contrast_error_message(err));
        if (ok(err)) save_raw("local_contrast_bilateral.rgb", dst);
    }

    printf("\nLocal contrast demos complete.\n");
    return 0;
}
