// color_temp_example.cpp
// Demonstrates color temperature adjustment using several algorithms.
// Links against im_operator library.

#include "color_temp.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <chrono>

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
    printf("=== Color Temperature Example ===\n");

    const int W = 640, H = 480, C = 3, BD = 8;
    size_t bytes = (size_t)W * H * C;
    auto src = load_raw(TEST_DATA("rgb_640x480_8bit.raw"), bytes);
    if (src.empty()) return 1;
    printf("Loaded rgb_640x480_8bit.raw (%zux%zux%d, %d-bit)\n\n", (size_t)W, (size_t)H, C, BD);

    std::vector<uint8_t> dst(bytes);
    using namespace color_temp;

    // --- KELVIN: warm 3500K ---
    auto err = process_color_temp(src.data(), dst.data(), W, H, C,
                                  ColorTempAlgorithm::KELVIN, BD, 3500);
    printf("  KELVIN 3500K (warm):  %s\n", color_temp_error_message(err));
    if (ok(err)) save_raw("color_temp_kelvin_3500k.rgb", dst);

    // --- KELVIN: cool 7500K ---
    err = process_color_temp(src.data(), dst.data(), W, H, C,
                             ColorTempAlgorithm::KELVIN, BD, 7500);
    printf("  KELVIN 7500K (cool):  %s\n", color_temp_error_message(err));
    if (ok(err)) save_raw("color_temp_kelvin_7500k.rgb", dst);

    // --- KELVIN: daylight 5500K ---
    err = process_color_temp(src.data(), dst.data(), W, H, C,
                             ColorTempAlgorithm::KELVIN, BD, 5500);
    printf("  KELVIN 5500K (daylight): %s\n", color_temp_error_message(err));
    if (ok(err)) save_raw("color_temp_kelvin_5500k.rgb", dst);

    // --- PRESET: TUNGSTEN (~2850K) ---
    err = process_color_temp(src.data(), dst.data(), W, H, C,
                             ColorTempAlgorithm::PRESET, BD, 0,
                             IlluminantPreset::TUNGSTEN_100W);
    printf("  PRESET TUNGSTEN_100W:    %s\n", color_temp_error_message(err));
    if (ok(err)) save_raw("color_temp_preset_tungsten.rgb", dst);

    // --- PRESET: WARM_FLUORESCENT (~3500K) ---
    err = process_color_temp(src.data(), dst.data(), W, H, C,
                             ColorTempAlgorithm::PRESET, BD, 0,
                             IlluminantPreset::WARM_FLUORESCENT);
    printf("  PRESET WARM_FLUORESCENT: %s\n", color_temp_error_message(err));
    if (ok(err)) save_raw("color_temp_preset_warm_fluo.rgb", dst);

    // --- PRESET: MIDDAY_SUN (~5500K daylight) ---
    err = process_color_temp(src.data(), dst.data(), W, H, C,
                             ColorTempAlgorithm::PRESET, BD, 0,
                             IlluminantPreset::MIDDAY_SUN);
    printf("  PRESET MIDDAY_SUN:       %s\n", color_temp_error_message(err));
    if (ok(err)) save_raw("color_temp_preset_daylight.rgb", dst);

    // --- MANUAL: r_gain=1.2, b_gain=0.8 (g=1.0 implicit) ---
    err = process_color_temp(src.data(), dst.data(), W, H, C,
                             ColorTempAlgorithm::MANUAL, BD, 0,
                             IlluminantPreset::CLOUDY, 1.2f, 0.8f);
    printf("  MANUAL (r=1.2, g=1.0, b=0.8): %s\n", color_temp_error_message(err));
    if (ok(err)) save_raw("color_temp_manual.rgb", dst);

    // --- WHITE_BALANCE: auto white-balance via gray-world ---
    {
        auto start = std::chrono::steady_clock::now();
        err = process_color_temp(src.data(), dst.data(), W, H, C,
                                 ColorTempAlgorithm::WHITE_BALANCE, BD);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("  WHITE_BALANCE (auto gray-world): %s (%lld ms)\n", color_temp_error_message(err), (long long)ms);
        if (ok(err)) save_raw("color_temp_white_balance.rgb", dst);
    }

    printf("\nAll color temperature demos complete.\n");
    return 0;
}
