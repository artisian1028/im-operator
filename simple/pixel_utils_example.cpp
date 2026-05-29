// pixel_utils_example.cpp
// Demonstrates pixel utility functions for raw Bayer data I/O
// Purely in-memory demo - no file I/O needed

#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
#include "im_operator.h"

int main() {
    printf("=== Pixel Utilities Example ===\n\n");
    using namespace imop;
    using namespace imop::pixel;

    // Create a small Bayer RGGB 64x64 8-bit image with a test pattern
    const int W = 64, H = 64;
    std::vector<uint8_t> bayer(W * H);

    // Fill with a test pattern: value = row*2 + col%256
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            bayer[y * W + x] = (uint8_t)((y * 2 + x) & 0xFF);
        }
    }
    printf("Created 64x64 Bayer RGGB 8-bit test image.\n\n");

    int passed = 0, failed = 0;
    auto check = [&](const char* name, bool cond) {
        if (cond) { printf("  PASS: %s\n", name); passed++; }
        else      { printf("  FAIL: %s\n", name); failed++; }
    };

    // ---------------------------------------------------------------
    // 1. get_raw_8: read individual Bayer pixel values
    // ---------------------------------------------------------------
    printf("--- get_raw_8 / set_raw_8-style access ---\n");

    int v00 = get_raw_8(bayer.data(), 0, 0, W);
    int v11 = get_raw_8(bayer.data(), 1, 1, W);
    int v6331 = get_raw_8(bayer.data(), 63, 31, W);

    printf("  Pixel (0,0) = %d\n", v00);
    printf("  Pixel (1,1) = %d\n", v11);
    printf("  Pixel (63,31) = %d\n", v6331);

    // Formula: pixel[y * W + x] = (y*2 + x) & 0xFF
    check("get_raw_8 (0,0)",     v00   == ((0 * 2 + 0) & 0xFF));
    check("get_raw_8 (1,1)",     v11   == ((1 * 2 + 1) & 0xFF));
    check("get_raw_8 (63,31)",  v6331  == ((31 * 2 + 63) & 0xFF));

    // Direct pixel write to demonstrate byte-level access
    bayer[10 * W + 20] = 99;
    check("direct byte write + get_raw_8 (20,10)", get_raw_8(bayer.data(), 20, 10, W) == 99);

    // ---------------------------------------------------------------
    // 2. 10-bit packed format: get_packed_raw / set_packed_raw
    // ---------------------------------------------------------------
    printf("\n--- 10-bit Packed Raw Access ---\n");

    // Create a 4x4 10-bit packed buffer
    // 4 pixels at 10 bits = 40 bits = 5 bytes
    // Packing: 4 pixels in 5 bytes, layout per 4-pixel group
    const int PW = 4, PH = 1;
    size_t packed_size = compute_packed_byte_size(PW, PH, 10); // = 5 bytes
    std::vector<uint8_t> packed_buf(packed_size, 0);

    printf("  10-bit packed: %dx%d, %zu bytes\n", PW, PH, packed_size);

    // Write known 10-bit values into the packed buffer using the pixel_utils API
    // We use get_raw / get_packed_raw to read, but we need raw byte manipulation to write
    // Values: pixel 0=512, pixel 1=100, pixel 2=1023, pixel 3=300
    // Manually pack 4 pixels into 5 bytes:
    // byte0 = p0[9:2], byte1 = (p0[1:0]<<6)|p1[9:4], byte2 = (p1[3:0]<<4)|p2[9:6],
    // byte3 = (p2[5:0]<<2)|p3[9:8], byte4 = p3[7:0]
    uint16_t p0 = 512, p1 = 100, p2 = 1023, p3 = 300;
    packed_buf[0] = (uint8_t)((p0 >> 2) & 0xFF);
    packed_buf[1] = (uint8_t)(((p0 & 0x3) << 6) | ((p1 >> 4) & 0x3F));
    packed_buf[2] = (uint8_t)(((p1 & 0xF) << 4) | ((p2 >> 6) & 0xF));
    packed_buf[3] = (uint8_t)(((p2 & 0x3F) << 2) | ((p3 >> 8) & 0x3));
    packed_buf[4] = (uint8_t)(p3 & 0xFF);

    // Read back with get_packed_raw
    int rp0 = get_packed_raw(packed_buf.data(), 0, 0, PW, 10);
    int rp1 = get_packed_raw(packed_buf.data(), 1, 0, PW, 10);
    int rp2 = get_packed_raw(packed_buf.data(), 2, 0, PW, 10);
    int rp3 = get_packed_raw(packed_buf.data(), 3, 0, PW, 10);

    printf("  Written: p0=%d p1=%d p2=%d p3=%d\n", p0, p1, p2, p3);
    printf("  Read:    p0=%d p1=%d p2=%d p3=%d\n", rp0, rp1, rp2, rp3);

    check("get_packed_raw pixel 0", rp0 == 512);
    check("get_packed_raw pixel 1", rp1 == 100);
    check("get_packed_raw pixel 2", rp2 == 1023);
    check("get_packed_raw pixel 3", rp3 == 300);

    // ---------------------------------------------------------------
    // 3. align_raw_value: bit-depth conversion
    // ---------------------------------------------------------------
    printf("\n--- align_raw_value (bit-depth conversion) ---\n");

    // 12-bit value 4095 (max for 12-bit) in a 16-bit container
    uint16_t val12 = 0x0FFF;  // stored in lower 12 bits
    uint16_t aligned_16 = align_raw_value(val12, 12);
    printf("  12-bit max 0x%04X -> align_raw_value -> 0x%04X\n", val12, aligned_16);
    // For 12-bit, max_val is (1<<12)-1 = 4095 = 0x0FFF, which fits in 16-bit container
    check("align_raw_value 12->16", aligned_16 == 0x0FFF);

    // 10-bit value 1023 in 16-bit container
    uint16_t val10 = 0x03FF;
    uint16_t aligned_10 = align_raw_value(val10, 10);
    printf("  10-bit max 0x%04X -> align_raw_value -> 0x%04X\n", val10, aligned_10);
    check("align_raw_value 10->16", aligned_10 == 0x03FF);

    // 8-bit value
    uint16_t val8 = 255;
    uint16_t aligned_8 = align_raw_value(val8, 8);
    printf("  8-bit max %d -> align_raw_value -> %d\n", val8, aligned_8);
    check("align_raw_value 8->16", aligned_8 == 255);

    // ---------------------------------------------------------------
    // 4. set_rgb_clamped: write RGB pixels with clamping
    // ---------------------------------------------------------------
    printf("\n--- set_rgb_clamped ---\n");

    const int RGB_W = 4, RGB_H = 2, RGB_CH = 3;
    std::vector<uint8_t> rgb_buf(RGB_W * RGB_H * RGB_CH, 0);

    // Set pixel (1,0) to valid RGB values
    set_rgb_clamped(rgb_buf.data(), 1, 0, RGB_W, 180, 100, 50, 8);
    int r = rgb_buf[(0 * RGB_W + 1) * 3 + 0];
    int g = rgb_buf[(0 * RGB_W + 1) * 3 + 1];
    int b = rgb_buf[(0 * RGB_W + 1) * 3 + 2];
    printf("  set_rgb_clamped (1,0): r=%d g=%d b=%d\n", r, g, b);
    check("set_rgb_clamped valid values", r == 180 && g == 100 && b == 50);

    // Set pixel (2,1) to out-of-range values (should clamp to 0-255)
    set_rgb_clamped(rgb_buf.data(), 2, 1, RGB_W, 300, -20, 128, 8);
    r = rgb_buf[(1 * RGB_W + 2) * 3 + 0];
    g = rgb_buf[(1 * RGB_W + 2) * 3 + 1];
    b = rgb_buf[(1 * RGB_W + 2) * 3 + 2];
    printf("  set_rgb_clamped (2,1) with overshoot: r=%d g=%d b=%d\n", r, g, b);
    check("set_rgb_clamped clamps >255", r == 255);
    check("set_rgb_clamped clamps <0",   g == 0);
    check("set_rgb_clamped preserves valid", b == 128);

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    printf("\n=== Results: %d passed, %d failed out of %d tests ===\n",
           passed, failed, passed + failed);

    return (failed > 0) ? 1 : 0;
}
