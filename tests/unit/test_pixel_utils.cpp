#include <gtest/gtest.h>
#include <cstring>
#include "imop/pixel_utils.hpp"
#include "imop/types.hpp"

using namespace imop;
using namespace imop::pixel;

TEST(PixelUtilsTest, SafeMaxVal) {
    EXPECT_EQ(safe_max_val(8), 255);
    EXPECT_EQ(safe_max_val(10), 1023);
    EXPECT_EQ(safe_max_val(12), 4095);
    EXPECT_EQ(safe_max_val(16), 65535);
    EXPECT_EQ(safe_max_val(0), 0);
    EXPECT_EQ(safe_max_val(17), 65535);
}

TEST(PixelUtilsTest, ComputeByteSizes) {
    EXPECT_EQ(compute_bayer_byte_size(1920, 1080, 8, false), 1920u * 1080u);
    EXPECT_EQ(compute_bayer_byte_size(1920, 1080, 16, false), 1920u * 1080u * 2u);
    EXPECT_EQ(compute_rgb_byte_size(1920, 1080, 8), 1920u * 1080u * 3u);
    EXPECT_EQ(compute_rgb_byte_size(1920, 1080, 16), 1920u * 1080u * 6u);
}

TEST(PixelUtilsTest, Packed10BitByteSize) {
    size_t sz = compute_packed_byte_size(100, 100, 10);
    EXPECT_GE(sz, 100u * 100u * 10u / 8u);
    EXPECT_LE(sz, 100u * 100u * 10u / 8u + 1u);
}

TEST(PixelUtilsTest, PixelPredicates) {
    auto po = PatternOffsets::from_pattern(BayerPattern::RGGB);
    EXPECT_TRUE(is_r_at(po, 0, 0));
    EXPECT_TRUE(is_b_at(po, 1, 1));
    EXPECT_TRUE(is_g_at(po, 0, 1));
    EXPECT_TRUE(is_g_at(po, 1, 0));

    EXPECT_FALSE(is_r_at(po, 1, 1));
    EXPECT_FALSE(is_b_at(po, 0, 0));
}

TEST(PixelUtilsTest, ReadWriteU16) {
    uint8_t buf[4] = {0x34, 0x12, 0xCD, 0xAB};
    EXPECT_EQ(read_u16(buf, 0), 0x1234);
    EXPECT_EQ(read_u16(buf, 2), 0xABCD);

    write_u16(buf, 0, 0x5678);
    EXPECT_EQ(read_u16(buf, 0), 0x5678);
}

TEST(PixelUtilsTest, GetRaw8bit) {
    // 3x3 grid: row0=[10,20,30], row1=[40,50,60], row2=[70,80,90]
    uint8_t buf[] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    EXPECT_EQ(get_raw(buf, 0, 0, 3, 8), 10);
    EXPECT_EQ(get_raw(buf, 2, 0, 3, 8), 30);
    EXPECT_EQ(get_raw(buf, 1, 2, 3, 8), 80);
}

TEST(PixelUtilsTest, GetRaw16bit) {
    uint8_t buf[18];
    write_u16(buf, 0, 100);   // (0,0)
    write_u16(buf, 2, 200);   // (1,0)
    write_u16(buf, 4, 300);   // (2,0)
    write_u16(buf, 6, 400);   // (0,1)

    EXPECT_EQ(get_raw(buf, 0, 0, 3, 16), 100);
    EXPECT_EQ(get_raw(buf, 1, 0, 3, 16), 200);
    EXPECT_EQ(get_raw(buf, 0, 1, 3, 16), 400);
}

TEST(PixelUtilsTest, GetClampedBoundaries) {
    uint8_t buf[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(get_clamped(buf, -1, 0, 3, 3, 8), 1);
    EXPECT_EQ(get_clamped(buf, 5, 0, 3, 3, 8), 3);
    EXPECT_EQ(get_clamped(buf, 0, -1, 3, 3, 8), 1);
    EXPECT_EQ(get_clamped(buf, 0, 5, 3, 3, 8), 7);
}

TEST(PixelUtilsTest, SetRgb8bit) {
    uint8_t rgb[12] = {0};
    set_rgb(rgb, 0, 0, 2, 10, 20, 30, 8, false);
    EXPECT_EQ(rgb[0], 10); EXPECT_EQ(rgb[1], 20); EXPECT_EQ(rgb[2], 30);

    set_rgb(rgb, 1, 0, 2, 300, -5, 0, 8, true);
    EXPECT_EQ(rgb[3], 255); EXPECT_EQ(rgb[4], 0); EXPECT_EQ(rgb[5], 0);
}

TEST(PixelUtilsTest, SetRgb16bit) {
    uint8_t rgb[12] = {0};
    set_rgb_raw(rgb, 0, 0, 2, 100, 200, 300, 16);
    auto* rgb16 = reinterpret_cast<uint16_t*>(rgb);
    EXPECT_EQ(rgb16[0], 100);
    EXPECT_EQ(rgb16[1], 200);
    EXPECT_EQ(rgb16[2], 300);
}

TEST(PixelUtilsTest, GetClamped8) {
    uint8_t buf[] = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(get_clamped_8(buf, 0, 0, 3, 2), 1);
    EXPECT_EQ(get_clamped_8(buf, 5, 0, 3, 2), 3);   // x out of range -> clamped
    EXPECT_EQ(get_clamped_8(buf, 0, 5, 3, 2), 4);    // y out of range -> clamped
}

TEST(PixelUtilsTest, GetRaw8Fast) {
    uint8_t buf[] = {5, 10, 15, 20};
    EXPECT_EQ(get_raw_8(buf, 0, 0, 2), 5);
    EXPECT_EQ(get_raw_8(buf, 1, 0, 2), 10);
    EXPECT_EQ(get_raw_8(buf, 0, 1, 2), 15);
}

TEST(PixelUtilsTest, GetPackedRaw10BitRoundtrip) {
    // 10-bit packed: 4 pixels -> 5 bytes per group
    // Encoding formula (from pixel_utils.hpp):
    //   sub=0: (b0 << 2) | (b4 & 0x3)  → b0 = val>>2,  b4 bits 1:0 = val & 3
    //   sub=1: (b1 << 2) | ((b4>>2)&3) → b1 = val>>2,  b4 bits 3:2 = val & 3
    //   sub=2: (b2 << 2) | ((b4>>4)&3) → b2 = val>>2,  b4 bits 5:4 = val & 3
    //   sub=3: (b3 << 2) | ((b4>>6)&3) → b3 = val>>2,  b4 bits 7:6 = val & 3
    uint8_t buf[5] = {0};
    // p0=0x2FF (767) → b0=767>>2=191=0xBF, b4 bits 1:0 = 767&3 = 3
    // p1=0x100 (256) → b1=256>>2=64=0x40,  b4 bits 3:2 = 256&3 = 0
    // p2=0x001 (1)   → b2=1>>2=0=0x00,    b4 bits 5:4 = 1&3 = 1
    // p3=0x3FF (1023)→ b3=1023>>2=255=0xFF, b4 bits 7:6 = 1023&3 = 3
    buf[0] = 0xBF; buf[1] = 0x40; buf[2] = 0x00; buf[3] = 0xFF;
    buf[4] = 0xD3;  // 0b11_01_00_11

    int p0 = get_packed_raw(buf, 0, 0, 100, 10, 5);
    int p1 = get_packed_raw(buf, 1, 0, 100, 10, 5);
    int p2 = get_packed_raw(buf, 2, 0, 100, 10, 5);
    int p3 = get_packed_raw(buf, 3, 0, 100, 10, 5);

    EXPECT_EQ(p0, 0x2FF);
    EXPECT_EQ(p1, 0x100);
    EXPECT_EQ(p2, 0x001);
    EXPECT_EQ(p3, 0x3FF);
}

TEST(PixelUtilsTest, GetPacked10BitBoundaryCheck) {
    uint8_t buf[2] = {0xFF, 0xFF};
    int val = get_packed_raw(buf, 2, 0, 1, 10, 0); // data_byte_size=0 but height=0 -> no check
    EXPECT_EQ(val, 0);
}

TEST(PixelUtilsTest, AlignRawValue) {
    uint16_t raw = 0x03FF;
    uint16_t aligned = align_raw_value(raw, 10);
    EXPECT_EQ(aligned, 0x03FF);
}

TEST(PixelUtilsTest, SetRgbClamped) {
    uint8_t rgb[9] = {0};
    set_rgb_clamped(rgb, 0, 0, 3, -10, 300, 128, 8);
    EXPECT_EQ(rgb[0], 0);
    EXPECT_EQ(rgb[1], 255);
    EXPECT_EQ(rgb[2], 128);
}
