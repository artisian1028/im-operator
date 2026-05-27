#include <gtest/gtest.h>
#include "imop/types.hpp"

using namespace imop;

TEST(TypesTest, PatternOffsetsFromPattern) {
    auto rggb = PatternOffsets::from_pattern(BayerPattern::RGGB);
    EXPECT_EQ(rggb.r_row, 0); EXPECT_EQ(rggb.r_col, 0);
    EXPECT_EQ(rggb.b_row, 1); EXPECT_EQ(rggb.b_col, 1);

    auto bggr = PatternOffsets::from_pattern(BayerPattern::BGGR);
    EXPECT_EQ(bggr.r_row, 1); EXPECT_EQ(bggr.r_col, 1);
    EXPECT_EQ(bggr.b_row, 0); EXPECT_EQ(bggr.b_col, 0);

    auto grbg = PatternOffsets::from_pattern(BayerPattern::GRBG);
    EXPECT_EQ(grbg.r_row, 0); EXPECT_EQ(grbg.r_col, 1);
    EXPECT_EQ(grbg.b_row, 1); EXPECT_EQ(grbg.b_col, 0);

    auto gbrg = PatternOffsets::from_pattern(BayerPattern::GBRG);
    EXPECT_EQ(gbrg.r_row, 1); EXPECT_EQ(gbrg.r_col, 0);
    EXPECT_EQ(gbrg.b_row, 0); EXPECT_EQ(gbrg.b_col, 1);
}

TEST(TypesTest, ValidBayerPattern) {
    EXPECT_TRUE(is_valid_bayer_pattern(BayerPattern::RGGB));
    EXPECT_TRUE(is_valid_bayer_pattern(BayerPattern::BGGR));
    EXPECT_TRUE(is_valid_bayer_pattern(BayerPattern::GRBG));
    EXPECT_TRUE(is_valid_bayer_pattern(BayerPattern::GBRG));
}

TEST(TypesTest, ValidBitDepth) {
    EXPECT_TRUE(is_valid_bit_depth(8));
    EXPECT_TRUE(is_valid_bit_depth(10));
    EXPECT_TRUE(is_valid_bit_depth(12));
    EXPECT_TRUE(is_valid_bit_depth(14));
    EXPECT_TRUE(is_valid_bit_depth(16));
    EXPECT_FALSE(is_valid_bit_depth(0));
    EXPECT_FALSE(is_valid_bit_depth(17));
    EXPECT_FALSE(is_valid_bit_depth(-1));
}

TEST(TypesTest, ValidDimensions) {
    EXPECT_TRUE(is_valid_dimensions(1920, 1080));
    EXPECT_FALSE(is_valid_dimensions(0, 1080));
    EXPECT_FALSE(is_valid_dimensions(1920, 0));
    EXPECT_FALSE(is_valid_dimensions(-1, 1080));
}

TEST(TypesTest, ErrorMessageCoverage) {
    EXPECT_STREQ(demosaic_error_message(DemosaicError::Ok), "Success");
    EXPECT_STREQ(demosaic_error_message(DemosaicError::NullInput), "Null input/output pointer");
    EXPECT_STREQ(demosaic_error_message(DemosaicError::InvalidDimensions), "Invalid image dimensions");
    EXPECT_STREQ(demosaic_error_message(DemosaicError::InvalidBitDepth), "Invalid bit depth (must be 1-16)");
    EXPECT_STREQ(demosaic_error_message(DemosaicError::InvalidPattern), "Invalid Bayer pattern");
    EXPECT_STREQ(demosaic_error_message(DemosaicError::ImageTooSmall), "Image too small for algorithm");
    EXPECT_STREQ(demosaic_error_message(DemosaicError::InternalError), "Internal processing error");
}

TEST(TypesTest, ErrorOperators) {
    EXPECT_TRUE(ok(DemosaicError::Ok));
    EXPECT_FALSE(ok(DemosaicError::NullInput));
    EXPECT_FALSE(!DemosaicError::Ok);
    EXPECT_TRUE(!DemosaicError::NullInput);
}

TEST(TypesTest, ImageBufferDefaults) {
    ImageBuffer buf;
    EXPECT_EQ(buf.width, 0);
    EXPECT_EQ(buf.height, 0);
    EXPECT_EQ(buf.channels, 1);
    EXPECT_EQ(buf.bit_depth, 8);
    EXPECT_FALSE(buf.is_packed);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0);
}

TEST(TypesTest, ImageBufferNonEmpty) {
    ImageBuffer buf;
    buf.width = 10;
    buf.height = 10;
    buf.channels = 1;
    buf.bit_depth = 8;
    buf.data.resize(100);
    EXPECT_EQ(buf.size(), 100);
    EXPECT_FALSE(buf.empty());
}
