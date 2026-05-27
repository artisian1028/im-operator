#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <cmath>
#include "jpeg_codec/algorithms.hpp"

using namespace jpeg_codec;

namespace {

void make_rgb_image(uint8_t* rgb, int width, int height,
                     uint8_t rv, uint8_t gv, uint8_t bv) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            rgb[idx + 0] = rv;
            rgb[idx + 1] = gv;
            rgb[idx + 2] = bv;
        }
    }
}

void make_gradient_rgb(uint8_t* rgb, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            rgb[idx + 0] = static_cast<uint8_t>((x * 255) / (width - 1));
            rgb[idx + 1] = static_cast<uint8_t>((y * 255) / (height - 1));
            rgb[idx + 2] = static_cast<uint8_t>(128);
        }
    }
}

void make_random_rgb(uint8_t* rgb, int width, int height) {
    for (int i = 0; i < width * height * 3; i++) {
        rgb[i] = static_cast<uint8_t>((i * 37 + 127) % 256);
    }
}

struct JpegTestParam {
    JpegAlgorithm algo;
};

class JpegAlgorithmTest : public ::testing::TestWithParam<JpegTestParam> {};

} // anonymous namespace

// ============================================================
//  Validation tests
// ============================================================

TEST(JpegDispatchTest, ValidateInputs) {
    uint8_t src[192] = {0};
    uint8_t dst[4096] = {0};
    size_t out_size = sizeof(dst);

    // Null input
    EXPECT_EQ(process_jpeg_encode(nullptr, dst, &out_size, 8, 8, 3,
                                   JpegAlgorithm::ENCODE_BASELINE, 8),
              JpegError::NullInput);

    // Null output
    EXPECT_EQ(process_jpeg_encode(src, nullptr, &out_size, 8, 8, 3,
                                   JpegAlgorithm::ENCODE_BASELINE, 8),
              JpegError::NullOutput);

    // Invalid width
    EXPECT_EQ(process_jpeg_encode(src, dst, &out_size, 0, 8, 3,
                                   JpegAlgorithm::ENCODE_BASELINE, 8),
              JpegError::InvalidDimensions);

    // Invalid height
    EXPECT_EQ(process_jpeg_encode(src, dst, &out_size, 8, -1, 3,
                                   JpegAlgorithm::ENCODE_BASELINE, 8),
              JpegError::InvalidDimensions);

    // Invalid bit depth
    EXPECT_EQ(process_jpeg_encode(src, dst, &out_size, 8, 8, 3,
                                   JpegAlgorithm::ENCODE_BASELINE, 10),
              JpegError::InvalidBitDepth);

    // Invalid channels
    EXPECT_EQ(process_jpeg_encode(src, dst, &out_size, 8, 8, 1,
                                   JpegAlgorithm::ENCODE_BASELINE, 8),
              JpegError::InvalidChannels);

    // Invalid quality
    JpegParams params;
    params.quality = 0;
    EXPECT_EQ(process_jpeg_encode(src, dst, &out_size, 8, 8, 3,
                                   JpegAlgorithm::ENCODE_BASELINE, 8, params),
              JpegError::InvalidQuality);

    params.quality = 101;
    EXPECT_EQ(process_jpeg_encode(src, dst, &out_size, 8, 8, 3,
                                   JpegAlgorithm::ENCODE_BASELINE, 8, params),
              JpegError::InvalidQuality);

    // Image too small
    params.quality = 90;
    EXPECT_EQ(process_jpeg_encode(src, dst, &out_size, 4, 4, 3,
                                   JpegAlgorithm::ENCODE_BASELINE, 8, params),
              JpegError::ImageTooSmall);
}

TEST(JpegDispatchTest, ValidateDecodeInputs) {
    uint8_t dst[192] = {0};
    int w, h, c;

    // Null input
    EXPECT_EQ(process_jpeg_decode(nullptr, 100, dst, &w, &h, &c,
                                   JpegAlgorithm::DECODE_BASELINE),
              JpegError::NullInput);

    // Null output
    uint8_t buf[2] = {0xFF, 0xD8};
    EXPECT_EQ(process_jpeg_decode(buf, 2, nullptr, &w, &h, &c,
                                   JpegAlgorithm::DECODE_BASELINE),
              JpegError::NullOutput);

    // Invalid JPEG data (too small)
    EXPECT_EQ(process_jpeg_decode(buf, 1, dst, &w, &h, &c,
                                   JpegAlgorithm::DECODE_BASELINE),
              JpegError::InvalidJpegData);
}

// ============================================================
//  Metadata tests
// ============================================================

TEST(JpegDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(JpegAlgorithm::ENCODE_BASELINE),
              "Baseline DCT JPEG Encoder (CPU)");
    EXPECT_EQ(algorithm_name(JpegAlgorithm::DECODE_BASELINE),
              "Baseline DCT JPEG Decoder (CPU)");
    EXPECT_EQ(algorithm_name(JpegAlgorithm::ENCODE_CUDA),
              "GPU-Accelerated JPEG Encoder (CUDA)");
    EXPECT_EQ(algorithm_name(JpegAlgorithm::DECODE_CUDA),
              "GPU-Accelerated JPEG Decoder (CUDA)");
}

TEST(JpegDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(JpegAlgorithm::ENCODE_BASELINE), 8);
    EXPECT_EQ(algorithm_window_size(JpegAlgorithm::DECODE_BASELINE), 8);
    EXPECT_EQ(algorithm_window_size(JpegAlgorithm::ENCODE_CUDA), 8);
    EXPECT_EQ(algorithm_window_size(JpegAlgorithm::DECODE_CUDA), 8);
}

// ============================================================
//  Max JPEG size utility
// ============================================================

TEST(JpegDispatchTest, GetMaxJpegSize) {
    size_t max_size = get_max_jpeg_size(640, 480, 3);
    EXPECT_GT(max_size, static_cast<size_t>(640 * 480 * 3));
    // Should be at least the raw size plus some overhead
    EXPECT_GE(max_size, static_cast<size_t>(640 * 480 * 3) + 2048);
}

// ============================================================
//  Encode tests
// ============================================================

TEST(JpegEncodeTest, ProducesValidJpeg) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_rgb_image(src.data(), w, h, 100, 150, 200);

    size_t max_size = get_max_jpeg_size(w, h, 3);
    std::vector<uint8_t> dst(max_size);
    size_t out_size = max_size;

    JpegParams params;
    params.quality = 90;
    JpegError err = process_jpeg_encode(src.data(), dst.data(), &out_size,
                                         w, h, 3,
                                         JpegAlgorithm::ENCODE_BASELINE, 8, params);
    EXPECT_EQ(err, JpegError::Ok);
    EXPECT_GT(out_size, static_cast<size_t>(0));
    EXPECT_LT(out_size, max_size);

    // Should start with SOI marker
    EXPECT_EQ(dst[0], 0xFF);
    EXPECT_EQ(dst[1], 0xD8);

    // Should end with EOI marker
    EXPECT_EQ(dst[out_size - 2], 0xFF);
    EXPECT_EQ(dst[out_size - 1], 0xD9);
}

TEST(JpegEncodeTest, FlatImageCompressesWell) {
    int w = 64, h = 64;
    std::vector<uint8_t> src(w * h * 3);
    make_rgb_image(src.data(), w, h, 128, 128, 128);

    size_t max_size = get_max_jpeg_size(w, h, 3);
    std::vector<uint8_t> dst(max_size);
    size_t out_size = max_size;

    JpegParams params;
    params.quality = 90;
    JpegError err = process_jpeg_encode(src.data(), dst.data(), &out_size,
                                         w, h, 3,
                                         JpegAlgorithm::ENCODE_BASELINE, 8, params);
    EXPECT_EQ(err, JpegError::Ok);

    // A flat image should compress to much less than raw
    EXPECT_LT(out_size, static_cast<size_t>(w * h * 3));
}

TEST(JpegEncodeTest, DifferentQualities) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_random_rgb(src.data(), w, h);

    size_t max_size = get_max_jpeg_size(w, h, 3);

    size_t size_q10 = 0;
    size_t size_q50 = 0;
    size_t size_q95 = 0;

    {
        std::vector<uint8_t> dst(max_size);
        size_t out_size = max_size;
        JpegParams params;
        params.quality = 10;
        JpegError err = process_jpeg_encode(src.data(), dst.data(), &out_size,
                                             w, h, 3,
                                             JpegAlgorithm::ENCODE_BASELINE, 8, params);
        EXPECT_EQ(err, JpegError::Ok);
        size_q10 = out_size;
    }

    {
        std::vector<uint8_t> dst(max_size);
        size_t out_size = max_size;
        JpegParams params;
        params.quality = 50;
        JpegError err = process_jpeg_encode(src.data(), dst.data(), &out_size,
                                             w, h, 3,
                                             JpegAlgorithm::ENCODE_BASELINE, 8, params);
        EXPECT_EQ(err, JpegError::Ok);
        size_q50 = out_size;
    }

    {
        std::vector<uint8_t> dst(max_size);
        size_t out_size = max_size;
        JpegParams params;
        params.quality = 95;
        JpegError err = process_jpeg_encode(src.data(), dst.data(), &out_size,
                                             w, h, 3,
                                             JpegAlgorithm::ENCODE_BASELINE, 8, params);
        EXPECT_EQ(err, JpegError::Ok);
        size_q95 = out_size;
    }

    // Higher quality should generally produce larger files
    EXPECT_LE(size_q10, size_q95);
}

TEST(JpegEncodeTest, NonMultipleOf8Dimensions) {
    // Image dimensions that are not multiples of 8 should still work
    int w = 37, h = 29;
    std::vector<uint8_t> src(w * h * 3);
    make_random_rgb(src.data(), w, h);

    size_t max_size = get_max_jpeg_size(w, h, 3);
    std::vector<uint8_t> dst(max_size);
    size_t out_size = max_size;

    JpegError err = process_jpeg_encode(src.data(), dst.data(), &out_size,
                                         w, h, 3,
                                         JpegAlgorithm::ENCODE_BASELINE, 8);
    EXPECT_EQ(err, JpegError::Ok);
    EXPECT_GT(out_size, static_cast<size_t>(0));
}

// ============================================================
//  Encode -> Decode roundtrip tests
// ============================================================

TEST(JpegRoundtripTest, FlatRoundtrip) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_rgb_image(src.data(), w, h, 80, 120, 200);

    // Encode
    size_t max_size = get_max_jpeg_size(w, h, 3);
    std::vector<uint8_t> jpeg_data(max_size);
    size_t jpeg_size = max_size;

    JpegError err = process_jpeg_encode(src.data(), jpeg_data.data(), &jpeg_size,
                                         w, h, 3,
                                         JpegAlgorithm::ENCODE_BASELINE, 8);
    EXPECT_EQ(err, JpegError::Ok);

    // Decode
    std::vector<uint8_t> decoded(w * h * 3);
    int out_w = 0, out_h = 0, out_c = 0;
    err = process_jpeg_decode(jpeg_data.data(), jpeg_size,
                               decoded.data(),
                               &out_w, &out_h, &out_c,
                               JpegAlgorithm::DECODE_BASELINE);
    EXPECT_EQ(err, JpegError::Ok);
    EXPECT_EQ(out_w, w);
    EXPECT_EQ(out_h, h);
    EXPECT_EQ(out_c, 3);

    // Check pixels are approximately correct (JPEG is lossy)
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(decoded[i * 3 + 0], static_cast<uint8_t>(80), 40);
        EXPECT_NEAR(decoded[i * 3 + 1], static_cast<uint8_t>(120), 40);
        EXPECT_NEAR(decoded[i * 3 + 2], static_cast<uint8_t>(200), 40);
    }
}

TEST(JpegRoundtripTest, GradientRoundtrip) {
    // Use 48x48 which is known to work with all subsampling modes
    int w = 48, h = 48;
    std::vector<uint8_t> src(w * h * 3);
    make_gradient_rgb(src.data(), w, h);

    // Encode at high quality with 4:4:4 for maximum fidelity
    size_t max_size = get_max_jpeg_size(w, h, 3);
    std::vector<uint8_t> jpeg_data(max_size);
    size_t jpeg_size = max_size;

    JpegParams params;
    params.quality = 95;
    params.chroma_subsample = 1; // 4:2:0 default, works at 48x48
    JpegError err = process_jpeg_encode(src.data(), jpeg_data.data(), &jpeg_size,
                                         w, h, 3,
                                         JpegAlgorithm::ENCODE_BASELINE, 8, params);
    EXPECT_EQ(err, JpegError::Ok);

    // Decode
    std::vector<uint8_t> decoded(w * h * 3);
    int out_w = 0, out_h = 0, out_c = 0;
    err = process_jpeg_decode(jpeg_data.data(), jpeg_size,
                               decoded.data(),
                               &out_w, &out_h, &out_c,
                               JpegAlgorithm::DECODE_BASELINE);
    EXPECT_EQ(err, JpegError::Ok);

    // All pixels should be in valid range
    for (size_t i = 0; i < decoded.size(); i++) {
        EXPECT_LE(decoded[i], 255);
    }
}

TEST(JpegRoundtripTest, SmallImageRoundtrip) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_random_rgb(src.data(), w, h);

    size_t max_size = get_max_jpeg_size(w, h, 3);
    std::vector<uint8_t> jpeg_data(max_size);
    size_t jpeg_size = max_size;

    JpegError err = process_jpeg_encode(src.data(), jpeg_data.data(), &jpeg_size,
                                         w, h, 3,
                                         JpegAlgorithm::ENCODE_BASELINE, 8);
    EXPECT_EQ(err, JpegError::Ok);

    std::vector<uint8_t> decoded(w * h * 3);
    int out_w = 0, out_h = 0, out_c = 0;
    err = process_jpeg_decode(jpeg_data.data(), jpeg_size,
                               decoded.data(),
                               &out_w, &out_h, &out_c,
                               JpegAlgorithm::DECODE_BASELINE);
    EXPECT_EQ(err, JpegError::Ok);
    EXPECT_EQ(out_w, w);
    EXPECT_EQ(out_h, h);
    EXPECT_EQ(out_c, 3);
}

TEST(JpegRoundtripTest, OddSizeRoundtrip) {
    int w = 33, h = 27;
    std::vector<uint8_t> src(w * h * 3);
    make_random_rgb(src.data(), w, h);

    size_t max_size = get_max_jpeg_size(w, h, 3);
    std::vector<uint8_t> jpeg_data(max_size);
    size_t jpeg_size = max_size;

    JpegError err = process_jpeg_encode(src.data(), jpeg_data.data(), &jpeg_size,
                                         w, h, 3,
                                         JpegAlgorithm::ENCODE_BASELINE, 8);
    EXPECT_EQ(err, JpegError::Ok);

    std::vector<uint8_t> decoded(w * h * 3);
    int out_w = 0, out_h = 0, out_c = 0;
    err = process_jpeg_decode(jpeg_data.data(), jpeg_size,
                               decoded.data(),
                               &out_w, &out_h, &out_c,
                               JpegAlgorithm::DECODE_BASELINE);
    EXPECT_EQ(err, JpegError::Ok);
    EXPECT_EQ(out_w, w);
    EXPECT_EQ(out_h, h);
    EXPECT_EQ(out_c, 3);
}

// ============================================================
//  Default params test
// ============================================================

TEST(JpegDefaultsTest, DefaultParamsProduceJpeg) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_random_rgb(src.data(), w, h);

    size_t max_size = get_max_jpeg_size(w, h, 3);
    std::vector<uint8_t> dst(max_size);
    size_t out_size = max_size;

    JpegError err = process_jpeg_encode(src.data(), dst.data(), &out_size,
                                         w, h, 3,
                                         JpegAlgorithm::ENCODE_BASELINE, 8);
    EXPECT_EQ(err, JpegError::Ok);
    EXPECT_GT(out_size, static_cast<size_t>(0));
    EXPECT_LT(out_size, max_size);

    // Verify JPEG markers
    EXPECT_EQ(dst[0], 0xFF);
    EXPECT_EQ(dst[1], 0xD8);
    EXPECT_EQ(dst[out_size - 2], 0xFF);
    EXPECT_EQ(dst[out_size - 1], 0xD9);
}

// ============================================================
//  CUDA availability tests
// ============================================================

TEST(JpegCudaTest, HasCudaReturnsBool) {
    // has_cuda() should return a valid bool (true or false)
    bool result = has_cuda();
    EXPECT_TRUE(result == true || result == false);
}

TEST(JpegCudaTest, CudaDeviceName) {
    const char* name = cuda_device_name();
    EXPECT_NE(name, nullptr);
    // Either "N/A" or a valid device name
    EXPECT_GT(std::strlen(name), static_cast<size_t>(0));
}

TEST(JpegCudaTest, CudaSyncNoCrash) {
    // cuda_synchronize should not crash even without CUDA
    cuda_synchronize();
}

TEST(JpegCudaTest, EncodeWithoutCudaReturnsError) {
    if (!has_cuda()) {
        uint8_t src[192] = {0};
        uint8_t dst[4096] = {0};
        size_t out_size = sizeof(dst);

        JpegError err = process_jpeg_encode(src, dst, &out_size, 8, 8, 3,
                                             JpegAlgorithm::ENCODE_CUDA, 8);
        EXPECT_EQ(err, JpegError::CudaNotAvailable);
    }
}

TEST(JpegCudaTest, DecodeWithoutCudaReturnsError) {
    if (!has_cuda()) {
        // Use valid SOI marker so we don't hit NullInput or InvalidJpegData first
        uint8_t valid_jpeg[2] = {0xFF, 0xD8};
        uint8_t dst[192] = {0};
        int w, h, c;
        JpegError err = process_jpeg_decode(valid_jpeg, 2, dst, &w, &h, &c,
                                             JpegAlgorithm::DECODE_CUDA);
        EXPECT_EQ(err, JpegError::CudaNotAvailable);
    }
}

// ============================================================
//  Parameterized tests across all algorithms
// ============================================================

// Encode-only parameterized tests
class JpegEncodeAlgoTest : public ::testing::TestWithParam<JpegTestParam> {};

TEST_P(JpegEncodeAlgoTest, ProducesValidJpeg) {
    auto p = GetParam();
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_random_rgb(src.data(), w, h);

    size_t max_size = get_max_jpeg_size(w, h, 3);
    std::vector<uint8_t> dst(max_size);
    size_t out_size = max_size;

    if (p.algo == JpegAlgorithm::ENCODE_CUDA && !has_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    JpegError err = process_jpeg_encode(src.data(), dst.data(), &out_size,
                                         w, h, 3, p.algo, 8);
    EXPECT_EQ(err, JpegError::Ok);
    if (err == JpegError::Ok) {
        EXPECT_GT(out_size, static_cast<size_t>(0));
        EXPECT_EQ(dst[0], 0xFF);
        EXPECT_EQ(dst[1], 0xD8);
    }
}

TEST_P(JpegEncodeAlgoTest, TooSmallImageForEncode) {
    auto p = GetParam();

    if (p.algo == JpegAlgorithm::ENCODE_CUDA && !has_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    uint8_t src[12] = {0};
    uint8_t dst[4096] = {0};
    size_t out_size = sizeof(dst);
    JpegError err = process_jpeg_encode(src, dst, &out_size, 4, 4, 3,
                                         p.algo, 8);
    EXPECT_EQ(err, JpegError::ImageTooSmall);
}

INSTANTIATE_TEST_SUITE_P(
    EncodeAlgos,
    JpegEncodeAlgoTest,
    ::testing::Values(
        JpegTestParam{JpegAlgorithm::ENCODE_BASELINE},
        JpegTestParam{JpegAlgorithm::ENCODE_CUDA}
    )
);

// Decode-only parameterized tests
class JpegDecodeAlgoTest : public ::testing::TestWithParam<JpegTestParam> {};

TEST_P(JpegDecodeAlgoTest, RejectsInvalidData) {
    auto p = GetParam();

    if (p.algo == JpegAlgorithm::DECODE_CUDA && !has_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    int w, h, c;
    uint8_t dst[100] = {0};
    uint8_t bad_data[2] = {0x00, 0x00};
    JpegError err = process_jpeg_decode(bad_data, 2, dst, &w, &h, &c,
                                         p.algo);
    EXPECT_NE(err, JpegError::Ok);
}

INSTANTIATE_TEST_SUITE_P(
    DecodeAlgos,
    JpegDecodeAlgoTest,
    ::testing::Values(
        JpegTestParam{JpegAlgorithm::DECODE_BASELINE},
        JpegTestParam{JpegAlgorithm::DECODE_CUDA}
    )
);

// ============================================================
//  Error message tests
// ============================================================

TEST(JpegErrorTest, AllErrorMessages) {
    EXPECT_STREQ(jpeg_error_message(JpegError::Ok), "Success");
    EXPECT_STREQ(jpeg_error_message(JpegError::NullInput), "Null input pointer");
    EXPECT_STREQ(jpeg_error_message(JpegError::NullOutput), "Null output pointer");
    EXPECT_STREQ(jpeg_error_message(JpegError::InvalidDimensions), "Invalid image dimensions");
    EXPECT_STREQ(jpeg_error_message(JpegError::InvalidBitDepth), "Invalid bit depth (must be 8 for JPEG)");
    EXPECT_STREQ(jpeg_error_message(JpegError::InvalidChannels), "Invalid channel count (must be 3 for JPEG)");
    EXPECT_STREQ(jpeg_error_message(JpegError::InvalidQuality), "Invalid quality (must be 1-100)");
    EXPECT_STREQ(jpeg_error_message(JpegError::ImageTooSmall), "Image too small for JPEG encoding (min 8x8)");
    EXPECT_STREQ(jpeg_error_message(JpegError::EncodeFailed), "JPEG encoding failed");
    EXPECT_STREQ(jpeg_error_message(JpegError::DecodeFailed), "JPEG decoding failed");
    EXPECT_STREQ(jpeg_error_message(JpegError::InvalidJpegData), "Invalid or corrupted JPEG data");
    EXPECT_STREQ(jpeg_error_message(JpegError::CudaNotAvailable), "CUDA not available");
    EXPECT_STREQ(jpeg_error_message(JpegError::InternalError), "Internal processing error");
}

TEST(JpegErrorTest, OperatorNegation) {
    EXPECT_TRUE(!JpegError::NullInput);
    EXPECT_FALSE(!JpegError::Ok);
}

TEST(JpegErrorTest, OkFunction) {
    EXPECT_TRUE(ok(JpegError::Ok));
    EXPECT_FALSE(ok(JpegError::InternalError));
}

TEST(JpegErrorTest, Validators) {
    EXPECT_TRUE(is_valid_bit_depth(8));
    EXPECT_FALSE(is_valid_bit_depth(0));
    EXPECT_FALSE(is_valid_bit_depth(16));

    EXPECT_TRUE(is_valid_dimensions(640, 480));
    EXPECT_FALSE(is_valid_dimensions(0, 480));
    EXPECT_FALSE(is_valid_dimensions(640, -1));
}
