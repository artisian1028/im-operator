#include "jpeg_codec/algorithms.hpp"
#include <string>
#include <array>

namespace jpeg_codec {

// ============================================================
//  Input validation
// ============================================================

JpegError validate_jpeg_inputs(const uint8_t* input, int width, int height,
                                int channels, int bit_depth) {
    if (!input) return JpegError::NullInput;
    if (!is_valid_dimensions(width, height)) return JpegError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return JpegError::InvalidBitDepth;
    if (channels != 3) return JpegError::InvalidChannels;
    if (width < 8 || height < 8) return JpegError::ImageTooSmall;
    return JpegError::Ok;
}

JpegError validate_jpeg_encode_inputs(const uint8_t* input, uint8_t* output,
                                       int width, int height, int channels,
                                       int bit_depth, int quality) {
    JpegError err = validate_jpeg_inputs(input, width, height, channels, bit_depth);
    if (err != JpegError::Ok) return err;
    if (!output) return JpegError::NullOutput;
    if (quality < 1 || quality > 100) return JpegError::InvalidQuality;
    return JpegError::Ok;
}

// ============================================================
//  Utility
// ============================================================

size_t get_max_jpeg_size(int width, int height, int channels) {
    (void)channels;
    // JPEG can be larger than raw for high-entropy content.
    // Allow up to 3x raw size for safety.
    size_t raw_size = static_cast<size_t>(width) * height * 3;
    return raw_size * 3 + 65536;
}

// ============================================================
//  Metadata
// ============================================================

std::string algorithm_name(JpegAlgorithm algo) {
    switch (algo) {
        case JpegAlgorithm::ENCODE_BASELINE: return "Baseline DCT JPEG Encoder (CPU)";
        case JpegAlgorithm::DECODE_BASELINE: return "Baseline DCT JPEG Decoder (CPU)";
        case JpegAlgorithm::ENCODE_CUDA:     return "GPU-Accelerated JPEG Encoder (CUDA)";
        case JpegAlgorithm::DECODE_CUDA:     return "GPU-Accelerated JPEG Decoder (CUDA)";
        default:                             return "Unknown";
    }
}

int algorithm_window_size(JpegAlgorithm algo) {
    switch (algo) {
        case JpegAlgorithm::ENCODE_BASELINE: return 8;
        case JpegAlgorithm::DECODE_BASELINE: return 8;
        case JpegAlgorithm::ENCODE_CUDA:     return 8;
        case JpegAlgorithm::DECODE_CUDA:     return 8;
        default:                             return 8;
    }
}

// ============================================================
//  Registry: Encode algorithms
// ============================================================

using JpegEncodeFunc = JpegError(*)(const uint8_t*, uint8_t*, size_t*, int, int, int, int, const JpegParams&);

struct EncodeEntry {
    JpegAlgorithm algorithm;
    JpegEncodeFunc func;
};

static const std::array<EncodeEntry, 2> kJpegEncodeRegistry = {{
    {JpegAlgorithm::ENCODE_BASELINE, process_encode_baseline},
    {JpegAlgorithm::ENCODE_CUDA,     process_encode_cuda}
}};

static_assert(kJpegEncodeRegistry.size() == 2,
              "kJpegEncodeRegistry size must match encode algorithm count");

// ============================================================
//  Registry: Decode algorithms
// ============================================================

using JpegDecodeFunc = JpegError(*)(const uint8_t*, size_t, uint8_t*, int*, int*, int*);

struct DecodeEntry {
    JpegAlgorithm algorithm;
    JpegDecodeFunc func;
};

static const std::array<DecodeEntry, 2> kJpegDecodeRegistry = {{
    {JpegAlgorithm::DECODE_BASELINE, process_decode_baseline},
    {JpegAlgorithm::DECODE_CUDA,     process_decode_cuda}
}};

static_assert(kJpegDecodeRegistry.size() == 2,
              "kJpegDecodeRegistry size must match decode algorithm count");

// ============================================================
//  Dispatch helpers
// ============================================================

static JpegEncodeFunc find_encode_func(JpegAlgorithm algorithm) {
    for (const auto& entry : kJpegEncodeRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

static JpegDecodeFunc find_decode_func(JpegAlgorithm algorithm) {
    for (const auto& entry : kJpegDecodeRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// ============================================================
//  Main dispatch: Encode
// ============================================================

JpegError process_jpeg_encode(const uint8_t* input, uint8_t* output,
                               size_t* output_size,
                               int width, int height, int channels,
                               JpegAlgorithm algorithm, int bit_depth,
                               const JpegParams& params) {
    JpegError err = validate_jpeg_encode_inputs(input, output, width, height,
                                                 channels, bit_depth, params.quality);
    if (err != JpegError::Ok) return err;

    JpegEncodeFunc func = find_encode_func(algorithm);
    if (!func) return JpegError::InternalError;

    return func(input, output, output_size, width, height, channels, bit_depth, params);
}

// ============================================================
//  Main dispatch: Decode
// ============================================================

JpegError process_jpeg_decode(const uint8_t* input, size_t input_size,
                               uint8_t* output,
                               int* width, int* height, int* channels,
                               JpegAlgorithm algorithm) {
    if (!input) return JpegError::NullInput;
    if (!output) return JpegError::NullOutput;
    if (!width || !height || !channels) return JpegError::NullInput;
    if (input_size < 2) return JpegError::InvalidJpegData;

    JpegDecodeFunc func = find_decode_func(algorithm);
    if (!func) return JpegError::InternalError;

    return func(input, input_size, output, width, height, channels);
}

} // namespace jpeg_codec
