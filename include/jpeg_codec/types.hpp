#ifndef JPEG_CODEC_TYPES_HPP
#define JPEG_CODEC_TYPES_HPP

#include <cstdint>

namespace jpeg_codec {

enum class JpegAlgorithm {
    ENCODE_BASELINE,  // CPU: Baseline DCT JPEG encoder (RGB -> JPEG)
    DECODE_BASELINE,  // CPU: Baseline DCT JPEG decoder (JPEG -> RGB)
    ENCODE_CUDA,      // CUDA: GPU-accelerated JPEG encoder
    DECODE_CUDA       // CUDA: GPU-accelerated JPEG decoder
};

enum class JpegError {
    Ok = 0,
    NullInput,
    NullOutput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    InvalidQuality,
    ImageTooSmall,
    EncodeFailed,
    DecodeFailed,
    InvalidJpegData,
    CudaNotAvailable,
    InternalError
};

inline const char* jpeg_error_message(JpegError err) {
    switch (err) {
        case JpegError::Ok:               return "Success";
        case JpegError::NullInput:        return "Null input pointer";
        case JpegError::NullOutput:       return "Null output pointer";
        case JpegError::InvalidDimensions: return "Invalid image dimensions";
        case JpegError::InvalidBitDepth:   return "Invalid bit depth (must be 8 for JPEG)";
        case JpegError::InvalidChannels:   return "Invalid channel count (must be 3 for JPEG)";
        case JpegError::InvalidQuality:    return "Invalid quality (must be 1-100)";
        case JpegError::ImageTooSmall:     return "Image too small for JPEG encoding (min 8x8)";
        case JpegError::EncodeFailed:      return "JPEG encoding failed";
        case JpegError::DecodeFailed:      return "JPEG decoding failed";
        case JpegError::InvalidJpegData:   return "Invalid or corrupted JPEG data";
        case JpegError::CudaNotAvailable:  return "CUDA not available";
        case JpegError::InternalError:     return "Internal processing error";
        default:                           return "Unknown error";
    }
}

inline bool operator!(JpegError err) {
    return err != JpegError::Ok;
}

inline bool ok(JpegError err) {
    return err == JpegError::Ok;
}

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth == 8;
}

inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

struct JpegParams {
    int quality = 90;         // 1-100, higher = better quality + larger file
    bool progressive = false; // Progressive JPEG (CPU only)
    bool optimize = true;     // Optimize Huffman tables
    int chroma_subsample = 1; // 0=4:4:4, 1=4:2:0, 2=4:2:2
};

} // namespace jpeg_codec

#endif // JPEG_CODEC_TYPES_HPP
