#ifndef JPEG_CODEC_ALGORITHMS_HPP
#define JPEG_CODEC_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <cstddef>
#include <string>

namespace jpeg_codec {

// --- Metadata helpers ---

std::string algorithm_name(JpegAlgorithm algo);
int algorithm_window_size(JpegAlgorithm algo);

// --- Input validation ---

JpegError validate_jpeg_inputs(const uint8_t* input, int width, int height,
                                int channels, int bit_depth);

JpegError validate_jpeg_encode_inputs(const uint8_t* input, uint8_t* output,
                                       int width, int height, int channels,
                                       int bit_depth, int quality);

// --- Main dispatch ---

/// Encode RGB image to JPEG.
/// @param input  RGB pixel data (interleaved, width*height*channels bytes)
/// @param output Output buffer for JPEG data (caller-allocated, sized by get_max_jpeg_size)
/// @param output_size [in/out] On input: capacity of output buffer.
///                    On output: actual JPEG size written.
/// @param width   Image width in pixels
/// @param height  Image height in pixels
/// @param channels Must be 3 (RGB)
/// @param algorithm Encoding algorithm
/// @param bit_depth Must be 8
/// @param params  JPEG encoding parameters
JpegError process_jpeg_encode(const uint8_t* input, uint8_t* output,
                               size_t* output_size,
                               int width, int height, int channels,
                               JpegAlgorithm algorithm, int bit_depth = 8,
                               const JpegParams& params = {});

/// Decode JPEG to RGB image.
/// @param input  JPEG compressed data
/// @param input_size Size of JPEG data in bytes
/// @param output Output RGB buffer (caller-allocated, width*height*channels bytes)
/// @param width  [out] Decoded image width
/// @param height [out] Decoded image height
/// @param channels [out] Decoded channel count
/// @param algorithm Decoding algorithm
JpegError process_jpeg_decode(const uint8_t* input, size_t input_size,
                               uint8_t* output,
                               int* width, int* height, int* channels,
                               JpegAlgorithm algorithm);

/// Get maximum possible JPEG size for given image dimensions (worst case).
size_t get_max_jpeg_size(int width, int height, int channels);

// --- Individual algorithm functions ---

// CPU encoder
JpegError process_encode_baseline(const uint8_t* input, uint8_t* output,
                                   size_t* output_size,
                                   int width, int height, int channels,
                                   int bit_depth, const JpegParams& params);

// CPU decoder
JpegError process_decode_baseline(const uint8_t* input, size_t input_size,
                                   uint8_t* output,
                                   int* width, int* height, int* channels);

// CUDA encoder
JpegError process_encode_cuda(const uint8_t* input, uint8_t* output,
                               size_t* output_size,
                               int width, int height, int channels,
                               int bit_depth, const JpegParams& params);

// CUDA decoder
JpegError process_decode_cuda(const uint8_t* input, size_t input_size,
                               uint8_t* output,
                               int* width, int* height, int* channels);

// --- CUDA runtime queries ---

bool has_cuda();
const char* cuda_device_name();
void cuda_synchronize();

} // namespace jpeg_codec

#endif // JPEG_CODEC_ALGORITHMS_HPP
