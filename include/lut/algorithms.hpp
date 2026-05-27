#ifndef LUT_ALGORITHMS_HPP
#define LUT_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace lut {

// Metadata helpers
std::string algorithm_name(LUTAlgorithm algo);
int algorithm_window_size(LUTAlgorithm algo);

// Input validation
LUTError validate_lut_inputs(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth);

// Main dispatch: input RGB (channels=3) -> output RGB (channels=3)
// lut_data: for CUSTOM_3D, pass a LUT3D*; for CUBE_FILE, pass a const char* filepath;
//            for built-in styles, pass nullptr (ignored)
// lut_size: number of samples per dimension in the LUT (default 33)
LUTError process_lut(const uint8_t* input, uint8_t* output,
                      int width, int height, int channels,
                      LUTAlgorithm algorithm, int bit_depth = 8,
                      const void* lut_data = nullptr,
                      int lut_size = 33);

// Individual algorithm functions
LUTError process_cube_file(const uint8_t* input, uint8_t* output,
                            int width, int height, int channels,
                            int bit_depth, const void* lut_data, int lut_size);

LUTError process_custom_3d(const uint8_t* input, uint8_t* output,
                            int width, int height, int channels,
                            int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_sepia(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_cool(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_warm(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_high_contrast(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_low_contrast(const uint8_t* input, uint8_t* output,
                                     int width, int height, int channels,
                                     int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_invert(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_vintage_fade(const uint8_t* input, uint8_t* output,
                                     int width, int height, int channels,
                                     int bit_depth, const void* lut_data, int lut_size);

// --- LUT construction & I/O ---

// Build a 3D LUT from a .cube file path. Returns the parsed LUT.
// Returns empty LUT on failure.
LUT3D load_cube_file(const char* filepath);

// Generate built-in style LUTs
LUT3D build_sepia_lut(int size = 33);
LUT3D build_cool_lut(int size = 33);
LUT3D build_warm_lut(int size = 33);
LUT3D build_high_contrast_lut(int size = 33);
LUT3D build_low_contrast_lut(int size = 33);
LUT3D build_invert_lut(int size = 33);
LUT3D build_vintage_fade_lut(int size = 33);

// Identity LUT (no-op)
LUT3D build_identity_lut(int size = 33);

// Apply a 3D LUT to an RGB image (trilinear interpolation)
LUTError apply_lut(const LUT3D& lut, const uint8_t* input, uint8_t* output,
                   int width, int height, int bit_depth);

} // namespace lut

#endif // LUT_ALGORITHMS_HPP
