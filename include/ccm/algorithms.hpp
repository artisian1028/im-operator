#ifndef CCM_ALGORITHMS_HPP
#define CCM_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace ccm {

// Metadata helpers
std::string algorithm_name(CCMAlgorithm algo);
int algorithm_window_size(CCMAlgorithm algo);

// Input validation shared across algorithms
CCMError validate_ccm_inputs(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth);

// Main dispatch: input RGB (channels=3) -> output RGB (channels=3)
// matrix: user-supplied matrix for MANUAL algorithm, or pre-computed matrix
CCMError process_ccm(const uint8_t* input, uint8_t* output,
                      int width, int height, int channels,
                      CCMAlgorithm algorithm, int bit_depth = 8,
                      const void* matrix = nullptr);

// Individual algorithm functions
CCMError process_linear_3x3(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const void* matrix);

CCMError process_linear_4x3(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const void* matrix);

CCMError process_polynomial_3x9(const uint8_t* input, uint8_t* output,
                                 int width, int height, int channels,
                                 int bit_depth, const void* matrix);

CCMError process_manual_ccm(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const void* matrix);

// Predefined standard matrices

// sRGB to XYZ (D65) conversion matrix
CCMatrix3x3 srgb_to_xyz_d65();

// XYZ (D65) to sRGB conversion matrix
CCMatrix3x3 xyz_to_srgb_d65();

// Typical color space conversion examples
CCMatrix3x3 srgb_to_bt709();
CCMatrix3x3 bt709_to_srgb();

// Identity matrix
CCMatrix3x3 identity_3x3();

// Compute a saturation-boosting matrix (sat: 0 = grayscale, 1 = original, >1 = boosted)
CCMatrix3x3 saturation_matrix(float sat);

// CUDA support
bool has_cuda();
CCMError process_ccm_cuda(const uint8_t* input, uint8_t* output,
                           int width, int height, int channels,
                           int bit_depth, const float* matrix);

} // namespace ccm

#endif // CCM_ALGORITHMS_HPP
