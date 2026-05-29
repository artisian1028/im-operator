#ifndef DEFECT_CORRECT_ALGORITHMS_HPP
#define DEFECT_CORRECT_ALGORITHMS_HPP

#include "types.hpp"
#include "imop/types.hpp" // for BayerPattern
#include <cstdint>
#include <string>

namespace defect_correct {

using imop::BayerPattern;

std::string algorithm_name(DefectCorrectAlgorithm algo);
int algorithm_window_size(DefectCorrectAlgorithm algo);

DefectCorrectError validate_defect_correct_inputs(const uint8_t* data,
                                                   int width, int height,
                                                   int channels, int bit_depth);

DefectCorrectError process_defect_correct(uint8_t* data,
                                           int width, int height,
                                           BayerPattern pattern,
                                           DefectCorrectAlgorithm algorithm,
                                           int bit_depth,
                                           const DefectCorrectParams& params);

DefectCorrectError process_adaptive(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     int bit_depth,
                                     const DefectCorrectParams& params);

DefectCorrectError process_map_based(uint8_t* data,
                                      int width, int height,
                                      BayerPattern pattern,
                                      int bit_depth,
                                      const DefectCorrectParams& params);

// CUDA support
bool has_cuda();
DefectCorrectError process_defect_correct_cuda(uint8_t* data,
                                                int width, int height,
                                                BayerPattern pattern,
                                                DefectCorrectAlgorithm algorithm,
                                                int bit_depth,
                                                const DefectCorrectParams& params);

} // namespace defect_correct

#endif
