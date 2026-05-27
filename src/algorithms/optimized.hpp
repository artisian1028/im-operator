#ifndef IMOP_ALGORITHMS_OPTIMIZED_HPP
#define IMOP_ALGORITHMS_OPTIMIZED_HPP

#include "imop/types.hpp"
#include <cstdint>

namespace imop {

// Internal optimized entry points called from algorithm implementations.
// These take PatternOffsets directly (not BayerPattern) and are NOT part of
// the public API. They are declared here (internal header) rather than in
// include/imop/algorithms.hpp.

void process_super_fast_optimized(const uint8_t* bayer, uint8_t* rgb,
                                  int width, int height, const PatternOffsets& po, int bit_depth,
                                  bool is_packed = false);
void process_mg_optimized(const uint8_t* bayer, uint8_t* rgb,
                          int width, int height, const PatternOffsets& po, int bit_depth,
                          bool is_packed = false);
void process_hqli_optimized(const uint8_t* bayer, uint8_t* rgb,
                            int width, int height, const PatternOffsets& po, int bit_depth,
                            bool is_packed = false);
void process_l7_optimized(const uint8_t* bayer, uint8_t* rgb,
                          int width, int height, const PatternOffsets& po, int bit_depth,
                          bool is_packed = false);

} // namespace imop

#endif // IMOP_ALGORITHMS_OPTIMIZED_HPP
