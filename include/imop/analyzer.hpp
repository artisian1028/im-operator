#ifndef IMOP_ANALYZER_HPP
#define IMOP_ANALYZER_HPP

#include "types.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>

namespace imop {

DataInfo analyze_data(const uint8_t* data, size_t byte_size);

int detect_bit_depth(const uint8_t* data, size_t byte_size);

std::vector<std::pair<int, int>> suggest_dimensions(size_t pixel_count);

BayerPattern guess_pattern(const uint8_t* data, int width, int height,
                            int bit_depth = 8, bool is_packed = false);

} // namespace imop

#endif // IMOP_ANALYZER_HPP
