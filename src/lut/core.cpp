#include "common.hpp"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace lut {

// --- Trilinear LUT application (shared by all algorithms) ---

LUTError apply_lut(const LUT3D& lut, const uint8_t* input, uint8_t* output,
                   int width, int height, int bit_depth) {
    if (lut.empty()) return LUTError::InternalError;
    if (!input || !output) return LUTError::NullInput;
    if (width <= 0 || height <= 0) return LUTError::InvalidDimensions;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float rn = detail::read_pixel(input, x, y, width, 3, bit_depth, 0) / mv;
            float gn = detail::read_pixel(input, x, y, width, 3, bit_depth, 1) / mv;
            float bn = detail::read_pixel(input, x, y, width, 3, bit_depth, 2) / mv;

            float ro, go, bo;
            detail::trilinear_lookup(lut.data.data(), lut.size, rn, gn, bn, ro, go, bo);

            detail::write_pixel(output, x, y, width, 3, bit_depth, 0,
                                detail::clamp_val(static_cast<int>(ro * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, 3, bit_depth, 1,
                                detail::clamp_val(static_cast<int>(go * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, 3, bit_depth, 2,
                                detail::clamp_val(static_cast<int>(bo * mv + 0.5f), max_val));
        }
    }

    return LUTError::Ok;
}

// --- Identity LUT builder ---

LUT3D build_identity_lut(int size) {
    LUT3D lut;
    lut.size = size;
    lut.data.resize(static_cast<size_t>(size) * size * size * 3);

    for (int ri = 0; ri < size; ri++) {
        float r = static_cast<float>(ri) / static_cast<float>(size - 1);
        for (int gi = 0; gi < size; gi++) {
            float g = static_cast<float>(gi) / static_cast<float>(size - 1);
            for (int bi = 0; bi < size; bi++) {
                float b = static_cast<float>(bi) / static_cast<float>(size - 1);
                size_t idx = (static_cast<size_t>(ri) * size * size +
                               static_cast<size_t>(gi) * size + bi) * 3;
                lut.data[idx + 0] = r;
                lut.data[idx + 1] = g;
                lut.data[idx + 2] = b;
            }
        }
    }

    return lut;
}

// --- .cube file parser ---

LUT3D load_cube_file(const char* filepath) {
    LUT3D lut;

    if (!filepath) return lut;

    std::ifstream file(filepath);
    if (!file.is_open()) return lut;

    std::string line;
    int size = 0;
    float min_rgb[3] = {0, 0, 0};
    float max_rgb[3] = {1, 1, 1};

    // Parse header
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty() || line[0] == '#') continue;

        if (line.rfind("LUT_3D_SIZE", 0) == 0) {
            std::istringstream iss(line.substr(11));
            iss >> size;
        } else if (line.rfind("DOMAIN_MIN", 0) == 0) {
            std::istringstream iss(line.substr(10));
            iss >> min_rgb[0] >> min_rgb[1] >> min_rgb[2];
        } else if (line.rfind("DOMAIN_MAX", 0) == 0) {
            std::istringstream iss(line.substr(10));
            iss >> max_rgb[0] >> max_rgb[1] >> max_rgb[2];
        } else if (line.rfind("TITLE", 0) == 0) {
            // Skip TITLE line
            continue;
        } else {
            // Assume this is a data line (starts with number or '-')
            break;
        }
    }

    if (size < 2 || size > 256) return lut;

    lut.size = size;
    lut.data.resize(static_cast<size_t>(size) * size * size * 3);

    // Parse data values. First data line may already be in `line`.
    size_t entry = 0;
    float domain_scale[3] = {
        max_rgb[0] - min_rgb[0],
        max_rgb[1] - min_rgb[1],
        max_rgb[2] - min_rgb[2]
    };

    auto parse_data_line = [&](const std::string& l) {
        std::istringstream iss(l);
        float rv, gv, bv;
        while (iss >> rv >> gv >> bv) {
            if (entry < lut.total_samples()) {
                // Normalize to [0, 1]
                lut.data[entry * 3 + 0] = (domain_scale[0] > 0) ? (rv - min_rgb[0]) / domain_scale[0] : rv;
                lut.data[entry * 3 + 1] = (domain_scale[1] > 0) ? (gv - min_rgb[1]) / domain_scale[1] : gv;
                lut.data[entry * 3 + 2] = (domain_scale[2] > 0) ? (bv - min_rgb[2]) / domain_scale[2] : bv;
                entry++;
            }
        }
    };

    // Process the partial line we already read (if any data on it)
    if (!line.empty() && ((line[0] >= '0' && line[0] <= '9') || line[0] == '-' || line[0] == '+')) {
        parse_data_line(line);
    }

    while (entry < lut.total_samples() && std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty() || line[0] == '#') continue;
        parse_data_line(line);
    }

    // Pad with identity if incomplete
    while (entry < lut.total_samples()) {
        lut.data[entry * 3 + 0] = static_cast<float>((entry / lut.size / lut.size) % lut.size) / static_cast<float>(lut.size - 1);
        lut.data[entry * 3 + 1] = static_cast<float>((entry / lut.size) % lut.size) / static_cast<float>(lut.size - 1);
        lut.data[entry * 3 + 2] = static_cast<float>(entry % lut.size) / static_cast<float>(lut.size - 1);
        entry++;
    }

    return lut;
}

} // namespace lut
