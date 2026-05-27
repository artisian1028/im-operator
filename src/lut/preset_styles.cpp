#include "common.hpp"
#include <cmath>

namespace lut {

// --- Built-in style LUT generators ---

// Helper: map 1D index to 3D coordinate
static inline void idx_to_rgb(int idx, int size, int& ri, int& gi, int& bi) {
    ri = idx / (size * size);
    gi = (idx / size) % size;
    bi = idx % size;
}

LUT3D build_sepia_lut(int size) {
    LUT3D lut;
    lut.size = size;
    lut.data.resize(static_cast<size_t>(size) * size * size * 3);

    for (int ri = 0; ri < size; ri++) {
        float r = static_cast<float>(ri) / static_cast<float>(size - 1);
        for (int gi = 0; gi < size; gi++) {
            float g = static_cast<float>(gi) / static_cast<float>(size - 1);
            for (int bi = 0; bi < size; bi++) {
                float b = static_cast<float>(bi) / static_cast<float>(size - 1);

                // Luminance
                float luma = 0.299f * r + 0.587f * g + 0.114f * b;
                // Sepia tone
                float out_r = std::min(1.0f, luma * 1.07f + 0.08f);
                float out_g = std::min(1.0f, luma * 0.74f + 0.02f);
                float out_b = std::min(1.0f, luma * 0.43f);

                size_t idx = (static_cast<size_t>(ri) * size * size +
                               static_cast<size_t>(gi) * size + bi) * 3;
                lut.data[idx + 0] = out_r;
                lut.data[idx + 1] = out_g;
                lut.data[idx + 2] = out_b;
            }
        }
    }

    return lut;
}

LUT3D build_cool_lut(int size) {
    LUT3D lut;
    lut.size = size;
    lut.data.resize(static_cast<size_t>(size) * size * size * 3);

    for (int ri = 0; ri < size; ri++) {
        float r = static_cast<float>(ri) / static_cast<float>(size - 1);
        for (int gi = 0; gi < size; gi++) {
            float g = static_cast<float>(gi) / static_cast<float>(size - 1);
            for (int bi = 0; bi < size; bi++) {
                float b = static_cast<float>(bi) / static_cast<float>(size - 1);

                // Boost blue, suppress red
                float out_r = std::max(0.0f, std::min(1.0f, r * 0.80f));
                float out_g = std::max(0.0f, std::min(1.0f, g * 0.95f));
                float out_b = std::max(0.0f, std::min(1.0f, b * 1.20f + 0.03f));

                size_t idx = (static_cast<size_t>(ri) * size * size +
                               static_cast<size_t>(gi) * size + bi) * 3;
                lut.data[idx + 0] = out_r;
                lut.data[idx + 1] = out_g;
                lut.data[idx + 2] = out_b;
            }
        }
    }

    return lut;
}

LUT3D build_warm_lut(int size) {
    LUT3D lut;
    lut.size = size;
    lut.data.resize(static_cast<size_t>(size) * size * size * 3);

    for (int ri = 0; ri < size; ri++) {
        float r = static_cast<float>(ri) / static_cast<float>(size - 1);
        for (int gi = 0; gi < size; gi++) {
            float g = static_cast<float>(gi) / static_cast<float>(size - 1);
            for (int bi = 0; bi < size; bi++) {
                float b = static_cast<float>(bi) / static_cast<float>(size - 1);

                // Boost red/amber, suppress blue
                float out_r = std::max(0.0f, std::min(1.0f, r * 1.15f + 0.04f));
                float out_g = std::max(0.0f, std::min(1.0f, g * 1.05f));
                float out_b = std::max(0.0f, std::min(1.0f, b * 0.75f));

                size_t idx = (static_cast<size_t>(ri) * size * size +
                               static_cast<size_t>(gi) * size + bi) * 3;
                lut.data[idx + 0] = out_r;
                lut.data[idx + 1] = out_g;
                lut.data[idx + 2] = out_b;
            }
        }
    }

    return lut;
}

LUT3D build_high_contrast_lut(int size) {
    LUT3D lut;
    lut.size = size;
    lut.data.resize(static_cast<size_t>(size) * size * size * 3);

    for (int ri = 0; ri < size; ri++) {
        float r = static_cast<float>(ri) / static_cast<float>(size - 1);
        for (int gi = 0; gi < size; gi++) {
            float g = static_cast<float>(gi) / static_cast<float>(size - 1);
            for (int bi = 0; bi < size; bi++) {
                float b = static_cast<float>(bi) / static_cast<float>(size - 1);

                // S-curve per channel
                auto scurve = [](float v) -> float {
                    float t = 2.0f * v - 1.0f; // map to [-1, 1]
                    float s = std::tanh(2.5f * t) / std::tanh(2.5f);
                    return (s + 1.0f) * 0.5f;
                };

                float out_r = std::max(0.0f, std::min(1.0f, scurve(r)));
                float out_g = std::max(0.0f, std::min(1.0f, scurve(g)));
                float out_b = std::max(0.0f, std::min(1.0f, scurve(b)));

                size_t idx = (static_cast<size_t>(ri) * size * size +
                               static_cast<size_t>(gi) * size + bi) * 3;
                lut.data[idx + 0] = out_r;
                lut.data[idx + 1] = out_g;
                lut.data[idx + 2] = out_b;
            }
        }
    }

    return lut;
}

LUT3D build_low_contrast_lut(int size) {
    LUT3D lut;
    lut.size = size;
    lut.data.resize(static_cast<size_t>(size) * size * size * 3);

    for (int ri = 0; ri < size; ri++) {
        float r = static_cast<float>(ri) / static_cast<float>(size - 1);
        for (int gi = 0; gi < size; gi++) {
            float g = static_cast<float>(gi) / static_cast<float>(size - 1);
            for (int bi = 0; bi < size; bi++) {
                float b = static_cast<float>(bi) / static_cast<float>(size - 1);

                // Compress range + lift blacks (faded/bleached look)
                auto fade = [](float v) -> float {
                    return 0.15f + v * 0.70f;
                };

                float out_r = std::max(0.0f, std::min(1.0f, fade(r)));
                float out_g = std::max(0.0f, std::min(1.0f, fade(g)));
                float out_b = std::max(0.0f, std::min(1.0f, fade(b)));

                size_t idx = (static_cast<size_t>(ri) * size * size +
                               static_cast<size_t>(gi) * size + bi) * 3;
                lut.data[idx + 0] = out_r;
                lut.data[idx + 1] = out_g;
                lut.data[idx + 2] = out_b;
            }
        }
    }

    return lut;
}

LUT3D build_invert_lut(int size) {
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
                lut.data[idx + 0] = 1.0f - r;
                lut.data[idx + 1] = 1.0f - g;
                lut.data[idx + 2] = 1.0f - b;
            }
        }
    }

    return lut;
}

LUT3D build_vintage_fade_lut(int size) {
    LUT3D lut;
    lut.size = size;
    lut.data.resize(static_cast<size_t>(size) * size * size * 3);

    for (int ri = 0; ri < size; ri++) {
        float r = static_cast<float>(ri) / static_cast<float>(size - 1);
        for (int gi = 0; gi < size; gi++) {
            float g = static_cast<float>(gi) / static_cast<float>(size - 1);
            for (int bi = 0; bi < size; bi++) {
                float b = static_cast<float>(bi) / static_cast<float>(size - 1);

                // Desaturate + warm shift + slightly raised blacks
                float luma = 0.299f * r + 0.587f * g + 0.114f * b;
                float desat_r = luma + 0.3f * (r - luma);
                float desat_g = luma + 0.3f * (g - luma);
                float desat_b = luma + 0.3f * (b - luma);

                // Warm tint + raise blacks
                float out_r = std::max(0.0f, std::min(1.0f, desat_r + 0.06f));
                float out_g = std::max(0.0f, std::min(1.0f, desat_g + 0.02f));
                float out_b = std::max(0.0f, std::min(1.0f, desat_b - 0.05f + 0.04f));

                size_t idx = (static_cast<size_t>(ri) * size * size +
                               static_cast<size_t>(gi) * size + bi) * 3;
                lut.data[idx + 0] = out_r;
                lut.data[idx + 1] = out_g;
                lut.data[idx + 2] = out_b;
            }
        }
    }

    return lut;
}

} // namespace lut
