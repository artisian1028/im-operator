#include "im_operator.h"
#include "denoise.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

// ── TIFF writer (same format as synthetic_test) ────────────────────────

static void write16le(std::ofstream& f, uint16_t v) {
    f.put(static_cast<uint8_t>(v & 0xFF));
    f.put(static_cast<uint8_t>((v >> 8) & 0xFF));
}
static void write32le(std::ofstream& f, uint32_t v) {
    f.put(static_cast<uint8_t>(v & 0xFF));
    f.put(static_cast<uint8_t>((v >> 8) & 0xFF));
    f.put(static_cast<uint8_t>((v >> 16) & 0xFF));
    f.put(static_cast<uint8_t>((v >> 24) & 0xFF));
}

static bool save_tiff(const std::string& filename, const uint8_t* rgb,
                       int width, int height, int channels, int bit_depth) {
    std::ofstream f(filename, std::ios::binary);
    if (!f) return false;

    bool use16 = (bit_depth > 8);
    int bps = use16 ? 16 : 8;
    int samples_per_pixel = channels;
    size_t sample_count = static_cast<size_t>(width) * height * samples_per_pixel;
    size_t strip_bytes = use16 ? sample_count * 2 : sample_count;
    uint32_t rows_per_strip = static_cast<uint32_t>(height);

    f.put('I'); f.put('I');
    write16le(f, 42);
    write32le(f, 8);

    constexpr int NUM_TAGS = 12;
    uint32_t ifd_size = 2 + 12 * NUM_TAGS + 4;
    uint32_t bps_byte_count = static_cast<uint32_t>(samples_per_pixel) * 2;
    uint32_t bps_offset   = 8 + ifd_size;
    uint32_t xres_offset  = bps_offset + bps_byte_count;
    uint32_t yres_offset  = xres_offset + 8;
    uint32_t data_offset  = yres_offset + 8;

    write16le(f, NUM_TAGS);

    write16le(f, 256); write16le(f, 4); write32le(f, 1); write32le(f, static_cast<uint32_t>(width));
    write16le(f, 257); write16le(f, 4); write32le(f, 1); write32le(f, static_cast<uint32_t>(height));
    write16le(f, 258); write16le(f, 3); write32le(f, static_cast<uint32_t>(samples_per_pixel)); write32le(f, bps_offset);
    write16le(f, 259); write16le(f, 3); write32le(f, 1); write32le(f, 1);
    // PhotometricInterpretation: 1=BlackIsZero (gray), 2=RGB
    write16le(f, 262); write16le(f, 3); write32le(f, 1); write32le(f, (samples_per_pixel == 1) ? 1u : 2u);
    write16le(f, 273); write16le(f, 4); write32le(f, 1); write32le(f, data_offset);
    write16le(f, 277); write16le(f, 3); write32le(f, 1); write32le(f, static_cast<uint32_t>(samples_per_pixel));
    write16le(f, 278); write16le(f, 4); write32le(f, 1); write32le(f, rows_per_strip);
    write16le(f, 279); write16le(f, 4); write32le(f, 1); write32le(f, static_cast<uint32_t>(strip_bytes));
    write16le(f, 282); write16le(f, 5); write32le(f, 1); write32le(f, xres_offset);
    write16le(f, 283); write16le(f, 5); write32le(f, 1); write32le(f, yres_offset);
    write16le(f, 296); write16le(f, 3); write32le(f, 1); write32le(f, 2);

    write32le(f, 0);

    for (int s = 0; s < samples_per_pixel; s++) {
        write16le(f, static_cast<uint16_t>(bps));
    }
    write32le(f, 72); write32le(f, 1);
    write32le(f, 72); write32le(f, 1);

    if (use16 && bit_depth < 16) {
        int max_val = (1 << bit_depth) - 1;
        std::vector<uint8_t> scaled(strip_bytes);
        for (size_t i = 0; i < sample_count; i++) {
            uint16_t val;
            std::memcpy(&val, rgb + i * 2, sizeof(val));
            uint32_t sv = (static_cast<uint32_t>(val) * 65535u + static_cast<uint32_t>(max_val) / 2u) / static_cast<uint32_t>(max_val);
            if (sv > 65535u) sv = 65535u;
            uint16_t out = static_cast<uint16_t>(sv);
            std::memcpy(scaled.data() + i * 2, &out, sizeof(out));
        }
        f.write(reinterpret_cast<const char*>(scaled.data()), strip_bytes);
    } else {
        f.write(reinterpret_cast<const char*>(rgb), strip_bytes);
    }
    return f.good();
}

// ── Noise generators ────────────────────────────────────────────────────

// Add Gaussian noise to an RGB/gray image buffer
// noise_sigma: standard deviation of Gaussian noise
static void add_gaussian_noise(uint8_t* data, size_t pixel_count, int channels,
                                int bit_depth, float noise_sigma,
                                std::mt19937& rng) {
    int max_val = (1 << bit_depth) - 1;
    std::normal_distribution<float> dist(0.0f, noise_sigma);

    if (bit_depth <= 8) {
        for (size_t i = 0; i < pixel_count * channels; i++) {
            int noisy = static_cast<int>(data[i] + dist(rng) + 0.5f);
            noisy = std::max(0, std::min(max_val, noisy));
            data[i] = static_cast<uint8_t>(noisy);
        }
    } else {
        auto* data16 = reinterpret_cast<uint16_t*>(data);
        for (size_t i = 0; i < pixel_count * channels; i++) {
            int noisy = static_cast<int>(data16[i] + dist(rng) + 0.5f);
            noisy = std::max(0, std::min(max_val, noisy));
            data16[i] = static_cast<uint16_t>(noisy);
        }
    }
}

// Add salt-and-pepper noise
// density: fraction of pixels to corrupt (0.0 - 1.0)
static void add_salt_pepper_noise(uint8_t* data, size_t pixel_count, int channels,
                                   int bit_depth, float density,
                                   std::mt19937& rng) {
    int max_val = (1 << bit_depth) - 1;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (bit_depth <= 8) {
        for (size_t i = 0; i < pixel_count * channels; i++) {
            float r = dist(rng);
            if (r < density * 0.5f) {
                data[i] = 0;                     // pepper
            } else if (r < density) {
                data[i] = static_cast<uint8_t>(max_val); // salt
            }
        }
    } else {
        auto* data16 = reinterpret_cast<uint16_t*>(data);
        for (size_t i = 0; i < pixel_count * channels; i++) {
            float r = dist(rng);
            if (r < density * 0.5f) {
                data16[i] = 0;
            } else if (r < density) {
                data16[i] = static_cast<uint16_t>(max_val);
            }
        }
    }
}

// ── Scene generators (produce clean RGB images) ─────────────────────────

// Generate a color bar image
static std::vector<uint8_t> make_color_bars(int W, int H, int channels, int bit_depth) {
    int max_val = (1 << bit_depth) - 1;
    int bps = (bit_depth <= 8) ? 1 : 2;
    std::vector<uint8_t> image(static_cast<size_t>(W) * H * channels * bps, 0);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int bar = x * 8 / W;
            int r = 0, g = 0, b = 0;
            switch (bar) {
                case 0: r = 0;       g = 0;       b = 0;       break;  // black
                case 1: r = max_val; g = max_val; b = max_val; break;  // white
                case 2: r = max_val; g = 0;       b = 0;       break;  // red
                case 3: r = 0;       g = max_val; b = 0;       break;  // green
                case 4: r = 0;       g = 0;       b = max_val; break;  // blue
                case 5: r = max_val; g = max_val; b = 0;       break;  // yellow
                case 6: r = max_val; g = 0;       b = max_val; break;  // magenta
                case 7: r = 0;       g = max_val; b = max_val; break;  // cyan
            }

            if (channels == 1) {
                // Grayscale: use luminance
                int gray = static_cast<int>(0.299f * r + 0.587f * g + 0.114f * b);
                if (bit_depth <= 8) {
                    image[(static_cast<size_t>(y) * W + x)] = static_cast<uint8_t>(gray);
                } else {
                    auto* d16 = reinterpret_cast<uint16_t*>(image.data());
                    d16[static_cast<size_t>(y) * W + x] = static_cast<uint16_t>(gray);
                }
            } else {
                if (bit_depth <= 8) {
                    size_t idx = (static_cast<size_t>(y) * W + x) * 3;
                    image[idx + 0] = static_cast<uint8_t>(r);
                    image[idx + 1] = static_cast<uint8_t>(g);
                    image[idx + 2] = static_cast<uint8_t>(b);
                } else {
                    auto* d16 = reinterpret_cast<uint16_t*>(image.data());
                    size_t idx = (static_cast<size_t>(y) * W + x) * 3;
                    d16[idx + 0] = static_cast<uint16_t>(r);
                    d16[idx + 1] = static_cast<uint16_t>(g);
                    d16[idx + 2] = static_cast<uint16_t>(b);
                }
            }
        }
    }
    return image;
}

// Generate a smooth gradient image
static std::vector<uint8_t> make_gradient(int W, int H, int channels, int bit_depth) {
    int max_val = (1 << bit_depth) - 1;
    int bps = (bit_depth <= 8) ? 1 : 2;
    std::vector<uint8_t> image(static_cast<size_t>(W) * H * channels * bps, 0);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double cx = static_cast<double>(x) / (W - 1);
            double cy = static_cast<double>(y) / (H - 1);
            int r = static_cast<int>(cx * max_val + 0.5);
            int g = static_cast<int>(cy * max_val + 0.5);
            int b = static_cast<int>((0.5 * cx + 0.5 * cy) * max_val + 0.5);

            if (channels == 1) {
                int gray = static_cast<int>((0.299 * r + 0.587 * g + 0.114 * b));
                if (bit_depth <= 8) {
                    image[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(gray);
                } else {
                    auto* d16 = reinterpret_cast<uint16_t*>(image.data());
                    d16[static_cast<size_t>(y) * W + x] = static_cast<uint16_t>(gray);
                }
            } else {
                if (bit_depth <= 8) {
                    size_t idx = (static_cast<size_t>(y) * W + x) * 3;
                    image[idx + 0] = static_cast<uint8_t>(r);
                    image[idx + 1] = static_cast<uint8_t>(g);
                    image[idx + 2] = static_cast<uint8_t>(b);
                } else {
                    auto* d16 = reinterpret_cast<uint16_t*>(image.data());
                    size_t idx = (static_cast<size_t>(y) * W + x) * 3;
                    d16[idx + 0] = static_cast<uint16_t>(r);
                    d16[idx + 1] = static_cast<uint16_t>(g);
                    d16[idx + 2] = static_cast<uint16_t>(b);
                }
            }
        }
    }
    return image;
}

// Generate a Siemens star for checking detail preservation
static std::vector<uint8_t> make_siemens_star(int W, int H, int channels, int bit_depth) {
    int max_val = (1 << bit_depth) - 1;
    int bps = (bit_depth <= 8) ? 1 : 2;
    std::vector<uint8_t> image(static_cast<size_t>(W) * H * channels * bps, 0);

    double cx = W / 2.0 - 0.5;
    double cy = H / 2.0 - 0.5;
    int num_spokes = 36;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double dx = x - cx;
            double dy = y - cy;
            double angle = std::atan2(dy, dx);
            double spoke = std::sin(angle * num_spokes * 0.5);
            int val = static_cast<int>((spoke > 0 ? 1.0 : 0.0) * max_val + 0.5);

            if (channels == 1) {
                if (bit_depth <= 8) {
                    image[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(val);
                } else {
                    auto* d16 = reinterpret_cast<uint16_t*>(image.data());
                    d16[static_cast<size_t>(y) * W + x] = static_cast<uint16_t>(val);
                }
            } else {
                if (bit_depth <= 8) {
                    size_t idx = (static_cast<size_t>(y) * W + x) * 3;
                    image[idx + 0] = static_cast<uint8_t>(val);
                    image[idx + 1] = static_cast<uint8_t>(val);
                    image[idx + 2] = static_cast<uint8_t>(val);
                } else {
                    auto* d16 = reinterpret_cast<uint16_t*>(image.data());
                    size_t idx = (static_cast<size_t>(y) * W + x) * 3;
                    d16[idx + 0] = static_cast<uint16_t>(val);
                    d16[idx + 1] = static_cast<uint16_t>(val);
                    d16[idx + 2] = static_cast<uint16_t>(val);
                }
            }
        }
    }
    return image;
}

// ── Bayer-domain noise scene (add noise to Bayer raw data) ────────────

static std::vector<uint8_t> make_bayer_scene(int W, int H, int bit_depth,
                                              imop::BayerPattern pattern) {
    using namespace imop;
    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    size_t bs = pixel::compute_bayer_byte_size(W, H, bit_depth, false);
    std::vector<uint8_t> bayer(bs, 0);
    int max_val = (1 << bit_depth) - 1;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double cx = static_cast<double>(x) / (W - 1);
            double cy = static_cast<double>(y) / (H - 1);
            double r = 0.5 + 0.5 * std::cos(cx * 12.0);
            double g = 0.5 + 0.5 * std::sin(cx * 8.0 + cy * 6.0);
            double b = 0.5 + 0.5 * std::cos(cy * 10.0);

            double sv;
            if (pixel::is_r_at(po, y, x)) sv = r;
            else if (pixel::is_b_at(po, y, x)) sv = b;
            else sv = g;

            int iv = static_cast<int>(sv * max_val + 0.5);
            iv = std::max(0, std::min(max_val, iv));

            if (bit_depth <= 8) {
                bayer[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(iv);
            } else {
                uint16_t v16 = static_cast<uint16_t>(iv);
                std::memcpy(&bayer[(static_cast<size_t>(y) * W + x) * 2], &v16, 2);
            }
        }
    }
    return bayer;
}

// Add noise to Bayer data (same as RGB but single-channel)
static void add_noise_to_bayer(std::vector<uint8_t>& bayer, int W, int H,
                                int bit_depth, float gauss_sigma, float sp_density,
                                std::mt19937& rng) {
    int max_val = (1 << bit_depth) - 1;
    size_t pixel_count = static_cast<size_t>(W) * H;
    std::normal_distribution<float> gauss_dist(0.0f, gauss_sigma);
    std::uniform_real_distribution<float> unif_dist(0.0f, 1.0f);

    if (bit_depth <= 8) {
        for (size_t i = 0; i < pixel_count; i++) {
            // Gaussian
            int noisy = static_cast<int>(bayer[i] + gauss_dist(rng) + 0.5f);
            // Salt & pepper
            float rn = unif_dist(rng);
            if (rn < sp_density * 0.5f) noisy = 0;
            else if (rn < sp_density) noisy = max_val;
            noisy = std::max(0, std::min(max_val, noisy));
            bayer[i] = static_cast<uint8_t>(noisy);
        }
    } else {
        auto* d16 = reinterpret_cast<uint16_t*>(bayer.data());
        for (size_t i = 0; i < pixel_count; i++) {
            int noisy = static_cast<int>(d16[i] + gauss_dist(rng) + 0.5f);
            float rn = unif_dist(rng);
            if (rn < sp_density * 0.5f) noisy = 0;
            else if (rn < sp_density) noisy = max_val;
            noisy = std::max(0, std::min(max_val, noisy));
            d16[i] = static_cast<uint16_t>(noisy);
        }
    }
}

// ── PSNR computation for reporting ─────────────────────────────────────

static double compute_psnr(const uint8_t* clean, const uint8_t* noisy,
                            const uint8_t* denoised, size_t count) {
    double mse_denoised = 0.0, mse_noisy = 0.0;
    for (size_t i = 0; i < count; i++) {
        double d_denoised = static_cast<double>(clean[i]) - static_cast<double>(denoised[i]);
        double d_noisy = static_cast<double>(clean[i]) - static_cast<double>(noisy[i]);
        mse_denoised += d_denoised * d_denoised;
        mse_noisy += d_noisy * d_noisy;
    }
    mse_denoised /= static_cast<double>(count);
    mse_noisy /= static_cast<double>(count);
    if (mse_denoised < 1e-10) return 100.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse_denoised);
}

static double compute_psnr_16bit(const uint8_t* clean, const uint8_t* noisy,
                                  const uint8_t* denoised, size_t pixel_count,
                                  int bit_depth) {
    auto* pc = reinterpret_cast<const uint16_t*>(clean);
    auto* pn = reinterpret_cast<const uint16_t*>(noisy);
    auto* pd = reinterpret_cast<const uint16_t*>(denoised);
    double max_val = static_cast<double>((1 << bit_depth) - 1);
    double mse_denoised = 0.0, mse_noisy = 0.0;
    for (size_t i = 0; i < pixel_count; i++) {
        double dd = static_cast<double>(pc[i]) - static_cast<double>(pd[i]);
        double dn = static_cast<double>(pc[i]) - static_cast<double>(pn[i]);
        mse_denoised += dd * dd;
        mse_noisy += dn * dn;
    }
    mse_denoised /= static_cast<double>(pixel_count);
    mse_noisy /= static_cast<double>(pixel_count);
    if (mse_denoised < 1e-10) return 100.0;
    return 10.0 * std::log10(max_val * max_val / mse_denoised);
}

// ── Scene runner ────────────────────────────────────────────────────────

static const char* algo_short_name(denoise::DenoiseAlgorithm algo) {
    switch (algo) {
        case denoise::DenoiseAlgorithm::GAUSSIAN:       return "gaussian";
        case denoise::DenoiseAlgorithm::MEDIAN:         return "median";
        case denoise::DenoiseAlgorithm::BILATERAL:      return "bilateral";
        case denoise::DenoiseAlgorithm::NLM:            return "nlm";
        case denoise::DenoiseAlgorithm::WAVELET:        return "wavelet";
        case denoise::DenoiseAlgorithm::BAYER_DENOISE:  return "bayer";
        default: return "unknown";
    }
}

static void run_denoise_scene(const std::string& dir, const std::string& scene_name,
                               const std::vector<uint8_t>& clean, int W, int H,
                               int channels, int bit_depth,
                               float gauss_sigma, float sp_density,
                               bool skip_bayer_denoise = false) {
    size_t pixel_count = static_cast<size_t>(W) * H;
    int bps = (bit_depth <= 8) ? 1 : 2;
    size_t data_size = pixel_count * channels * bps;

    // Copy clean image
    std::vector<uint8_t> noisy(clean);
    std::vector<uint8_t> denoised(data_size);

    // Seed for reproducibility
    std::mt19937 rng(42);

    // Add noise
    if (gauss_sigma > 0.0f) {
        add_gaussian_noise(noisy.data(), pixel_count, channels, bit_depth,
                           gauss_sigma * ((1 << bit_depth) - 1) / 255.0f, rng);
    }
    if (sp_density > 0.0f) {
        add_salt_pepper_noise(noisy.data(), pixel_count, channels, bit_depth,
                               sp_density, rng);
    }

    // Save clean and noisy
    char buf[512];
    if (channels == 1) {
        // For Bayer, save as TIFF
        save_tiff(dir + "/" + scene_name + "_clean.tiff",
                  clean.data(), W, H, 1, bit_depth);
        save_tiff(dir + "/" + scene_name + "_noisy.tiff",
                  noisy.data(), W, H, 1, bit_depth);
    } else {
        save_tiff(dir + "/" + scene_name + "_clean.tiff",
                  clean.data(), W, H, 3, bit_depth);
        save_tiff(dir + "/" + scene_name + "_noisy.tiff",
                  noisy.data(), W, H, 3, bit_depth);
    }

    // Run all denoise algorithms
    denoise::DenoiseAlgorithm algos[] = {
        denoise::DenoiseAlgorithm::GAUSSIAN,
        denoise::DenoiseAlgorithm::MEDIAN,
        denoise::DenoiseAlgorithm::BILATERAL,
        denoise::DenoiseAlgorithm::NLM,
        denoise::DenoiseAlgorithm::WAVELET,
        denoise::DenoiseAlgorithm::BAYER_DENOISE
    };

    for (auto algo : algos) {
        if (skip_bayer_denoise && algo == denoise::DenoiseAlgorithm::BAYER_DENOISE) {
            continue;
        }
        std::memset(denoised.data(), 0, data_size);
        denoise::DenoiseError err = denoise::process_denoise(
            noisy.data(), denoised.data(), W, H, channels, algo, bit_depth, 1.0f);

        if (err == denoise::DenoiseError::Ok) {
            snprintf(buf, sizeof(buf), "%s/%s_%s.tiff",
                     dir.c_str(), scene_name.c_str(), algo_short_name(algo));
            save_tiff(buf, denoised.data(), W, H, channels, bit_depth);

            // Compute PSNR
            double psnr;
            if (bit_depth <= 8) {
                psnr = compute_psnr(clean.data(), noisy.data(), denoised.data(),
                                     data_size);
            } else {
                psnr = compute_psnr_16bit(clean.data(), noisy.data(), denoised.data(),
                                           pixel_count * channels, bit_depth);
            }
            std::cout << "    " << denoise::algorithm_name(algo) << ": PSNR " << psnr << " dB" << std::endl;
        } else {
            std::cout << "    " << denoise::algorithm_name(algo) << ": ERROR - "
                      << denoise::denoise_error_message(err) << std::endl;
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────

int main() {
#ifdef _WIN32
    CreateDirectoryA("denoise_output", nullptr);
#endif
    std::cout << "=== Denoise Synthetic Test Images ===\n" << std::endl;

    // ── RGB scenes ───────────────────────────────────────────────────
    std::cout << "--- RGB Color Bars (640x360, 8-bit) ---" << std::endl;
    std::cout << "  Noise: gauss=15, salt-pepper=0.02\n";
    {
        auto clean = make_color_bars(640, 360, 3, 8);
        run_denoise_scene("denoise_output", "colorbar_8bit_rgb",
                          clean, 640, 360, 3, 8, 15.0f, 0.02f, true);
    }

    std::cout << "\n--- RGB Gradient (640x480, 8-bit) ---" << std::endl;
    std::cout << "  Noise: gauss=15, salt-pepper=0.02\n";
    {
        auto clean = make_gradient(640, 480, 3, 8);
        run_denoise_scene("denoise_output", "gradient_8bit_rgb",
                          clean, 640, 480, 3, 8, 15.0f, 0.02f, true);
    }

    std::cout << "\n--- RGB Siemens Star (600x600, 8-bit) ---" << std::endl;
    std::cout << "  Noise: gauss=10, salt-pepper=0.01\n";
    {
        auto clean = make_siemens_star(600, 600, 3, 8);
        run_denoise_scene("denoise_output", "siemens_8bit_rgb",
                          clean, 600, 600, 3, 8, 10.0f, 0.01f, true);
    }

    std::cout << "\n--- RGB Color Bars (640x360, 12-bit) ---" << std::endl;
    std::cout << "  Noise: gauss=30\n";
    {
        auto clean = make_color_bars(640, 360, 3, 12);
        run_denoise_scene("denoise_output", "colorbar_12bit_rgb",
                          clean, 640, 360, 3, 12, 30.0f, 0.0f, true);
    }

    // ── Salt & pepper specific test ──────────────────────────────────
    std::cout << "\n--- RGB Salt & Pepper only (480x320, 8-bit) ---" << std::endl;
    std::cout << "  Noise: salt-pepper=0.05\n";
    {
        auto clean = make_color_bars(480, 320, 3, 8);
        run_denoise_scene("denoise_output", "saltpepper_8bit_rgb",
                          clean, 480, 320, 3, 8, 0.0f, 0.05f, true);
    }

    // ── Gaussian only test ───────────────────────────────────────────
    std::cout << "\n--- RGB Gaussian only (600x400, 8-bit) ---" << std::endl;
    std::cout << "  Noise: gauss=25\n";
    {
        auto clean = make_gradient(600, 400, 3, 8);
        run_denoise_scene("denoise_output", "gaussian_only_8bit_rgb",
                          clean, 600, 400, 3, 8, 25.0f, 0.0f, true);
    }

    // ── Bayer-domain scene ───────────────────────────────────────────
    std::cout << "\n--- Bayer-domain Scene (800x600, 12-bit, RGGB) ---" << std::endl;
    std::cout << "  Noise: gauss=20, salt-pepper=0.01\n";
    std::cout << "  Pipeline: clean_bayer -> add_noise -> bayer_denoise -> demosaic(HQLI)\n";
    {
        const int W = 800, H = 600;
        const int bit_depth = 12;
        imop::BayerPattern pattern = imop::BayerPattern::RGGB;
        size_t bayer_size = imop::pixel::compute_bayer_byte_size(W, H, bit_depth, false);
        size_t rgb_size = static_cast<size_t>(W) * H * 3 * 2;

        // Clean Bayer
        auto clean_bayer = make_bayer_scene(W, H, bit_depth, pattern);

        // Noisy Bayer
        auto noisy_bayer = clean_bayer;
        std::mt19937 rng(42);
        add_noise_to_bayer(noisy_bayer, W, H, bit_depth, 20.0f, 0.01f, rng);

        // Save Bayer as single-channel TIFF
        save_tiff("denoise_output/bayer_scene_clean.tiff",
                  clean_bayer.data(), W, H, 1, bit_depth);
        save_tiff("denoise_output/bayer_scene_noisy.tiff",
                  noisy_bayer.data(), W, H, 1, bit_depth);

        // Bayer denoise the noisy Bayer data
        std::vector<uint8_t> denoised_bayer(bayer_size);
        denoise::DenoiseError dn_err = denoise::process_denoise(
            noisy_bayer.data(), denoised_bayer.data(), W, H, 1,
            denoise::DenoiseAlgorithm::BAYER_DENOISE, bit_depth, 1.0f);

        if (dn_err == denoise::DenoiseError::Ok) {
            save_tiff("denoise_output/bayer_scene_bayer_denoised.tiff",
                      denoised_bayer.data(), W, H, 1, bit_depth);

            // Demosaic all three for visual comparison
            std::vector<uint8_t> rgb_clean(rgb_size);
            std::vector<uint8_t> rgb_noisy(rgb_size);
            std::vector<uint8_t> rgb_denoised(rgb_size);

            imop::demosaic_cpu(clean_bayer.data(), rgb_clean.data(), W, H,
                                          pattern, imop::DemosaicAlgorithm::HQLI, bit_depth);
            imop::demosaic_cpu(noisy_bayer.data(), rgb_noisy.data(), W, H,
                                          pattern, imop::DemosaicAlgorithm::HQLI, bit_depth);
            imop::demosaic_cpu(denoised_bayer.data(), rgb_denoised.data(), W, H,
                                          pattern, imop::DemosaicAlgorithm::HQLI, bit_depth);

            save_tiff("denoise_output/bayer_scene_clean_debayered.tiff",
                      rgb_clean.data(), W, H, 3, bit_depth);
            save_tiff("denoise_output/bayer_scene_noisy_debayered.tiff",
                      rgb_noisy.data(), W, H, 3, bit_depth);
            save_tiff("denoise_output/bayer_scene_denoised_debayered.tiff",
                      rgb_denoised.data(), W, H, 3, bit_depth);

            // PSNR comparison
            double psnr_noisy = compute_psnr_16bit(rgb_clean.data(), rgb_noisy.data(),
                                                    rgb_noisy.data(), rgb_size / 2, bit_depth);
            double psnr_denoised = compute_psnr_16bit(rgb_clean.data(), rgb_noisy.data(),
                                                       rgb_denoised.data(), rgb_size / 2, bit_depth);
            std::cout << "    Noisy debayered PSNR: " << psnr_noisy << " dB" << std::endl;
            std::cout << "    Denoised debayered PSNR: " << psnr_denoised << " dB" << std::endl;
        } else {
            std::cout << "    Bayer denoise failed: "
                      << denoise::denoise_error_message(dn_err) << std::endl;
        }
    }

    std::cout << "\nDone! Files in denoise_output/" << std::endl;
    return 0;
}
