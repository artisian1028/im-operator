#include "im_operator.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using namespace imop;

static std::string pat_str(BayerPattern p) {
    switch (p) {
        case BayerPattern::RGGB: return "RGGB";
        case BayerPattern::BGGR: return "BGGR";
        case BayerPattern::GRBG: return "GRBG";
        case BayerPattern::GBRG: return "GBRG";
    }
    return "???";
}

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
                       int width, int height, int bit_depth) {
    std::ofstream f(filename, std::ios::binary);
    if (!f) return false;

    bool use16 = (bit_depth > 8);
    int bps = use16 ? 16 : 8;
    size_t sample_count = static_cast<size_t>(width) * height * 3;
    size_t strip_bytes = use16 ? sample_count * 2 : sample_count;
    uint32_t rows_per_strip = static_cast<uint32_t>(height);

    f.put('I'); f.put('I');
    write16le(f, 42);
    write32le(f, 8);

    constexpr int NUM_TAGS = 12;
    uint32_t ifd_size = 2 + 12 * NUM_TAGS + 4;
    uint32_t bps_offset = 8 + ifd_size;
    uint32_t xres_offset = bps_offset + 6;
    uint32_t yres_offset = xres_offset + 8;
    uint32_t data_offset = yres_offset + 8;

    write16le(f, NUM_TAGS);

    write16le(f, 256); write16le(f, 4); write32le(f, 1); write32le(f, static_cast<uint32_t>(width));
    write16le(f, 257); write16le(f, 4); write32le(f, 1); write32le(f, static_cast<uint32_t>(height));
    write16le(f, 258); write16le(f, 3); write32le(f, 3); write32le(f, bps_offset);
    write16le(f, 259); write16le(f, 3); write32le(f, 1); write32le(f, 1);
    write16le(f, 262); write16le(f, 3); write32le(f, 1); write32le(f, 2);
    write16le(f, 273); write16le(f, 4); write32le(f, 1); write32le(f, data_offset);
    write16le(f, 277); write16le(f, 3); write32le(f, 1); write32le(f, 3);
    write16le(f, 278); write16le(f, 4); write32le(f, 1); write32le(f, rows_per_strip);
    write16le(f, 279); write16le(f, 4); write32le(f, 1); write32le(f, static_cast<uint32_t>(strip_bytes));
    write16le(f, 282); write16le(f, 5); write32le(f, 1); write32le(f, xres_offset);
    write16le(f, 283); write16le(f, 5); write32le(f, 1); write32le(f, yres_offset);
    write16le(f, 296); write16le(f, 3); write32le(f, 1); write32le(f, 2);

    write32le(f, 0);

    write16le(f, static_cast<uint16_t>(bps));
    write16le(f, static_cast<uint16_t>(bps));
    write16le(f, static_cast<uint16_t>(bps));
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

static void run_all_algos(const char* dir, const char* prefix,
                          const std::vector<uint8_t>& bayer, int W, int H,
                          BayerPattern pattern, int bit_depth,
                          bool is_packed = false) {
    size_t rgb_sz = static_cast<size_t>(W) * H * 3 * (bit_depth <= 8 ? 1 : 2);
    std::vector<uint8_t> rgb(rgb_sz);
    DemosaicAlgorithm algos[] = {DemosaicAlgorithm::SUPER_FAST, DemosaicAlgorithm::HQLI, DemosaicAlgorithm::MG,
        DemosaicAlgorithm::L7, DemosaicAlgorithm::DFPD, DemosaicAlgorithm::AHD,
        DemosaicAlgorithm::AMAZE, DemosaicAlgorithm::RCD, DemosaicAlgorithm::PRISM};
    const char* an[] = {"SUPER_FAST","HQLI","MG","L7","DFPD","AHD","AMAZE","RCD","PRISM"};

    for (int ai = 0; ai < 9; ai++) {
        memset(rgb.data(), 0, rgb_sz);
        if (!demosaic(bayer.data(), rgb.data(), W, H, pattern, algos[ai], bit_depth, is_packed)) continue;
        char buf[512];
        snprintf(buf, sizeof(buf), "%s/%s_%s_%s.tiff", dir, prefix, pat_str(pattern).c_str(), an[ai]);
        save_tiff(buf, rgb.data(), W, H, bit_depth);
    }
}

static void write_bayer_pixel(std::vector<uint8_t>& bayer, int x, int y, int W,
                               int iv, int bit_depth) {
    if (bit_depth <= 8) {
        bayer[y * W + x] = static_cast<uint8_t>(iv);
    } else {
        uint16_t v16 = static_cast<uint16_t>(iv);
        memcpy(&bayer[(y * W + x) * 2], &v16, 2);
    }
}

static void generate_color_bars(const char* dir, BayerPattern pattern, int bit_depth) {
    const int W = 1280, H = 720;
    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    size_t bs = pixel::compute_bayer_byte_size(W, H, bit_depth, false);
    std::vector<uint8_t> bayer(bs, 0);
    int max_val = (1 << bit_depth) - 1;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int bar = x * 8 / W;
            double sr = 0, sg = 0, sb = 0;
            switch (bar) {
                case 0: sr=1; sg=1; sb=1; break; case 1: sr=1; sg=1; sb=0; break;
                case 2: sr=0; sg=1; sb=1; break; case 3: sr=0; sg=1; sb=0; break;
                case 4: sr=1; sg=0; sb=1; break; case 5: sr=1; sg=0; sb=0; break;
                case 6: sr=0; sg=0; sb=1; break; case 7: sr=0; sg=0; sb=0; break;
            }
            double sv;
            if (pixel::is_r_at(po, y, x)) sv = sr;
            else if (pixel::is_b_at(po, y, x)) sv = sb;
            else sv = sg;
            int iv = static_cast<int>(sv * max_val + 0.5);
            iv = std::max(0, std::min(max_val, iv));
            write_bayer_pixel(bayer, x, y, W, iv, bit_depth);
        }
    }

    char prefix[64];
    snprintf(prefix, sizeof(prefix), "colorbar_%dbpp", bit_depth);
    run_all_algos(dir, prefix, bayer, W, H, pattern, bit_depth);
}

static void generate_complex(const char* dir, int W, int H, int bit_depth, BayerPattern pattern) {
    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    size_t bs = pixel::compute_bayer_byte_size(W, H, bit_depth, false);
    std::vector<uint8_t> bayer(bs, 0);
    int max_val = (1 << bit_depth) - 1;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double cx = (double)x / W, cy = (double)y / H;
            double sr, sg, sb;
            if (cx < 0.33 && cy < 0.33)      { sr=1; sg=0; sb=0; }
            else if (cx >= 0.67 && cy < 0.33) { sr=0; sg=1; sb=0; }
            else if (cx < 0.33 && cy >= 0.67) { sr=0; sg=0; sb=1; }
            else if (cx >= 0.67 && cy >= 0.67) { sr=1; sg=1; sb=1; }
            else if (cx >= 0.33 && cx < 0.67 && cy >= 0.33 && cy < 0.67) { sr=0.5; sg=0.5; sb=0.5; }
            else {
                double r = sqrt((cx-0.5)*(cx-0.5) + (cy-0.5)*(cy-0.5));
                double ang = atan2(cy-0.5, cx-0.5);
                sr = 0.5 + 0.5*sin(ang*8); sg = 0.5 + 0.5*cos(ang*8); sb = 0.5 + 0.5*sin(r*20);
            }
            sr = std::max(0.0, std::min(1.0, sr));
            sg = std::max(0.0, std::min(1.0, sg));
            sb = std::max(0.0, std::min(1.0, sb));

            double sv;
            if (pixel::is_r_at(po, y, x)) sv = sr;
            else if (pixel::is_b_at(po, y, x)) sv = sb;
            else sv = sg;
            int iv = static_cast<int>(sv * max_val + 0.5);
            iv = std::max(0, std::min(max_val, iv));
            write_bayer_pixel(bayer, x, y, W, iv, bit_depth);
        }
    }

    char prefix[64];
    snprintf(prefix, sizeof(prefix), "complex_%dd", bit_depth);
    run_all_algos(dir, prefix, bayer, W, H, pattern, bit_depth);
}

static void generate_gradient(const char* dir, BayerPattern pattern, int bit_depth) {
    const int W = 1600, H = 600;
    size_t bs = pixel::compute_bayer_byte_size(W, H, bit_depth, false);
    std::vector<uint8_t> bayer(bs, 0);
    int max_val = (1 << bit_depth) - 1;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double t = static_cast<double>(x) / (W - 1);
            int iv = static_cast<int>(t * max_val + 0.5);
            iv = std::max(0, std::min(max_val, iv));
            write_bayer_pixel(bayer, x, y, W, iv, bit_depth);
        }
    }

    char prefix[64];
    snprintf(prefix, sizeof(prefix), "gradient_%dbpp", bit_depth);
    run_all_algos(dir, prefix, bayer, W, H, pattern, bit_depth);
}

static void generate_checkerboard(const char* dir, BayerPattern pattern, int bit_depth) {
    const int W = 1600, H = 900;
    size_t bs = pixel::compute_bayer_byte_size(W, H, bit_depth, false);
    std::vector<uint8_t> bayer(bs, 0);
    int max_val = (1 << bit_depth) - 1;

    int cell_sizes[] = {2, 4, 8, 16, 32};
    int num_sizes = sizeof(cell_sizes) / sizeof(cell_sizes[0]);
    int region_w = W / num_sizes;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int region = x / region_w;
            int cs = cell_sizes[std::min(region, num_sizes - 1)];
            int cx = x / cs;
            int cy = y / cs;
            int iv = ((cx + cy) & 1) ? max_val : 0;
            iv = std::max(0, std::min(max_val, iv));
            write_bayer_pixel(bayer, x, y, W, iv, bit_depth);
        }
    }

    char prefix[64];
    snprintf(prefix, sizeof(prefix), "checkerboard_%dbpp", bit_depth);
    run_all_algos(dir, prefix, bayer, W, H, pattern, bit_depth);
}

static void generate_siemens_star(const char* dir, BayerPattern pattern, int bit_depth) {
    const int W = 1200, H = 1200;
    size_t bs = pixel::compute_bayer_byte_size(W, H, bit_depth, false);
    std::vector<uint8_t> bayer(bs, 0);
    int max_val = (1 << bit_depth) - 1;

    double cx = W / 2.0 - 0.5;
    double cy = H / 2.0 - 0.5;
    double radius = W / 2.0;
    int num_spokes = 36;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double dx = x - cx;
            double dy = y - cy;
            double r = std::sqrt(dx * dx + dy * dy);
            double angle = std::atan2(dy, dx);
            double spoke = std::sin(angle * num_spokes * 0.5);
            int iv;
            if (r > radius - 4) {
                iv = max_val / 4;
            } else {
                double envelope = std::max(0.0, 1.0 - r / radius);
                double pval = (spoke > 0) ? 1.0 : 0.0;
                iv = static_cast<int>((0.5 + 0.5 * pval * envelope) * max_val + 0.5);
            }
            iv = std::max(0, std::min(max_val, iv));
            write_bayer_pixel(bayer, x, y, W, iv, bit_depth);
        }
    }

    char prefix[64];
    snprintf(prefix, sizeof(prefix), "siemensstar_%dbpp", bit_depth);
    run_all_algos(dir, prefix, bayer, W, H, pattern, bit_depth);
}

static void generate_zone_plate(const char* dir, BayerPattern pattern, int bit_depth) {
    const int W = 1200, H = 1200;
    size_t bs = pixel::compute_bayer_byte_size(W, H, bit_depth, false);
    std::vector<uint8_t> bayer(bs, 0);
    int max_val = (1 << bit_depth) - 1;

    double cx = W / 2.0 - 0.5;
    double cy = H / 2.0 - 0.5;
    double k = 0.001;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double dx = (x - cx) / cx;
            double dy = (y - cy) / cy;
            double r2 = dx * dx + dy * dy;
            double v = 0.5 + 0.5 * std::cos(k * r2 * W * H);
            int iv = static_cast<int>(v * max_val + 0.5);
            iv = std::max(0, std::min(max_val, iv));
            write_bayer_pixel(bayer, x, y, W, iv, bit_depth);
        }
    }

    char prefix[64];
    snprintf(prefix, sizeof(prefix), "zoneplate_%dbpp", bit_depth);
    run_all_algos(dir, prefix, bayer, W, H, pattern, bit_depth);
}

static void generate_knife_edge(const char* dir, BayerPattern pattern, int bit_depth) {
    const int W = 1600, H = 900;
    size_t bs = pixel::compute_bayer_byte_size(W, H, bit_depth, false);
    std::vector<uint8_t> bayer(bs, 0);
    int max_val = (1 << bit_depth) - 1;

    double dx = 1.0;
    double dy = 0.6;
    double norm = std::sqrt(dx * dx + dy * dy);
    dx /= norm; dy /= norm;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double proj = dx * (x - W / 2.0) + dy * (y - H / 2.0);
            double v = (proj > 0) ? 1.0 : 0.0;
            int iv = static_cast<int>(v * max_val + 0.5);
            iv = std::max(0, std::min(max_val, iv));
            write_bayer_pixel(bayer, x, y, W, iv, bit_depth);
        }
    }

    char prefix[64];
    snprintf(prefix, sizeof(prefix), "knifeedge_%dbpp", bit_depth);
    run_all_algos(dir, prefix, bayer, W, H, pattern, bit_depth);
}

static void generate_color_ramp(const char* dir, BayerPattern pattern, int bit_depth) {
    const int W = 1600, H = 600;
    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    size_t bs = pixel::compute_bayer_byte_size(W, H, bit_depth, false);
    std::vector<uint8_t> bayer(bs, 0);
    int max_val = (1 << bit_depth) - 1;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double t = static_cast<double>(x) / (W - 1);
            double sr, sg, sb;

            if (t < 0.125) {
                double u = t / 0.125;
                sr = 1.0; sg = u; sb = 0.0;
            } else if (t < 0.25) {
                double u = (t - 0.125) / 0.125;
                sr = 1.0 - u; sg = 1.0; sb = 0.0;
            } else if (t < 0.375) {
                double u = (t - 0.25) / 0.125;
                sr = 0.0; sg = 1.0; sb = u;
            } else if (t < 0.5) {
                double u = (t - 0.375) / 0.125;
                sr = 0.0; sg = 1.0 - u; sb = 1.0;
            } else if (t < 0.625) {
                double u = (t - 0.5) / 0.125;
                sr = u; sg = 0.0; sb = 1.0;
            } else if (t < 0.75) {
                double u = (t - 0.625) / 0.125;
                sr = 1.0; sg = 0.0; sb = 1.0 - u;
            } else if (t < 0.875) {
                double u = (t - 0.75) / 0.125;
                sr = 1.0; sg = u; sb = 0.5 * u;
            } else {
                double u = (t - 0.875) / 0.125;
                sr = 1.0 - 0.5 * u; sg = 1.0 - 0.5 * u; sb = 0.5 + 0.5 * u;
            }

            double sv;
            if (pixel::is_r_at(po, y, x)) sv = sr;
            else if (pixel::is_b_at(po, y, x)) sv = sb;
            else sv = sg;
            int iv = static_cast<int>(sv * max_val + 0.5);
            iv = std::max(0, std::min(max_val, iv));
            write_bayer_pixel(bayer, x, y, W, iv, bit_depth);
        }
    }

    char prefix[64];
    snprintf(prefix, sizeof(prefix), "colorramp_%dbpp", bit_depth);
    run_all_algos(dir, prefix, bayer, W, H, pattern, bit_depth);
}

static void generate_packed(const char* dir, BayerPattern pattern) {
    const int bit_depth = 12;
    const int W = 1200, H = 900;
    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    int max_val = (1 << bit_depth) - 1;

    size_t packed_size = pixel::compute_packed_byte_size(W, H, bit_depth);
    std::vector<uint8_t> bayer(packed_size, 0);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double sr = 0, sg = 0, sb = 0;
            int bar = x * 9 / W;
            switch (bar) {
                case 0: sr=1; sg=1; sb=1; break; case 1: sr=1; sg=1; sb=0; break;
                case 2: sr=0; sg=1; sb=1; break; case 3: sr=0; sg=1; sb=0; break;
                case 4: sr=1; sg=0; sb=1; break; case 5: sr=1; sg=0; sb=0; break;
                case 6: sr=0; sg=0; sb=1; break; case 7: sr=0; sg=0; sb=0; break;
                case 8: sr=0.5; sg=0.5; sb=0.5; break;
            }
            double sv;
            if (pixel::is_r_at(po, y, x)) sv = sr;
            else if (pixel::is_b_at(po, y, x)) sv = sb;
            else sv = sg;
            int iv = static_cast<int>(sv * max_val + 0.5);
            iv = std::max(0, std::min(max_val, iv));

            size_t pixel_index = static_cast<size_t>(y) * W + x;
            size_t byte_offset = pixel_index / 2 * 3;
            if ((pixel_index & 1) == 0) {
                bayer[byte_offset]     = (iv >> 4) & 0xFF;
                bayer[byte_offset + 1] = (bayer[byte_offset + 1] & 0xF0) | (iv & 0xF);
            } else {
                bayer[byte_offset + 1] = (bayer[byte_offset + 1] & 0x0F) | ((iv & 0xF) << 4);
                bayer[byte_offset + 2] = (iv >> 4) & 0xFF;
            }
        }
    }

    run_all_algos(dir, "packed_12bpp", bayer, W, H, pattern, bit_depth, true);
}

int main() {
#ifdef _WIN32
    CreateDirectoryA("synthetic_output", nullptr);
#endif
    std::cout << "=== Synthetic Bayer Test Images ===\n" << std::endl;

    BayerPattern p4[] = {BayerPattern::RGGB, BayerPattern::BGGR, BayerPattern::GRBG, BayerPattern::GBRG};
    int bd4[] = {8, 10, 12, 16};

    std::cout << "--- Color bars         | 1280x720  | 4 patterns x 4 bpp x 9 algos = 144 files ---\n";
    for (auto p : p4) {
        for (int bd : bd4) {
            std::cout << "  " << pat_str(p) << " " << bd << "-bit" << std::endl;
            generate_color_bars("synthetic_output", p, bd);
        }
    }

    std::cout << "\n--- Complex scenes      | 1920x1280 | 4 patterns x 8-bit x 9 algos = 36 files ---\n";
    for (auto p : p4) {
        std::cout << "  " << pat_str(p) << std::endl;
        generate_complex("synthetic_output", 1920, 1280, 8, p);
    }

    int gbd[] = {8, 10, 12, 16};
    std::cout << "\n--- Grayscale gradients | 1600x600  | 4 patterns x 4 bpp x 9 algos = 144 files ---\n";
    for (auto p : p4) {
        for (int bd : gbd) {
            std::cout << "  " << pat_str(p) << " " << bd << "-bit" << std::endl;
            generate_gradient("synthetic_output", p, bd);
        }
    }

    int cbd[] = {8, 10, 12};
    std::cout << "\n--- Checkerboard        | 1600x900  | 4 patterns x 3 bpp x 9 algos = 108 files ---\n";
    for (auto p : p4) {
        for (int bd : cbd) {
            std::cout << "  " << pat_str(p) << " " << bd << "-bit" << std::endl;
            generate_checkerboard("synthetic_output", p, bd);
        }
    }

    int sbd[] = {8, 12};
    std::cout << "\n--- Siemens stars       | 1200x1200 | 4 patterns x 2 bpp x 9 algos = 72 files ---\n";
    for (auto p : p4) {
        for (int bd : sbd) {
            std::cout << "  " << pat_str(p) << " " << bd << "-bit" << std::endl;
            generate_siemens_star("synthetic_output", p, bd);
        }
    }

    int zbd[] = {8, 12};
    std::cout << "\n--- Zone plates         | 1200x1200 | 4 patterns x 2 bpp x 9 algos = 72 files ---\n";
    for (auto p : p4) {
        for (int bd : zbd) {
            std::cout << "  " << pat_str(p) << " " << bd << "-bit" << std::endl;
            generate_zone_plate("synthetic_output", p, bd);
        }
    }

    int kbd[] = {8, 10, 12};
    std::cout << "\n--- Knife edges         | 1600x900  | 4 patterns x 3 bpp x 9 algos = 108 files ---\n";
    for (auto p : p4) {
        for (int bd : kbd) {
            std::cout << "  " << pat_str(p) << " " << bd << "-bit" << std::endl;
            generate_knife_edge("synthetic_output", p, bd);
        }
    }

    int rbd[] = {8, 10, 12};
    std::cout << "\n--- Color ramps         | 1600x600  | 4 patterns x 3 bpp x 9 algos = 108 files ---\n";
    for (auto p : p4) {
        for (int bd : rbd) {
            std::cout << "  " << pat_str(p) << " " << bd << "-bit" << std::endl;
            generate_color_ramp("synthetic_output", p, bd);
        }
    }

    std::cout << "\n--- Packed Bayer        | 1200x900  | 4 patterns x 12-bit packed x 9 algos = 36 files ---\n";
    for (auto p : p4) {
        std::cout << "  " << pat_str(p) << std::endl;
        generate_packed("synthetic_output", p);
    }

    std::cout << "\nDone! 828 TIFF files in synthetic_output/" << std::endl;
    return 0;
}
