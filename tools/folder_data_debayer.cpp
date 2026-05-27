#include "im_operator.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstdio>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

using namespace imop;

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static const char* pattern_name(BayerPattern p) {
    switch (p) {
        case BayerPattern::RGGB: return "RGGB";
        case BayerPattern::BGGR: return "BGGR";
        case BayerPattern::GRBG: return "GRBG";
        case BayerPattern::GBRG: return "GBRG";
        default: return "???";
    }
}

static void write16le(FILE* f, uint16_t v) {
    fputc(v & 0xFF, f);
    fputc((v >> 8) & 0xFF, f);
}
static void write32le(FILE* f, uint32_t v) {
    fputc(v & 0xFF, f);
    fputc((v >> 8) & 0xFF, f);
    fputc((v >> 16) & 0xFF, f);
    fputc((v >> 24) & 0xFF, f);
}

static bool save_tiff(const std::string& path, const uint8_t* rgb,
                       int width, int height, int bit_depth) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    bool use16 = (bit_depth > 8);
    int bps = use16 ? 16 : 8;
    size_t sample_count = static_cast<size_t>(width) * height * 3;
    size_t strip_bytes = use16 ? sample_count * 2 : sample_count;
    uint32_t rows_per_strip = static_cast<uint32_t>(height);

    fputc('I', f); fputc('I', f);
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
        fwrite(scaled.data(), 1, strip_bytes, f);
    } else {
        fwrite(rgb, 1, strip_bytes, f);
    }
    fclose(f);
    return true;
}

static std::string sanitize_filename(const std::string& s) {
    std::string result;
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '-') {
            result += c;
        } else {
            result += '_';
        }
    }
    size_t pos;
    while ((pos = result.find("..")) != std::string::npos)
        result.replace(pos, 2, "__");
    return result;
}

static const DemosaicAlgorithm kAllAlgos[] = {
    DemosaicAlgorithm::SUPER_FAST,
    DemosaicAlgorithm::HQLI,
    DemosaicAlgorithm::MG,
    DemosaicAlgorithm::L7,
    DemosaicAlgorithm::DFPD,
    DemosaicAlgorithm::AHD,
    DemosaicAlgorithm::AMAZE,
    DemosaicAlgorithm::RCD,
    DemosaicAlgorithm::PRISM,
};
static constexpr int kNumAlgos = sizeof(kAllAlgos) / sizeof(kAllAlgos[0]);

static std::string algo_label(DemosaicAlgorithm a) {
    return Demosaic::algorithm_name(a);
}

static std::vector<std::string> discover_files(const std::string& dir) {
    std::vector<std::string> result;

#ifdef _WIN32
    std::string search_pattern = dir + "\\*.raw";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(search_pattern.c_str(), &fd);
    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "ERROR: Cannot access directory or no .raw files found: " << dir << std::endl;
        return result;
    }
    do {
        std::string name = fd.cFileName;
        if (ends_with(name, ".raw")) {
            result.push_back(dir + "\\" + name);
        }
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);
#else
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        std::cerr << "ERROR: Cannot open directory: " << dir << std::endl;
        return result;
    }
    struct dirent* entry;
    while ((entry = readdir(dp)) != NULL) {
        std::string name = entry->d_name;
        if (ends_with(name, ".raw")) {
            result.push_back(dir + "/" + name);
        }
    }
    closedir(dp);
#endif

    return result;
}

static void ensure_output_dir(const std::string& dir) {
#ifdef _WIN32
    CreateDirectoryA(dir.c_str(), nullptr);
#else
    mkdir(dir.c_str(), 0755);
#endif
}

static void process_file(const std::string& filepath, const std::string& output_dir) {
    std::string fname = filepath;
    auto slash_pos = fname.find_last_of("/\\");
    if (slash_pos != std::string::npos) fname = fname.substr(slash_pos + 1);

    std::cout << "--- " << fname << " ---" << std::endl;

    std::ifstream f(filepath, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "  ERROR: Cannot open file" << std::endl;
        return;
    }

    size_t file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> raw_data(file_size);
    f.read(reinterpret_cast<char*>(raw_data.data()), file_size);
    f.close();

    DataInfo info = Demosaic::analyze_data(raw_data.data(), file_size);

    if (info.detected_bit_depth <= 0) {
        std::cerr << "  ERROR: Cannot detect bit depth" << std::endl;
        return;
    }
    if (info.suggested_width <= 0 || info.suggested_height <= 0) {
        std::cerr << "  ERROR: Cannot determine image dimensions";
        if (!info.possible_dimensions.empty()) {
            std::cerr << " (candidates:";
            for (auto& d : info.possible_dimensions)
                std::cerr << " " << d.first << "x" << d.second;
            std::cerr << ")";
        }
        std::cerr << std::endl;
        return;
    }

    int width = info.suggested_width;
    int height = info.suggested_height;
    int bit_depth = info.detected_bit_depth;
    bool is_packed = info.is_packed;

    BayerPattern guessed_pattern = Demosaic::guess_pattern(
        raw_data.data(), width, height, bit_depth, is_packed);

    std::cout << "    Detected: " << width << "x" << height
              << " | " << bit_depth << "-bit"
              << (is_packed ? " | packed" : "")
              << " | " << pattern_name(guessed_pattern) << std::endl;

    Demosaic dm;

    for (int ai = 0; ai < kNumAlgos; ai++) {
        DemosaicAlgorithm algo = kAllAlgos[ai];
        int ws = Demosaic::algorithm_window_size(algo);

        if (width < ws + 1 || height < ws + 1) {
            std::cout << "  [" << algo_label(algo) << "] SKIP: image too small for "
                      << ws << "x" << ws << " window" << std::endl;
            continue;
        }

        size_t rgb_bytes = static_cast<size_t>(width) * height * 3 *
                           ((bit_depth > 8) ? 2 : 1);
        std::vector<uint8_t> rgb(rgb_bytes);

        auto start = std::chrono::high_resolution_clock::now();

        auto res = dm.process(raw_data.data(), rgb.data(),
                                   width, height,
                                   guessed_pattern, algo, bit_depth,
                                   is_packed);

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (!ok(res)) {
            std::cerr << "  [" << algo_label(algo) << "] FAIL: "
                      << demosaic_error_message(res) << std::endl;
            continue;
        }

        double mpix = static_cast<double>(width) * height / 1e6;
        std::cout << "  [" << algo_label(algo) << "] "
                  << dur.count() << "ms ("
                  << std::fixed << std::setprecision(2)
                  << (dur.count() > 0 ? mpix / (dur.count() / 1000.0) : 0.0)
                  << " MP/s)" << std::endl;

        std::string tiff_name = output_dir + "/" + fname + "_" +
                                sanitize_filename(algo_label(algo)) + ".tiff";
        if (save_tiff(tiff_name, rgb.data(), width, height, bit_depth)) {
            std::cout << "       -> " << tiff_name << std::endl;
        } else {
            std::cerr << "       ERROR: Failed to save " << tiff_name << std::endl;
        }
    }
    std::cout << std::endl;
}

static void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <input_dir> <output_dir>" << std::endl;
    std::cout << std::endl;
    std::cout << "  input_dir   Directory containing .raw Bayer files" << std::endl;
    std::cout << "  output_dir  Directory to save debayered TIFF images" << std::endl;
    std::cout << std::endl;
    std::cout << "All parameters (width, height, bit depth, Bayer pattern, packed)" << std::endl;
    std::cout << "are auto-detected via analyze_data() and guess_pattern()." << std::endl;
    std::cout << "Any .raw filename is supported." << std::endl;
    std::cout << std::endl;
    std::cout << "Algorithms:";
    for (int i = 0; i < kNumAlgos; i++)
        std::cout << " " << algo_label(kAllAlgos[i]);
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::ios::sync_with_stdio(false);

    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    auto files = discover_files(input_dir);
    if (files.empty()) {
        std::cerr << "ERROR: No .raw files found in " << input_dir << std::endl;
        return 1;
    }

    ensure_output_dir(output_dir);

    std::cout << "============================================" << std::endl;
    std::cout << "  Folder Data Processor" << std::endl;
    std::cout << "  Input:  " << input_dir << std::endl;
    std::cout << "  Output: " << output_dir << std::endl;
    std::cout << "  Files:  " << files.size() << std::endl;
    std::cout << "============================================" << std::endl;

    for (auto& filepath : files) {
        process_file(filepath, output_dir);
    }

    std::cout << "============================================" << std::endl;
    std::cout << "  All files processed." << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
