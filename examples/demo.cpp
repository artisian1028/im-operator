#include "im_operator.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace imop;

// ============================================================
// 辅助：保存 8-bit RGB 为 BMP 文件（用于可视化验证）
// ============================================================

#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t file_type = 0x4D42;
    uint32_t file_size = 0;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t offset_data = 54;
};

struct BMPInfoHeader {
    uint32_t size = 40;
    int32_t width = 0;
    int32_t height = 0;
    uint16_t planes = 1;
    uint16_t bit_count = 24;
    uint32_t compression = 0;
    uint32_t size_image = 0;
    int32_t x_pixels_per_meter = 0;
    int32_t y_pixels_per_meter = 0;
    uint32_t colors_used = 0;
    uint32_t colors_important = 0;
};
#pragma pack(pop)

bool save_bmp(const std::string& filename, const uint8_t* rgb_data,
              int width, int height, int bit_depth) {
    std::ofstream f(filename, std::ios::binary);
    if (!f) return false;

    int row_size = ((width * 3 + 3) / 4) * 4;
    int image_size = row_size * height;

    BMPFileHeader fh;
    BMPInfoHeader ih;
    fh.file_size = 54 + image_size;
    ih.width = width;
    ih.height = height;
    ih.size_image = image_size;

    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    f.write(reinterpret_cast<const char*>(&ih), sizeof(ih));

    std::vector<uint8_t> row(row_size, 0);

    int max_val = (1 << bit_depth) - 1;
    bool msb_aligned = false;
    uint32_t actual_max = 0;

    if (bit_depth > 8) {
        const uint16_t* d16 = reinterpret_cast<const uint16_t*>(rgb_data);
        size_t n = static_cast<size_t>(width) * height * 3;
        for (size_t i = 0; i < n; i++) {
            if (d16[i] > actual_max) actual_max = d16[i];
        }
        msb_aligned = (actual_max > static_cast<uint32_t>(max_val));
    }

    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int r, g, b;
            if (bit_depth > 8) {
                const uint16_t* d16 = reinterpret_cast<const uint16_t*>(rgb_data);
                size_t idx = (static_cast<size_t>(y) * width + x) * 3;
                int rv = d16[idx], gv = d16[idx + 1], bv = d16[idx + 2];
                uint32_t div = msb_aligned ? actual_max : static_cast<uint32_t>(max_val);
                r = static_cast<int>((static_cast<uint64_t>(rv) * 255 + div / 2) / div);
                g = static_cast<int>((static_cast<uint64_t>(gv) * 255 + div / 2) / div);
                b = static_cast<int>((static_cast<uint64_t>(bv) * 255 + div / 2) / div);
            } else {
                r = rgb_data[(static_cast<size_t>(y) * width + x) * 3];
                g = rgb_data[(static_cast<size_t>(y) * width + x) * 3 + 1];
                b = rgb_data[(static_cast<size_t>(y) * width + x) * 3 + 2];
            }
            r = std::clamp(r, 0, 255);
            g = std::clamp(g, 0, 255);
            b = std::clamp(b, 0, 255);
            row[x * 3 + 0] = static_cast<uint8_t>(b);
            row[x * 3 + 1] = static_cast<uint8_t>(g);
            row[x * 3 + 2] = static_cast<uint8_t>(r);
        }
        f.write(reinterpret_cast<const char*>(row.data()), row_size);
    }
    return f.good();
}

// ============================================================
// 辅助：计时工具
// ============================================================

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
    const char* label_;
public:
    Timer(const char* label) : label_(label) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << "  [" << label_ << "] " << ms.count() << " ms" << std::endl;
    }
};

// ============================================================
// 辅助：生成模拟 Bayer 测试数据
// ============================================================

std::vector<uint8_t> generate_test_bayer(int width, int height, int bit_depth, bool packed) {
    size_t bayer_size;
    if (packed) {
        bayer_size = imop::pixel::compute_packed_byte_size(width, height, bit_depth);
    } else {
        bayer_size = imop::pixel::compute_bayer_byte_size(width, height, bit_depth, false);
    }
    std::vector<uint8_t> bayer(bayer_size);

    int max = (1 << bit_depth) - 1;

    if (!packed && bit_depth <= 8) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int val = ((x * 7 + y * 13 + (x ^ y) * 3) % max);
                bayer[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(val);
            }
        }
    }
    else if (!packed && bit_depth > 8) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int val = ((x * 7 + y * 13 + (x ^ y) * 3) % max);
                size_t offset = (static_cast<size_t>(y) * width + x) * 2;
                imop::pixel::write_u16(bayer.data(), offset, static_cast<uint16_t>(val));
            }
        }
    }
    else {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int val = ((x * 7 + y * 13 + (x ^ y) * 3) % max);
                (void)val;
            }
        }
    }
    return bayer;
}

// ============================================================
// 辅助：从文件加载 RAW 数据
// ============================================================

std::vector<uint8_t> load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// ============================================================
// Demo 1: 基础用法 — 单帧去马赛克
// ============================================================

void demo_basic_usage() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 1: 基础单帧去马赛克" << std::endl;
    std::cout << "========================================" << std::endl;

    const int width = 1920, height = 1080;
    const int bit_depth = 8;

    auto bayer = generate_test_bayer(width, height, bit_depth, false);
    std::vector<uint8_t> rgb(width * height * 3);

    Demosaic dm;
    {
        Timer t("process DFPD 8bit");
        auto err = dm.process(bayer.data(), rgb.data(),
                                   width, height,
                                   BayerPattern::RGGB,
                                   DemosaicAlgorithm::DFPD,
                                   bit_depth);
        if (!ok(err)) {
            std::cerr << "  错误: " << demosaic_error_message(err) << std::endl;
            return;
        }
    }
    std::cout << "  输出 RGB 大小: " << rgb.size() << " 字节" << std::endl;
    std::cout << "  结果: 成功" << std::endl;
}

// ============================================================
// Demo 2: 12-bit 图像处理
// ============================================================

void demo_12bit_processing() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 2: 12-bit 高位深处理" << std::endl;
    std::cout << "========================================" << std::endl;

    const int width = 4024, height = 3032;
    const int bit_depth = 12;

    auto bayer = generate_test_bayer(width, height, bit_depth, false);
    size_t rgb_size = static_cast<size_t>(width) * height * 3 * 2;
    std::vector<uint8_t> rgb(rgb_size);

    Demosaic dm;
    {
        Timer t("process DFPD 12bit");
        auto err = dm.process(bayer.data(), rgb.data(),
                                   width, height,
                                   BayerPattern::RGGB,
                                   DemosaicAlgorithm::DFPD,
                                   bit_depth);
        if (!ok(err)) {
            std::cerr << "  错误: " << demosaic_error_message(err) << std::endl;
            return;
        }
    }

    const uint16_t* rgb16 = reinterpret_cast<const uint16_t*>(rgb.data());
    std::cout << "  RGB[0] = (" << rgb16[0] << ", " << rgb16[1] << ", " << rgb16[2] << ")" << std::endl;
    std::cout << "  输出 16-bit 大小: " << rgb.size() << " 字节" << std::endl;
}

// ============================================================
// Demo 3: 使用 ImageBuffer 自动管理内存
// ============================================================

void demo_image_buffer() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 3: ImageBuffer 自动管理内存" << std::endl;
    std::cout << "========================================" << std::endl;

    const int width = 1006, height = 758;
    const int bit_depth = 10;

    ImageBuffer bayer;
    bayer.data = generate_test_bayer(width, height, bit_depth, false);
    bayer.width = width;
    bayer.height = height;
    bayer.channels = 1;
    bayer.bit_depth = bit_depth;

    ImageBuffer rgb;
    Demosaic dm;

    {
        Timer t("process ImageBuffer HQLI 10bit");
        auto err = dm.process(bayer, rgb, BayerPattern::BGGR, DemosaicAlgorithm::HQLI);
        if (!ok(err)) {
            std::cerr << "  错误: " << demosaic_error_message(err) << std::endl;
            return;
        }
    }

    std::cout << "  自动分配: rgb.width=" << rgb.width
              << " rgb.height=" << rgb.height
              << " rgb.channels=" << rgb.channels
              << " rgb.bit_depth=" << rgb.bit_depth
              << " rgb.data.size()=" << rgb.data.size() << std::endl;
}

// ============================================================
// Demo 4: 所有算法对比
// ============================================================

void demo_all_algorithms() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 4: 全部 9 种算法对比" << std::endl;
    std::cout << "========================================" << std::endl;

    const int width = 1920, height = 1080, bit_depth = 8;
    auto bayer = generate_test_bayer(width, height, bit_depth, false);

    std::vector<DemosaicAlgorithm> algorithms = {
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

    std::cout << std::left << std::setw(16) << "Algorithm"
              << std::setw(10) << "Window"
              << std::setw(14) << "Time"
              << "Status" << std::endl;
    std::cout << std::string(56, '-') << std::endl;

    Demosaic dm;
    for (auto algo : algorithms) {
        std::vector<uint8_t> rgb(width * height * 3);

        auto start = std::chrono::high_resolution_clock::now();
        auto err = dm.process(bayer.data(), rgb.data(),
                                   width, height,
                                   BayerPattern::RGGB, algo, bit_depth);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << std::left << std::setw(16) << Demosaic::algorithm_name(algo)
                  << std::setw(10) << (std::to_string(Demosaic::algorithm_window_size(algo)) + "x" + std::to_string(Demosaic::algorithm_window_size(algo)))
                  << std::setw(14) << (std::to_string(ms.count()) + " ms")
                  << (ok(err) ? "OK" : demosaic_error_message(err))
                  << std::endl;
    }
}

// ============================================================
// Demo 5: 使用自由函数（无需 Demosaic 类）
// ============================================================

void demo_free_functions() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 5: 自由函数调用（无类封装）" << std::endl;
    std::cout << "========================================" << std::endl;

    const int width = 640, height = 480, bit_depth = 8;
    auto bayer = generate_test_bayer(width, height, bit_depth, false);
    std::vector<uint8_t> rgb(width * height * 3);

    auto err = demosaic_cpu(bayer.data(), rgb.data(),
                                   width, height,
                                   BayerPattern::GRBG,
                                   DemosaicAlgorithm::MG,
                                   bit_depth);
    if (ok(err)) {
        std::cout << "  demosaic_cpu(MG, GRBG) 成功" << std::endl;
    }

    err = process_mg(bayer.data(), rgb.data(),
                     width, height,
                     BayerPattern::GRBG,
                     bit_depth);
    if (ok(err)) {
        std::cout << "  process_mg() 成功" << std::endl;
    }

    std::cout << "  MG 算法窗口: " << algorithm_window_size(DemosaicAlgorithm::MG) << "x"
              << algorithm_window_size(DemosaicAlgorithm::MG) << std::endl;
}

// ============================================================
// Demo 6: 系统能力查询
// ============================================================

void demo_capability_query() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 6: 系统能力查询" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "  AVX2 可用:    " << (has_avx2() ? "Yes" : "No") << std::endl;
    std::cout << "  CUDA 可用:    " << (has_cuda() ? "Yes" : "No") << std::endl;
    std::cout << "  CUDA 设备:    " << cuda_device_name() << std::endl;
    std::cout << "  CPU 并发数:   " << compute_hardware_concurrency() << std::endl;

    std::cout << "  算法列表:" << std::endl;
    std::array<DemosaicAlgorithm, 9> algos = {
        DemosaicAlgorithm::SUPER_FAST, DemosaicAlgorithm::HQLI, DemosaicAlgorithm::MG,
        DemosaicAlgorithm::L7, DemosaicAlgorithm::DFPD, DemosaicAlgorithm::AHD,
        DemosaicAlgorithm::AMAZE, DemosaicAlgorithm::RCD, DemosaicAlgorithm::PRISM,
    };
    for (auto a : algos) {
        std::cout << "    " << algorithm_name(a)
                  << " (窗口 " << algorithm_window_size(a) << "x" << algorithm_window_size(a) << ")" << std::endl;
    }
}

// ============================================================
// Demo 7: 错误处理
// ============================================================

void demo_error_handling() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 7: 错误处理" << std::endl;
    std::cout << "========================================" << std::endl;

    Demosaic dm;
    std::vector<uint8_t> bayer(100);
    std::vector<uint8_t> rgb(300);

    auto test = [&](const char* desc, auto call) {
        auto err = call();
        std::cout << "  " << desc << ": "
                  << (ok(err) ? "OK" : demosaic_error_message(err))
                  << std::endl;
    };

    test("空指针输入", [&]() {
        return dm.process(nullptr, rgb.data(), 100, 100,
                               BayerPattern::RGGB, DemosaicAlgorithm::DFPD, 8);
    });
    test("零宽度", [&]() {
        return dm.process(bayer.data(), rgb.data(), 0, 100,
                               BayerPattern::RGGB, DemosaicAlgorithm::DFPD, 8);
    });
    test("无效位深", [&]() {
        return dm.process(bayer.data(), rgb.data(), 100, 100,
                               BayerPattern::RGGB, DemosaicAlgorithm::DFPD, 20);
    });
    test("有效参数", [&]() {
        std::vector<uint8_t> b(100 * 100);
        std::vector<uint8_t> r(100 * 100 * 3);
        return dm.process(b.data(), r.data(), 100, 100,
                               BayerPattern::RGGB, DemosaicAlgorithm::SUPER_FAST, 8);
    });
}

// ============================================================
// Demo 8: RAW 文件自动分析
// ============================================================

void demo_raw_analysis(const std::string& test_data_dir) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 8: RAW 文件自动分析" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<std::string> test_files = {
        "Image_20260510144302933_w1006_h758_pBayerRG8.raw",
        "Image_20260510144315897_w1006_h758_pBayerRG12.raw",
        "Image_20260510144329680_w1006_h758_pBayerRG10_Packed.raw",
    };

    for (auto& fname : test_files) {
        std::string path = test_data_dir + "/" + fname;
        auto data = load_file(path);
        if (data.empty()) {
            std::cout << "  [跳过] 文件不存在: " << path << std::endl;
            continue;
        }

        DataInfo info = Demosaic::analyze_data(data.data(), data.size());
        std::cout << "\n  文件: " << fname << std::endl;
        std::cout << "    大小: " << data.size() << " 字节" << std::endl;
        std::cout << "    检测位深: " << info.detected_bit_depth << " bit" << std::endl;
        std::cout << "    建议分辨率: " << info.suggested_width << "x" << info.suggested_height << std::endl;
        std::cout << "    像素数: " << info.pixel_count << std::endl;
        std::cout << "    值范围: [" << info.min_value << ", " << info.max_value << "]" << std::endl;
        if (!info.possible_dimensions.empty()) {
            std::cout << "    可能分辨率:" << std::endl;
            for (auto& d : info.possible_dimensions) {
                std::cout << "      " << d.first << " x " << d.second << std::endl;
            }
        }

        if (info.suggested_width > 0 && info.suggested_height > 0) {
            auto pattern = Demosaic::guess_pattern(data.data(),
                                                   info.suggested_width,
                                                   info.suggested_height,
                                                   info.detected_bit_depth,
                                                   info.is_packed);
            const char* pname = "RGGB";
            if (pattern == BayerPattern::BGGR) pname = "BGGR";
            else if (pattern == BayerPattern::GRBG) pname = "GRBG";
            else if (pattern == BayerPattern::GBRG) pname = "GBRG";
            std::cout << "    猜测 Bayer 模式: " << pname << std::endl;
        }
    }
}

// ============================================================
// Demo 9: CUDA 批量帧处理
// ============================================================

void demo_cuda_batch() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 9: CUDA 批量帧流水线处理" << std::endl;
    std::cout << "========================================" << std::endl;

    if (!has_cuda()) {
        std::cout << "  [跳过] CUDA 不可用" << std::endl;
        return;
    }

    const int N = 50;
    const int width = 1920, height = 1080, bit_depth = 8;
    size_t bayer_size = static_cast<size_t>(width) * height;
    size_t rgb_size = bayer_size * 3;

    std::vector<std::vector<uint8_t>> bayer_frames(N);
    std::vector<std::vector<uint8_t>> rgb_frames(N);
    std::vector<const uint8_t*> bayer_ptrs(N);
    std::vector<uint8_t*> rgb_ptrs(N);

    for (int i = 0; i < N; i++) {
        bayer_frames[i] = generate_test_bayer(width, height, bit_depth, false);
        rgb_frames[i].resize(rgb_size);
        bayer_ptrs[i] = bayer_frames[i].data();
        rgb_ptrs[i] = rgb_frames[i].data();
    }

    {
        Timer t("CUDA batch 50 frames DFPD");
        auto err = demosaic_cuda_batch(
            bayer_ptrs.data(), rgb_ptrs.data(),
            N, width, height,
            BayerPattern::RGGB, DemosaicAlgorithm::DFPD, bit_depth
        );
        if (!ok(err)) {
            std::cerr << "  错误: " << demosaic_error_message(err) << std::endl;
            return;
        }
    }

    std::cout << "  帧数: " << N << std::endl;
    std::cout << "  每帧分辨率: " << width << "x" << height << std::endl;
    std::cout << "  总处理像素: " << (1LL * width * height * N) << std::endl;
}

// ============================================================
// Demo 10: 使用像素工具进行自定义处理
// ============================================================

void demo_pixel_utils() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 10: 像素工具底层操作" << std::endl;
    std::cout << "========================================" << std::endl;

    const int width = 64, height = 64, bit_depth = 12;
    auto bayer = generate_test_bayer(width, height, bit_depth, false);

    auto po = PatternOffsets::from_pattern(BayerPattern::RGGB);
    std::cout << "  RGGB 模式偏移:" << std::endl;
    std::cout << "    R: (" << po.r_row << ", " << po.r_col << ")" << std::endl;
    std::cout << "    B: (" << po.b_row << ", " << po.b_col << ")" << std::endl;

    int pixel_count = 0;
    int r_count = 0, g_count = 0, b_count = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int val = pixel::get_raw(bayer.data(), x, y, width, bit_depth, false);
            if (val < 0) continue;
            pixel_count++;
            if (pixel::is_r_at(po, y, x)) r_count++;
            else if (pixel::is_b_at(po, y, x)) b_count++;
            else g_count++;
        }
    }

    std::cout << "  总像素: " << pixel_count << std::endl;
    std::cout << "  R: " << r_count << "  G: " << g_count << "  B: " << b_count << std::endl;

    size_t bayer_bytes = pixel::compute_bayer_byte_size(width, height, bit_depth, false);
    size_t rgb_bytes = pixel::compute_rgb_byte_size(width, height, bit_depth);
    size_t packed_bytes = pixel::compute_packed_byte_size(width, height, bit_depth);
    std::cout << "  Bayer 字节数 (unpacked): " << bayer_bytes << std::endl;
    std::cout << "  RGB 字节数:               " << rgb_bytes << std::endl;
    std::cout << "  Bayer 字节数 (packed):    " << packed_bytes << std::endl;

    int max_val = pixel::safe_max_val(12);
    std::cout << "  12-bit max value: " << max_val << std::endl;
}

// ============================================================
// Demo 11: 强制 CPU 模式
// ============================================================

void demo_cpu_only() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 11: 强制 CPU 处理模式" << std::endl;
    std::cout << "========================================" << std::endl;

    const int width = 1920, height = 1080, bit_depth = 8;
    auto bayer = generate_test_bayer(width, height, bit_depth, false);
    std::vector<uint8_t> rgb(width * height * 3);

    Demosaic dm;

    {
        Timer t("process_cpu DFPD");
        auto err = dm.process_cpu(bayer.data(), rgb.data(),
                                       width, height,
                                       BayerPattern::RGGB,
                                       DemosaicAlgorithm::DFPD,
                                       bit_depth);
        if (ok(err)) {
            std::cout << "  CPU 处理成功（跳过 CUDA）" << std::endl;
        }
    }

    {
        Timer t("process(HQLI)");
        auto err = dm.process(bayer.data(), rgb.data(),
                                   width, height,
                                   BayerPattern::RGGB,
                                   DemosaicAlgorithm::HQLI,
                                   bit_depth);
        if (ok(err)) {
            std::cout << "  process(HQLI) 调用成功" << std::endl;
        }
    }
}

// ============================================================
// Demo 12: Packed 格式处理
// ============================================================

void demo_packed_format() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 12: Packed 格式支持" << std::endl;
    std::cout << "========================================" << std::endl;

    const int width = 1006, height = 758;

    std::string test_data_dir = "test_data";
    std::string packed_file = test_data_dir + "/Image_20260510144329680_w1006_h758_pBayerRG10_Packed.raw";
    auto data = load_file(packed_file);

    if (data.empty()) {
        std::cout << "  [跳过] packed 测试文件不存在: " << packed_file << std::endl;
        std::cout << "  Packed 格式使用说明:" << std::endl;
        std::cout << "    只需在 process() 中设置 is_packed=true，bit_depth=10/12/14/16 即可" << std::endl;
        std::cout << "    示例代码:" << std::endl;
        std::cout << "      dm.process(bayer, rgb, w, h, pattern, algo, 10, true);" << std::endl;
        return;
    }

    std::cout << "  文件: " << packed_file << std::endl;
    std::cout << "  大小: " << data.size() << " 字节" << std::endl;

    size_t rgb_size = static_cast<size_t>(width) * height * 3;
    if (10 > 8) rgb_size *= 2;
    std::vector<uint8_t> rgb(rgb_size);

    Demosaic dm;
    {
        Timer t("Packed 10bit DFPD");
        auto err = dm.process(data.data(), rgb.data(),
                                   width, height,
                                   BayerPattern::RGGB,
                                   DemosaicAlgorithm::DFPD,
                                   10, true);
        if (ok(err)) {
            std::cout << "  Packed 10-bit 处理成功" << std::endl;
        } else {
            std::cerr << "  错误: " << demosaic_error_message(err) << std::endl;
        }
    }
}

// ============================================================
// Demo 13: 输入验证
// ============================================================

void demo_input_validation() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 13: 输入验证工具" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<uint8_t> buf(100);

    auto check = [](int w, int h, int bd) {
        std::vector<uint8_t> b(w * h);
        std::vector<uint8_t> r(w * h * 3);
        auto err = validate_demosaic_inputs(b.data(), r.data(), w, h, bd);
        const char* status = ok(err) ? "OK" : demosaic_error_message(err);
        std::cout << "  (" << w << "x" << h << ", " << bd << "bit): " << status << std::endl;
    };

    check(100, 100, 8);
    check(0, 100, 8);
    check(100, 0, 8);
    check(100, 100, 0);
    check(100, 100, 17);
    check(100, 100, 16);

    auto err = validate_demosaic_inputs(nullptr, buf.data(), 100, 100, 8);
    std::cout << "  (nullptr, 100x100, 8bit): " << demosaic_error_message(err) << std::endl;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "╔══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        Demosaic 库 API 演示程序 v2.0.0               ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════╝" << std::endl;

    std::string test_data_dir = "test_data";

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--test-data") == 0 && i + 1 < argc) {
            test_data_dir = argv[++i];
        }
    }

    std::cout << "\n系统环境:" << std::endl;
    std::cout << "  AVX2: " << (has_avx2() ? "ON" : "OFF") << std::endl;
    std::cout << "  CUDA: " << (has_cuda() ? "ON (" + std::string(cuda_device_name()) + ")" : "OFF") << std::endl;
    std::cout << "  CPU 线程: " << compute_hardware_concurrency() << std::endl;

    demo_basic_usage();
    demo_12bit_processing();
    demo_image_buffer();
    demo_all_algorithms();
    demo_free_functions();
    demo_capability_query();
    demo_error_handling();
    demo_raw_analysis(test_data_dir);
    demo_cuda_batch();
    demo_pixel_utils();
    demo_cpu_only();
    demo_packed_format();
    demo_input_validation();

    std::cout << "\n========================================" << std::endl;
    std::cout << "全部 Demo 执行完毕!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
