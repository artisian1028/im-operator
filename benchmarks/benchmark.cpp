#include "im_operator.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#endif

using namespace imop;

// ============================================================================
// Named constants
// ============================================================================

namespace constants {

constexpr double kTimeBudgetSec       = 2.0;
constexpr int    kMinIterations       = 5;
constexpr int    kMaxIterations       = 200;
constexpr int    kGpuWarmupIters      = 5;
constexpr int    kCpuWarmupIters      = 2;
constexpr double kIqrMultiplier       = 1.5;
constexpr double kOutlierFallbackRatio = 0.70;
constexpr int    kSymmetricTrimTenth  = 10;
constexpr double kMegaPixel           = 1e6;
constexpr double kFastSpeedupThreshold = 2.0;
constexpr size_t kBatchMemoryLimitGB  = 4;
constexpr int    kBatchMinFrames      = 10;
constexpr int    kBatchMaxFrames      = 500;

}

// ============================================================================
// Statistics
// ============================================================================

struct BenchStats {
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double median_ms = 0.0;
    double stddev_ms = 0.0;
    double cv_pct = 0.0;
    double mpix_per_sec = 0.0;
    double ci95_low = 0.0;
    double ci95_high = 0.0;
    int iterations = 0;
    int outliers_removed = 0;
};

static double t_critical_95(int df) {
    static const double table[] = {
        12.706, 4.303, 3.182, 2.776, 2.571,
         2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060,
         2.056, 2.052, 2.048, 2.045, 2.042,
    };
    if (df <= 0) return 999.0;
    if (df <= 30) return table[df - 1];
    if (df <= 40) return 2.021;
    if (df <= 50) return 2.009;
    if (df <= 60) return 2.000;
    if (df <= 80) return 1.990;
    if (df <= 100) return 1.984;
    if (df <= 200) return 1.972;
    return 1.960;
}

static double compute_median(const std::vector<double>& sorted) {
    size_t n = sorted.size();
    if (n == 0) return 0.0;
    size_t mid = n / 2;
    if (n % 2 == 1) {
        return sorted[mid];
    } else {
        return (sorted[mid - 1] + sorted[mid]) / 2.0;
    }
}

// Quartile computation uses exclusive-median method (Tukey Type 2):
//   Q1 = median of lower half (excluding median if n is odd)
//   Q3 = median of upper half (excluding median if n is odd)
static double compute_median_subrange(const std::vector<double>& sorted,
                                       size_t lo, size_t hi) {
    if (lo >= hi) return 0.0;
    size_t n = hi - lo;
    size_t mid = lo + n / 2;
    if (n % 2 == 1) {
        return sorted[mid];
    } else {
        return (sorted[mid - 1] + sorted[mid]) / 2.0;
    }
}

static BenchStats compute_stats(std::vector<double>& raw_times,
                                 int width, int height) {
    BenchStats st;

    if (raw_times.empty()) return st;

    std::vector<double> sorted = raw_times;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();

    double q1 = compute_median_subrange(sorted, 0, n / 2);
    double q3 = compute_median_subrange(sorted, (n + 1) / 2, n);
    double iqr = q3 - q1;
    double lower_fence = q1 - constants::kIqrMultiplier * iqr;
    double upper_fence = q3 + constants::kIqrMultiplier * iqr;

    std::vector<double> filtered;
    filtered.reserve(n);
    std::copy_if(sorted.begin(), sorted.end(), std::back_inserter(filtered),
                 [lower_fence, upper_fence](double t) {
                     return t >= lower_fence && t <= upper_fence;
                 });

    if (filtered.size() < n * constants::kOutlierFallbackRatio) {
        size_t trim = n / constants::kSymmetricTrimTenth;
        if (trim * 2 >= n) trim = 0;
        using diff_t = std::vector<double>::difference_type;
        filtered.assign(sorted.begin() + static_cast<diff_t>(trim),
                        sorted.end() - static_cast<diff_t>(trim));
    }

    st.outliers_removed = static_cast<int>(n - filtered.size());
    size_t fn = filtered.size();
    if (fn == 0) {
        filtered = sorted;
        fn = sorted.size();
        st.outliers_removed = 0;
    }

    double sum = std::accumulate(filtered.begin(), filtered.end(), 0.0);
    st.mean_ms = sum / static_cast<double>(fn);
    st.min_ms = filtered.front();
    st.max_ms = filtered.back();
    st.median_ms = compute_median(filtered);
    st.iterations = static_cast<int>(fn);

    double sq_sum = std::inner_product(filtered.begin(), filtered.end(),
                                        filtered.begin(), 0.0);
    double mean_sq = st.mean_ms * st.mean_ms;
    st.stddev_ms = (fn > 1)
        ? std::sqrt((sq_sum / static_cast<double>(fn)) - mean_sq)
          * std::sqrt(static_cast<double>(fn) / static_cast<double>(fn - 1))
        : 0.0;
    st.cv_pct = (st.mean_ms > 0.0) ? (st.stddev_ms / st.mean_ms * 100.0) : 0.0;

    double t_val = t_critical_95(static_cast<int>(fn) - 1);
    double ci_margin = t_val * st.stddev_ms / std::sqrt(static_cast<double>(fn));
    st.ci95_low = st.mean_ms - ci_margin;
    st.ci95_high = st.mean_ms + ci_margin;

    double mpix = static_cast<double>(width) * height / constants::kMegaPixel;
    st.mpix_per_sec = (st.mean_ms > 0.0) ? mpix / (st.mean_ms / 1000.0) : 0.0;

    return st;
}

// ============================================================================
// Process priority / CPU frequency hint
// ============================================================================

static void set_benchmark_priority() {
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#else
    setpriority(PRIO_PROCESS, static_cast<id_t>(getpid()), -10);
#endif
}

static std::string build_env_hint() {
    std::ostringstream os;
#ifdef _WIN32
    os << "High Priority Class | ";
    os << "Tip: set Control Panel -> Power Options -> High Performance";
#else
    os << "nice -10 | ";
    os << "Tip: sudo cpupower frequency-set -g performance";
#endif
    return os.str();
}

// ============================================================================
// CUDA synchronization helper
// ============================================================================

static void sync_cuda() {
    cuda_synchronize();
}

// ============================================================================
// Test pattern generation (templated)
// ============================================================================

template<typename T>
static void fill_test_pattern(T* data, int width, int height,
                               const PatternOffsets& po, T max_val) {
    double cx = width / 2.0;
    double cy = height / 2.0;
    double max_r = width / std::sqrt(2.0);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double angle = std::atan2(y - cy, x - cx);
            double radius = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            double r_norm = radius / max_r;
            double val;

            if (pixel::is_r_at(po, y, x))
                val = std::sin(angle * 4 + r_norm * 10);
            else if (pixel::is_b_at(po, y, x))
                val = std::cos(angle * 4 + r_norm * 10);
            else
                val = std::sin(r_norm * 15);

            double v = (val + 1.0) / 2.0;
            int iv = static_cast<int>(v * max_val + 0.5);
            data[static_cast<size_t>(y) * width + x] =
                static_cast<T>(std::clamp(iv, 0, static_cast<int>(max_val)));
        }
    }
}

static std::vector<uint8_t> generate_test_bayer(int width, int height,
                                                 int bit_depth,
                                                 BayerPattern pattern) {
    assert(bit_depth > 0 && bit_depth <= 16 && "bit_depth must be in [1, 16]");

    size_t bayer_size = static_cast<size_t>(width) * height;
    if (bit_depth > 8) bayer_size *= 2;
    std::vector<uint8_t> bayer(bayer_size);
    PatternOffsets po = PatternOffsets::from_pattern(pattern);

    if (bit_depth <= 8) {
        fill_test_pattern<uint8_t>(bayer.data(), width, height, po,
                                    static_cast<uint8_t>(255));
    } else {
        int max_val = (1 << bit_depth) - 1;
        fill_test_pattern<uint16_t>(
            reinterpret_cast<uint16_t*>(bayer.data()),
            width, height, po, static_cast<uint16_t>(max_val));
    }
    return bayer;
}

// ============================================================================
// Correctness validation (prevent dead-code elimination)
// ============================================================================

static bool validate_output_nonzero(const std::vector<uint8_t>& rgb) {
    return std::any_of(rgb.begin(), rgb.end(),
                       [](uint8_t v) { return v != 0; });
}

// ============================================================================
// Core benchmark runner
// ============================================================================

static BenchStats run_single_benchmark(DemosaicAlgorithm algo,
                                        const std::vector<uint8_t>& bayer,
                                        std::vector<uint8_t>& rgb,
                                        int width, int height,
                                        BayerPattern pattern, int bit_depth,
                                        bool is_packed, bool use_cpu,
                                        bool validate) {
    const double time_budget_sec = constants::kTimeBudgetSec;
    const int min_iterations = constants::kMinIterations;
    const int max_iterations = constants::kMaxIterations;
    const int warmup_iters = use_cpu
        ? constants::kCpuWarmupIters
        : constants::kGpuWarmupIters;

    for (int i = 0; i < warmup_iters; i++) {
        DemosaicError err;
        if (use_cpu)
            err = demosaic_cpu(bayer.data(), rgb.data(), width, height,
                                      pattern, algo, bit_depth, is_packed);
        else
            err = demosaic_cuda(bayer.data(), rgb.data(), width, height,
                                       pattern, algo, bit_depth, is_packed);
        if (err != DemosaicError::Ok) {
            std::cerr << "WARNING: " << algorithm_name(algo)
                      << " warmup failed (error " << static_cast<int>(err)
                      << ")\n";
        }
    }

    std::vector<double> times;
    times.reserve(max_iterations);

    auto batch_start = std::chrono::high_resolution_clock::now();
    int iterations = 0;

    while (iterations < max_iterations) {
        auto start = std::chrono::high_resolution_clock::now();

        DemosaicError err;
        if (use_cpu)
            err = demosaic_cpu(bayer.data(), rgb.data(), width, height,
                                      pattern, algo, bit_depth, is_packed);
        else
            err = demosaic_cuda(bayer.data(), rgb.data(), width, height,
                                       pattern, algo, bit_depth, is_packed);

        if (!use_cpu) sync_cuda();

        auto end = std::chrono::high_resolution_clock::now();

        if (err != DemosaicError::Ok) {
            std::cerr << "WARNING: " << algorithm_name(algo)
                      << " iteration " << iterations
                      << " failed (error " << static_cast<int>(err) << ")\n";
            continue;
        }

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
        iterations++;

        auto elapsed = std::chrono::duration<double>(end - batch_start).count();
        if (elapsed >= time_budget_sec && iterations >= min_iterations) break;
    }

    if (validate && !validate_output_nonzero(rgb)) {
        std::cerr << "WARNING: " << algorithm_name(algo)
                  << " output is all zeros -- possible dead-code elimination\n";
    }

    return compute_stats(times, width, height);
}

// ============================================================================
// Data structures
// ============================================================================

struct AlgoResult {
    std::string name;
    BenchStats cpu;
    BenchStats gpu;
    double speedup = 0.0;
    bool gpu_available = false;
    bool skipped = false;
    bool cpu_failed = false;
    bool gpu_failed = false;
};

struct ConfigResult {
    std::string label;
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    std::vector<AlgoResult> algo_results;
};

struct BatchRow {
    std::string algo_name;
    std::string method;
    int batch_size = 0;
    double total_ms = 0.0;
    double avg_ms_per_frame = 0.0;
    double fps = 0.0;
    double speedup_vs_single = 0.0;
};

struct BatchConfig {
    std::string label;
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    std::vector<BatchRow> rows;
};

// ============================================================================
// Printing helpers
// ============================================================================

static void print_separator(int width) {
    std::cout << std::string(width, '-') << '\n';
}

static std::string current_time_str() {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return buf;
}

static std::string report_stem() {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "benchmark_report_%Y%m%d_%H%M%S", &tm_buf);
    return buf;
}

static void print_system_info(bool have_gpu, const char* gpu_name,
                               int hw_threads) {
    std::cout << "==================================================================\n";
    std::cout << "  Demosaic Benchmark Suite  (v2.1)\n";
    std::cout << "------------------------------------------------------------------\n";
    std::cout << "  CPU threads : " << hw_threads << "\n";
    std::cout << "  AVX2        : " << (has_avx2() ? "available" : "not available") << "\n";
    std::cout << "  OpenMP      : " <<
#ifdef _OPENMP
        "enabled"
#else
        "not enabled"
#endif
        << "\n";
    std::cout << "  CUDA GPU    : " << (have_gpu ? gpu_name : "NOT AVAILABLE") << "\n";
    std::cout << "  Env hint    : " << build_env_hint() << "\n";
    std::cout << "==================================================================\n\n";
}

// ============================================================================
// CPU / GPU comparison benchmarks
// ============================================================================

struct Config {
    int width;
    int height;
    int bit_depth;
    bool is_packed;
    std::string label;
};

static std::vector<ConfigResult> run_cpu_gpu_benchmarks(
    const Config* configs, int num_configs,
    const DemosaicAlgorithm* algos, int num_algos,
    BayerPattern pattern, bool have_gpu) {

    std::vector<ConfigResult> all_configs;

    for (int ci = 0; ci < num_configs; ci++) {
        const auto& cfg = configs[ci];

        std::cout << "=== " << cfg.label << " (" << cfg.width << "x"
                  << cfg.height << ") ===\n\n";

        auto bayer = generate_test_bayer(cfg.width, cfg.height,
                                          cfg.bit_depth, pattern);
        size_t rgb_size = static_cast<size_t>(cfg.width) * cfg.height * 3;
        if (cfg.bit_depth > 8) rgb_size *= 2;
        std::vector<uint8_t> rgb_cpu(rgb_size);
        std::vector<uint8_t> rgb_gpu(rgb_size);

        std::cout << std::left
                  << std::setw(22) << "Algorithm"
                  << std::right
                  << std::setw(10) << "CPU ms"
                  << std::setw(10) << "GPU ms"
                  << std::setw(9) << "Speedup"
                  << std::setw(10) << "CV%"
                  << std::setw(12) << "CPU MP/s"
                  << std::setw(12) << "GPU MP/s"
                  << "\n";
        print_separator(90);

        ConfigResult cr;
        cr.label = cfg.label;
        cr.width = cfg.width;
        cr.height = cfg.height;
        cr.bit_depth = cfg.bit_depth;

        for (int ai = 0; ai < num_algos; ai++) {
            AlgoResult ar;
            ar.name = algorithm_name(algos[ai]);

            int ws = algorithm_window_size(algos[ai]);
            if (cfg.width < ws + 1 || cfg.height < ws + 1) {
                std::cout << std::left << std::setw(22) << ar.name
                          << "  (image too small -- needs >=" << (ws + 1)
                          << ")\n";
                ar.skipped = true;
                cr.algo_results.push_back(ar);
                continue;
            }

            ar.skipped = false;

            ar.cpu = run_single_benchmark(algos[ai], bayer, rgb_cpu,
                                           cfg.width, cfg.height, pattern,
                                           cfg.bit_depth, cfg.is_packed,
                                           true, true);

            if (ar.cpu.iterations == 0) {
                std::cout << std::left << std::setw(22) << ar.name
                          << "  (CPU benchmark failed)\n";
                ar.cpu_failed = true;
                cr.algo_results.push_back(ar);
                continue;
            }

            std::cout << std::left << std::setw(22) << ar.name
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(10) << ar.cpu.mean_ms;

            if (have_gpu && !cfg.is_packed) {
                ar.gpu = run_single_benchmark(algos[ai], bayer, rgb_gpu,
                                               cfg.width, cfg.height, pattern,
                                               cfg.bit_depth, cfg.is_packed,
                                               false, true);

                if (ar.gpu.iterations == 0) {
                    ar.gpu_failed = true;
                    std::cout << std::setw(10) << "FAIL"
                              << std::setw(9) << "-"
                              << std::setprecision(1)
                              << std::setw(9) << ar.cpu.cv_pct << "%"
                              << std::setw(12) << ar.cpu.mpix_per_sec
                              << std::setw(12) << "-";
                } else {
                    double speedup = (ar.gpu.mean_ms > 0.0)
                        ? ar.cpu.mean_ms / ar.gpu.mean_ms
                        : 0.0;
                    ar.gpu_available = true;
                    ar.speedup = speedup;

                    std::cout << std::setw(10) << ar.gpu.mean_ms
                              << std::setprecision(1)
                              << std::setw(8) << speedup << "x"
                              << std::setprecision(1)
                              << std::setw(9) << ar.cpu.cv_pct << "%"
                              << std::setprecision(1)
                              << std::setw(12) << ar.cpu.mpix_per_sec
                              << std::setw(12) << ar.gpu.mpix_per_sec;
                }
            } else {
                ar.gpu_available = false;
                std::cout << std::setw(10) << "-"
                          << std::setw(9) << "-"
                          << std::setprecision(1)
                          << std::setw(9) << ar.cpu.cv_pct << "%"
                          << std::setw(12) << ar.cpu.mpix_per_sec
                          << std::setw(12) << "-";
            }

            if (ar.cpu.outliers_removed > 0) {
                std::cout << "  [" << ar.cpu.outliers_removed << " outliers]";
            }

            std::cout << "\n";
            cr.algo_results.push_back(ar);
        }
        all_configs.push_back(cr);
        std::cout << "\n\n";
    }

    if (!have_gpu) {
        std::cout << "NOTE: No CUDA GPU detected. GPU columns show \"-\".\n";
        std::cout << "      Rebuild without -NoCuda and ensure an NVIDIA GPU is present.\n\n";
    }

    return all_configs;
}

// ============================================================================
// Batch throughput benchmarks
// ============================================================================

static int compute_max_frames(int width, int height, int bit_depth) {
    size_t bayer_bytes = static_cast<size_t>(width) * height * (bit_depth > 8 ? 2 : 1);
    size_t rgb_bytes = static_cast<size_t>(width) * height * 3 * (bit_depth > 8 ? 2 : 1);
    size_t frame_bytes = bayer_bytes + rgb_bytes;
    if (frame_bytes == 0) return constants::kBatchMinFrames;
    constexpr size_t limit = constants::kBatchMemoryLimitGB * 1024ULL * 1024ULL * 1024ULL;
    int frames = static_cast<int>(limit / frame_bytes);
    return std::clamp(frames, constants::kBatchMinFrames, constants::kBatchMaxFrames);
}

static std::vector<BatchConfig> run_batch_benchmarks(
    const Config* configs, int num_configs,
    const DemosaicAlgorithm* algos, int num_algos,
    BayerPattern pattern) {

    int batch_sizes[] = {10, 50, 100, 500};

    std::vector<BatchConfig> batch_results;

    std::cout << "\n==================================================================\n";
    std::cout << "  Batch Throughput Benchmark\n";
    std::cout << "  single = N-frame single-API loop (pageable mem)\n";
    std::cout << "  batch N = batch API (pinned mem + CUDA Graphs + dual-buffer)\n";
    std::cout << "==================================================================\n\n";

    for (int ci = 0; ci < num_configs; ci++) {
        const auto& cfg = configs[ci];
        int max_frames = compute_max_frames(cfg.width, cfg.height, cfg.bit_depth);

        std::cout << "=== " << cfg.label << " (" << cfg.width << "x"
                  << cfg.height << ", " << cfg.bit_depth << "-bit, "
                  << max_frames << " frames) ===\n\n";

        size_t bayer_bytes = static_cast<size_t>(cfg.width) * cfg.height;
        if (cfg.bit_depth > 8) bayer_bytes *= 2;
        size_t rgb_bytes = static_cast<size_t>(cfg.width) * cfg.height * 3;
        if (cfg.bit_depth > 8) rgb_bytes *= 2;

        std::vector<std::vector<uint8_t>> frame_data(max_frames);
        std::vector<uint8_t*> input_ptrs(max_frames);
        std::vector<uint8_t*> output_ptrs(max_frames);
        std::vector<std::vector<uint8_t>> rgb_bufs(max_frames);

        for (int f = 0; f < max_frames; f++) {
            frame_data[f] = generate_test_bayer(cfg.width, cfg.height,
                                                 cfg.bit_depth, pattern);
            input_ptrs[f] = frame_data[f].data();
            rgb_bufs[f].resize(rgb_bytes);
            output_ptrs[f] = rgb_bufs[f].data();
        }

        BatchConfig bc;
        bc.label = cfg.label;
        bc.width = cfg.width;
        bc.height = cfg.height;
        bc.bit_depth = cfg.bit_depth;

        std::cout << std::left
                  << std::setw(22) << "Algorithm"
                  << std::setw(12) << "Method"
                  << std::right
                  << std::setw(12) << "Total ms"
                  << std::setw(14) << "Avg/frame ms"
                  << std::setw(10) << "FPS"
                  << std::setw(14) << "vs ref"
                  << "\n";
        print_separator(90);

        for (int ai = 0; ai < num_algos; ai++) {
            int ws = algorithm_window_size(algos[ai]);
            if (cfg.width < ws + 1 || cfg.height < ws + 1) continue;

            for (int w = 0; w < constants::kGpuWarmupIters; w++) {
                demosaic_cuda(input_ptrs[0], output_ptrs[0],
                    cfg.width, cfg.height, pattern, algos[ai],
                    cfg.bit_depth, false);
            }
            sync_cuda();

            double single_total = 0.0;
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                for (int f = 0; f < max_frames; f++) {
                    demosaic_cuda(input_ptrs[f], output_ptrs[f],
                        cfg.width, cfg.height, pattern, algos[ai],
                        cfg.bit_depth, false);
                }
                sync_cuda();
                auto t2 = std::chrono::high_resolution_clock::now();
                single_total =
                    std::chrono::duration<double, std::milli>(t2 - t1).count();
            }
            double single_avg = (single_total > 0.0)
                ? single_total / max_frames : 0.0;
            double single_fps = (single_total > 0.0)
                ? max_frames / (single_total / 1000.0) : 0.0;

            bool first = true;
            for (int bs : batch_sizes) {
                if (bs > max_frames) continue;

                if (first) {
                    std::cout << std::left << std::setw(22)
                              << algorithm_name(algos[ai]);
                } else {
                    std::cout << std::left << std::setw(22) << "";
                }
                first = false;

                double batch_total = 0.0;
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    demosaic_cuda_batch(input_ptrs.data(),
                        output_ptrs.data(), bs, cfg.width, cfg.height,
                        pattern, algos[ai], cfg.bit_depth);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    batch_total =
                        std::chrono::duration<double, std::milli>(t2 - t1).count();
                }
                double batch_avg = (batch_total > 0.0)
                    ? batch_total / bs : 0.0;
                double batch_fps = (batch_total > 0.0)
                    ? bs / (batch_total / 1000.0) : 0.0;
                double speedup_vs_single = (single_avg > 0.0 && batch_avg > 0.0)
                    ? single_avg / batch_avg : 0.0;

                std::ostringstream method_label;
                method_label << "batch " << bs;

                std::cout << std::setw(12) << method_label.str()
                          << std::right << std::fixed << std::setprecision(2)
                          << std::setw(12) << batch_total
                          << std::setw(14) << batch_avg
                          << std::setprecision(1)
                          << std::setw(10) << batch_fps
                          << std::setprecision(1)
                          << std::setw(13) << speedup_vs_single << "x";

                if (speedup_vs_single >= constants::kFastSpeedupThreshold)
                    std::cout << " [FAST]";
                std::cout << "\n";

                bc.rows.push_back({algorithm_name(algos[ai]),
                    method_label.str(), bs, batch_total, batch_avg,
                    batch_fps, speedup_vs_single});
            }

            std::ostringstream ref_label;
            ref_label << "single " << max_frames;
            std::cout << std::left << std::setw(22) << ""
                      << std::setw(12) << ref_label.str()
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(12) << single_total
                      << std::setw(14) << single_avg
                      << std::setprecision(1)
                      << std::setw(10) << single_fps
                      << std::setw(14) << "base"
                      << "\n\n";

            bc.rows.push_back({algorithm_name(algos[ai]),
                ref_label.str(), max_frames, single_total, single_avg,
                single_fps, 1.0});
        }
        batch_results.push_back(bc);
        std::cout << "\n";
    }

    return batch_results;
}

// ============================================================================
// Markdown report
// ============================================================================

static void write_markdown_report(const std::string& filename,
                                   bool have_gpu, const char* gpu_name,
                                   int hw_threads,
                                   const std::vector<ConfigResult>& configs,
                                   const std::vector<BatchConfig>& batch_results) {
    std::ofstream rpt(filename);
    if (!rpt.is_open()) {
        std::cerr << "WARNING: Failed to write report: " << filename << "\n";
        return;
    }

    rpt << "# Demosaic Benchmark Report\n\n";
    rpt << "**Date:** " << current_time_str() << "\n\n";

    rpt << "## System Information\n\n";
    rpt << "| Item | Value |\n|------|-------|\n";
    rpt << "| CPU Threads | " << hw_threads << " |\n";
    rpt << "| AVX2 | " << (has_avx2() ? "available" : "not available") << " |\n";
    rpt << "| OpenMP | " <<
#ifdef _OPENMP
        "enabled"
#else
        "not enabled"
#endif
        << " |\n";
    rpt << "| CUDA GPU | " << (have_gpu ? std::string(gpu_name) : "NOT AVAILABLE") << " |\n";
    rpt << "| Env hint | " << build_env_hint() << " |\n\n";

    for (const auto& cfg : configs) {
        rpt << "## " << cfg.label << " (" << cfg.width << "x" << cfg.height
            << ", " << cfg.bit_depth << "-bit)\n\n";
        rpt << "| Algorithm | CPU (ms) | GPU (ms) | Speedup | CV% | CPU (MP/s) | GPU (MP/s) | CI95 |\n";
        rpt << "|-----------|----------|----------|---------|-----|------------|------------|------|\n";

        for (const auto& ar : cfg.algo_results) {
            rpt << "| " << ar.name;
            if (ar.skipped) {
                rpt << " | (too small) | - | - | - | - | - | - |\n";
            } else if (ar.cpu_failed) {
                rpt << " | (CPU failed) | - | - | - | - | - | - |\n";
            } else {
                rpt << std::fixed
                    << " | " << std::setprecision(2) << ar.cpu.mean_ms;
                if (ar.gpu_available) {
                    rpt << " | " << std::setprecision(2) << ar.gpu.mean_ms
                        << " | " << std::setprecision(1) << ar.speedup << "x"
                        << " | " << std::setprecision(1) << ar.cpu.cv_pct << "%"
                        << " | " << std::setprecision(1) << ar.cpu.mpix_per_sec
                        << " | " << std::setprecision(1) << ar.gpu.mpix_per_sec
                        << " | +/-" << std::setprecision(1)
                        << (ar.cpu.ci95_high - ar.cpu.mean_ms) << "ms"
                        << " |\n";
                } else if (ar.gpu_failed) {
                    rpt << " | (GPU failed)"
                        << " | -"
                        << " | " << std::setprecision(1) << ar.cpu.cv_pct << "%"
                        << " | " << std::setprecision(1) << ar.cpu.mpix_per_sec
                        << " | -"
                        << " | +/-" << std::setprecision(1)
                        << (ar.cpu.ci95_high - ar.cpu.mean_ms) << "ms"
                        << " |\n";
                } else {
                    rpt << " | - | -"
                        << " | " << std::setprecision(1) << ar.cpu.cv_pct << "%"
                        << " | " << std::setprecision(1) << ar.cpu.mpix_per_sec
                        << " | -"
                        << " | +/-" << std::setprecision(1)
                        << (ar.cpu.ci95_high - ar.cpu.mean_ms) << "ms"
                        << " |\n";
                }
            }
        }
        rpt << "\n";
    }

    if (!have_gpu) {
        rpt << "> **Note:** No CUDA GPU detected.\n\n";
    }

    if (!batch_results.empty()) {
        rpt << "## Batch Throughput Benchmark\n\n";
        rpt << "**Single-API loop (pageable) vs batch API (pinned + CUDA Graphs + dual-buffer).**\n\n";

        for (const auto& bc : batch_results) {
            rpt << "### " << bc.label << " (" << bc.width << "x" << bc.height
                << ", " << bc.bit_depth << "-bit)\n\n";
            rpt << "| Algorithm | Method | Batch Size | Total (ms) | Avg/Frame (ms) | FPS | vs Single |\n";
            rpt << "|-----------|--------|-----------|------------|----------------|-----|-----------|\n";

            std::string prev_algo;
            for (const auto& row : bc.rows) {
                rpt << "| "
                    << (row.algo_name != prev_algo ? row.algo_name : "")
                    << " | " << row.method
                    << " | " << row.batch_size;
                if (row.speedup_vs_single == 1.0 && row.method.find("single") == 0) {
                    rpt << " | " << std::fixed << std::setprecision(2) << row.total_ms
                        << " | " << std::setprecision(3) << row.avg_ms_per_frame
                        << " | " << std::setprecision(1) << row.fps
                        << " | base |\n";
                } else {
                    rpt << " | " << std::fixed << std::setprecision(2) << row.total_ms
                        << " | " << std::setprecision(3) << row.avg_ms_per_frame
                        << " | " << std::setprecision(1) << row.fps
                        << " | " << std::setprecision(1) << row.speedup_vs_single << "x |\n";
                }
                prev_algo = row.algo_name;
            }
            rpt << "\n";
        }
    }

    rpt << "---\n*Report generated by im_operator_benchmark.*\n";
    rpt.close();
    std::cout << "\nMarkdown report saved to: " << filename << "\n";
}

// ============================================================================
// JSON report
// ============================================================================

static std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    o += buf;
                } else {
                    o += c;
                }
                break;
        }
    }
    return o;
}

static void write_json_report(const std::string& filename,
                               bool have_gpu, const char* gpu_name,
                               int hw_threads,
                               const std::vector<ConfigResult>& configs,
                               const std::vector<BatchConfig>& batch_results) {
    std::ofstream js(filename);
    if (!js.is_open()) {
        std::cerr << "WARNING: Failed to write JSON: " << filename << "\n";
        return;
    }

    js << "{\n";
    js << "  \"timestamp\": \"" << current_time_str() << "\",\n";
    js << "  \"system\": {\n";
    js << "    \"cpu_threads\": " << hw_threads << ",\n";
    js << "    \"avx2\": " << (has_avx2() ? "true" : "false") << ",\n";
    js << "    \"openmp\": " <<
#ifdef _OPENMP
        "true"
#else
        "false"
#endif
        << ",\n";
    js << "    \"cuda\": " << (have_gpu ? "true" : "false") << ",\n";
    js << "    \"gpu_name\": \"" << json_escape(gpu_name ? gpu_name : "N/A") << "\"\n";
    js << "  },\n";

    js << "  \"configs\": [\n";
    for (size_t ci = 0; ci < configs.size(); ci++) {
        const auto& cfg = configs[ci];
        js << "    {\n";
        js << "      \"label\": \"" << json_escape(cfg.label) << "\",\n";
        js << "      \"width\": " << cfg.width << ",\n";
        js << "      \"height\": " << cfg.height << ",\n";
        js << "      \"bit_depth\": " << cfg.bit_depth << ",\n";
        js << "      \"algorithms\": [\n";
        for (size_t ai = 0; ai < cfg.algo_results.size(); ai++) {
            const auto& ar = cfg.algo_results[ai];
            js << "        {\n";
            js << "          \"name\": \"" << json_escape(ar.name) << "\",\n";
            js << "          \"skipped\": " << (ar.skipped ? "true" : "false") << ",\n";
            if (!ar.skipped && !ar.cpu_failed) {
                js << std::fixed << std::setprecision(4);
                js << "          \"cpu_mean_ms\": " << ar.cpu.mean_ms << ",\n";
                js << "          \"cpu_median_ms\": " << ar.cpu.median_ms << ",\n";
                js << "          \"cpu_stddev_ms\": " << ar.cpu.stddev_ms << ",\n";
                js << "          \"cpu_cv_pct\": " << std::setprecision(2) << ar.cpu.cv_pct << ",\n";
                js << "          \"cpu_ci95_ms\": " << std::setprecision(4)
                   << (ar.cpu.ci95_high - ar.cpu.mean_ms) << ",\n";
                js << "          \"cpu_mpix_per_sec\": " << std::setprecision(2) << ar.cpu.mpix_per_sec << ",\n";
                js << "          \"cpu_iterations\": " << ar.cpu.iterations << ",\n";
                js << "          \"cpu_outliers_removed\": " << ar.cpu.outliers_removed << ",\n";
                if (ar.gpu_available) {
                    js << "          \"gpu_mean_ms\": " << ar.gpu.mean_ms << ",\n";
                    js << "          \"gpu_median_ms\": " << ar.gpu.median_ms << ",\n";
                    js << "          \"gpu_stddev_ms\": " << ar.gpu.stddev_ms << ",\n";
                    js << "          \"gpu_cv_pct\": " << std::setprecision(2) << ar.gpu.cv_pct << ",\n";
                    js << "          \"gpu_ci95_ms\": " << std::setprecision(4)
                       << (ar.gpu.ci95_high - ar.gpu.mean_ms) << ",\n";
                    js << "          \"gpu_mpix_per_sec\": " << std::setprecision(2) << ar.gpu.mpix_per_sec << ",\n";
                    js << "          \"gpu_iterations\": " << ar.gpu.iterations << ",\n";
                    js << "          \"gpu_outliers_removed\": " << ar.gpu.outliers_removed << ",\n";
                    js << "          \"speedup\": " << std::setprecision(2) << ar.speedup << "\n";
                } else {
                    js << "          \"gpu_available\": false\n";
                }
            } else if (ar.cpu_failed) {
                js << "          \"cpu_failed\": true\n";
            }
            js << "        }" << (ai + 1 < cfg.algo_results.size() ? "," : "") << "\n";
        }
        js << "      ]\n";
        js << "    }" << (ci + 1 < configs.size() ? "," : "") << "\n";
    }
    js << "  ]";

    if (!batch_results.empty()) {
        js << ",\n  \"batch_configs\": [\n";
        for (size_t bi = 0; bi < batch_results.size(); bi++) {
            const auto& bc = batch_results[bi];
            js << "    {\n";
            js << "      \"label\": \"" << json_escape(bc.label) << "\",\n";
            js << "      \"width\": " << bc.width << ",\n";
            js << "      \"height\": " << bc.height << ",\n";
            js << "      \"bit_depth\": " << bc.bit_depth << ",\n";
            js << "      \"rows\": [\n";
            for (size_t ri = 0; ri < bc.rows.size(); ri++) {
                const auto& row = bc.rows[ri];
                js << "        {\n";
                js << "          \"algo\": \"" << json_escape(row.algo_name) << "\",\n";
                js << "          \"method\": \"" << json_escape(row.method) << "\",\n";
                js << "          \"batch_size\": " << row.batch_size << ",\n";
                js << std::fixed << std::setprecision(4);
                js << "          \"total_ms\": " << row.total_ms << ",\n";
                js << "          \"avg_ms_per_frame\": " << row.avg_ms_per_frame << ",\n";
                js << std::setprecision(2);
                js << "          \"fps\": " << row.fps << ",\n";
                js << "          \"speedup_vs_single\": " << row.speedup_vs_single << "\n";
                js << "        }" << (ri + 1 < bc.rows.size() ? "," : "") << "\n";
            }
            js << "      ]\n";
            js << "    }" << (bi + 1 < batch_results.size() ? "," : "") << "\n";
        }
        js << "  ]\n";
    } else {
        js << "\n";
    }

    js << "}\n";
    js.close();
    std::cout << "JSON report saved to:    " << filename << "\n";
}

// ============================================================================
// Command-line options
// ============================================================================

struct BenchOptions {
    bool no_gpu = false;
    bool no_batch = false;
    bool quick = false;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --no-gpu    Skip GPU benchmarks\n"
              << "  --no-batch  Skip batch throughput benchmarks\n"
              << "  --quick     Reduce iterations for quick validation\n"
              << "  --help      Show this help message\n\n";
}

static BenchOptions parse_args(int argc, char* argv[]) {
    BenchOptions opts;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-gpu")       opts.no_gpu = true;
        else if (arg == "--no-batch") opts.no_batch = true;
        else if (arg == "--quick")    opts.quick = true;
        else if (arg == "--help")     { print_usage(argv[0]); std::exit(0); }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return opts;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    BenchOptions opts = parse_args(argc, argv);

    set_benchmark_priority();

    bool have_gpu = has_cuda() && !opts.no_gpu;
    const char* gpu_name_raw = cuda_device_name();
    const char* gpu_name = (have_gpu && gpu_name_raw) ? gpu_name_raw : "N/A";

    int hw_threads = compute_hardware_concurrency();

    print_system_info(have_gpu, gpu_name, hw_threads);

    const Config configs[] = {
        {1920, 1080,  8, false, "1080p 8-bit"},
        {1920, 1080, 12, false, "1080p 12-bit"},
        {3840, 2160,  8, false, "4K 8-bit"},
        {3840, 2160, 12, false, "4K 12-bit"},
        {7680, 4320,  8, false, "8K 8-bit"},
        {7680, 4320, 12, false, "8K 12-bit"},
    };
    constexpr int num_configs = std::size(configs);

    const DemosaicAlgorithm algos[] = {
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
    constexpr int num_algos = std::size(algos);

    BayerPattern pattern = BayerPattern::RGGB;

    std::vector<ConfigResult> all_configs =
        run_cpu_gpu_benchmarks(configs, num_configs, algos, num_algos,
                               pattern, have_gpu);

    std::vector<BatchConfig> batch_results;
    if (have_gpu && !opts.no_batch) {
        batch_results = run_batch_benchmarks(configs, num_configs,
                                              algos, num_algos, pattern);
    }

    std::cout << "Benchmark complete.\n";

    std::string stem = report_stem();
    write_markdown_report(stem + ".md", have_gpu, gpu_name, hw_threads,
                           all_configs, batch_results);
    write_json_report(stem + ".json", have_gpu, gpu_name, hw_threads,
                       all_configs, batch_results);

    return 0;
}
