# Demosaic 库 API 参考手册

## 版本 2.0.0

---

## 目录

1. [概述](#1-概述)
2. [快速开始](#2-快速开始)
3. [数据类型](#3-数据类型)
4. [类 Demosaic（推荐入口）](#4-类-Demosaic推荐入口)
5. [自由函数 API](#5-自由函数-api)
6. [像素工具函数 (namespace pixel)](#6-像素工具函数-namespace-pixel)
7. [原始数据分析 API](#7-原始数据分析-api)
8. [CUDA GPU 加速 API](#8-cuda-gpu-加速-api)
9. [AVX2 SIMD 优化 API](#9-avx2-simd-优化-api)
10. [能力查询函数](#10-能力查询函数)
11. [构建与集成](#11-构建与集成)
12. [错误处理](#12-错误处理)
13. [性能基准参考](#13-性能基准参考)

---

## 1. 概述

Demosaic 是一个高性能 **Bayer 图像去马赛克（Demosaicing/Demosaicing）** 算法库，将单通道 Bayer raw 数据转换为三通道 RGB 彩色图像。

### 核心特性

| 特性 | 说明 |
|------|------|
| **9 种算法** | SUPER_FAST、HQLI、MG、L7、DFPD、AHD、AMAZE、RCD、PRISM |
| **多级加速** | CPU + OpenMP 多线程 + AVX2 SIMD + CUDA GPU |
| **多位深支持** | 8 / 10 / 12 / 14 / 16 bit |
| **多种存储** | Unpacked（逐像素）和 Packed（压缩）格式 |
| **零外部依赖** | 核心库仅依赖 C++17 标准库 |
| **自动分析** | 可自动检测未知 RAW 文件的位深、分辨率、Bayer 模式 |
| **CUDA Graph** | 批量帧处理的流水线加速 |

### 支持的 Bayer 模式

| 模式 | 说明 |
|------|------|
| `RGGB` | 第 0 行：R G R G... / 第 1 行：G B G B... |
| `BGGR` | 第 0 行：B G B G... / 第 1 行：G R G R... |
| `GRBG` | 第 0 行：G R G R... / 第 1 行：B G B G... |
| `GBRG` | 第 0 行：G B G B... / 第 1 行：R G R G... |

### 支持的算法

| 算法 | 窗口 | PSNR | 速度 | 适用场景 |
|------|------|------|------|---------|
| `SUPER_FAST` | 1×1 | 最低 | 最快 | 实时预览、低延迟需求 |
| `HQLI` | 5×5 | ~36 dB | 快 | 一般高质量需求 |
| `MG` | 5×5 | ~37 dB | 快 | Malvar-He-Cutler 经典算法 |
| `L7` | 7×7 | ~37.1 dB | 中等 | 大窗口线性插值 |
| `DFPD` | 11×11 | ~39 dB | 中等 | 方向滤波，平衡质量与速度 |
| `AHD` | 5×5 | 高 | 中等 | 自适应同源性，边缘保持好 |
| `AMAZE` | 5×5 | 高 | 中等 | 自适应梯度，复杂纹理表现好 |
| `RCD` | 9×9 | 高 | 慢 | 比率校正，色彩还原优秀 |
| `PRISM` | 9×9 | 最高 | 最慢 | 极坐标比率光谱融合，最高质量 |

---

## 2. 快速开始

### 最简单的 5 行调用

```cpp
#include "im_operator.h"

std::vector<uint8_t> bayer_data = /* 加载你的 Bayer raw 数据 */;
std::vector<uint8_t> rgb_data(width * height * 3);  // 8-bit 输出

imop::Demosaic dm;
auto err = dm.process(bayer_data.data(), rgb_data.data(),
                           width, height,
                           imop::BayerPattern::RGGB,
                           imop::DemosaicAlgorithm::DFPD,
                           8);  // bit_depth
if (imop::ok(err)) {
    // rgb_data 现在包含去马赛克后的 RGB 数据
}
```

### 头文件

你的程序只需包含一个头文件：

```cpp
#include "im_operator.h"
```

### 链接

链接静态库 `im_operator.lib`：

```cmake
target_link_libraries(your_target PRIVATE im_operator)
```

如果启用 CUDA，还需链接 `CUDA::cudart`（CMake 自动处理）。

---

## 3. 数据类型

### 3.1 BayerPattern 枚举

```cpp
namespace imop {

enum class BayerPattern {
    RGGB,  // R G   第0行
           // G B   第1行

    BGGR,  // B G   第0行
           // G R   第1行

    GRBG,  // G R   第0行
           // B G   第1行

    GBRG   // G B   第0行
           // R G   第1行
};

// 工具函数
bool is_valid_bayer_pattern(BayerPattern p);
```

### 3.2 DemosaicAlgorithm 枚举

```cpp
enum class DemosaicAlgorithm {
    SUPER_FAST,  // 最近邻插值（1x1）
    HQLI,        // 高质量线性插值（5x5）
    MG,          // Malvar-He-Cutler（5x5）
    L7,          // 7x7 线性插值
    DFPD,        // 方向滤波后验决策（11x11）
    AHD,         // 自适应同源性导向（5x5）
    AMAZE,       // 自适应梯度（5x5）
    RCD,         // 比率校正去马赛克（9x9）
    PRISM        // 极坐标比率光谱融合（9x9）
};
```

### 3.3 DemosaicError 枚举

```cpp
enum class DemosaicError {
    Ok = 0,               // 成功
    NullInput,            // 空指针
    InvalidDimensions,    // 无效尺寸
    InvalidBitDepth,      // 无效位深（须 1-16）
    InvalidPattern,       // 无效 Bayer 模式
    ImageTooSmall,        // 图像太小（不满足算法窗口要求）
    InternalError         // 内部处理错误
};

// 获取错误消息
const char* demosaic_error_message(DemosaicError err);

// 便捷判断
bool operator!(DemosaicError err);   // 等同于 err != Ok
bool ok(DemosaicError err);          // 等同于 err == Ok
```

### 3.4 ImageBuffer 结构体

自管理的图像缓冲区，自动分配和释放内存。

```cpp
struct ImageBuffer {
    std::vector<uint8_t> data;  // 像素数据
    int width = 0;              // 图像宽度
    int height = 0;             // 图像高度
    int channels = 1;           // 通道数（Bayer 为 1，RGB 为 3）
    int bit_depth = 8;          // 位深
    bool is_packed = false;     // 是否 packed 格式

    size_t size() const;              // 返回 data.size()
    bool empty() const;               // 返回 data.empty()
    uint8_t* ptr();                   // 返回 data.data()
    const uint8_t* ptr() const;       // 返回 data.data() const
};
```

### 3.5 DataInfo 结构体

原始数据分析结果。

```cpp
struct DataInfo {
    int detected_bit_depth = 0;          // 检测到的位深
    int suggested_width = 0;             // 建议宽度
    int suggested_height = 0;            // 建议高度
    int pixel_count = 0;                 // 像素总数
    int max_value = 0;                   // 最大像素值
    int min_value = 0;                   // 最小像素值
    bool is_likely_16bit = false;        // 是否疑似 16-bit
    bool is_packed = false;              // 是否 packed 格式
    std::vector<std::pair<int, int>> possible_dimensions;  // 可能的 (宽, 高) 组合
};
```

### 3.6 PatternOffsets 结构体

Bayer 模式中 R/B 像素在 2×2 块中的偏移。

```cpp
struct PatternOffsets {
    int r_row, r_col;  // R 像素偏移（行, 列）
    int b_row, b_col;  // B 像素偏移（行, 列）

    static PatternOffsets from_pattern(BayerPattern pattern);
};
```

---

## 4. 类 Demosaic（推荐入口）

类 `imop::Demosaic` 是推荐的主要 API 入口，封装了所有处理功能。

```cpp
#include "im_operator.h"

namespace imop {
class Demosaic final {
public:
    Demosaic() = default;
    ~Demosaic() = default;
    // ... 所有成员函数
};
}
```

### 4.1 process() — 核心处理（自动选 CUDA/CPU）

```cpp
DemosaicError process(const uint8_t* bayer_data, uint8_t* rgb_data,
                     int width, int height,
                     BayerPattern pattern,
                     DemosaicAlgorithm algorithm,
                     int bit_depth = 8,
                     bool is_packed = false);
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `bayer_data` | `const uint8_t*` | 输入 Bayer raw 数据 |
| `rgb_data` | `uint8_t*` | 输出 RGB 数据（需预分配） |
| `width` | `int` | 图像宽度（像素） |
| `height` | `int` | 图像高度（像素） |
| `pattern` | `BayerPattern` | Bayer 滤色阵列模式 |
| `algorithm` | `DemosaicAlgorithm` | 去马赛克算法 |
| `bit_depth` | `int` | 位深（1-16，默认 8） |
| `is_packed` | `bool` | 是否 packed 格式（默认 false） |

**输出缓冲区大小：**
- 8-bit：`width * height * 3` 字节
- >8-bit：`width * height * 3 * 2` 字节（16-bit per channel）

**行为：** 自动检测 CUDA 可用性，优先使用 GPU，CPU 作为后备。

**示例：**

```cpp
Demosaic dm;

// 8-bit 全分辨率处理
std::vector<uint8_t> rgb(5320 * 4600 * 3);
auto err = dm.process(bayer, rgb.data(), 5320, 4600,
                           BayerPattern::RGGB,
                           DemosaicAlgorithm::DFPD, 8);
if (!ok(err)) {
    std::cerr << "处理失败: " << demosaic_error_message(err) << std::endl;
}
```

### 4.2 process_cpu() — 强制 CPU 处理

```cpp
DemosaicError process_cpu(const uint8_t* bayer_data, uint8_t* rgb_data,
                         int width, int height,
                         BayerPattern pattern,
                         DemosaicAlgorithm algorithm,
                         int bit_depth = 8,
                         bool is_packed = false);
```

参数与 `process()` 完全相同，但**始终使用 CPU** 执行，不尝试 CUDA。

### 4.3 process(ImageBuffer) — 自动管理内存版本

```cpp
DemosaicError process(const ImageBuffer& bayer, ImageBuffer& rgb,
                     BayerPattern pattern,
                     DemosaicAlgorithm algorithm);
```

自动填充 `rgb.width`、`rgb.height`、`rgb.channels`（=3）、`rgb.bit_depth`，并 resize `rgb.data`。

**示例：**

```cpp
ImageBuffer bayer;
bayer.data = load_raw_file("input.raw");
bayer.width = 4024;
bayer.height = 3032;
bayer.bit_depth = 12;

ImageBuffer rgb;
auto err = dm.process(bayer, rgb, BayerPattern::RGGB, DemosaicAlgorithm::PRISM);
// rgb.data 已自动分配，可直接使用
```

### 4.4 各算法独立调用

每个算法都有独立的成员函数，**始终使用 CPU** 执行：

```cpp
DemosaicError process_super_fast(const uint8_t* bayer, uint8_t* rgb,
                                int width, int height, BayerPattern pattern,
                                int bit_depth, bool is_packed = false);

DemosaicError process_hqli(...);
DemosaicError process_mg(...);
DemosaicError process_l7(...);
DemosaicError process_dfpd(...);
DemosaicError process_ahd(...);
DemosaicError process_amaze(...);
DemosaicError process_rcd(...);
DemosaicError process_prism(...);
```

> **注意：** 这些方法绕过了自动调度，直接调用 CPU 实现。
> 参数与 `process_cpu()` 相同（省略算法参数，因为已固定）。

### 4.5 静态工具方法

```cpp
// 获取算法名称
static std::string algorithm_name(DemosaicAlgorithm algo);

// 获取算法窗口大小
static int algorithm_window_size(DemosaicAlgorithm algo);

// 分析原始数据
static DataInfo analyze_data(const uint8_t* data, size_t byte_size);

// 检测位深
static int detect_bit_depth(const uint8_t* data, size_t byte_size);

// 推测分辨率
static std::vector<std::pair<int, int>> suggest_dimensions(size_t pixel_count);

// 猜测 Bayer 模式
static BayerPattern guess_pattern(const uint8_t* data, int width, int height,
                                   int bit_depth = 8, bool is_packed = false);

// 获取硬件并发数
static int compute_hardware_concurrency();
```

---

## 5. 自由函数 API

如果你不想使用 `Demosaic` 类，也可以直接调用命名空间 `im_operator` 下的自由函数。这些函数与 `Demosaic` 类的成员函数一一对应。

### 5.1 统一处理入口

```cpp
namespace imop {

// 自动选 CUDA/CPU
DemosaicError demosaic(const uint8_t* bayer_data, uint8_t* rgb_data,
                             int width, int height, BayerPattern pattern,
                             DemosaicAlgorithm algorithm, int bit_depth = 8,
                             bool is_packed = false);

// 强制 CPU
DemosaicError demosaic_cpu(const uint8_t* bayer_data, uint8_t* rgb_data,
                                 int width, int height, BayerPattern pattern,
                                 DemosaicAlgorithm algorithm, int bit_depth = 8,
                                 bool is_packed = false);

}
```

### 5.2 各算法独立函数

```cpp
DemosaicError process_hqli(const uint8_t* bayer, uint8_t* rgb,
                          int width, int height, BayerPattern pattern,
                          int bit_depth, bool is_packed = false);

DemosaicError process_mg(...);
DemosaicError process_l7(...);
DemosaicError process_dfpd(...);
DemosaicError process_ahd(...);
DemosaicError process_amaze(...);
DemosaicError process_rcd(...);
DemosaicError process_prism(...);
DemosaicError process_super_fast(...);
```

### 5.3 输入验证

```cpp
DemosaicError validate_demosaic_inputs(const uint8_t* bayer_data, uint8_t* rgb_data,
                                     int width, int height, int bit_depth);
```

### 5.4 工具函数

```cpp
std::string algorithm_name(DemosaicAlgorithm algo);
int algorithm_window_size(DemosaicAlgorithm algo);
int compute_hardware_concurrency();
```

---

## 6. 像素工具函数 (namespace pixel)

`imop::pixel` 命名空间提供底层像素操作函数，适合编写自定义处理流水线。

### 6.1 大小计算

```cpp
namespace imop::pixel {

// 最大有效像素值
int safe_max_val(int bit_depth);
// 例如：safe_max_val(8)  => 255
//       safe_max_val(12) => 4095

// Bayer 数据字节数
size_t compute_bayer_byte_size(int width, int height, int bit_depth, bool is_packed);

// RGB 数据字节数
size_t compute_rgb_byte_size(int width, int height, int bit_depth);

// Packed 格式字节数
size_t compute_packed_byte_size(int width, int height, int bit_depth);

}
```

### 6.2 像素类型判断（基于 Bayer 模式）

```cpp
// 判断 (row, col) 位置是否为 R / B / G 像素
bool is_r_at(const PatternOffsets& po, int row, int col);
bool is_b_at(const PatternOffsets& po, int row, int col);
bool is_g_at(const PatternOffsets& po, int row, int col);
```

### 6.3 数据读取

```cpp
// 通用读取（自动适配 8-bit / 16-bit / packed）
int get_raw(const uint8_t* data, int x, int y, int width, int bit_depth,
            bool is_packed = false, size_t data_byte_size = 0, int height = 0);

// 边界夹持读取（越界时返回最近有效像素）
int get_clamped(const uint8_t* data, int x, int y, int width, int height,
                int bit_depth, bool is_packed = false, size_t data_byte_size = 0);

// Packed 格式读取
int get_packed_raw(const uint8_t* data, int x, int y, int width, int bit_depth,
                   size_t data_byte_size = 0, int height = 0);

// 8-bit / 16-bit 快速通路
int get_raw_8(const uint8_t* data, int x, int y, int width);
int get_raw_16(const uint8_t* data, int x, int y, int width, int bit_depth);
int get_clamped_8(const uint8_t* data, int x, int y, int width, int height);
int get_clamped_16(const uint8_t* data, int x, int y, int width, int height, int bit_depth);

// MSB 对齐校正（将 16-bit 容器中的高 bps 值右移到 LSB 对齐）
uint16_t align_raw_value(uint16_t val, int bit_depth);
```

### 6.4 RGB 数据写入

```cpp
// 写入 RGB 像素（可选 clamp）
void set_rgb(uint8_t* rgb, int x, int y, int width,
             int r, int g, int b, int bit_depth, bool should_clamp,
             size_t rgb_byte_size = 0);

// 写入 RGB 像素（带 clamp）
void set_rgb_clamped(uint8_t* rgb, int x, int y, int width,
                     int r, int g, int b, int bit_depth,
                     size_t rgb_byte_size = 0);

// 写入 RGB 像素（不 clamp）
void set_rgb_raw(uint8_t* rgb, int x, int y, int width,
                 int r, int g, int b, int bit_depth,
                 size_t rgb_byte_size = 0);

// 8-bit / 16-bit 快速通路
void set_rgb_8(uint8_t* rgb, int x, int y, int width, int r, int g, int b);
void set_rgb_8_clamp(uint8_t* rgb, int x, int y, int width, int r, int g, int b);
void set_rgb_16(uint8_t* rgb, int x, int y, int width, int r, int g, int b);
```

### 6.5 16-bit 内存操作

```cpp
uint16_t read_u16(const uint8_t* data, size_t byte_offset);
void write_u16(uint8_t* data, size_t byte_offset, uint16_t val);
```

---

## 7. 原始数据分析 API

自动分析未知 RAW 文件，检测位深、分辨率、Bayer 模式。

### 7.1 analyze_data()

```cpp
DataInfo analyze_data(const uint8_t* data, size_t byte_size);
```

分析策略：
1. 匹配已知传感器分辨率（5320×4600, 4024×3032, 1006×758 等）
2. 16-bit 值范围检测
3. Packed 格式启发式检测
4. 通用回退

**示例：**

```cpp
std::ifstream file("unknown.raw", std::ios::binary | std::ios::ate);
size_t size = file.tellg();
file.seekg(0);
std::vector<uint8_t> data(size);
file.read((char*)data.data(), size);

DataInfo info = analyze_data(data.data(), size);
std::cout << "检测位深: " << info.detected_bit_depth << " bit" << std::endl;
std::cout << "建议分辨率: " << info.suggested_width << "x"
          << info.suggested_height << std::endl;
std::cout << "像素数: " << info.pixel_count << std::endl;
std::cout << "值范围: [" << info.min_value << ", " << info.max_value << "]" << std::endl;

for (auto& dim : info.possible_dimensions) {
    std::cout << "可能分辨率: " << dim.first << " x " << dim.second << std::endl;
}
```

### 7.2 detect_bit_depth()

```cpp
int detect_bit_depth(const uint8_t* data, size_t byte_size);
```

快速检测 RAW 数据的位深。

### 7.3 suggest_dimensions()

```cpp
std::vector<std::pair<int, int>> suggest_dimensions(size_t pixel_count);
```

给定像素总数，推测可能的分辨率组合。优先匹配常见分辨率，然后尝试因数分解。

### 7.4 guess_pattern()

```cpp
BayerPattern guess_pattern(const uint8_t* data, int width, int height,
                           int bit_depth = 8, bool is_packed = false);
```

启发式检测 Bayer 模式。通过分析 2×2 块的亮度模式判断滤色阵列类型。

---

## 8. CUDA GPU 加速 API

> **前置条件：** 构建时启用 `IM_OPERATOR_ENABLE_CUDA=ON`（默认），且系统安装了 CUDA Toolkit。

### 8.1 单帧 CUDA 处理

```cpp
DemosaicError demosaic_cuda(const uint8_t* bayer_data, uint8_t* rgb_data,
                                  int width, int height, BayerPattern pattern,
                                  DemosaicAlgorithm algorithm, int bit_depth = 8,
                                  bool is_packed = false);
```

与 `demosaic_cpu()` 参数相同，但**仅在 GPU 上执行**。如果 CUDA 不可用，返回错误。

### 8.2 批量帧 CUDA 处理（流水线加速）

```cpp
DemosaicError demosaic_cuda_batch(const uint8_t* const* bayer_data_array,
                                         uint8_t* const* rgb_data_array,
                                         int num_frames,
                                         int width, int height,
                                         BayerPattern pattern,
                                         DemosaicAlgorithm algorithm,
                                         int bit_depth = 8);
```

**特点：**
- 使用双缓冲 slot + CUDA Graph 缓存
- Pinned memory 实现 H2D / Kernel / D2H 流水线重叠
- 适合视频流或连续帧处理

**注意：** 批量模式**不支持 packed 格式**。

**示例（处理 100 帧）：**

```cpp
const int N = 100;
std::vector<std::vector<uint8_t>> bayer_frames(N);
std::vector<std::vector<uint8_t>> rgb_frames(N);
std::vector<const uint8_t*> bayer_ptrs(N);
std::vector<uint8_t*> rgb_ptrs(N);

for (int i = 0; i < N; i++) {
    bayer_frames[i].resize(bayer_size);
    rgb_frames[i].resize(rgb_size);
    // 加载帧数据...
    bayer_ptrs[i] = bayer_frames[i].data();
    rgb_ptrs[i] = rgb_frames[i].data();
}

auto err = demosaic_cuda_batch(
    bayer_ptrs.data(), rgb_ptrs.data(),
    N, width, height,
    BayerPattern::RGGB, DemosaicAlgorithm::DFPD, 8
);
```

---

## 9. AVX2 SIMD 优化 API

> **前置条件：** 构建时启用 `IM_OPERATOR_ENABLE_SIMD=ON`（默认），且 CPU 支持 AVX2 指令集。

以下函数仅在 **8-bit unpacked** 数据时被内部调度自动使用：

```cpp
void process_super_fast_optimized(const uint8_t* bayer, uint8_t* rgb,
                                   int width, int height, const PatternOffsets& po,
                                   int bit_depth, bool is_packed = false);

void process_mg_optimized(...);
void process_hqli_optimized(...);
void process_l7_optimized(...);
```

> 一般用户无需直接调用这些函数。使用 `demosaic()` 会自动选择 SIMD 优化路径。

---

## 10. 能力查询函数

```cpp
// 当前 CPU 是否支持 AVX2？
bool has_avx2();

// 当前系统是否安装了 CUDA？
bool has_cuda();

// 获取 CUDA 设备名称（如 "NVIDIA GeForce RTX 4090"）
// 如果 CUDA 不可用，返回 "No CUDA device"
const char* cuda_device_name();
```

**示例：**

```cpp
std::cout << "AVX2: " << (has_avx2() ? "Yes" : "No") << std::endl;
std::cout << "CUDA: " << (has_cuda() ? "Yes" : "No") << std::endl;
if (has_cuda()) {
    std::cout << "CUDA Device: " << cuda_device_name() << std::endl;
}
std::cout << "CPU Cores: " << compute_hardware_concurrency() << std::endl;
```

---

## 11. 构建与集成

### 11.1 CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `IM_OPERATOR_ENABLE_CUDA` | ON | CUDA GPU 加速 |
| `IM_OPERATOR_BUILD_TESTS` | ON | 构建测试程序 |
| `IM_OPERATOR_BUILD_BENCHMARKS` | ON | 构建基准测试 |
| `IM_OPERATOR_ENABLE_SIMD` | ON | AVX2 SIMD 优化 |
| `IM_OPERATOR_ENABLE_OPENMP` | ON | OpenMP 多线程 |

### 11.2 构建命令

```powershell
# 默认构建（Release + CUDA + SIMD + OpenMP）
.\build.ps1

# 无 CUDA 构建
.\build.ps1 -NoCuda

# Debug 构建
.\build.ps1 -Config Debug

# 清理重建
.\build.ps1 -Clean -Rebuild

# 构建并运行测试
.\build.ps1 -Target test_im_operator -Run
```

### 11.3 集成到你的 CMake 项目

```cmake
# 方式一：add_subdirectory
add_subdirectory(path/to/SoolinOperator)
target_link_libraries(your_app PRIVATE im_operator)

# 方式二：find_package（安装后）
find_package(Demosaic 2.0 REQUIRED)
target_link_libraries(your_app PRIVATE im_operator)
```

### 11.4 编译宏

你的代码可以通过以下宏判断编译配置：

| 宏 | 含义 |
|---|------|
| `IM_OPERATOR_HAS_CUDA` | CUDA 编译已启用 |
| `__AVX2__` | AVX2 SIMD 已启用 |
| `_OPENMP` | OpenMP 已启用 |

---

## 12. 错误处理

### 12.1 检查返回码

```cpp
auto err = dm.process(...);

// 方式 1：使用 ok()
if (ok(err)) { /* 成功 */ }

// 方式 2：使用 !
if (!err) { /* 失败 */ }

// 方式 3：直接比较
if (err == DemosaicError::Ok) { /* 成功 */ }
```

### 12.2 获取错误描述

```cpp
if (!ok(err)) {
    std::cerr << "错误: " << demosaic_error_message(err) << std::endl;
}
```

### 12.3 错误码含义

| 错误码 | 含义 | 常见原因 |
|--------|------|---------|
| `Ok` | 成功 | - |
| `NullInput` | 输入/输出指针为空 | `bayer_data` 或 `rgb_data` 为 `nullptr` |
| `InvalidDimensions` | 无效图像尺寸 | `width` 或 `height` ≤ 0 |
| `InvalidBitDepth` | 无效位深 | `bit_depth` 不在 1-16 范围 |
| `InvalidPattern` | 无效 Bayer 模式 | 传入了未定义的模式值 |
| `ImageTooSmall` | 图像太小 | 尺寸小于算法窗口要求 |
| `InternalError` | 内部错误 | CUDA 内存分配失败等 |

---

## 13. 性能基准参考

以下数据在 **Intel Core i9 + NVIDIA RTX 4090** 上测得（5320×4600，8-bit）：

| 算法 | CPU 时间 | CPU 吞吐量 | GPU 时间 | GPU 吞吐量 |
|------|----------|-----------|----------|-----------|
| SUPER_FAST | ~25 ms | ~980 MP/s | ~1 ms | ~24.5 GP/s |
| HQLI | ~45 ms | ~540 MP/s | ~2 ms | ~12.3 GP/s |
| MG | ~50 ms | ~490 MP/s | ~2 ms | ~12.3 GP/s |
| L7 | ~70 ms | ~350 MP/s | ~3 ms | ~8.2 GP/s |
| DFPD | ~120 ms | ~200 MP/s | ~5 ms | ~4.9 GP/s |
| AHD | ~150 ms | ~160 MP/s | ~6 ms | ~4.1 GP/s |
| AMAZE | ~180 ms | ~136 MP/s | ~7 ms | ~3.5 GP/s |
| RCD | ~250 ms | ~98 MP/s | ~10 ms | ~2.5 GP/s |
| PRISM | ~350 ms | ~70 MP/s | ~14 ms | ~1.7 GP/s |

> 实际性能取决于 CPU 型号、GPU 型号、OpenMP 线程数、图像尺寸和位深。

---

## 附录 A：Packed 格式详解

### A.1 10-bit Packed

每 4 个像素用 5 字节存储。布局：

```
Byte 0   Byte 1   Byte 2   Byte 3   Byte 4  (共享字节)
[P0_h8][P1_h8][P2_h8][P3_h8][P3_l2|P2_l2|P1_l2|P0_l2]
```

### A.2 12-bit Packed

每 2 个像素用 3 字节存储。布局：

```
Byte 0  Byte 1    Byte 2
[P0_h8][P1_l4|P0_l4][P1_h8]
```

### A.3 14/16-bit Packed（通用）

使用位偏移读取：`bit_offset = pixel_index * bit_depth`

### A.4 函数参数说明

当 `is_packed = true` 时，`bayer_data` 的大小应为：

```
compute_packed_byte_size(width, height, bit_depth)
```

即 `ceil(width * height * bit_depth / 8)` 字节。

---

## 附录 B：RGB 输出格式

| 位深 | 每像素字节 | 通道布局 | 值范围 |
|------|-----------|---------|--------|
| 8-bit | 3 bytes | R, G, B 交错 | 0-255 |
| 10-bit | 6 bytes (3×uint16) | R, G, B 交错 | 0-1023 |
| 12-bit | 6 bytes (3×uint16) | R, G, B 交错 | 0-4095 |
| 14-bit | 6 bytes (3×uint16) | R, G, B 交错 | 0-16383 |
| 16-bit | 6 bytes (3×uint16) | R, G, B 交错 | 0-65535 |

> **注意：** 16-bit 输出数据存储在 `uint8_t*` 缓冲区中，可通过 `reinterpret_cast<uint16_t*>` 访问。
> 值已自动 MSB 对齐到 16-bit 范围。

---

## 附录 C：处理流程图

```
用户调用 process(bayer, rgb, w, h, pattern, algo, bit_depth, packed)
    │
    ▼
validate_demosaic_inputs() ─── 空指针、尺寸、位深校验
    │
    ▼
    has_cuda() ?
    │
    ├── YES ──► demosaic_cuda()
    │               │
    │               ├── cudaMemcpy H2D
    │               ├── launch_{ALGO}_kernel<<<>>>
    │               └── cudaMemcpy D2H
    │               │
    │               ▼ 成功? ──YES──► 返回 Ok
    │               │
    │               NO（回退 CPU）
    │
    └──► find_demosaic_func(algo) ──► 算法注册表查找
                │
                ▼
        ┌───────────────────────────┐
        │ 根据 bit_depth + packed   │
        │ 选择对应的核心实现        │
        ├───────────────────────────┤
        │ • 8-bit unpacked          │
        │   └─► AVX2 优化版（优先） │
        │   └─► 纯 C++ 向量化版    │
        │ • >8-bit unpacked         │
        │   └─► 16-bit 核心        │
        │ • packed 格式             │
        │   └─► packed 核心        │
        └───────────────────────────┘
                │
                ▼
        fill_rgb_borders() ─── 边界像素填充
                │
                ▼
            返回 Ok
```
