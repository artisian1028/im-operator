# iris-memo Operator API 参考手册

## 目录

1. [概述](#概述)
2. [Demosaic - 去马赛克](#1-demosaic---去马赛克)
3. [Denoise - 降噪](#2-denoise---降噪)
4. [White Balance - 白平衡](#3-white-balance---白平衡)
5. [CCM - 色彩校正矩阵](#4-ccm---色彩校正矩阵)
6. [LUT - 3D查找表](#5-lut---3d查找表)
7. [Sharpen - 锐化](#6-sharpen---锐化)
8. [Tone - 色调映射](#7-tone---色调映射)
9. [Color Temperature - 色温调整](#8-color-temperature---色温调整)
10. [Saturation - 饱和度](#9-saturation---饱和度)
11. [JPEG Codec - JPEG编解码](#10-jpeg-codec---jpeg编解码)
12. [HDR Tone Mapping - HDR色调映射](#11-hdr-tone-mapping---hdr色调映射)
13. [Black Level - 黑电平校正](#12-black-level---黑电平校正)
14. [Defect Correction - 坏点校正](#13-defect-correction---坏点校正)
15. [Highlight Reconstruction - 高光重建](#14-highlight-reconstruction---高光重建)
16. [Local Contrast - 局部对比度](#15-local-contrast---局部对比度)
17. [Lens Shading - 镜头阴影校正](#16-lens-shading---镜头阴影校正)
18. [Color Calibration - 颜色校准](#17-color-calibration---颜色校准)
19. [Analyzer - 数据分析工具](#18-analyzer---数据分析工具)
20. [Pixel Utils - 像素工具](#19-pixel-utils---像素工具)
21. [Calibration - 相机标定](#20-calibration---相机标定)
22. [通用模式与约定](#通用模式与约定)

---

## 概述

iris-memo Operator（项目名 `im_operator`，版本 2.0.0）是一个 C++17 图像处理算子库，提供从原始 Bayer 数据到最终 RGB 图像的全套处理流水线。库以静态库形式提供，支持 CPU（AVX2 加速 + OpenMP 多线程）和 CUDA GPU 加速两种后端。

### 命名空间体系

| 模块 | 主命名空间 | 头文件 |
|------|-----------|--------|
| 去马赛克 / 分析器 / 像素工具 | `imop` | `im_operator.h` |
| 降噪 | `denoise` | `denoise.h` |
| 白平衡 | `white_balance` | `white_balance.h` |
| 色彩校正矩阵 | `ccm` | `ccm.h` |
| 3D LUT | `lut` | `lut.h` |
| 锐化 | `sharpen` | `sharpen.h` |
| 色调映射 | `tone` | `tone.h` |
| 色温调整 | `color_temp` | `color_temp.h` |
| 饱和度 | `saturation` | `saturation.h` |
| JPEG编解码 | `jpeg_codec` | `jpeg_codec.h` |
| HDR色调映射 | `hdr` | `hdr.h` |
| 黑电平 | `black_level` | `black_level.h` |
| 坏点校正 | `defect_correct` | `defect_correct.h` |
| 高光重建 | `highlight_reconstruct` | `highlight_reconstruct.h` |
| 局部对比度 | `local_contrast` | `local_contrast.h` |
| 镜头阴影 | `lens_shading` | `lens_shading.h` |
| 颜色校准 | `color_calibration` | `color_calibration.h` |
| 相机标定 | `calibration` | `calibration.h` |

### 通用约定

- 所有错误返回值为枚举类型，`Ok` 表示成功
- 支持 `operator!` 和 `ok()` 辅助函数快速判错
- `bit_depth` 范围为 1-16（除 JPEG 固定为 8，HDR 支持 0=float32）
- 输入/输出均为 `const uint8_t*` / `uint8_t*` 裸指针，调用方负责内存管理
- 每个模块均有 `has_cuda()` 和对应的 `cuda` 变体函数

### 通用辅助函数

以下辅助函数在各模块中按相同模式提供：

**输入验证函数** — 在调用主处理函数前预处理检查参数有效性：

```cpp
// 标准模式（input + output 分离的模块）
XxxError validate_xxx_inputs(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels, int bit_depth);

// 就地修改模式（black_level, defect_correct, lens_shading）
XxxError validate_xxx_inputs(const uint8_t* data,
                               int width, int height, int channels, int bit_depth);

// 特例（demosaic）
DemosaicError validate_demosaic_inputs(const uint8_t* bayer_data, uint8_t* rgb_data,
                                        int width, int height, int bit_depth);
```

**元数据函数** — 获取算法可读名称和所需最小窗口尺寸：

```cpp
std::string algorithm_name(XxxAlgorithm algo);   // 算法可读名称
int algorithm_window_size(XxxAlgorithm algo);      // 算法所需最小窗口尺寸
```

**全局运行时查询函数：**

```cpp
bool has_avx2();                    // CPU: 检测当前CPU是否支持AVX2指令集
bool has_cuda();                    // GPU: 检测CUDA是否可用
const char* cuda_device_name();     // GPU: 获取当前CUDA设备名称
void cuda_synchronize();            // GPU: 同步CUDA设备（等待所有CUDA操作完成）
int compute_hardware_concurrency(); // CPU: 获取硬件并发线程数（OpenMP线程数）
```

---

## 1. Demosaic - 去马赛克

将各像元只含单一颜色通道的 Bayer 原始数据插值还原为每个像素三个通道的全彩 RGB 图像。

### 1.1 命名空间与头文件

```cpp
#include <im_operator.h>
// 或分量包含:
#include <imop/types.hpp>
#include <imop/algorithms.hpp>

namespace imop { ... }
```

### 1.2 枚举定义

**BayerPattern** - Bayer 色彩滤波阵列排列模式：

| 值 | 说明 | 左上角2x2布局 |
|----|------|-------------|
| `RGGB` | 红-绿/绿-蓝 | R G / G B |
| `BGGR` | 蓝-绿/绿-红 | B G / G R |
| `GRBG` | 绿-红/蓝-绿 | G R / B G |
| `GBRG` | 绿-蓝/红-绿 | G B / R G |

**DemosaicAlgorithm** - 去马赛克算法：

| 值 | 说明 | 特点 |
|----|------|------|
| `SUPER_FAST` | 超快速双线性插值 | 速度最快，质量最简 |
| `HQLI` | 高质量线性插值（Malvar-He-Cutler） | 梯度校正线性插值 |
| `MG` | 基于中值的梯度插值 | 边缘敏感插值 |
| `L7` | 七阶滤波器插值 | 多方向加权 |
| `DFPD` | 方向滤波伪像素检测 | 方向自适应 |
| `AHD` | 自适应同质化导向 | 同质区域选择 |
| `AMAZE` | AMaZE（Aliasing Minimization and Zipper Elimination） | 抗锯齿首选 |
| `RCD` | 残差通道差分 | 残差插值 |
| `PRISM` | PRISM算法 | 近边插值 |

**DemosaicError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidPattern` | Bayer模式无效 |
| `ImageTooSmall` | 图像太小，无法支持所选算法 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_demosaic_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）、`has_avx2()` / `has_cuda()`（运行时检测）、`compute_hardware_concurrency()`（并发线程数）。

### 1.3 主分发函数

```cpp
DemosaicError demosaic(
    const uint8_t* bayer_data,   // 输入：Bayer原始数据
    uint8_t* rgb_data,           // 输出：RGB交错数据（3通道）
    int width,                   // 图像宽度（像素）
    int height,                  // 图像高度（像素）
    BayerPattern pattern,        // Bayer排列模式
    DemosaicAlgorithm algorithm, // 去马赛克算法
    int bit_depth = 8,           // 位深：1-16
    bool is_packed = false       // 是否为压缩格式（10/12bit打包原始数据）
);
```

```cpp
// CPU专用（不尝试使用CUDA）
DemosaicError demosaic_cpu(
    const uint8_t* bayer_data, uint8_t* rgb_data,
    int width, int height, BayerPattern pattern,
    DemosaicAlgorithm algorithm, int bit_depth = 8,
    bool is_packed = false
);
```

```cpp
// CUDA GPU加速
DemosaicError demosaic_cuda(
    const uint8_t* bayer_data, uint8_t* rgb_data,
    int width, int height, BayerPattern pattern,
    DemosaicAlgorithm algorithm, int bit_depth = 8,
    bool is_packed = false
);

// CUDA批量处理
DemosaicError demosaic_cuda_batch(
    const uint8_t* const* bayer_data_array,  // 多帧输入数组
    uint8_t* const* rgb_data_array,          // 多帧输出数组
    int num_frames, int width, int height,
    BayerPattern pattern,
    DemosaicAlgorithm algorithm,
    int bit_depth = 8
);
```

### 1.4 单独算法函数

每种算法均有独立的函数接口，签名一致：

```cpp
DemosaicError process_hqli(const uint8_t* bayer, uint8_t* rgb,
    int width, int height, BayerPattern pattern, int bit_depth,
    bool is_packed = false);

DemosaicError process_mg(const uint8_t* bayer, uint8_t* rgb,
    int width, int height, BayerPattern pattern, int bit_depth,
    bool is_packed = false);

DemosaicError process_l7(const uint8_t* bayer, uint8_t* rgb,
    int width, int height, BayerPattern pattern, int bit_depth,
    bool is_packed = false);

DemosaicError process_dfpd(const uint8_t* bayer, uint8_t* rgb,
    int width, int height, BayerPattern pattern, int bit_depth,
    bool is_packed = false);

DemosaicError process_ahd(const uint8_t* bayer, uint8_t* rgb,
    int width, int height, BayerPattern pattern, int bit_depth,
    bool is_packed = false);

DemosaicError process_amaze(const uint8_t* bayer, uint8_t* rgb,
    int width, int height, BayerPattern pattern, int bit_depth,
    bool is_packed = false);

DemosaicError process_rcd(const uint8_t* bayer, uint8_t* rgb,
    int width, int height, BayerPattern pattern, int bit_depth,
    bool is_packed = false);

DemosaicError process_prism(const uint8_t* bayer, uint8_t* rgb,
    int width, int height, BayerPattern pattern, int bit_depth,
    bool is_packed = false);

DemosaicError process_super_fast(const uint8_t* bayer, uint8_t* rgb,
    int width, int height, BayerPattern pattern, int bit_depth,
    bool is_packed = false);
```

### 1.5 数据格式说明

- **输入（Bayer）**：每像素1个值（单通道），共 width * height 个值
  - 8bit：每值1字节
  - 10/12bit打包：按位打包存储，每值占用 bit_depth 位
  - 16bit：每值2字节（小端序 uint16）
- **输出（RGB）**：每像素3个值（R, G, B 交错），共 width * height * 3 个值
  - 8bit：每像素3字节
  - 16bit：每像素6字节（3个uint16小端序）

### 1.6 使用示例

```cpp
#include <im_operator.h>

// 场景：处理一张 1920x1080 的 RGGB Bayer 12bit 打包图像
int width = 1920, height = 1080, bit_depth = 12;
size_t bayer_bytes = imop::pixel::compute_bayer_byte_size(width, height, bit_depth, true);
size_t rgb_bytes = imop::pixel::compute_rgb_byte_size(width, height, bit_depth);

std::vector<uint8_t> bayer_data(bayer_bytes);
std::vector<uint8_t> rgb_data(rgb_bytes);

// 从文件读取 bayer_data ...

imop::DemosaicError err = imop::demosaic(
    bayer_data.data(), rgb_data.data(),
    width, height,
    imop::BayerPattern::RGGB,
    imop::DemosaicAlgorithm::AMAZE,
    bit_depth,
    true   // is_packed = true (12bit packed)
);

if (err != imop::DemosaicError::Ok) {
    fprintf(stderr, "去马赛克失败: %s\n", imop::demosaic_error_message(err));
    return 1;
}

// 或使用 Demosaic 类封装
imop::Demosaic dm;
auto err2 = dm.process(bayer_data.data(), rgb_data.data(),
    width, height, imop::BayerPattern::RGGB,
    imop::DemosaicAlgorithm::HQLI, bit_depth, true);
```

---

## 2. Denoise - 降噪

对 RGB 或灰度图像执行降噪处理，支持空域和频域多种算法。

### 2.1 命名空间与头文件

```cpp
#include <denoise.h>
namespace denoise { ... }
```

### 2.2 枚举定义

**DenoiseAlgorithm** - 降噪算法：

| 值 | 说明 | 适用场景 |
|----|------|---------|
| `GAUSSIAN` | 高斯模糊（可分离、快速） | 高斯噪声，通用降噪 |
| `MEDIAN` | 中值滤波 | 椒盐噪声（Salt & Pepper） |
| `BILATERAL` | 双边滤波（保边） | 边缘保留降噪 |
| `NLM` | 非局部均值（基于块） | 纹理保持，质量最高 |
| `WAVELET` | 小波阈值（频域） | 宽带噪声，多尺度 |
| `BAYER_DENOISE` | Bayer域降噪（原始CFA数据） | 原始数据预降噪 |

**DenoiseError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须1或3） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_denoise_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 2.3 主分发函数

```cpp
DenoiseError process_denoise(
    const uint8_t* input,        // 输入：RGB或灰度数据
    uint8_t* output,             // 输出：与输入同格式
    int width,                   // 图像宽度
    int height,                  // 图像高度
    int channels,                // 通道数：1=灰度，3=RGB
    DenoiseAlgorithm algorithm,  // 降噪算法
    int bit_depth = 8,           // 位深：1-16
    float strength = 1.0f        // 降噪强度 [0.0, 2.0]，1.0为默认
);
```

### 2.4 单独算法函数

```cpp
DenoiseError process_gaussian(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float strength);

DenoiseError process_median(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float strength);

DenoiseError process_bilateral(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float strength);

DenoiseError process_nlm(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float strength);

DenoiseError process_bayer_denoise(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float strength);

DenoiseError process_wavelet(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float strength);
```

### 2.5 CUDA 支持

```cpp
bool has_cuda();
DenoiseError process_denoise_cuda(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    DenoiseAlgorithm algorithm, int bit_depth, float strength);
```

### 2.6 数据格式说明

- **输入/输出**：均为 RGB 交错格式（channels=3）或灰度（channels=1）
- 8bit：每通道1字节，RGB每像素3字节
- 16bit：每通道2字节，RGB每像素6字节

### 2.7 使用示例

```cpp
#include <denoise.h>

int width = 1024, height = 768;
size_t rgb_size = static_cast<size_t>(width) * height * 3; // 8bit

std::vector<uint8_t> input(rgb_size);
std::vector<uint8_t> output(rgb_size);

// 使用双边滤波降噪，强度1.5
denoise::DenoiseError err = denoise::process_denoise(
    input.data(), output.data(),
    width, height, 3,
    denoise::DenoiseAlgorithm::BILATERAL,
    8,   // bit_depth
    1.5f // strength
);

if (denoise::ok(err)) {
    // output 中包含降噪后的RGB数据
}
```

---

## 3. White Balance - 白平衡

对 RGB 图像执行白平衡校正，补偿光源色温偏差。

### 3.1 命名空间与头文件

```cpp
#include <white_balance.h>
namespace white_balance { ... }
```

### 3.2 枚举定义

**WhiteBalanceAlgorithm** - 白平衡算法：

| 值 | 说明 | 原理 |
|----|------|------|
| `GRAY_WORLD` | 灰度世界假设 | 各通道均值应相等，据此计算增益 |
| `WHITE_PATCH` | 白块法（Max-RGB） | 最亮像素为白点 |
| `SHADE_OF_GRAY` | 灰度阴影法 | Minkowski范数推广的灰度世界 |
| `MANUAL` | 手动增益 | 用户直接指定 R/G/B 增益系数 |

**WhiteBalanceError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_white_balance_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 3.3 参数结构体

```cpp
struct WBCoefficients {
    float r_gain = 1.0f;   // 红色增益
    float g_gain = 1.0f;   // 绿色增益
    float b_gain = 1.0f;   // 蓝色增益
};
```

### 3.4 主分发函数

```cpp
WhiteBalanceError process_white_balance(
    const uint8_t* input,             // 输入：RGB数据（channels=3）
    uint8_t* output,                  // 输出：白平衡校正后RGB数据
    int width,                        // 图像宽度
    int height,                       // 图像高度
    int channels,                     // 通道数（必须3）
    WhiteBalanceAlgorithm algorithm,  // 白平衡算法
    int bit_depth = 8,                // 位深：1-16
    float p = 6.0f,                   // Minkowski范数参数（用于SHADE_OF_GRAY）
    const WBCoefficients& manual_gains = {}  // 手动增益（用于MANUAL）
);
```

### 3.5 单独算法函数

```cpp
WhiteBalanceError process_gray_world(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float p,
    const WBCoefficients& manual_gains);

WhiteBalanceError process_white_patch(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float p,
    const WBCoefficients& manual_gains);

WhiteBalanceError process_shade_of_gray(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float p,
    const WBCoefficients& manual_gains);

WhiteBalanceError process_manual_wb(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float p,
    const WBCoefficients& manual_gains);
```

### 3.6 分析工具函数

```cpp
// 仅计算增益系数，不修改图像（用于分析/预览）
WBCoefficients compute_white_balance_gains(
    const uint8_t* input, int width, int height, int bit_depth,
    WhiteBalanceAlgorithm algorithm, float p = 6.0f
);
```

### 3.7 CUDA 支持

```cpp
bool has_cuda();
WhiteBalanceError process_white_balance_cuda(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float r_gain, float g_gain, float b_gain);
```

### 3.8 使用示例

```cpp
#include <white_balance.h>

// 自动白平衡
white_balance::WhiteBalanceError err = white_balance::process_white_balance(
    input_rgb, output_rgb,
    width, height, 3,
    white_balance::WhiteBalanceAlgorithm::GRAY_WORLD
);

// 手动白平衡
white_balance::WBCoefficients gains;
gains.r_gain = 1.8f;
gains.g_gain = 1.0f;
gains.b_gain = 1.2f;
white_balance::process_white_balance(
    input_rgb, output_rgb,
    width, height, 3,
    white_balance::WhiteBalanceAlgorithm::MANUAL,
    8, 6.0f, gains
);
```

---

## 4. CCM - 色彩校正矩阵

对 RGB 图像应用线性色彩校正矩阵，实现色彩空间转换或传感器颜色校正。

### 4.1 命名空间与头文件

```cpp
#include <ccm.h>
namespace ccm { ... }
```

### 4.2 枚举定义

**CCMAlgorithm** - 色彩校正矩阵算法：

| 值 | 说明 | 矩阵维度 |
|----|------|---------|
| `LINEAR_3X3` | 标准 3x3 线性矩阵 | 3x3 |
| `LINEAR_4X3` | 4x3 矩阵（含偏置行） | 3x4 |
| `POLYNOMIAL_3X9` | 二阶多项式展开（3x9） | 3x9 |
| `MANUAL` | 用户自定义矩阵 | 用户指定 |

**CCMError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_ccm_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 4.3 矩阵结构体

```cpp
// 3x3 线性矩阵（行主序）
// out_r = m[0]*r + m[1]*g + m[2]*b
// out_g = m[3]*r + m[4]*g + m[5]*b
// out_b = m[6]*r + m[7]*g + m[8]*b
struct CCMatrix3x3 {
    CCMAlgorithm matrix_type = CCMAlgorithm::LINEAR_3X3;
    float m[9] = { 1,0,0,  0,1,0,  0,0,1 };  // 默认为单位矩阵
};

// 3x4 矩阵（含偏置）
// out_r = m[0]*r + m[1]*g + m[2]*b + m[3]
// out_g = m[4]*r + m[5]*g + m[6]*b + m[7]
// out_b = m[8]*r + m[9]*g + m[10]*b + m[11]
struct CCMatrix3x4 {
    CCMAlgorithm matrix_type = CCMAlgorithm::LINEAR_4X3;
    float m[12] = { 1,0,0,0,  0,1,0,0,  0,0,1,0 };
};

// 3x9 多项式矩阵（二阶展开）
// 特征向量: [R, G, B, RG, RB, GB, R², G², B²]
// out_r = M_r · features, out_g = M_g · features, out_b = M_b · features
struct CCMatrix3x9 {
    CCMAlgorithm matrix_type = CCMAlgorithm::POLYNOMIAL_3X9;
    float m[27] = { /* 3行x9列，行主序，默认为只取R通道的单位映射 */ };
};
```

### 4.4 主分发函数

```cpp
CCMError process_ccm(
    const uint8_t* input,        // 输入：RGB数据（channels=3）
    uint8_t* output,             // 输出：校正后RGB数据
    int width,                   // 图像宽度
    int height,                  // 图像高度
    int channels,                // 通道数（必须3）
    CCMAlgorithm algorithm,      // CCM算法
    int bit_depth = 8,           // 位深：1-16
    const void* matrix = nullptr // 矩阵数据指针（类型依算法而定）
);
```

### 4.5 单独算法函数

```cpp
CCMError process_linear_3x3(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* matrix);

CCMError process_linear_4x3(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* matrix);

CCMError process_polynomial_3x9(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* matrix);

CCMError process_manual_ccm(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* matrix);
```

### 4.6 预定义矩阵函数

```cpp
CCMatrix3x3 srgb_to_xyz_d65();    // sRGB → XYZ (D65)
CCMatrix3x3 xyz_to_srgb_d65();    // XYZ (D65) → sRGB
CCMatrix3x3 srgb_to_bt709();      // sRGB → BT.709
CCMatrix3x3 bt709_to_srgb();      // BT.709 → sRGB
CCMatrix3x3 identity_3x3();       // 单位矩阵
CCMatrix3x3 saturation_matrix(float sat); // 饱和度矩阵（0=灰度,1=原图,>1=增强）
```

### 4.7 使用示例

```cpp
#include <ccm.h>

// 使用 sRGB→XYZ 转换矩阵
ccm::CCMatrix3x3 matrix = ccm::srgb_to_xyz_d65();
ccm::CCMError err = ccm::process_ccm(
    input_rgb, output_rgb,
    width, height, 3,
    ccm::CCMAlgorithm::LINEAR_3X3,
    8,
    &matrix
);

// 或者直接调用单独函数
ccm::process_linear_3x3(input_rgb, output_rgb,
    width, height, 3, 8, &matrix);
```

---

## 5. LUT - 3D查找表

对 RGB 图像应用 3D LUT 进行色彩风格化处理，支持 .cube 文件导入和内置预设风格。

### 5.1 命名空间与头文件

```cpp
#include <lut.h>
namespace lut { ... }
```

### 5.2 枚举定义

**LUTAlgorithm** - LUT算法：

| 值 | 说明 | 参数 |
|----|------|------|
| `CUBE_FILE` | 导入 .cube 格式 LUT 文件 | lut_data = 文件路径字符串 |
| `CUSTOM_3D` | 用户自定义 3D LUT 数据 | lut_data = LUT3D* 指针 |
| `SEPIA` | 内置怀旧棕色调 | lut_data = nullptr |
| `COOL` | 内置冷色调（蓝调） | lut_data = nullptr |
| `WARM` | 内置暖色调（琥珀调） | lut_data = nullptr |
| `HIGH_CONTRAST` | 内置高对比度 S 曲线 | lut_data = nullptr |
| `LOW_CONTRAST` | 内置低对比度（褪色感） | lut_data = nullptr |
| `INVERT` | 内置色彩反转 | lut_data = nullptr |
| `VINTAGE_FADE` | 内置复古褪色 | lut_data = nullptr |

**LUTError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `LUTSizeMismatch` | LUT 尺寸不匹配 |
| `FileNotFound` | LUT 文件未找到 |
| `FileParseError` | LUT 文件解析错误 |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_lut_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 5.3 参数结构体

```cpp
// 3D LUT 数据结构（R-major存储）
// 数据: lut[ri][gi][bi] = {r, g, b}，使用三线性插值
struct LUT3D {
    int size = 0;                  // 每维度采样点数（如33 = 33³ 个格点）
    std::vector<float> data;       // size³ * 3 个浮点数，R主序排列
    bool empty() const;
    size_t total_samples() const;  // = size * size * size
};
```

### 5.4 主分发函数

```cpp
LUTError process_lut(
    const uint8_t* input,        // 输入：RGB数据（channels=3）
    uint8_t* output,             // 输出：LUT处理后RGB数据
    int width,                   // 图像宽度
    int height,                  // 图像高度
    int channels,                // 通道数（必须3）
    LUTAlgorithm algorithm,      // LUT算法
    int bit_depth = 8,           // 位深：1-16
    const void* lut_data = nullptr, // 算法相关数据
    int lut_size = 33            // LUT尺寸（每维度采样点数，默认33）
);
```

### 5.5 单独算法函数

```cpp
LUTError process_cube_file(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* lut_data, int lut_size);

LUTError process_custom_3d(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* lut_data, int lut_size);

// 各内置风格函数
LUTError process_style_sepia(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_cool(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_warm(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_high_contrast(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_low_contrast(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_invert(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* lut_data, int lut_size);

LUTError process_style_vintage_fade(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const void* lut_data, int lut_size);
```

### 5.6 LUT 构建与 I/O 函数

```cpp
LUT3D load_cube_file(const char* filepath);          // 从 .cube 文件加载
LUT3D build_sepia_lut(int size = 33);               // 构建怀旧棕 LUT
LUT3D build_cool_lut(int size = 33);                // 构建冷色调 LUT
LUT3D build_warm_lut(int size = 33);                // 构建暖色调 LUT
LUT3D build_high_contrast_lut(int size = 33);        // 构建高对比度 LUT
LUT3D build_low_contrast_lut(int size = 33);         // 构建低对比度 LUT
LUT3D build_invert_lut(int size = 33);              // 构建反转 LUT
LUT3D build_vintage_fade_lut(int size = 33);         // 构建复古褪色 LUT
LUT3D build_identity_lut(int size = 33);            // 构建恒等（无效果）LUT

// 直接应用 LUT3D 到RGB图像
LUTError apply_lut(const LUT3D& lut, const uint8_t* input, uint8_t* output,
                   int width, int height, int bit_depth);
```

### 5.7 CUDA 支持

```cpp
bool has_cuda();
LUTError process_lut_cuda(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const LUT3D& lut_data);
```

### 5.8 使用示例

```cpp
#include <lut.h>

// 方式一：使用内置风格
lut::LUTError err = lut::process_lut(
    input_rgb, output_rgb,
    width, height, 3,
    lut::LUTAlgorithm::WARM
);

// 方式二：从 .cube 文件加载
const char* path = "path/to/lut.cube";
lut::LUTError err2 = lut::process_lut(
    input_rgb, output_rgb,
    width, height, 3,
    lut::LUTAlgorithm::CUBE_FILE,
    8, path, 33
);

// 方式三：构建并使用自定义 LUT
lut::LUT3D my_lut = lut::build_sepia_lut(33);
lut::apply_lut(my_lut, input_rgb, output_rgb, width, height, 8);
```

---

## 6. Sharpen - 锐化

对 RGB 图像执行锐化处理，增强边缘和细节。

### 6.1 命名空间与头文件

```cpp
#include <sharpen.h>
namespace sharpen { ... }
```

### 6.2 枚举定义

**SharpenAlgorithm** - 锐化算法：

| 值 | 说明 | 原理 |
|----|------|------|
| `UNSHARP_MASK` | 经典反锐化掩模 | sharpen = original + amount * (original - blurred) |
| `LAPLACIAN` | 拉普拉斯边缘增强 | 3x3/5x5 拉普拉斯算子 |
| `HIGH_PASS` | 高通滤波叠加 | 高通滤波提取细节叠加到原图 |
| `ADAPTIVE` | 自适应锐化 | 内容自适应（边缘感知）锐化 |

**SharpenError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_sharpen_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 6.3 参数结构体

```cpp
struct SharpenParams {
    float amount = 1.0f;     // 锐化强度 [0, 3]
    float radius = 1.0f;     // 模糊半径（反锐化掩模）/ 拉普拉斯sigma
    float threshold = 0.0f;  // 自适应模式下的边缘阈值 [0, 1]
};
```

### 6.4 主分发函数

```cpp
SharpenError process_sharpen(
    const uint8_t* input,          // 输入：RGB数据（channels=3）
    uint8_t* output,               // 输出：锐化后RGB数据
    int width,                     // 图像宽度
    int height,                    // 图像高度
    int channels,                  // 通道数（必须3）
    SharpenAlgorithm algorithm,    // 锐化算法
    int bit_depth = 8,             // 位深：1-16
    const SharpenParams& params = {} // 锐化参数
);
```

### 6.5 单独算法函数

```cpp
SharpenError process_unsharp_mask(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const SharpenParams& params);

SharpenError process_laplacian(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const SharpenParams& params);

SharpenError process_high_pass(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const SharpenParams& params);

SharpenError process_adaptive(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const SharpenParams& params);
```

### 6.6 使用示例

```cpp
#include <sharpen.h>

sharpen::SharpenParams params;
params.amount = 1.5f;   // 增强锐化强度
params.radius = 1.2f;   // 稍大的模糊半径
params.threshold = 0.05f;

sharpen::SharpenError err = sharpen::process_sharpen(
    input_rgb, output_rgb,
    width, height, 3,
    sharpen::SharpenAlgorithm::UNSHARP_MASK,
    8, params
);
```

---

## 7. Tone - 色调映射

对 RGB 图像执行色调调整，包括 Gamma 校正、S 曲线、色阶和阴影/高光控制。

### 7.1 命名空间与头文件

```cpp
#include <tone.h>
namespace tone { ... }
```

### 7.2 枚举定义

**ToneAlgorithm** - 色调算法：

| 值 | 说明 | 原理 |
|----|------|------|
| `GAMMA` | 幂律 Gamma 校正 | out = in^gamma |
| `S_CURVE` | S 曲线对比度 | 阴影压缩 + 高光压缩 |
| `LEVELS` | 色阶调整 | 黑场/白场/中间调裁切 |
| `CURVES_3POINT` | 三点贝塞尔色调曲线 | 阴影/中间调/高光三点控制 |
| `SHADOWS_HIGHLIGHTS` | 阴影提升 / 高光恢复 | 独立阴影区域提亮和高光区域压暗 |

**ToneError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_tone_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 7.3 参数结构体

```cpp
struct ToneParams {
    float gamma = 1.0f;        // Gamma值 [0.1, 10.0]，1.0=恒等
    float contrast = 0.0f;     // 对比度 [-1, 2]，0=恒等
    float shadows = 0.0f;      // 阴影提升 [-1, 1]
    float highlights = 0.0f;   // 高光恢复 [-1, 1]
    float black_point = 0.0f;  // 色阶黑场裁切 [0, 1]
    float white_point = 1.0f;  // 色阶白场裁切 [0, 1]
    float mid_point = 0.5f;    // 色阶中间调 [0, 1]
};
```

### 7.4 主分发函数

```cpp
ToneError process_tone(
    const uint8_t* input,        // 输入：RGB数据（channels=3）
    uint8_t* output,             // 输出：色调调整后RGB数据
    int width,                   // 图像宽度
    int height,                  // 图像高度
    int channels,                // 通道数（必须3）
    ToneAlgorithm algorithm,     // 色调算法
    int bit_depth = 8,           // 位深：1-16
    const ToneParams& params = {} // 色调参数
);
```

### 7.5 单独算法函数

```cpp
ToneError process_gamma(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const ToneParams& params);

ToneError process_s_curve(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const ToneParams& params);

ToneError process_levels(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const ToneParams& params);

ToneError process_curves_3point(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const ToneParams& params);

ToneError process_shadows_highlights(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const ToneParams& params);
```

### 7.6 使用示例

```cpp
#include <tone.h>

tone::ToneParams params;
params.gamma = 2.2f;           // sRGB标准gamma
params.contrast = 0.3f;        // 适度增加对比度
params.black_point = 0.02f;    // 轻微黑场裁切

tone::ToneError err = tone::process_tone(
    input_rgb, output_rgb,
    width, height, 3,
    tone::ToneAlgorithm::GAMMA,
    8, params
);
```

---

## 8. Color Temperature - 色温调整

对 RGB 图像执行色温调整，通过色温值（Kelvin）、预设光源或手动增益改变图像白平衡偏向。

### 8.1 命名空间与头文件

```cpp
#include <color_temp.h>
namespace color_temp { ... }
```

### 8.2 枚举定义

**ColorTempAlgorithm** - 色温算法：

| 值 | 说明 | 原理 |
|----|------|------|
| `KELVIN` | 基于 Kelvin 值的 RGB 调整 | 黑体辐射模型转换为 RGB 增益 |
| `PRESET` | 命名标准光源预设 | 预定义的色温值 |
| `MANUAL` | 用户自定义 RGB 乘数 | 手动指定 R/B 增益 |
| `WHITE_BALANCE` | 自动白平衡 | 对输入图像执行灰度世界白平衡 |

**IlluminantPreset** - 标准光源预设：

| 值 | 色温（约） | 说明 |
|----|-----------|------|
| `CANDLE` | ~1850K | 烛光 |
| `TUNGSTEN_40W` | ~2600K | 40W 钨丝灯 |
| `TUNGSTEN_100W` | ~2850K | 100W 钨丝灯 |
| `HALOGEN` | ~3200K | 卤素灯 |
| `WARM_FLUORESCENT` | ~3500K | 暖色荧光灯 |
| `COOL_WHITE_FLUO` | ~4200K | 冷白荧光灯 |
| `MIDDAY_SUN` | ~5500K | 正午日光 |
| `CLOUDY` | ~6500K | 阴天（D65标准光源） |
| `SHADE` | ~7500K | 阴影处 |
| `OVERCAST` | ~8000K | 多云 |
| `BLUE_SKY` | ~10000K | 蓝天 |

**ColorTempError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_color_temp_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 8.3 主分发函数

```cpp
ColorTempError process_color_temp(
    const uint8_t* input,                // 输入：RGB数据（channels=3）
    uint8_t* output,                     // 输出：色温调整后RGB数据
    int width,                           // 图像宽度
    int height,                          // 图像高度
    int channels,                        // 通道数（必须3）
    ColorTempAlgorithm algorithm,        // 色温算法
    int bit_depth = 8,                   // 位深：1-16
    int kelvin = 6500,                   // 色温值（Kelvin，1000-40000）
    IlluminantPreset preset = IlluminantPreset::CLOUDY,  // 光源预设
    float r_gain = 1.0f,                 // 手动R增益（用于MANUAL）
    float b_gain = 1.0f                  // 手动B增益（用于MANUAL）
);
```

### 8.4 单独算法函数

```cpp
ColorTempError process_kelvin(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, int kelvin,
    IlluminantPreset preset, float r_gain, float b_gain);

ColorTempError process_preset(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, int kelvin,
    IlluminantPreset preset, float r_gain, float b_gain);

ColorTempError process_manual_temp(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, int kelvin,
    IlluminantPreset preset, float r_gain, float b_gain);

ColorTempError process_white_balance_auto(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, int kelvin,
    IlluminantPreset preset, float r_gain, float b_gain);
```

### 8.5 工具函数

```cpp
// Kelvin → 线性 RGB 乘数（黑体辐射近似）
void kelvin_to_rgb_multipliers(int kelvin, float& r_mult, float& b_mult);
// G 乘数始终为 1.0

// 光源预设 → 线性 RGB 乘数
void illuminant_to_rgb_multipliers(IlluminantPreset preset, float& r_mult, float& b_mult);

// 获取光源预设的标称 Kelvin 值
int illuminant_kelvin(IlluminantPreset preset);

// 获取光源预设的显示名称
const char* illuminant_name(IlluminantPreset preset);
```

### 8.6 使用示例

```cpp
#include <color_temp.h>

// 将图像色温调至 3200K（钨丝灯）
color_temp::ColorTempError err = color_temp::process_color_temp(
    input_rgb, output_rgb,
    width, height, 3,
    color_temp::ColorTempAlgorithm::KELVIN,
    8, 3200
);

// 使用阴天预设
err = color_temp::process_preset(
    input_rgb, output_rgb,
    width, height, 3,
    8, 6500,
    color_temp::IlluminantPreset::CLOUDY,
    1.0f, 1.0f
);
```

---

## 9. Saturation - 饱和度

对 RGB 图像执行饱和度调整，支持 HSL、鲜艳度、通道混合和选择性饱和等算法。

### 9.1 命名空间与头文件

```cpp
#include <saturation.h>
namespace saturation { ... }
```

### 9.2 枚举定义

**SaturationAlgorithm** - 饱和度算法：

| 值 | 说明 | 特点 |
|----|------|------|
| `HSL` | 基于HSL色彩空间 | 转为HSL → 缩放S → 转回RGB |
| `VIBRANCE` | 智能鲜艳度 | 保护肤色，增强低饱和区域 |
| `CHANNEL_MIXER` | 通道混合 | 跨通道饱和度混合 |
| `SELECTIVE` | 选择性饱和度 | R、G、B 通道独立调节 |

**SaturationError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_saturation_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 9.3 参数结构体

```cpp
struct SaturationParams {
    float saturation = 1.0f;   // 全局饱和度 [0, 3]，1.0=恒等
    float r_sat = 1.0f;        // R 通道饱和度（SELECTIVE模式）
    float g_sat = 1.0f;        // G 通道饱和度（SELECTIVE模式）
    float b_sat = 1.0f;        // B 通道饱和度（SELECTIVE模式）
    float vibrance = 1.0f;     // 鲜艳度强度 [0, 3]（VIBRANCE模式）
};
```

### 9.4 主分发函数

```cpp
SaturationError process_saturation(
    const uint8_t* input,              // 输入：RGB数据（channels=3）
    uint8_t* output,                   // 输出：饱和度调整后RGB数据
    int width,                         // 图像宽度
    int height,                        // 图像高度
    int channels,                      // 通道数（必须3）
    SaturationAlgorithm algorithm,     // 饱和度算法
    int bit_depth = 8,                 // 位深：1-16
    const SaturationParams& params = {} // 饱和度参数
);
```

### 9.5 单独算法函数

```cpp
SaturationError process_hsl(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const SaturationParams& params);

SaturationError process_vibrance(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const SaturationParams& params);

SaturationError process_channel_mixer(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const SaturationParams& params);

SaturationError process_selective(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const SaturationParams& params);
```

### 9.6 CUDA 支持

```cpp
bool has_cuda();
SaturationError process_saturation_cuda(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, float sat, float vib);
```

### 9.7 使用示例

```cpp
#include <saturation.h>

saturation::SaturationParams params;
params.saturation = 1.3f;   // 增加30%全局饱和度

saturation::SaturationError err = saturation::process_saturation(
    input_rgb, output_rgb,
    width, height, 3,
    saturation::SaturationAlgorithm::HSL,
    8, params
);

// 使用鲜明度（保护肤色）
params.vibrance = 1.5f;
saturation::process_vibrance(input_rgb, output_rgb,
    width, height, 3, 8, params);
```

---

## 10. JPEG Codec - JPEG编解码

对 RGB 图像进行 JPEG 编码（压缩）和解码（解压缩）。

### 10.1 命名空间与头文件

```cpp
#include <jpeg_codec.h>
namespace jpeg_codec { ... }
```

### 10.2 枚举定义

**JpegAlgorithm** - JPEG算法：

| 值 | 说明 |
|----|------|
| `ENCODE_BASELINE` | CPU 基线 DCT JPEG 编码器（RGB → JPEG） |
| `DECODE_BASELINE` | CPU 基线 DCT JPEG 解码器（JPEG → RGB） |
| `ENCODE_CUDA` | CUDA GPU 加速 JPEG 编码器 |
| `DECODE_CUDA` | CUDA GPU 加速 JPEG 解码器 |

**JpegError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入指针为空 |
| `NullOutput` | 输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（JPEG必须为8） |
| `InvalidChannels` | 通道数无效（JPEG必须为3） |
| `InvalidQuality` | 质量参数无效（必须1-100） |
| `ImageTooSmall` | 图像太小（最小8x8） |
| `EncodeFailed` | JPEG编码失败 |
| `DecodeFailed` | JPEG解码失败 |
| `InvalidJpegData` | 无效或损坏的JPEG数据 |
| `CudaNotAvailable` | CUDA不可用 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_jpeg_inputs()` / `validate_jpeg_encode_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）、`get_max_jpeg_size()`（计算JPEG输出缓冲区大小）。

### 10.3 参数结构体

```cpp
struct JpegParams {
    int quality = 90;              // JPEG质量（1-100，越高越好）
    bool progressive = false;      // 渐进式JPEG（仅CPU）
    bool optimize = true;          // 优化Huffman表
    int chroma_subsample = 1;      // 色度子采样：0=4:4:4, 1=4:2:0, 2=4:2:2
};
```

### 10.4 主函数

```cpp
// JPEG 编码
// output_size: [in/out] 输入为输出缓冲区容量，输出为实际写入的JPEG大小
JpegError process_jpeg_encode(
    const uint8_t* input,        // 输入：RGB像素数据（交错）
    uint8_t* output,             // 输出：JPEG压缩数据（调用方分配）
    size_t* output_size,         // [in/out] JPEG输出大小
    int width,                   // 图像宽度
    int height,                  // 图像高度
    int channels,                // 通道数（必须3）
    JpegAlgorithm algorithm,     // 编码算法
    int bit_depth = 8,           // 位深（必须8）
    const JpegParams& params = {} // JPEG参数
);

// JPEG 解码
// width, height, channels: [out] 解码后的图像属性
JpegError process_jpeg_decode(
    const uint8_t* input,        // 输入：JPEG压缩数据
    size_t input_size,           // JPEG数据大小（字节）
    uint8_t* output,             // 输出：解码后RGB数据（调用方分配）
    int* width,                  // [out] 解码后宽度
    int* height,                 // [out] 解码后高度
    int* channels,               // [out] 解码后通道数
    JpegAlgorithm algorithm      // 解码算法
);

// 获取给定图像尺寸下的最大JPEG输出大小（最坏情况）
size_t get_max_jpeg_size(int width, int height, int channels);
```

### 10.5 单独算法函数

```cpp
// CPU编码器
JpegError process_encode_baseline(const uint8_t* input, uint8_t* output,
    size_t* output_size, int width, int height, int channels,
    int bit_depth, const JpegParams& params);

// CPU解码器
JpegError process_decode_baseline(const uint8_t* input, size_t input_size,
    uint8_t* output, int* width, int* height, int* channels);

// CUDA编码器
JpegError process_encode_cuda(const uint8_t* input, uint8_t* output,
    size_t* output_size, int width, int height, int channels,
    int bit_depth, const JpegParams& params);

// CUDA解码器
JpegError process_decode_cuda(const uint8_t* input, size_t input_size,
    uint8_t* output, int* width, int* height, int* channels);
```

### 10.6 使用示例

```cpp
#include <jpeg_codec.h>

// --- 编码 ---
int width = 1920, height = 1080;
size_t rgb_size = static_cast<size_t>(width) * height * 3;
size_t max_jpeg_size = jpeg_codec::get_max_jpeg_size(width, height, 3);

std::vector<uint8_t> rgb_data(rgb_size);
std::vector<uint8_t> jpeg_data(max_jpeg_size);

jpeg_codec::JpegParams params;
params.quality = 95;

size_t actual_jpeg_size = max_jpeg_size;
jpeg_codec::JpegError err = jpeg_codec::process_jpeg_encode(
    rgb_data.data(), jpeg_data.data(), &actual_jpeg_size,
    width, height, 3,
    jpeg_codec::JpegAlgorithm::ENCODE_BASELINE,
    8, params
);

// --- 解码 ---
int decoded_w, decoded_h, decoded_c;
size_t decoded_rgb_size = static_cast<size_t>(width) * height * 3;
std::vector<uint8_t> decoded_rgb(decoded_rgb_size);

err = jpeg_codec::process_jpeg_decode(
    jpeg_data.data(), actual_jpeg_size,
    decoded_rgb.data(),
    &decoded_w, &decoded_h, &decoded_c,
    jpeg_codec::JpegAlgorithm::DECODE_BASELINE
);
```

---

## 11. HDR Tone Mapping - HDR色调映射

将高动态范围（HDR）图像压缩映射为低动态范围（LDR）显示设备可呈现的图像。支持多种色调映射算子，以及标准 HDR 传递函数（PQ、HLG）。

### 11.1 命名空间与头文件

```cpp
#include <hdr.h>
namespace hdr { ... }
```

### 11.2 枚举定义

**HdrAlgorithm** - HDR算法：

| 值 | 说明 | 特点 |
|----|------|------|
| `REINHARD` | 经典 Reinhard 全局映射 | L/(1+L)，简单快速 |
| `REINHARD_EXT` | Reinhard 扩展 | 带 key 值和白点控制 |
| `FILMIC_ACES` | ACES RRT+ODT（Narkowicz近似） | 电影感色调曲线 |
| `HABLE` | Uncharted 2 电影曲线 | John Hable 的电影色调映射 |
| `DRAGO` | 自适应对数映射 | 对高亮度区域自适应 |
| `ADAPTIVE_LOCAL` | 双边分解局部映射 | 保持局部对比度 |
| `EXPONENTIAL` | 指数映射 | 1 - exp(-k * L) |
| `LOGARITHMIC` | 对数映射 | log(1+k*L) / log(1+k) |
| `LINEAR_TO_PQ` | 线性 → ST.2084 PQ | HDR显示器标准 |
| `PQ_TO_LINEAR` | ST.2084 PQ → 线性 | HDR显示器反变换 |
| `LINEAR_TO_HLG` | 线性 → BT.2100 HLG OETF | 广播HDR标准 |
| `HLG_TO_LINEAR` | BT.2100 HLG EOTF → 线性 | 广播HDR反变换 |

**HdrError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须0=float32或1-32） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ImageTooSmall` | 图像太小 |
| `CudaNotAvailable` | CUDA不可用 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_hdr_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 11.3 参数结构体

```cpp
struct HdrParams {
    float exposure = 0.0f;       // EV曝光调整 [-8, 8]
    float gamma = 2.2f;          // 输出gamma [0.5, 4.0]
    float saturation = 1.0f;     // 色调映射后饱和度 [0, 2]
    float key = 0.18f;           // 中灰key值 [0.01, 1.0]
    float white_point = 1.0f;    // Reinhard白点 [0.5, 20.0]
    float strength = 1.0f;       // 通用强度 [0, 2]
};

// 亮度系数（BT.709 基色）
constexpr float kLumaR = 0.2126f;
constexpr float kLumaG = 0.7152f;
constexpr float kLumaB = 0.0722f;
```

### 11.4 主分发函数

```cpp
HdrError process_hdr(
    const uint8_t* input,        // 输入：RGB数据（channels=3）
    uint8_t* output,             // 输出：色调映射后RGB数据
    int width,                   // 图像宽度
    int height,                  // 图像高度
    int channels,                // 通道数（必须3）
    HdrAlgorithm algorithm,      // HDR算法
    int bit_depth = 16,          // 位深：0=float32，1-32为整数
    const HdrParams& params = {} // HDR参数
);
```

### 11.5 单独算法函数

```cpp
HdrError process_reinhard(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_reinhard_ext(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_filmic_aces(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_hable(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_drago(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_adaptive_local(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_exponential(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_logarithmic(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_linear_to_pq(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_pq_to_linear(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_linear_to_hlg(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);

HdrError process_hlg_to_linear(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HdrParams& params);
```

### 11.6 CUDA 支持

```cpp
bool has_cuda();
HdrError process_hdr_cuda(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    HdrAlgorithm algorithm, int bit_depth,
    const HdrParams& params);
```

### 11.7 数据格式说明

- **float32 模式（bit_depth=0）**：每通道4字节（float），RGB每像素12字节
- **16bit 模式（bit_depth=16）**：每通道2字节（uint16 小端序），RGB每像素6字节
- **8bit 模式（bit_depth=8）**：每通道1字节，RGB每像素3字节

### 11.8 使用示例

```cpp
#include <hdr.h>

hdr::HdrParams params;
params.exposure = 1.0f;      // +1EV曝光
params.gamma = 2.2f;         // sRGB gamma
params.key = 0.18f;          // 中灰

// 使用Reinhard扩展色调映射
hdr::HdrError err = hdr::process_hdr(
    input_hdr, output_ldr,
    width, height, 3,
    hdr::HdrAlgorithm::REINHARD_EXT,
    16,    // 16bit HDR输入
    params
);

// 使用ACES电影感色调映射
hdr::process_filmic_aces(input_hdr, output_ldr,
    width, height, 3, 16, params);
```

---

## 12. Black Level - 黑电平校正

对 Bayer 原始数据执行黑电平偏移减除，可针对每个颜色通道独立设置偏移量或使用全局统一偏移。

**注意：此模块操作 Bayer 原始数据（每像素单通道），就地修改（in-place）。**

### 12.1 命名空间与头文件

```cpp
#include <black_level.h>
namespace black_level { ... }
// 使用 imop::BayerPattern
```

### 12.2 枚举定义

**BlackLevelAlgorithm** - 黑电平算法：

| 值 | 说明 |
|----|------|
| `PER_CHANNEL` | R、Gr、Gb、B 各通道独立偏移 |
| `GLOBAL` | 所有像素使用统一偏移值 |

**BlackLevelError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（Bayer必须为1） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_black_level_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 12.3 参数结构体

```cpp
struct BlackLevelParams {
    float r_offset = 0.0f;   // R 通道偏移量
    float gr_offset = 0.0f;  // Gr 通道偏移量
    float gb_offset = 0.0f;  // Gb 通道偏移量
    float b_offset = 0.0f;   // B 通道偏移量
};
```

### 12.4 主分发函数

```cpp
BlackLevelError process_black_level(
    uint8_t* data,                      // [in/out] Bayer原始数据（就地修改）
    int width,                          // 图像宽度
    int height,                         // 图像高度
    BayerPattern pattern,               // Bayer排列模式
    BlackLevelAlgorithm algorithm,      // 黑电平算法
    int bit_depth,                      // 位深：1-16
    const BlackLevelParams& params      // 偏移参数
);
```

### 12.5 单独算法函数

```cpp
BlackLevelError process_per_channel(uint8_t* data,
    int width, int height, BayerPattern pattern,
    int bit_depth, const BlackLevelParams& params);

BlackLevelError process_global(uint8_t* data,
    int width, int height, BayerPattern pattern,
    int bit_depth, const BlackLevelParams& params);
```

> 对于 GLOBAL 模式，使用 `r_offset` 作为全局偏移值。

### 12.6 数据格式说明

- 输入为 Bayer 原始数据，每像素1个值
- **就地修改**：输出直接覆写输入缓冲区
- 偏移量为浮点数，会自动转换为整数值后进行减除，结果 clamp 到 [0, max_val]

### 12.7 使用示例

```cpp
#include <black_level.h>

black_level::BlackLevelParams params;
params.r_offset = 64.0f;     // R通道减去64
params.gr_offset = 64.0f;    // Gr通道减去64
params.gb_offset = 64.0f;    // Gb通道减去64
params.b_offset = 64.0f;     // B通道减去64

black_level::BlackLevelError err = black_level::process_black_level(
    bayer_data, width, height,
    imop::BayerPattern::RGGB,
    black_level::BlackLevelAlgorithm::PER_CHANNEL,
    12, params
);

if (black_level::ok(err)) {
    // bayer_data 已就地完成黑电平校正
}
```

---

## 13. Defect Correction - 坏点校正

检测并修复 Bayer 原始数据中的坏点（缺陷像素），支持自适应检测和基于坏点图的两种模式。

**注意：此模块操作 Bayer 原始数据（每像素单通道），就地修改（in-place）。**

### 13.1 命名空间与头文件

```cpp
#include <defect_correct.h>
namespace defect_correct { ... }
// 使用 imop::BayerPattern
```

### 13.2 枚举定义

**DefectCorrectAlgorithm** - 坏点校正算法：

| 值 | 说明 |
|----|------|
| `ADAPTIVE` | 自动检测并修复：比较同色相邻像素差值 |
| `MAP_BASED` | 基于坏点图修复已知位置的坏点 |

**DefectCorrectError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（Bayer必须为1） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_defect_correct_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 13.3 参数结构体

```cpp
struct DefectPoint {
    int x, y;   // 坏点坐标
};

struct DefectCorrectParams {
    float threshold = 0.3f;            // 自适应模式：相对偏差阈值 [0.05, 1.0]
    const DefectPoint* map = nullptr;  // 基于坏点图模式：坏点坐标数组
    int map_count = 0;                 // 坏点数量
};
```

### 13.4 主分发函数

```cpp
DefectCorrectError process_defect_correct(
    uint8_t* data,                         // [in/out] Bayer原始数据（就地修改）
    int width,                             // 图像宽度
    int height,                            // 图像高度
    BayerPattern pattern,                  // Bayer排列模式
    DefectCorrectAlgorithm algorithm,      // 坏点校正算法
    int bit_depth,                         // 位深：1-16
    const DefectCorrectParams& params      // 校正参数
);
```

### 13.5 单独算法函数

```cpp
DefectCorrectError process_adaptive(uint8_t* data,
    int width, int height, BayerPattern pattern,
    int bit_depth, const DefectCorrectParams& params);

DefectCorrectError process_map_based(uint8_t* data,
    int width, int height, BayerPattern pattern,
    int bit_depth, const DefectCorrectParams& params);
```

### 13.6 使用示例

```cpp
#include <defect_correct.h>

// 自适应坏点检测与修复
defect_correct::DefectCorrectParams params;
params.threshold = 0.25f;  // 较低的检测阈值（更敏感）

defect_correct::DefectCorrectError err = defect_correct::process_defect_correct(
    bayer_data, width, height,
    imop::BayerPattern::RGGB,
    defect_correct::DefectCorrectAlgorithm::ADAPTIVE,
    12, params
);
```

---

## 14. Highlight Reconstruction - 高光重建

对因过曝导致单通道或多通道裁切的 RGB 图像进行高光区域色彩重建。

### 14.1 命名空间与头文件

```cpp
#include <highlight_reconstruct.h>
namespace highlight_reconstruct { ... }
```

### 14.2 枚举定义

**HighlightReconstructAlgorithm** - 高光重建算法：

| 值 | 说明 | 原理 |
|----|------|------|
| `CHANNEL_GUIDED` | 通道引导重建 | 利用未裁切通道的比例关系估算裁切通道值 |
| `GRADIENT_BASED` | 基于梯度重建 | 从裁切边界向内传播梯度信息 |

**HighlightReconstructError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_highlight_reconstruct_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 14.3 参数结构体

```cpp
struct HighlightReconstructParams {
    float threshold = 0.95f;  // 裁切检测阈值（相对于 max_val 的比例值）
};
```

### 14.4 主分发函数

```cpp
HighlightReconstructError process_highlight_reconstruct(
    const uint8_t* input,                        // 输入：RGB数据（channels=3）
    uint8_t* output,                             // 输出：高光重建后RGB数据
    int width,                                   // 图像宽度
    int height,                                  // 图像高度
    int channels,                                // 通道数（必须3）
    HighlightReconstructAlgorithm algorithm,     // 重建算法
    int bit_depth,                               // 位深：1-16
    const HighlightReconstructParams& params     // 重建参数
);
```

### 14.5 单独算法函数

```cpp
HighlightReconstructError process_channel_guided(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HighlightReconstructParams& params);

HighlightReconstructError process_gradient_based(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const HighlightReconstructParams& params);
```

### 14.6 使用示例

```cpp
#include <highlight_reconstruct.h>

highlight_reconstruct::HighlightReconstructParams hparams;
hparams.threshold = 0.98f;  // 将亮度超过最大值98%的像素视为裁切

highlight_reconstruct::HighlightReconstructError err =
    highlight_reconstruct::process_highlight_reconstruct(
        input_rgb, output_rgb,
        width, height, 3,
        highlight_reconstruct::HighlightReconstructAlgorithm::CHANNEL_GUIDED,
        16, hparams
    );
```

---

## 15. Local Contrast - 局部对比度

对大半径范围内的图像局部对比度进行增强（类似 Lightroom 的"清晰度/Clarity"效果）。

### 15.1 命名空间与头文件

```cpp
#include <local_contrast.h>
namespace local_contrast { ... }
```

### 15.2 枚举定义

**LocalContrastAlgorithm** - 局部对比度算法：

| 值 | 说明 | 原理 |
|----|------|------|
| `UNSHARP` | 大半径反锐化掩模 | 原图 + amount * (原图 - 大幅模糊) |
| `BILATERAL` | 双边分解增强 | 利用双边滤波分解出细节层进行增强 |

**LocalContrastError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_local_contrast_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 15.3 参数结构体

```cpp
struct LocalContrastParams {
    float amount = 0.0f;     // 强度 [0, 2]，0=关闭，1=默认Clarity
    float radius = 20.0f;    // 模糊sigma [3, 50]（像素）
    float threshold = 0.0f;  // 细节边缘阈值 [0, 0.5]
};
```

### 15.4 主分发函数

```cpp
LocalContrastError process_local_contrast(
    const uint8_t* input,                     // 输入：RGB数据（channels=3）
    uint8_t* output,                          // 输出：对比度增强后RGB数据
    int width,                                // 图像宽度
    int height,                               // 图像高度
    int channels,                             // 通道数（必须3）
    LocalContrastAlgorithm algorithm,         // 局部对比度算法
    int bit_depth,                            // 位深：1-16
    const LocalContrastParams& params         // 对比度参数
);
```

### 15.5 单独算法函数

```cpp
LocalContrastError process_unsharp(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const LocalContrastParams& params);

LocalContrastError process_bilateral(const uint8_t* input, uint8_t* output,
    int width, int height, int channels,
    int bit_depth, const LocalContrastParams& params);
```

### 15.6 使用示例

```cpp
#include <local_contrast.h>

local_contrast::LocalContrastParams lc_params;
lc_params.amount = 0.8f;     // 80%强度
lc_params.radius = 30.0f;    // 30像素半径
lc_params.threshold = 0.02f; // 边缘保护阈值

local_contrast::LocalContrastError err = local_contrast::process_local_contrast(
    input_rgb, output_rgb,
    width, height, 3,
    local_contrast::LocalContrastAlgorithm::UNSHARP,
    8, lc_params
);
```

---

## 16. Lens Shading - 镜头阴影校正

对 Bayer 原始数据执行镜头阴影（暗角）校正，补偿镜头边缘亮度衰减。

**注意：此模块操作 Bayer 原始数据（每像素单通道），就地修改（in-place）。**

### 16.1 命名空间与头文件

```cpp
#include <lens_shading.h>
namespace lens_shading { ... }
// 使用 imop::BayerPattern
```

### 16.2 枚举定义

**LensShadingAlgorithm** - 镜头阴影校正算法：

| 值 | 说明 | 原理 |
|----|------|------|
| `POLYNOMIAL` | 径向多项式增益 | gain(r) = 1 + a2*r² + a4*r⁴ + a6*r⁶ |
| `FLAT_FIELD` | 平场参考图像 | 使用均匀白光参考图像计算增益 |

**LensShadingError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入/输出指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须1-16） |
| `InvalidChannels` | 通道数无效（Bayer必须为1） |
| `ImageTooSmall` | 图像太小 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_lens_shading_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 16.3 参数结构体

```cpp
// 每通道多项式系数: gain(r) = 1 + a2*r² + a4*r⁴ + a6*r⁶
// r 是从光轴中心归一化距离，r ∈ [0, 1]
struct ShadingPolynomial {
    float a2 = 0.0f;
    float a4 = 0.0f;
    float a6 = 0.0f;
};

struct LensShadingParams {
    ShadingPolynomial r_coef;    // R 通道多项式
    ShadingPolynomial gr_coef;   // Gr 通道多项式
    ShadingPolynomial gb_coef;   // Gb 通道多项式
    ShadingPolynomial b_coef;    // B 通道多项式
    float center_x = 0.5f;       // 光轴中心 x（归一化 0-1）
    float center_y = 0.5f;       // 光轴中心 y（归一化 0-1）
    // 平场参考图像（FLAT_FIELD模式）
    const uint8_t* flat_field = nullptr;   // 平场图像（与输入格式相同）
    int flat_field_width = 0;
    int flat_field_height = 0;
};
```

### 16.4 主分发函数

```cpp
LensShadingError process_lens_shading(
    uint8_t* data,                          // [in/out] Bayer原始数据（就地修改）
    int width,                              // 图像宽度
    int height,                             // 图像高度
    BayerPattern pattern,                   // Bayer排列模式
    LensShadingAlgorithm algorithm,         // 镜头阴影校正算法
    int bit_depth,                          // 位深：1-16
    const LensShadingParams& params         // 校正参数
);
```

### 16.5 单独算法函数

```cpp
LensShadingError process_polynomial(uint8_t* data,
    int width, int height, BayerPattern pattern,
    int bit_depth, const LensShadingParams& params);

LensShadingError process_flat_field(uint8_t* data,
    int width, int height, BayerPattern pattern,
    int bit_depth, const LensShadingParams& params);
```

### 16.6 使用示例

```cpp
#include <lens_shading.h>

lens_shading::LensShadingParams ls_params;
ls_params.center_x = 0.5f;
ls_params.center_y = 0.5f;

// 设置多项式系数（边缘亮度衰减补偿）
ls_params.r_coef.a2 = 0.3f;
ls_params.r_coef.a4 = 0.1f;
ls_params.r_coef.a6 = 0.05f;
ls_params.gr_coef = ls_params.r_coef;
ls_params.gb_coef = ls_params.r_coef;
ls_params.b_coef = ls_params.r_coef;

lens_shading::LensShadingError err = lens_shading::process_lens_shading(
    bayer_data, width, height,
    imop::BayerPattern::RGGB,
    lens_shading::LensShadingAlgorithm::POLYNOMIAL,
    12, ls_params
);
```

---

## 17. Color Calibration - 颜色校准

对包含 X-Rite ColorChecker Classic 色卡的图像进行自动色块检测、颜色提取、CCM 矩阵求解和线性化 LUT 生成。

### 17.1 命名空间与头文件

```cpp
#include <color_calibration.h>
namespace color_calibration { ... }
```

### 17.2 枚举定义

**ColorCalibrationAlgorithm** - 颜色校准算法：

| 值 | 说明 |
|----|------|
| `DETECT_CHART` | 检测图像中的 X-Rite ColorChecker Classic 色卡 |
| `EXTRACT_PATCHES` | 从检测到的色块区域提取平均颜色 |
| `SOLVE_CCM` | 通过最小二乘法求解最优 CCM 矩阵 |
| `GENERATE_LINEARIZATION` | 从灰色色块生成线性化 LUT |

**MatrixType** - 矩阵类型：

| 值 | 说明 |
|----|------|
| `LINEAR_3X3` | 3x3 线性矩阵 |
| `LINEAR_4X3` | 3x4 矩阵（含偏置列） |
| `POLYNOMIAL_3X9` | 3x9 二阶多项式矩阵 |

**ColorCalibrationError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入指针为空 |
| `InvalidDimensions` | 图像尺寸无效 |
| `InvalidBitDepth` | 位深无效（必须8-16） |
| `InvalidChannels` | 通道数无效（必须3） |
| `ChartNotFound` | 图像中未检测到色卡 |
| `InsufficientPatches` | 提取到的色块不足 |
| `SingularMatrix` | 求解器遇到奇异矩阵 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `algorithm_name()`（算法名称查询）；各步骤函数不共用统一的 validate 函数，参数检查在每个函数内完成。

### 17.3 参数结构体

```cpp
// 色块区域（归一化坐标 0-1）
struct PatchRegion {
    float cx, cy;     // 中心点
    float half_w;     // 半宽
    float half_h;     // 半高
};

// 24个色块检测结果
struct ChartDetection {
    PatchRegion patches[24];
    bool valid[24];   // 各色块检测是否成功
};

// 单个色块的 RGB 测量值
struct PatchColor {
    float r, g, b;
};

// 24个色块的测量值集合
struct ChartMeasurements {
    PatchColor colors[24];
    int count;   // 有效测量数
};

// 求解后的 CCM
struct SolvedMatrix {
    MatrixType type;
    float m[27];    // 最大尺寸（3x9），超出部分为0
    int rows;       // 3
    int cols;       // 3, 4, 或 9
};

// 线性化 LUT
struct LinearizationLUT {
    std::vector<float> lut; // 大小为 lut_size，值域 [0, 1]
    int lut_size = 256;
};

// CCM 求解参数
struct SolveCCMParams {
    const PatchColor* measured = nullptr;   // 24个测量RGB值
    const PatchColor* reference = nullptr;  // 24个参考RGB值
    int patch_count = 24;
    MatrixType matrix_type = MatrixType::LINEAR_3X3;
};

// 线性化参数
struct LinearizationParams {
    const PatchColor* measured = nullptr;   // 灰色色块测量值（第19-24行）
    const PatchColor* reference = nullptr;  // 灰色色块参考值
    int gray_count = 6;
    int lut_size = 256;
};
```

### 17.4 单独算法函数

```cpp
// 检测色卡
ColorCalibrationError process_detect_chart(
    const uint8_t* input, int width, int height,
    int channels, int bit_depth,
    ChartDetection* result
);

// 提取色块颜色
ColorCalibrationError process_extract_patches(
    const uint8_t* input, int width, int height,
    int channels, int bit_depth,
    const ChartDetection* detection,
    ChartMeasurements* result
);

// 求解 CCM
ColorCalibrationError process_solve_ccm(
    const SolveCCMParams& params,
    SolvedMatrix* result
);

// 生成线性化 LUT
ColorCalibrationError process_generate_linearization(
    const LinearizationParams& params,
    LinearizationLUT* result
);
```

### 17.5 便捷全流程函数

```cpp
// 全流程: 检测 + 提取 + 求解 → 输出 CCMatrix3x3
ColorCalibrationError calibrate_from_chart(
    const uint8_t* input, int width, int height,
    int channels, int bit_depth,
    ccm::CCMatrix3x3* out_matrix,
    float* out_error = nullptr
);

// 全流程: 检测 + 提取 + 生成线性化 LUT
ColorCalibrationError linearize_from_chart(
    const uint8_t* input, int width, int height,
    int channels, int bit_depth,
    LinearizationLUT* out_lut
);

// 获取 ColorChecker Classic 标准参考值（sRGB D65，归一化 0-1）
void get_colorchecker_reference(PatchColor refs[24]);
```

### 17.6 使用示例

```cpp
#include <color_calibration.h>
#include <ccm.h>

// 全流程自动校准
ccm::CCMatrix3x3 matrix;
float calibration_error;

color_calibration::ColorCalibrationError err =
    color_calibration::calibrate_from_chart(
        chart_image,  // RGB 图像（含色卡）
        width, height, 3, 8,
        &matrix, &calibration_error
    );

if (color_calibration::ok(err)) {
    // 使用求解出的 CCM 矩阵校准后续图像
    ccm::process_ccm(input_rgb, output_rgb,
        width, height, 3,
        ccm::CCMAlgorithm::LINEAR_3X3,
        8, &matrix
    );
}
```

---

## 18. Analyzer - 数据分析工具

对原始 Bayer 数据进行分析，自动检测位深、推断尺寸和猜测 Bayer 模式。

### 18.1 命名空间与头文件

```cpp
#include <im_operator.h>
// 或分量包含:
#include <imop/analyzer.hpp>

namespace imop { ... }
```

### 18.2 函数

```cpp
// 通过数据统计推断位深（1-16）
int detect_bit_depth(const uint8_t* data, size_t byte_size);

// 根据像素总数建议可能的图像尺寸
std::vector<std::pair<int, int>> suggest_dimensions(size_t pixel_count);

// 综合数据分析，填充 DataInfo 结构
DataInfo analyze_data(const uint8_t* data, size_t byte_size);

// 从图像数据猜测 Bayer 模式
BayerPattern guess_pattern(const uint8_t* data, int width, int height,
                           int bit_depth = 8, bool is_packed = false);
```

### 18.3 参数结构体

```cpp
struct DataInfo {
    int detected_bit_depth = 0;            // 推断的位深
    int suggested_width = 0;               // 建议的宽度
    int suggested_height = 0;              // 建议的高度
    int pixel_count = 0;                   // 像素总数
    int max_value = 0;                     // 数据中的最大值
    int min_value = 0;                     // 数据中的最小值
    bool is_likely_16bit = false;          // 是否很可能是16位数据
    bool is_packed = false;                // 是否为打包格式
    std::vector<std::pair<int, int>> possible_dimensions;  // 所有可能的尺寸组合
};
```

### 18.4 使用示例

```cpp
#include <im_operator.h>

// 分析未知格式的原始数据
std::vector<uint8_t> raw_data = load_file("unknown.raw");
imop::DataInfo info = imop::analyze_data(raw_data.data(), raw_data.size());

printf("推断位深: %d\n", info.detected_bit_depth);
printf("像素总数: %d\n", info.pixel_count);

for (auto& dim : info.possible_dimensions) {
    printf("可能尺寸: %dx%d\n", dim.first, dim.second);
}

// 猜测 Bayer 模式（假设当前推测的尺寸和位深）
imop::BayerPattern pattern = imop::guess_pattern(
    raw_data.data(),
    info.suggested_width, info.suggested_height,
    info.detected_bit_depth, info.is_packed
);
```

---

## 19. Pixel Utils - 像素工具

提供 Bayer 原始数据和 RGB 数据的底层像素读写函数，支持打包/解包、对齐和 clamp 操作。

### 19.1 命名空间与头文件

```cpp
#include <im_operator.h>
// 或分量包含:
#include <imop/pixel_utils.hpp>

namespace imop::pixel { ... }
```

### 19.2 常量与辅助函数

```cpp
constexpr int kMaxBitDepth = 16;   // 支持的最大位深

// 计算最大有效值
int safe_max_val(int bit_depth);

// 计算字节大小
size_t compute_packed_byte_size(int width, int height, int bit_depth);
size_t compute_bayer_byte_size(int width, int height, int bit_depth, bool is_packed);
size_t compute_rgb_byte_size(int width, int height, int bit_depth);
```

### 19.3 Bayer 原始像素读写

```cpp
// 8-bit Bayer（每像素1字节）
int get_raw_8(const uint8_t* data, int x, int y, int width);

// 16-bit Bayer（每像素2字节，小端序）
int get_raw_16(const uint8_t* data, int x, int y, int width, int bit_depth);

// 通用 Bayer 读取（自动判断 bit_depth 和 is_packed）
int get_raw(const uint8_t* data, int x, int y, int width, int bit_depth,
            bool is_packed = false, size_t data_byte_size = 0, int height = 0);

// 带 clamp 的 Bayer 读取（坐标越界时 clamp 到边缘）
int get_clamped(const uint8_t* data, int x, int y, int width, int height,
                int bit_depth, bool is_packed = false, size_t data_byte_size = 0);
int get_clamped_8(const uint8_t* data, int x, int y, int width, int height);
int get_clamped_16(const uint8_t* data, int x, int y, int width, int height, int bit_depth);

// 打包格式读取（10bit / 12bit 等）
int get_packed_raw(const uint8_t* data, int x, int y, int width, int bit_depth,
                   size_t data_byte_size = 0, int height = 0);
int get_packed(const uint8_t* data, int x, int y, int width, int height,
               int bit_depth, size_t data_byte_size = 0);

// 原始 16-bit 值对齐
uint16_t align_raw_value(uint16_t val, int bit_depth);
```

### 19.4 RGB 像素写入

```cpp
// 8-bit RGB（带 clamp）
void set_rgb_8(uint8_t* rgb, int x, int y, int width, int r, int g, int b);
void set_rgb_8_clamp(uint8_t* rgb, int x, int y, int width, int r, int g, int b);

// 16-bit RGB
void set_rgb_16(uint8_t* rgb, int x, int y, int width, int r, int g, int b);

// 通用 RGB 写入（自动判断 bit_depth）
void set_rgb(uint8_t* rgb, int x, int y, int width,
             int r, int g, int b, int bit_depth, bool should_clamp);
void set_rgb_raw(uint8_t* rgb, int x, int y, int width,
                 int r, int g, int b, int bit_depth);
void set_rgb_clamped(uint8_t* rgb, int x, int y, int width,
                     int r, int g, int b, int bit_depth);
```

### 19.5 Bayer 模式查询

```cpp
// 判断当前坐标是否属于 R / G / B 通道
bool is_r_at(const PatternOffsets& po, int row, int col);
bool is_g_at(const PatternOffsets& po, int row, int col);
bool is_b_at(const PatternOffsets& po, int row, int col);
```

### 19.6 使用示例

```cpp
#include <imop/pixel_utils.hpp>
using namespace imop::pixel;

// 遍历 12-bit 打包 Bayer，读取每个像素值
size_t byte_size = compute_packed_byte_size(width, height, 12);
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        int pixel_val = get_packed(bayer_data, x, y, width, height, 12, byte_size);
        // 处理 pixel_val ...
    }
}

// 写入 16-bit RGB
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        set_rgb_16(rgb_data, x, y, width, r, g, b);
    }
}
```

---

## 20. Calibration - 相机标定

提供完整的相机几何标定流水线，包括三点标定杆、棋盘格检测与 Zhang 标定、DLT、立体标定、光束法平差以及大规模多相机系统的增量式 SfM 和稀疏 BA 优化。

### 20.1 命名空间与头文件

```cpp
#include <calibration.h>
namespace calibration { ... }
```

### 20.2 枚举定义

**CalibrationAlgorithm** - 标定算法：

| 值 | 说明 |
|----|------|
| `THREE_POINT` | 三点标定杆标定（内参 + 外参） |
| `RIGHT_TRIANGLE` | 直角三角形世界坐标系注册 |
| `CHECKERBOARD_DETECT` | 棋盘格角点亚像素检测 |
| `CHECKERBOARD_CALIBRATE` | Zhang 标定法（多视角棋盘格标定内参） |
| `DLT` | 直接线性变换：N 对 3D→2D 点对应求投影矩阵 |
| `STEREO_CALIBRATE` | 双目相机立体外参标定 |
| `BUNDLE_ADJUST` | 多相机联合优化（稀疏 LM 光束法平差） |

**CalibrationError** - 错误码：

| 值 | 说明 |
|----|------|
| `Ok` | 成功 |
| `NullInput` | 输入指针为空 |
| `InvalidCameraCount` | 相机数无效（必须 >= 1） |
| `InvalidFrameCount` | 帧数无效（必须 >= 3） |
| `InvalidConfiguration` | 标定配置参数无效 |
| `InsufficientObservations` | 观测数据不足 |
| `SingularMatrix` | 计算过程中出现奇异矩阵 |
| `OptimizationFailed` | 非线性优化未收敛 |
| `InternalError` | 内部处理错误 |

**辅助函数：** `validate_calibration_inputs()`（输入验证）、`algorithm_name()` / `algorithm_window_size()`（元数据查询）。

### 20.3 关键数据结构

```cpp
// 2D 图像点
struct Point2D { double x, y; };

// 摄像头内参（针孔模型 + 径向和切向畸变）
struct CameraIntrinsics {
    double fx, fy;     // 焦距（像素）
    double cx, cy;     // 主点
    double k1, k2, k3; // 径向畸变系数
    double p1, p2;     // 切向畸变系数
};

// 摄像头外参（世界 → 相机变换）
struct CameraExtrinsics {
    double rotation[9];    // 3x3 旋转矩阵（行主序）
    double translation[3]; // 平移向量
};

// 单相机标定结果
struct CameraCalibration {
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
    double reprojection_error;  // RMS重投影误差（像素）
};

// 三点标定杆一帧的观测
struct RodObservation {
    Point2D marker_a, marker_b, marker_c;  // 近端/中间/远端标记点
};

// 单相机所有帧观测
struct CameraObservations {
    const RodObservation* frames;
    int frame_count;
};

// 三点标定配置
struct ThreePointConfig {
    double ab_distance = 150.0;        // AB间距（mm）
    double bc_distance = 250.0;        // BC间距（mm）
    int image_width = 0, image_height = 0;
    int max_iterations = 100;
    double tolerance = 1e-6;
    bool fix_principal_point = false;
    bool fix_aspect_ratio = false;
    bool estimate_distortion = true;
};

// 直角三角形观测
struct TriangleObservation {
    Point2D marker_o;  // 直角顶点（世界原点）
    Point2D marker_x;  // 短边端点（X轴方向）
    Point2D marker_y;  // 长边端点（Y轴方向）
};

struct TriangleConfig {
    double ox_length = 300.0;  // OX距离（mm）
    double oy_length = 400.0;  // OY距离（mm）
    int image_width = 0, image_height = 0;
};

struct WorldRegistration {
    CameraExtrinsics world_to_camera;
    double fit_error;  // 刚性拟合RMS残差（mm）
};

// 棋盘格检测
struct CheckerboardCorners {
    Point2D* points;    // 角点（行主序）
    int rows, cols;     // 内部角点行列数
    bool valid;         // 是否成功检测
};

struct CheckerboardConfig {
    int cols = 9, rows = 6;                      // 内部角点数
    double square_size = 1.0;                     // 方格边长（mm）
    bool sub_pixel = true;                        // 亚像素精化
    int sub_pixel_window = 11;                    // 精化窗口半宽
    int sub_pixel_iters = 30;                     // 精化最大迭代
    double sub_pixel_eps = 0.001;                 // 精化收敛容差
};

// DLT（直接线性变换）
struct DltCorrespondence {
    double world_x, world_y, world_z;
    double image_x, image_y;
};

struct DltParams {
    const DltCorrespondence* correspondences;
    int count;  // >= 6
};

struct DltResult {
    double P[12];  // 3x4投影矩阵（行主序）
    CameraIntrinsics K;
    CameraExtrinsics extrinsics;
    double residual;
};

// 立体标定
struct StereoCalibrateParams {
    const CheckerboardCorners* left_corners;
    const CheckerboardCorners* right_corners;
    int view_count;
    const CameraIntrinsics* left_intrinsics;
    const CameraIntrinsics* right_intrinsics;
    bool fix_intrinsics = true;
};

// 光束法平差
struct BundleAdjustParams {
    const CameraObservations* cameras;
    int camera_count, frame_count;
    CameraIntrinsics* intrinsics;   // in/out
    CameraExtrinsics* extrinsics;   // in/out（每个相机）
    double ab_distance = 150.0, bc_distance = 250.0;
    int max_iterations = 50;
    double tolerance = 1e-6;
    bool fix_intrinsics = true;
};

// --- 大规模标定（50-100+ 相机）---
struct CameraGraph { /* 相机可见性图 */ };
struct PnPParams { /* N点→姿态 */ };
struct PnPResult { /* 姿态 + 内点 + RMS */ };
struct SfMConfig { /* 增量SfM配置 */ };
struct SfMResult { /* 重建结果：每视姿态 + 3D点 */ };
struct SparseBAParams { /* Schur补稀疏BA配置 */ };
```

### 20.4 主标定函数

```cpp
// 统一分发（legacy）
CalibrationError process_calibration(
    const CameraObservations* cameras, int camera_count,
    CameraCalibration* results,
    CalibrationAlgorithm algorithm,
    const ThreePointConfig* config
);

// 三点标定
CalibrationError process_three_point(
    const CameraObservations* cameras, int camera_count,
    CameraCalibration* results,
    const ThreePointConfig* config
);

// 直角三角形世界坐标系注册
CalibrationError process_right_triangle(
    const CameraObservations* cameras, int camera_count,
    const CameraCalibration* calibrations,
    const TriangleConfig* config,
    WorldRegistration* world_reg
);

// 棋盘格角点检测
CalibrationError process_checkerboard_detect(
    const uint8_t* image, int width, int height,
    int channels, int bit_depth,
    const CheckerboardConfig* config,
    CheckerboardCorners* corners
);

// Zhang 棋盘格标定
CalibrationError process_checkerboard_calibrate(
    const CheckerboardCorners* corners, int view_count,
    int image_width, int image_height,
    const CheckerboardConfig* config,
    CameraIntrinsics* intrinsics,
    CameraExtrinsics* extrinsics,
    bool estimate_distortion = true
);

// DLT：N对3D-2D对应点 → 3x4相机矩阵
CalibrationError process_dlt(const DltParams* params, DltResult* result);

// 立体标定
CalibrationError process_stereo_calibrate(
    const StereoCalibrateParams* params,
    CameraExtrinsics* stereo_R, CameraExtrinsics* stereo_t,
    double* rms_error = nullptr
);

// 光束法平差
CalibrationError process_bundle_adjust(
    BundleAdjustParams* params, double* final_rms = nullptr
);

// 相机可见性图构建
CalibrationError process_camera_graph(
    const SfMView* views, int view_count,
    const CheckerboardConfig* config,
    CameraGraph* graph
);

// PnP求解
CalibrationError process_pnp_solver(
    const PnPParams* params, PnPResult* result
);

// 增量式SfM
CalibrationError process_incremental_sfm(
    const SfMConfig* config, SfMResult* result
);

// 稀疏光束法平差（大规模）
CalibrationError process_sparse_ba(
    SparseBAParams* params, double* final_rms = nullptr
);
```

### 20.5 CUDA 支持

```cpp
bool has_cuda();
const char* cuda_device_name();
```

### 20.6 使用示例

```cpp
#include <calibration.h>

// Zhang 棋盘格标定
calibration::CheckerboardConfig cb_config;
cb_config.cols = 9;
cb_config.rows = 6;
cb_config.square_size = 24.0;  // 24mm方格

// 对每张标定图检测角点
std::vector<calibration::CheckerboardCorners> corners_views;
for (auto& img : calibration_images) {
    calibration::CheckerboardCorners corners;
    corners.points = new calibration::Point2D[9 * 6];
    corners.rows = 6;
    corners.cols = 9;

    calibration::process_checkerboard_detect(
        img.data(), img_width, img_height, 3, 8,
        &cb_config, &corners
    );

    if (corners.valid) {
        corners_views.push_back(corners);
    }
}

// 执行 Zhang 标定
calibration::CameraIntrinsics intrinsics;
calibration::CameraExtrinsics* extrinsics =
    new calibration::CameraExtrinsics[corners_views.size()];

calibration::CalibrationError err = calibration::process_checkerboard_calibrate(
    corners_views.data(), (int)corners_views.size(),
    img_width, img_height, &cb_config,
    &intrinsics, extrinsics, true
);

printf("内参: fx=%.2f fy=%.2f cx=%.2f cy=%.2f\n",
    intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);
```

---

## 通用模式与约定

### 错误处理模式

所有模块遵循统一的错误处理模式：

```cpp
// 方式一：直接比较
if (err != ModuleError::Ok) { /* 错误处理 */ }

// 方式二：使用辅助函数
if (!err) { /* 有错误 */ }
if (ok(err)) { /* 成功 */ }

// 方式三：获取错误描述
const char* msg = module_error_message(err);
```

### 内存管理约定

- **输入输出分离**：大多数模块接收独立的 `const uint8_t* input` 和 `uint8_t* output` 指针，调用方负责预分配输出缓冲区
- **就地修改**：`black_level`、`defect_correct`、`lens_shading` 三个模块采用就地修改（in-place）模式，`data` 参数既是输入也是输出
- **字节大小计算**：使用 `imop::pixel` 命名空间中的 `compute_*_byte_size` 函数计算所需缓冲区大小

### 位深（bit_depth）对照表

| bit_depth | 每通道字节数 | RGB每像素字节数 | 数值范围 |
|-----------|-------------|----------------|---------|
| 8 | 1 | 3 | [0, 255] |
| 12（打包） | 1.5 | N/A（Bayer only） | [0, 4095] |
| 10（打包） | 1.25 | N/A（Bayer only） | [0, 1023] |
| 16 | 2 | 6 | [0, 65535] |
| 0（float32，仅HDR） | 4 | 12 | float |

### CUDA 加速支持

以下模块支持 CUDA GPU 加速：

- `imop::demosaic_cuda` / `demosaic_cuda_batch`
- `denoise::process_denoise_cuda`
- `white_balance::process_white_balance_cuda`
- `ccm::process_ccm_cuda`
- `lut::process_lut_cuda`
- `sharpen::process_sharpen_cuda`
- `tone::process_tone_cuda`
- `color_temp::process_color_temp_cuda`
- `saturation::process_saturation_cuda`
- `jpeg_codec::process_encode_cuda` / `process_decode_cuda`
- `hdr::process_hdr_cuda`
- `black_level::process_black_level_cuda`
- `defect_correct::process_defect_correct_cuda`
- `highlight_reconstruct::process_highlight_reconstruct_cuda`
- `local_contrast::process_local_contrast_cuda`
- `lens_shading::process_lens_shading_cuda`
- `calibration` 模块包含 `calib_kernels.cu`

使用 `module::has_cuda()` 查询 CUDA 可用性后，调用对应的 `_cuda` 后缀函数即可。

### 处理流水线建议

典型的原始 Bayer 数据到最终 RGB 图像的处理顺序：

1. `black_level::process_black_level` - 黑电平校正（in-place，Bayer）
2. `defect_correct::process_defect_correct` - 坏点校正（in-place，Bayer）
3. `lens_shading::process_lens_shading` - 镜头阴影校正（in-place，Bayer）
4. `imop::demosaic` - 去马赛克（Bayer → RGB）
5. `white_balance::process_white_balance` - 白平衡（RGB）
6. `highlight_reconstruct::process_highlight_reconstruct` - 高光重建（RGB）
7. `denoise::process_denoise` - 降噪（RGB）
8. `ccm::process_ccm` - 色彩校正（RGB）
9. `hdr::process_hdr` / `tone::process_tone` - 色调映射（RGB）
10. `local_contrast::process_local_contrast` - 局部对比度（RGB）
11. `saturation::process_saturation` - 饱和度（RGB）
12. `sharpen::process_sharpen` - 锐化（RGB）
13. `lut::process_lut` - LUT风格化（RGB）
14. `jpeg_codec::process_jpeg_encode` - JPEG压缩输出

颜色校准流程（离线的、标定用）：

1. 拍摄 X-Rite ColorChecker Classic 色卡
2. `color_calibration::calibrate_from_chart` - 自动检测、提取、求解 CCM
3. 将求解得到的 CCM 矩阵用于后续步骤 8

---

*本文档基于 iris-memo Operator v2.0.0 生成，所有 API 签名以实际头文件为准。*
