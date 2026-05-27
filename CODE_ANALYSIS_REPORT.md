# SoolinOperator 代码分析报告

> 分析日期: 2026-05-27 | 版本: 2.0.0 | 语言: C++17

---

## 目录

1. [项目概览](#1-项目概览)
2. [架构与设计](#2-架构与设计)
3. [模块分析](#3-模块分析)
4. [关键 Bug 汇总](#4-关键-bug-汇总)
5. [性能分析](#5-性能分析)
6. [构建系统与测试](#6-构建系统与测试)
7. [改进建议](#7-改进建议)

---

## 1. 项目概览

**SoolinOperator** 是一个高性能图像处理算子库，采用 C++17 编写，使用 CMake 3.18+ 构建。输出为静态库 `im_operator.lib`。

### 1.1 技术栈

| 类别 | 技术选择 |
|------|----------|
| 语言标准 | C++17 |
| 构建系统 | CMake 3.18+ |
| 单元测试 | GoogleTest 1.15.2 |
| CPU 加速 | AVX2 SIMD + OpenMP 多线程 |
| GPU 加速 | CUDA 12.6 (Compute Capability 8.9) |
| 代码格式化 | clang-format (Google 风格, 4空格缩进) |
| 静态分析 | clang-tidy (bugprone/cert/cppcoreguidelines/modernize/performance/readability) |
| CI/CD | GitHub Actions (Win MSVC / Linux GCC / Linux Clang / Sanitizer) |

### 1.2 模块组成 (11个模块, 55种算法)

| 模块 | 命名空间 | 算法数量 | 算法 |
|------|----------|---------|------|
| **Demosaic** (去马赛克) | `imop` | 9 | SUPER_FAST, HQLI, MG, L7, DFPD, AHD, AMAZE, RCD, PRISM |
| **Denoise** (降噪) | `denoise` | 6 | GAUSSIAN, MEDIAN, BILATERAL, NLM, WAVELET, BAYER_DENOISE |
| **White Balance** (白平衡) | `white_balance` | 4 | GRAY_WORLD, WHITE_PATCH, SHADE_OF_GRAY, MANUAL |
| **CCM** (颜色校正) | `ccm` | 4 | LINEAR_3x3, LINEAR_4x3, POLYNOMIAL_3x9, MANUAL |
| **LUT** (查找表) | `lut` | 9 | CUBE_FILE, CUSTOM_3D, SEPIA, COOL, WARM 等 |
| **Sharpen** (锐化) | `sharpen` | 4 | UNSHARP_MASK, LAPLACIAN, HIGH_PASS, ADAPTIVE |
| **Tone** (色调映射) | `tone` | 5 | GAMMA, S_CURVE, LEVELS, CURVES_3POINT, SHADOWS_HIGHLIGHTS |
| **Color Temp** (色温) | `color_temp` | 4 | KELVIN, PRESET, MANUAL, WHITE_BALANCE |
| **Saturation** (饱和度) | `saturation` | 4 | HSL, VIBRANCE, CHANNEL_MIXER, SELECTIVE |
| **JPEG Codec** | `jpeg_codec` | 4 | ENCODE/DECODE_BASELINE, ENCODE/DECODE_CUDA |
| **Calibration** (标定) | `calibration` | 2 | THREE_POINT, RIGHT_TRIANGLE |

---

## 2. 架构与设计

### 2.1 整体架构评价: 良好

项目具有清晰的模块化架构，每个模块遵循一致的目录和文件组织模式：

```
include/<module>.h          # 聚合头文件
include/<module>/types.hpp  # 枚举/类型/错误码定义
include/<module>/algorithms.hpp  # 算法函数声明 (公共API)
src/<module>/common.hpp     # 内部工具函数
src/<module>/dispatch.cpp   # 验证 + 算法注册表 + 统一入口函数
src/<module>/<algorithm>.cpp # 各算法实现
```

### 2.2 设计模式

| 模式 | 应用场景 | 评价 |
|------|---------|------|
| **Facade** | `imop::Demosaic` 类封装底层函数 | 良好, 简化了公共API |
| **Strategy** | 每种算法都是相同签名的独立函数 | 良好, 算法可互换 |
| **Registry** | `kDemosaicRegistry` 等静态注册表 + `static_assert` 校验 | 优秀, 防止枚举与实现不同步 |
| **Object Pool** | `FloatBufferPool` 使用 `thread_local` 避免热路径分配 | 优秀设计 |
| **Template Policy** | `BPC` (Bytes Per Component) 模板消除运行时分支 | 良好 |

### 2.3 架构问题

1. **大量代码重复** — `read_pixel`/`write_pixel`/`clamp_val`/`safe_max_val` 在 8 个不同模块的 `common.hpp` 中逐字复制粘贴。应提取为共享内部库。

2. **文档仅覆盖 Demosaic 模块** — `API_Reference.md` 只记录了 Demosaic 模块的 API，其余 10 个模块完全没有 API 文档。

3. **`include/im_operator.h` 命名不当** — 作为聚合入口，仅导出 Demosaic 模块。其他模块 (denoise, white_balance 等) 需分别 include 各自的聚合头文件。

---

## 3. 模块分析

### 3.1 Demosaic (去马赛克) — 9 种算法

**代码规模:** 最大模块，约 8000+ 行，包含 AVX2 优化和 CUDA 内核。

**亮点:**
- 9 种算法从最近邻插值到 PRISM 光谱融合, 覆盖全部质量/性能梯度
- `FloatBufferPool` 使用 `thread_local` 为每个 OpenMP 线程提供独立缓冲区, 避免热路径堆分配
- CUDA Graph 支持批量处理, 双缓冲流水线实现 H2D/Kernel/D2H 重叠
- AVX2 加速 4 种算法 (SUPER_FAST, HQLI, MG, L7) 的 8-bit 非打包路径
- 打包格式 (10-bit 4像素5字节, 12-bit 2像素3字节) 完整支持

**关键问题:**

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| D1 | **严重** | PRISM 原地中值滤波 bug — 对 `L` 和 `M` 色彩差平面进行原地中值滤波, 后续邻居像素读取到已被修改的值, 导致色彩伪影 | `prism.cpp:267-291` |
| D2 | **严重** | 打包格式通用路径强制最小 3 字节读取 — bit_depth 较低时 (如 1-8 bit) 越界读取未初始化内存 | `pixel_utils.hpp:105` |
| D3 | **严重** | 10-bit/12-bit 打包路径边界检查差一错误 (`>=` 应为 `>`) — 遗漏末尾最后一个有效像素 | `pixel_utils.hpp:75,85` |
| D4 | **高** | 16-bit 非打包路径在图像边缘产生越界读取 — `Accessor16bit` 无边界钳制, `super_fast_core` 在全图范围内迭代 | `interp_core.hpp:135-141`, `super_fast.cpp:37-39` |
| D5 | **高** | 4/9 算法 (AHD, AMAZE, RCD, PRISM) 的单元测试被跳过 (MinGW toolchain crash) | `test_algorithms.cpp:37-38`, `test_correctness.cpp:71-73` |
| D6 | **中** | CUDA `hqli_kernel` 边界像素从尚未完全写入的输出缓冲区读取 — 依赖隐式块调度顺序 | `cuda_kernels.cu:287-307` |
| D7 | **中** | AVX2 检测在未定义 `__AVX2__` 宏时永远返回 false, 使运行时 CPUID 检测失效 | `avx2.cpp:689-696` |
| D8 | **中** | MG 8-bit 路径硬编码 `max_val = 255`, 忽略实际 bit_depth — 非 8-bit 深度时钳制值错误 | `mg.cpp:27` |
| D9 | **中** | AHD 边界像素钳制访问可能获取错误颜色通道的邻居值 — 导致边界行列色彩错误 | `ahd.cpp:78-83,108-112` |
| D10 | **低** | `DLOOP` 宏使用 `schedule(guided, 16)` — 对于均匀负载的图像循环, `static` 调度更高效 | `common.hpp:39-42` |
| D11 | **低** | 注册表大小硬编码为 9, 与枚举计数重复 — 维护风险 | `dispatch.cpp:67,79` |

### 3.2 Denoise (降噪) — 6 种算法

**代码规模:** 约 1000 行, 中规模模块。

**亮点:**
- Bilateral filter 的 8-bit LUT 优化 (用查表替代 `exp()` 调用) 是教科书级优化
- Wavelet denoise 的 Haar DWT 实现正确, 使用 MAD/VisuShrink 进行噪声估计和阈值处理
- 可分离 Gaussian filter 减少了 2D 卷积的计算量
- 模板 `BPC` 消除了像素级别的位深分支

**关键问题:**

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| N1 | **高** | NLM 中心块和邻居块距离对每个通道重新计算 — 对 RGB 图像成本增加约 3 倍 | `nlm.cpp:37-38, 80-121` |
| N2 | **中** | NLM 多通道距离计算有冗余 `sqrt` 后再 `square` 操作 | `nlm.cpp:99-118` |
| N3 | **中** | NLM `max_weight` 初始化为 0 — 退化情况下可能除零 | `nlm.cpp:140-150` |
| N4 | **低** | 16-bit Bilateral filter 无 LUT 优化, 每像素计算 `exp()` — 比 8-bit 路径慢 50-100 倍 | `bilateral.cpp:78-83` |
| N5 | **低** | `median.cpp` 和 `bayer_denoise.cpp` 中 `strength` 参数被忽略但未说明 | `median.cpp:34`, `bayer_denoise.cpp:43` |
| N6 | **低** | `read_pixel` 中 `y * width` 对大图像存在整数溢出风险 (在 `size_t` 转换前) | `denoise/common.hpp:26-33` |

### 3.3 White Balance (白平衡) — 4 种算法

**规模:** 小型模块, ~200 行。

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| W1 | **低** | 验证函数在每个算法函数中重复调用 — 每次处理调用两次验证 | `gray_world.cpp`, `white_patch.cpp` 等 |
| W2 | **低** | `SHADE_OF_GRAY` 在 p=0 时除零 | `manual.cpp:112` |
| W3 | **低** | 无并行化 — 单线程逐像素遍历 | 所有算法 |

### 3.4 CCM (颜色校正矩阵) — 4 种算法

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| C1 | **高** | `process_manual_ccm` 无条件将 `void* matrix` 转换为 `CCMatrix3x3*` — 若传入其他矩阵类型则产生错误输出 | `ccm/manual.cpp` |
| C2 | **低** | `srgb_to_bt709` 和 `bt709_to_srgb` 返回单位矩阵 — 命名暗示完整色彩空间转换但仅处理线性矩阵 | `ccm/manual.cpp` |

### 3.5 LUT (查找表) — 9 种算法

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| L1 | **中** | `.cube` 文件解析器不支持负值数据行 — 检查 `line[0] >= '0'` 会跳过以 `-` 开头的合法行 | `cube_file.cpp:135` |
| L2 | **中** | `.cube` 文件解析器在 EOF 上有无限循环风险 | `cube_file.cpp:139` |
| L3 | **低** | `LUT3D::total_samples()` 存在 `int*int*int` 溢出风险 (在转换为 `size_t` 之前) | `lut/types.hpp` |
| L4 | **低** | `preset_styles.cpp` 仅 `#include` 另一个 `.cpp` — C++ 反模式 | `preset_styles.cpp` |

### 3.6 Sharpen (锐化) — 4 种算法

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| S1 | **严重** | Laplacian 和 High-pass 在图像边界产生越界读取 — `read_pixel` 无坐标钳制, 边界处 `y-1` 导致无符号下溢 | `laplacian.cpp:41`, `high_pass.cpp:30-38` |
| S2 | **中** | Adaptive sharpen 仅从亮度平面计算细节 — 等亮度色边不会被锐化 | `adaptive.cpp:79-91` |
| S3 | **低** | Adaptive sharpen 方差计算对边界像素硬编码除以 9 — 边界像素被跳过 | `adaptive.cpp:41-42` |

### 3.7 Tone (色调映射) — 5 种算法

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| T1 | **中** | `process_levels` gamma 计算在 `mid_norm=1.0` 时除零 | `levels.cpp:33` |
| T2 | **中** | `process_curves_3point` 8-bit 快速路径在 `val=255` 时越界 — `idx=255` 对于 256 元素数组越界 | `curves_3point.cpp:86-89` |

### 3.8 Color Temp (色温) — 4 种算法

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| CT1 | **中** | 光源预设值语义模糊 — 是校正系数还是效果应用系数? CANDLE 的 `r_mult=0.55, b_mult=3.50` 对于校正合理但对于"应用烛光效果"则反了 | `preset.cpp` |
| CT2 | **低** | MIDDAY_SUN 和 CLOUDY 都返回恒等变换 — 不同色温应有不同系数 | `preset.cpp` |

### 3.9 Saturation (饱和度) — 4 种算法

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| SA1 | **低** | Vibrance 和 Saturation 参数在 HSL 路径中冲突 — 先应用 vibrance 再应用 saturation 可能过饱和 | `hsl.cpp` |

### 3.10 JPEG Codec — 4 种算法

**规模:** 中大型模块, 包含完整 DCT/IDCT/Huffman 编解码, 及 CUDA 加速。

**亮点:**
- 完整的 Baseline DCT JPEG 编码器/解码器
- CUDA 编码器使用两遍 Huffman (GPU 前缀和计算比特偏移), 避免 CPU 往返
- `__constant__` 内存用于 DCT 余弦表和 Huffman 编码表

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| J1 | **高** | CUDA JPEG 解码器是存根 — `process_decode_cuda` 仅委托给 CPU 路径, GPU 解码内核存在但未被调用 | `jpeg_codec/cuda_dispatch.cpp:253-258` |
| J2 | **中** | 编码器 DC 类别计算存在差一错误 (off-by-one error) | `encode_baseline.cpp` |
| J3 | **中** | 编码器在每次字节写入时检查 EOB 标记 — O(n^2) 复杂度 | `encode_baseline.cpp` |
| J4 | **低** | `write_sof0` 函数定义但从未被调用 — 死代码 | `encode_baseline.cpp` |

### 3.11 Calibration (标定) — 2 种算法

| # | 严重程度 | 描述 | 位置 |
|---|---------|------|------|
| CL1 | **严重** | LM 优化器在第一个可接受的步骤后即终止 — `cost - new_cost` 在 `cost` 已在行 356 更新后为 0, 收敛条件永远为真 | `three_point.cpp:356-360` |
| CL2 | **中** | 注册表只包含 `THREE_POINT`, `RIGHT_TRIANGLE` 无注册表条目 — 仅有 1/2 算法可被 dispatch | `calibration/dispatch.cpp` |
| CL3 | **中** | `validate_calibration_inputs` 对配置错误返回 `InvalidFrameCount` — 错误码与问题不匹配 | `calibration/dispatch.cpp:41-44` |
| CL4 | **低** | Jacobi SVD 收敛慢 — 对于近退化矩阵可能不够准确 | `three_point.cpp` (svd_3x3) |

---

## 4. 关键 Bug 汇总

### 4.1 严重 Bug (崩溃/错误输出)

| ID | 模块 | 描述 |
|----|------|------|
| D1 | Demosaic | PRISM 原地中值滤波 — 写入被后续邻居像素读取, 导致色彩伪影 |
| D2 | Demosaic | 打包格式通用路径强制 3 字节 — 低 bit_depth 越界读取 |
| D3 | Demosaic | 10/12-bit 打包边界 `>=` 应为 `>` — 遗漏末尾像素 |
| S1 | Sharpen | Laplacian/High-pass 边界越界读取 — 坐标 `-1` 导致 `size_t` 无符号下溢, 大概率崩溃 |
| CL1 | Calibration | LM 优化器收敛条件错误 — 第一次接受步骤后即终止, 标定参数未充分优化 |
| C1 | CCM | `process_manual_ccm` 无法区分 3x3/4x3/3x9 矩阵类型 — 传入错误类型产生错误输出, 无报错 |

### 4.2 高风险 Bug (功能缺失/结果错误)

| ID | 模块 | 描述 |
|----|------|------|
| D1 | Demosaic | 16-bit 非打包路径在图像边缘产生越界读取 |
| D2 | Demosaic | 4/9 去马赛克算法 (AHD, AMAZE, RCD, PRISM) 的单元测试被跳过 |
| J1 | JPEG | GPU 解码器是存根 — 并未实际使用 CUDA 解码内核 |
| N1 | Denoise | NLM 对每个通道重复计算 patch 距离 — RGB 图像成本增加 3 倍 |
| D4 | Demosaic | AVX2 运行时检测被编译时条件限制, 实际上无法在非 `/arch:AVX2` 构建中检测 AVX2 |

### 4.3 中风险 Bug

| ID | 模块 | 描述 |
|----|------|------|
| D3 | Demosaic | CUDA `hqli_kernel` 读取未完全输出的帧缓冲 — 依赖隐式调度顺序 |
| L1 | LUT | `.cube` 文件解析器不支持负值 — 合法数据被跳过 |
| L2 | LUT | `.cube` 解析器在 EOF 上有无限循环风险 |
| T1 | Tone | `levels.cpp` gamma 除零 |
| T2 | Tone | `curves_3point` 8-bit 路径在 val=255 时越界 |
| CT1 | Color Temp | 光源预设值语义模糊 (校正 vs 效果) |
| N2 | Denoise | NLM 冗余 sqrt-square 操作 |
| N3 | Denoise | NLM `max_weight=0` 初始化, 退化情况除零风险 |

---

## 5. 性能分析

### 5.1 加速栈

项目实现了完整的三级加速栈, 优先级从高到低:

1. **CUDA GPU** — 全部 9 种去马赛克算法, 部分 JPEG 编解码
2. **AVX2 SIMD** — 4 种去马赛克算法 (仅 8-bit 非打包)
3. **OpenMP 多线程** — 所有 CPU 路径
4. **纯标量 C++** — 兜底路径

### 5.2 性能问题

| 问题 | 影响 | 建议 |
|------|------|------|
| `DLOOP` 使用 `guided` 调度 | 均匀负载场景下 `static` 调度开销更低 | 改为 `schedule(static)` |
| Bilateral 16-bit 路径无 LUT | 比 8-bit 慢 50-100 倍 | 为 16-bit 添加 64KB LUT |
| NLM 重复 patch 计算 | 三通道成本增加 3 倍 | 将 patch 计算移到通道循环外 |
| L7 打包路径重复计算 `data_byte_size` | 每输出像素 49 次 | 在外层预计算 |
| AHD 方差重复计算 | 每像素 4 次 3x3 方差 (共 36 次读取) | 预计算方差图 |
| LM 有限差分 Jacobian | O(np * M) 每次 Jacobian 计算 | 使用解析导数 |
| `thread_local` CUDA 工作区 | 每线程分配 GPU 内存, 浪费显存 | 改为共享池或单例 |

### 5.3 打包格式支持缺口

- **不支持的路径**: CUDA GPU 完全不支持打包格式 (自动回退到 CPU)
- **性能警告**: L7 打包路径每像素调用 `get_raw` 49 次, 每次重新计算字节偏移
- **安全警告**: 通用打包路径在 `data_byte_size=0` 时边界检查弱

---

## 6. 构建系统与测试

### 6.1 构建系统

**CMakeLists.txt** (312 行):
- 选项分离良好 (`IM_OPERATOR_ENABLE_CUDA`, `IM_OPERATOR_BUILD_TESTS` 等)
- CUDA 集成小心处理 (pin 到 12.6 避免 NVCC 13.0 兼容问题)
- GoogleTest 通过 `FetchContent` 获取, 避免系统依赖
- `install()` 规则全面, 包含 `CMakePackageConfigHelpers`

**问题:**

| # | 描述 |
|---|------|
| B1 | **构建目标名称拼写错误**: `fold_data_processor` (缺少 "r" in "folder"), 与源文件 `tools/folder_data_debayer.cpp` 不匹配 — **构建会失败** |
| B2 | CUDA 路径硬编码为 `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6` |
| B3 | MSVC 特定 `/arch:AVX2` 和 CUDA Windows 路径 — GCC/Clang 分支存在但未经测试 |
| B4 | `build_cuda.ps1` 硬编码仅构建 `jpeg_bench` 目标 |
| B5 | `Directory.Build.props` 存在但未被 CMake 引用 |
| B6 | `gpu_jpeg_bench.cu` 和 `gpu_dct_test.cu` 在 CMake 之外编译 — 应集成或移除 |
| B7 | `VCPKG_APPLOCAL_DEPS OFF` 无条件设置, 但项目未使用 vcpkg |

### 6.2 测试覆盖

**16 个测试文件** 覆盖所有模块, 使用 GoogleTest 参数化测试套件。

**关键缺口:**

| # | 描述 |
|---|------|
| T1 | 4/9 去马赛克算法 (AHD, AMAZE, RCD, PRISM) 单元测试被跳过 — 引用 "MinGW toolchain crash" |
| T2 | 无 CUDA 特定单元测试 — 仅检查 `has_cuda()` 返回 bool |
| T3 | `test_data/` 目录中的真实 RAW 文件未在自动化测试中使用 |
| T4 | CUDA JPEG 解码器无测试 |
| T5 | 无性能回归测试 |

---

## 7. 改进建议

### 7.1 紧急修复 (P0)

1. **修复 PRISM 原地中值滤波** — `prism.cpp:267-291`: 使用独立输出缓冲区, 参考 AHD 的 `apply_median_filter_3x3` 实现
2. **修复打包格式通用路径越界** — `pixel_utils.hpp:105`: 移除 `bytes_to_read = 3` 强制最小读取
3. **修复 10/12-bit 打包路径差一错误** — `pixel_utils.hpp:75,85`: `>=` 改为 `>`
4. **修复构建目标名称** (`fold_data_processor` → `folder_data_processor`)
5. **修复 sharpen 边界越界** — `read_pixel` 需要添加坐标钳制
6. **修复 calibration LM 收敛** — 在更新 `cost` 前检查收敛条件
7. **修复 CCM manual 矩阵类型** — 添加类型鉴别参数或多重函数重载

### 7.2 高优先级 (P1)

5. **修复 4 个跳过的去马赛克测试** — 调查 MinGW crash 根因, 确保所有算法通过测试
6. **完�� CUDA JPEG 解码器** — 或从 API 中移除 CUDA 解码入口
7. **修复 `hqli_kernel` 边界像素依赖** — 使用两遍方法 (先计算所有内部像素, 再填边界)
8. **修复 AVX2 运行时检测** — 将 CPUID 检测从 `#ifdef __AVX2__` 中移出

### 7.3 中优先级 (P2)

9. **提取共享 pixel I/O 工具** — 消除 8 个模块间 `read_pixel`/`write_pixel`/`clamp_val` 的代码重复
10. **修复 NLM 重复通道计算** — 将 patch 距离计算移到通道循环外
11. **修复 `.cube` 文件解析器** — 支持负值, 添加 EOF 保护
12. **修复 `levels.cpp` gamma 除零** — 对 `mid_norm=1.0` 添加保护
13. **为其余 10 个模块编写 API 文档**

### 7.4 低优先级 (P3)

14. **修复 NLM `max_weight` 初始化为安全值**
15. **为 16-bit bilateral filter 添加 LUT**
16. **考虑 `DLOOP` 使用 `static` 调度**
17. **修复 LUT `total_samples()` 溢出风险**
18. **移除 `preset_styles.cpp` 的 `#include .cpp` 反模式**
19. **集成或移除独立 CUDA 工具** (`gpu_jpeg_bench.cu`, `gpu_dct_test.cu`)

---

## 附录: 代码质量总评

| 维度 | 评分 | 说明 |
|------|------|------|
| **架构设计** | ★★★★☆ | 统一的模块化架构, 一致的 dispatch 模式, 良好的关注点分离 |
| **代码规范** | ★★★★☆ | Google 风格, 命名一致, 复杂算法有详细注释 |
| **正确性** | ★★★☆☆ | 发现数个严重 bug (边界越界, LM 收敛, 矩阵类型), 去马赛克核心算法测试覆盖不足 |
| **性能优化** | ★★★★☆ | 三级加速栈设计优秀, 但部分模块存在已知性能瓶颈 (NLM, adaptive sharpen, LM) |
| **测试覆盖** | ★★★☆☆ | 16 个测试文件结构良好, 但 4/9 算法未测试, 无 CUDA/GPU 测试, 无真实数据测试 |
| **文档** | ★★☆☆☆ | 中文 API 文档仅有 Demosaic 模块, 其余 10 个模块无文档; 开发指南质量高但覆盖面窄 |
| **可维护性** | ★★★☆☆ | 跨模块代码重复严重; 打包格式支持碎片化; 注册表大小硬编码 |
| **安全性** | ★★★☆☆ | 存在越界读取风险; `reinterpret_cast` 违反严格别名规则; 边界检查不统一 |

**综合评分: 3.4 / 5.0** — 一个架构良好、算法丰富的图像处理库, 但存在数个需要修复的严重 bug, 且测试覆盖和文档需要显著增强。
