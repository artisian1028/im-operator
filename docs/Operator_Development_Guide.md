# SoolinOperator 算子开发规范文档

## 版本 2.0.0

---

## 1. 项目概述

SoolinOperator 是一个高性能图像处理算子库，当前包含两大算子模块：

- **Demosaic（去马赛克）**：将 Bayer CFA raw 数据转换为 RGB 彩色图像
- **Denoise（降噪）**：对 RGB 或灰度图像进行多种降噪处理

### 技术栈

| 项目 | 说明 |
|------|------|
| 语言 | C++17 |
| 构建系统 | CMake 3.18+ |
| 测试框架 | GoogleTest v1.15.2 |
| 加速选项 | AVX2 SIMD、OpenMP 多线程、CUDA GPU |
| 编码风格 | Google Style（clang-format，4空格缩进） |
| 静态分析 | clang-tidy（bugprone, cert, cppcoreguidelines, modernize, performance, readability） |

---

## 2. 核心设计模式

### 2.1 算子 = 算子模块

每个算子定义为**独立的命名空间 + 头文件目录 + 源文件目录**的三层结构：

```
include/<module>.h              # 模块聚合头文件
include/<module>/types.hpp      # 枚举/结构体/错误码
include/<module>/algorithms.hpp  # 算法函数声明（公共接口）
src/<module>/dispatch.cpp       # 验证 + 注册表 + 调度
src/<module>/common.hpp         # 内部工具函数（不对外暴露）
src/<module>/<algo_name>.cpp    # 各算法实现
tests/unit/test_<module>.cpp    # 单元测试
```

### 2.2 两份参考实现

以 **Demosaic** 和 **Denoise** 两个模块为范本，新增算子模块应完整复制这两者的设计模式：

| 模式要素 | Demosaic | Denoise |
|---------|---------|---------|
| 命名空间 | `im_operator` | `denoise` |
| 算法枚举 | `DemosaicAlgorithm` | `DenoiseAlgorithm` |
| 错误枚举 | `DemosaicError` | `DenoiseError` |
| 验证函数 | `validate_demosaic_inputs()` | `validate_denoise_inputs()` |
| 调度函数 | `demosaic()` | `process_denoise()` |
| 注册表 | `kDemosaicRegistry` (9项) | `kDenoiseRegistry` (6项) |
| 元数据函数 | `algorithm_name()`, `algorithm_window_size()` | 同名 |
| 算法签名 | `(bayer, rgb, w, h, pattern, bit_depth, packed)` | `(input, output, w, h, channels, bit_depth, strength)` |

---

## 3. 新增算子开发步骤（完整清单）

### Step 1: 定义枚举与类型 (`include/<module>/types.hpp`)

必须定义：

```cpp
namespace your_module {

// 1. 算法枚举
enum class YourAlgorithm {
    ALGO_A,
    ALGO_B,
    // ...
};

// 2. 错误码枚举
enum class YourError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    // ... 按需扩展
    InternalError
};

// 3. 错误消息函数（inline，与Demosaic/Denoise一致）
inline const char* your_error_message(YourError err) {
    switch (err) {
        case YourError::Ok: return "Success";
        // ...
    }
}

// 4. 便捷判断运算符
inline bool operator!(YourError err) { return err != YourError::Ok; }
inline bool ok(YourError err) { return err == YourError::Ok; }

// 5. 输入校验函数（inline）
inline bool is_valid_bit_depth(int bit_depth) { return bit_depth >= 1 && bit_depth <= 16; }
inline bool is_valid_dimensions(int width, int height) { return width > 0 && height > 0; }

} // namespace your_module
```

**要点：** 枚举使用 `enum class`，错误码 `Ok = 0` 表示成功。

### Step 2: 声明算法接口 (`include/<module>/algorithms.hpp`)

必须声明以下函数：

```cpp
namespace your_module {

// 算法元数据
std::string algorithm_name(YourAlgorithm algo);
int algorithm_window_size(YourAlgorithm algo);

// 输入验证
YourError validate_inputs(const uint8_t* input, uint8_t* output,
                          int width, int height, /* 其他参数 */);

// 主调度入口
YourError process(const uint8_t* input, uint8_t* output,
                  int width, int height, /* 其他参数 */,
                  YourAlgorithm algorithm, int bit_depth = 8, /* 其他参数 */);

// 各算法独立函数（每个算法一个）
YourError process_algo_a(const uint8_t* input, uint8_t* output,
                         int width, int height, /* 其他参数 */);
YourError process_algo_b(...);
// ...

} // namespace your_module
```

**要点：** 算法函数签名**保持统一**，参数顺序参考 `const uint8_t* input, uint8_t* output, int width, int height, ...` 的固定前缀。

### Step 3: 实现内部工具 (`src/<module>/common.hpp`)

这是**内部头文件**（不安装到 include），提供各算法实现共享的工具：

```cpp
namespace your_module {
namespace detail {

// 像素读写辅助（8-bit / 16-bit 统一接口）
inline int read_pixel(const uint8_t* data, int x, int y, int width,
                      int channels, int bit_depth, int channel) { ... }
inline void write_pixel(uint8_t* data, int x, int y, int width,
                        int channels, int bit_depth, int channel, int value) { ... }

// 边界处理辅助
inline int clamp_val(int v, int max_val) { ... }
inline int safe_max_val(int bit_depth) { ... }

} // namespace detail
} // namespace your_module
```

**要点：**
- Bit depth 分支用 `if (bit_depth <= 8)` / `else` 分两条路，避免每次循环都判断
- 对性能关键的循环，可考虑模板化 `template<int BPC>` 消除分支

### Step 4: 实现调度器 (`src/<module>/dispatch.cpp`)

**4.1 输入验证**

```cpp
YourError validate_inputs(const uint8_t* input, uint8_t* output, ...) {
    if (!input || !output) return YourError::NullInput;
    if (!is_valid_dimensions(width, height)) return YourError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return YourError::InvalidBitDepth;
    // 其他业务相关的校验...
    return YourError::Ok;
}
```

**4.2 元数据函数**

```cpp
std::string algorithm_name(YourAlgorithm algo) {
    switch (algo) {
        case YourAlgorithm::ALGO_A: return "ALGO_A (description)";
        // ...
    }
}
```

**4.3 算法注册表**

使用**静态数组 + 函数指针**的注册模式：

```cpp
using AlgoFunc = YourError(*)(const uint8_t*, uint8_t*, int, int, /* ... */);

struct AlgorithmEntry {
    YourAlgorithm algorithm;
    AlgoFunc func;
};

static const std::array<AlgorithmEntry, N> kRegistry = {{
    {YourAlgorithm::ALGO_A, process_algo_a},
    {YourAlgorithm::ALGO_B, process_algo_b},
    // ...
}};

// 编译期验证注册表与枚举数量一致
static_assert(kRegistry.size() == N,
              "kRegistry size must match YourAlgorithm enum count");
```

**要点：**
- 用 `static const std::array` 而非 `std::map`，确保编译期确定且无堆分配
- **必须加 `static_assert`** 验证注册表数量与枚举值匹配，防止新增算法后忘记注册

**4.4 调度函数**

```cpp
YourError process(const uint8_t* input, uint8_t* output, ...,
                  YourAlgorithm algorithm, ...) {
    YourError err = validate_inputs(input, output, ...);
    if (err != YourError::Ok) return err;

    AlgoFunc func = find_func(algorithm);  // 遍历注册表查找
    if (!func) return YourError::InternalError;

    return func(input, output, ...);
}
```

### Step 5: 实现算法 (`src/<module>/<algo_name>.cpp`)

每个算法文件遵循统一结构：

```cpp
#include "common.hpp"
// 其他必要的标准库头文件

namespace your_module {

YourError process_algo_a(const uint8_t* input, uint8_t* output,
                         int width, int height, /* ... */) {
    // 1. 输入校验
    YourError err = validate_inputs(input, output, ...);
    if (err != YourError::Ok) return err;

    // 2. 最小值检查（算法窗口要求）
    if (width < MIN_SIZE || height < MIN_SIZE)
        return YourError::ImageTooSmall;

    // 3. 核心逻辑
    // - 优先尝试硬件加速路径（AVX2 / CUDA）
    // - 8-bit / 16-bit / packed 分支处理
    // - OpenMP 并行化使用 DLOOP 宏

    // 4. 边界处理（若算法产生无效边界像素）
    // fill_rgb_borders() 或 fill_intermediate_borders()

    return YourError::Ok;
}

} // namespace your_module
```

**核心逻辑三要素：**

1. **硬件加速优先**：先检查 `has_avx2()` / GPU 可用性，有则走加速路径
2. **位深/格式分支**：`bit_depth <= 8 && !is_packed` → `8-bit 路径` → `packed 路径` → `else 16-bit 路径`
3. **OpenMP 并行**：使用 `DLOOP` 宏（在 `common.hpp` 中定义），自动适配 OpenMP 开/关

### Step 6: 聚合头文件 (`include/<module>.h`)

```cpp
#ifndef YOUR_MODULE_H
#define YOUR_MODULE_H

#include "your_module/types.hpp"
#include "your_module/algorithms.hpp"

#endif // YOUR_MODULE_H
```

### Step 7: 注册到 CMakeLists.txt

在 `CMakeLists.txt` 中：

```cmake
# 1. 添加源文件到库
target_sources(im_operator PRIVATE
    src/your_module/dispatch.cpp
    src/your_module/algo_a.cpp
    src/your_module/algo_b.cpp
    # ...
)

# 2. 添加私有 include 目录
target_include_directories(im_operator PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/your_module>
)

# 3. 添加公共头文件安装
install(FILES include/your_module.h DESTINATION include)
install(FILES include/your_module/types.hpp      DESTINATION include/your_module)
install(FILES include/your_module/algorithms.hpp  DESTINATION include/your_module)
```

### Step 8: 编写单元测试 (`tests/unit/test_<module>.cpp`)

**必须覆盖的测试类：**

| 测试类别 | 测试内容 | 最低数量 |
|---------|---------|---------|
| 校验测试 | Null input, invalid dims, invalid bit depth, invalid channels | 5+ |
| 元数据测试 | algorithm_name() 覆盖所有枚举值 | N |
| 元数据测试 | algorithm_window_size() 覆盖所有枚举值 | N |
| 基础处理 | 8-bit 正常输入产生非零输出 | 1 |
| 边界测试 | ImageTooSmall 检测 | 1 |
| 通道测试 | Grayscale (channels=1) 和 RGB (channels=3) | 2 |
| 参数测试 | strength 等参数的合法范围 | 2 |
| 高位深测试 | 12-bit / 16-bit 处理及输出范围 | 2 |
| 参数化测试 | 所有算法均通过基础测试 | N |

**测试结构：**

```cpp
#include <gtest/gtest.h>
#include "your_module/algorithms.hpp"

using namespace your_module;

// 参数化测试
struct TestParam { YourAlgorithm algo; int min_size; };
class AlgorithmTest : public ::testing::TestWithParam<TestParam> {};

TEST_P(AlgorithmTest, ProducesValidOutput) { ... }
TEST_P(AlgorithmTest, DetectsImageTooSmall) { ... }

INSTANTIATE_TEST_SUITE_P(AllAlgos, AlgorithmTest,
    ::testing::Values(
        TestParam{YourAlgorithm::ALGO_A, 3},
        TestParam{YourAlgorithm::ALGO_B, 5},
        // ...
    )
);
```

**要点：**
- 使用 `INSTANTIATE_TEST_SUITE_P` 对所有算法参数化执行
- 验证输出中至少有一个非零值
- 验证所有像素值在合法范围内（`[0, 255]` 或 `[0, max_val]`）

---

## 4. 代码规范

### 4.1 命名规范

| 类型 | 约定 | 示例 |
|------|------|------|
| 命名空间 | 小写 | `im_operator`, `denoise`, `pixel` |
| 类名 | 大驼峰 | `ImageBuffer`, `Demosaic` |
| 枚举类型 | 大驼峰 | `BayerPattern`, `DemosaicError` |
| 枚举值 | 大驼峰 | `RGGB`, `NullInput` |
| 函数名 | 小写下划线 | `demosaic()`, `get_raw()` |
| 常量 | `k` 前缀 + 大驼峰 | `kMaxBitDepth`, `kDemosaicRegistry` |
| 私有/内部 | `detail` 子命名空间 | `imop::detail::Accessor8bit` |
| 头文件保护 | 全大写 + `_HPP`/`_H` | `IMOP_TYPES_HPP` |

### 4.2 clang-format 要点

```yaml
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 120
PointerAlignment: Left
BreakBeforeBraces: Custom
  AfterClass: true
  AfterFunction: true
```

### 4.3 clang-tidy 要点

启用的检查集：`bugprone-*`, `cert-*`, `cppcoreguidelines-*`, `modernize-*`, `performance-*`, `readability-*`, `misc-*`

关键抑制项：
- `-readability-magic-numbers` — 图像处理中常量值不可避免
- `-readability-identifier-naming` — 不强制命名风格
- `-cppcoreguidelines-avoid-magic-numbers` — 同上
- `-cppcoreguidelines-pro-bounds-array-to-pointer-decay` — 允许数组退化
- `-misc-include-cleaner` — 避免误报
- `-misc-const-correctness` — 避免过度 const

### 4.4 错误处理规范

- 所有处理函数返回错误码枚举（如 `DemosaicError`）
- 成功返回 `Ok`（值为 0）
- 函数开头统一调用 `validate_xxx_inputs()` 进行参数校验
- 提供 `ok(err)` 和 `bool operator!()` 便捷检查
- 异常**不使用**，全部通过返回码传达错误

### 4.5 内存和性能

- **调用者分配输出缓冲区**，算子不负责内存分配（ImageBuffer 重载除外）
- `uint8_t*` 作为通用像素数据指针，位深 > 8 时通过 `reinterpret_cast<uint16_t*>` 访问
- 使用 `std::memcpy` 读写 `uint16_t` 值（避免未对齐访问 UB）
- 内部临时缓冲区用 `std::vector<float>` 或 `std::vector<int>`，利用 thread_local 池化减少分配
- 性能敏感的循环中，8/16 位分支用模板参数 `template<int BPC>` 消除运行时分支

---

## 5. 检查清单

新增算子模块时，逐一确认以下事项：

- [ ] `include/<module>.h` 聚合头文件已创建
- [ ] `include/<module>/types.hpp` — 枚举、错误码、校验函数完整
- [ ] `include/<module>/algorithms.hpp` — 所有算法函数声明且签名统一
- [ ] `src/<module>/common.hpp` — 内部工具函数提取完毕
- [ ] `src/<module>/dispatch.cpp` — 验证 + 注册表 + `static_assert` + 调度
- [ ] `src/<module>/<algo>.cpp` — 每个算法独立文件，开篇校验，结尾边界填充
- [ ] `CMakeLists.txt` — 源文件注册、include 路径、安装规则已添加
- [ ] `tests/unit/test_<module>.cpp` — 覆盖校验、元数据、基础处理、边界、高位深、参数化
- [ ] 注册表 `static_assert` 数量 = 枚举值数量
- [ ] 所有算法通过参数化测试
- [ ] `algorithm_name()` 和 `algorithm_window_size()` 的 switch 覆盖所有枚举值
- [ ] `validate_inputs()` 的 error 分支覆盖所有校验项
- [ ] clang-format 通过（Google Style, 4空格, 120列）
- [ ] 无编译警告（MSVC /W4 或 GCC -Wall -Wextra -Wpedantic）

---

## 6. 反例与常见错误

### 6.1 忘记更新注册表

```cpp
// 错误：新增枚举值但未加入注册表
enum class DemosaicAlgorithm {
    SUPER_FAST, HQLI, MG, L7, DFPD, AHD, AMAZE, RCD, PRISM,
    NEW_ALGO  // ← 新增，但 kDemosaicRegistry 未更新 → static_assert 崩溃
};
```

**正确做法：** 同时在 `dispatch.cpp` 的 `kRegistry` 数组中添加条目并更新 `static_assert` 的数量。

### 6.2 算法签名不一致

```cpp
// 错误：不同算法的参数顺序/类型不一致
DemosaicError process_a(const uint8_t* bayer, uint8_t* rgb, int w, int h, int bd);  // 参数名缩写
DemosaicError process_b(uint8_t* rgb, const uint8_t* bayer, int width, int height);  // 顺序不同
```

**正确做法：** 所有算法函数的参数名、类型、顺序**完全一致**，确保可以放入函数指针数组。

### 6.3 忘记边界填充

```cpp
// 错误：算法有 NxN 窗口但未填充边界，导致边缘有黑边/未定义值
DemosaicError process_xxx(...) {
    // ... 核心插值只处理 [border, w-border) x [border, h-border)
    return DemosaicError::Ok;  // ← 忘记 fill_rgb_borders()
}
```

**正确做法：** 有窗口的算法在返回前必须调用边界填充函数。

### 6.4 在公共头文件中暴露内部细节

```cpp
// include/your_module/algorithms.hpp — 公共API
// 错误：在此处 #include 内部头文件或声明内部函数
#include "src/your_module/common.hpp"  // ← 不该出现在公共头文件中
```

**正确做法：** 内部工具仅放在 `src/<module>/common.hpp`，通过源文件 `#include "common.hpp"` 使用。

---

## 7. 目录结构参考

```
SoolinOperator/
├── CMakeLists.txt                 # 项目构建配置
├── build.ps1                      # Windows 构建脚本
├── .clang-format                  # 格式化配置
├── .clang-tidy                    # 静态分析配置
├── include/
│   ├── im_operator.h                  # Demosaic 模块聚合头
│   ├── imop/
│   │   ├── types.hpp              # 枚举/结构体
│   │   ├── pixel_utils.hpp        # 像素工具（该模块特有）
│   │   ├── analyzer.hpp           # 分析工具（该模块特有）
│   │   └── algorithms.hpp         # 算法声明
│   ├── denoise.h                  # Denoise 模块聚合头
│   └── denoise/
│       ├── types.hpp
│       └── algorithms.hpp
├── src/
│   ├── analyzer.cpp
│   ├── algorithms/
│   │   ├── common.hpp             # Demosaic 内部工具
│   │   ├── optimized.hpp          # SIMD 优化声明
│   │   ├── interp_core.hpp        # 插值核心模板
│   │   ├── dispatch.cpp           # 调度器
│   │   ├── super_fast.cpp
│   │   ├── hqli.cpp
│   │   └── ...
│   └── denoise/
│       ├── common.hpp             # Denoise 内部工具
│       ├── dispatch.cpp           # 调度器
│       ├── gaussian.cpp
│       ├── median.cpp
│       └── ...
├── tests/
│   └── unit/
│       ├── main.cpp
│       ├── test_types.cpp
│       ├── test_pixel_utils.cpp
│       ├── test_dispatch.cpp
│       ├── test_algorithms.cpp
│       ├── test_correctness.cpp
│       └── test_denoise.cpp
├── benchmarks/
│   └── benchmark.cpp
├── examples/
│   └── demo.cpp
├── tools/
│   ├── synthetic_test.cpp
│   ├── folder_data_processor.cpp
│   ├── cuda_verify.cpp
│   └── denoise_synthetic_test.cpp
├── test_data/
└── docs/
    ├── API_Reference.md
    └── Operator_Development_Guide.md  ← 本文档
```
