# QDQ 布局优化示例

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
本示例演示了使用 TensorRT 性能测量方法对 ONNX 模型进行自动 Q/DQ（量化/反量化）节点放置优化。

## 目录

<div align="center">

| **章节** | **描述** | **链接** | **文档** |
| :------------: | :------------: | :------------: | :------------: |
| 前提条件 | 获取模型、设置固定批处理大小和目录概览 | [Link](#prerequisites) | |
| 快速入门 | 基本用法、FP8 量化和快速探索 | [Link](#quick-start) | |
| 输出结构 | 输出工作区布局和文件 | [Link](#output-structure) | |
| 区域检查 | 调试区域发现和分区 | [Link](#region-inspection) | |
| 使用优化模型 | 通过 TensorRT 部署 | [Link](#using-the-optimized-model) | |
| 模式缓存 | 在类似模型上重用已学习的模式 | [Link](#pattern-cache) | |
| 基于现有QDQ模型进行优化 | 从现有的量化模型开始 | [Link](#optimize-from-existing-qdq-model) | |
| 使用 TensorRT 进行远程自动调优 | 将自动调优任务卸载到远程硬件 | [Link](#remote-autotuning-with-tensorrt) | |
| 程序化 API 使用 | Python API 和底层控制 | [Link](#programmatic-api-usage) | |
| 文档 | 用户指南和 API 参考 | [Link](#documentation) | [docs](https://nvidia.github.io/Model-Optimizer/) |

</div>

## 先决条件

### 获取模型

从 ONNX 模型库下载 ResNet50 模型：

```bash
# Download ResNet50 from ONNX Model Zoo
curl -L -o resnet50_Opset17.onnx https://github.com/onnx/models/raw/main/Computer_Vision/resnet50_Opset17_torch_hub/resnet50_Opset17.onnx
```

### 设置固定批量大小

下载的模型具有动态批处理大小。为了获得最佳的 TensorRT 基准测试性能，请使用 Polygraphy 设置固定批处理大小：

```bash
polygraphy surgeon sanitize --override-input-shapes x:[128,3,1024,1024] -o resnet50_Opset17_bs128.onnx resnet50_Opset17.onnx
```

对于其他批次大小，请更改形状中的第一个维度（例如） `x:[1,3,1024,1024]` 批次大小为 1）。

### 本目录包含哪些内容

- `README.md` 本指南

**注意：** ONNX 模型文件未包含在存储库中（已通过以下方式排除）： `.gitignore`请按照上述说明下载并准备它们。

## 快速入门

### 基本用法

使用 INT8 量化优化 ResNet50 模型：

```bash
# Using the fixed batch size model
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50_Opset17_bs128.onnx \
    --output_dir ./resnet50_results \
    --quant_type int8 \
    --schemes_per_region 30

# Or use the original dynamic batch size model, batch is set to 1 in benchmark
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50_Opset17.onnx \
    --output_dir ./resnet50_results \
    --quant_type int8 \
    --schemes_per_region 30
```

短线选项： `-m` 为了 `--onnx_path`， `-o` 为了 `--output_dir`， `-s` 为了 `--schemes_per_region`默认输出目录是 `./autotuner_output` 如果 `--output_dir` 省略。

这将：

1. 自动发现模型中的优化区域
2. 每个区域模式测试 30 种不同的 Q/DQ 放置方案
3. 测量每种方案的 TensorRT 性能
4. 将最佳优化模型导出到 `./resnet50_results/optimized_final.onnx`

### FP8 量化

对于 FP8 量化：

```bash
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50_Opset17_bs128.onnx \
    --output_dir ./resnet50_fp8_results \
    --quant_type fp8 \
    --schemes_per_region 50
```

### 更快的探索

为了快速进行实验，请减少方案数量：

```bash
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50_Opset17_bs128.onnx \
    --output_dir ./resnet50_quick \
    --schemes_per_region 15
```

## 输出结构

运行后，输出工作区将如下所示：

```log
resnet50_results/
├── optimized_final.onnx              # Optimized model
├── baseline.onnx                     # Baseline for comparison
├── autotuner_state.yaml              # Resume checkpoint
├── autotuner_state_pattern_cache.yaml # Reusable pattern cache
├── logs/
│   ├── baseline.log                  # TensorRT baseline log
│   ├── region_*_scheme_*.log         # Per-scheme logs
│   └── final.log                     # Final model log
└── region_models/                    # Best model per region (intermediate)
    └── region_*_level_*.onnx
```

## 区域巡查

要调试自动调优器如何发现和划分模型中的区域，请使用以下方法： `region_inspect` 该工具运行与自动调优器相同的区域搜索，并打印区域层次结构、节点计数和汇总统计信息（不运行基准测试）。

```bash
# Basic inspection (regions with quantizable ops only)
python3 -m modelopt.onnx.quantization.autotune.region_inspect --model resnet50_Opset17_bs128.onnx

# Verbose mode for detailed debug logging
python3 -m modelopt.onnx.quantization.autotune.region_inspect --model resnet50_Opset17_bs128.onnx --verbose

# Custom maximum sequence region size
python3 -m modelopt.onnx.quantization.autotune.region_inspect --model resnet50_Opset17_bs128.onnx --max-sequence-size 20

# Include all regions (including those without Conv/MatMul etc.)
python3 -m modelopt.onnx.quantization.autotune.region_inspect --model resnet50_Opset17_bs128.onnx --include-all-regions
```

简称： `-m` 为了 `--model`， `-v` 为了 `--verbose`. 使用此功能可在自动调优之前或期间验证区域边界和计数。

## 使用优化模型

使用 TensorRT 进行部署：

```bash
trtexec --onnx=resnet50_results/optimized_final.onnx \
        --saveEngine=resnet50.engine \
        --stronglyTyped
```

## 模式缓存

在类似模型上重用已学习的模式（热启动）：

```bash
# First optimization on ResNet50
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50_Opset17_bs128.onnx \
    --output_dir ./resnet50_run

# Download and prepare ResNet101 (or any similar model)
curl -L -o resnet101_Opset17.onnx https://github.com/onnx/models/raw/main/Computer_Vision/resnet101_Opset17_torch_hub/resnet101_Opset17.onnx
polygraphy surgeon sanitize --override-input-shapes x:[128,3,1024,1024] -o resnet101_Opset17_bs128.onnx resnet101_Opset17.onnx

# Reuse patterns from ResNet50 on ResNet101 
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet101_Opset17_bs128.onnx \
    --output_dir ./resnet101_run \
    --pattern_cache ./resnet50_run/autotuner_state_pattern_cache.yaml
```

## 基于现有QDQ模型进行优化

如果用户已经拥有量化模型，他可以以此为起点，寻找更优的 Q/DQ 位置：

```bash
# Use an existing QDQ model as baseline (imports quantization patterns)
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50_Opset17_bs128.onnx \
    --output_dir ./resnet50_improved \
    --qdq_baseline resnet50_quantized.onnx \
    --schemes_per_region 40
```

这将：

1. 从基线模型中提取 Q/DQ 插入点
2. 将它们作为种子方案导入模式缓存
3. 生成并测试各种方案，以找到更佳的布局。
4. 与基准性能进行比较

**使用案例：**

- **改进现有量化**：微调手动量化模型
- **工具对比**：测试自动调谐器是否能胜过其他量化方法
- **引导优化**：从专家调优的方案开始

**示例工作流程：**

```bash
# Step 1: Create initial quantized model with modelopt 
# For example, using modelopt's quantize function:
python3 -c "
import numpy as np
from modelopt.onnx.quantization import quantize

# Create dummy calibration data (replace with real data for production)
dummy_input = np.random.randn(128, 3, 224, 224).astype(np.float32)
quantize(
    'resnet50_Opset17_bs128.onnx',
    calibration_data=dummy_input,
    calibration_method='entropy',
    output_path='resnet50_quantized.onnx'
)
"

# Step 2: Use the quantized baseline for autotuning
# The autotuner will try to find better Q/DQ placements than the initial quantization
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50_Opset17_bs128.onnx \
    --output_dir ./resnet50_autotuned \
    --qdq_baseline resnet50_quantized.onnx \
    --schemes_per_region 50
```

**注意：**本示例使用虚拟校准数据。在生产环境中使用时，请提供能够代表推理工作负载的真实校准数据。

## 使用 TensorRT 进行远程自动调优

TensorRT 10.15+ 支持安全模式下的远程自动调优（`--safe`这使得 TensorRT 的优化过程可以卸载到远程硬件上。这在针对不同目标 GPU 优化模型而无法直接访问这些 GPU 时非常有用。

要在 Q/DQ 位置优化期间使用远程自动调谐，请运行以下命令 `trtexec` 并传递额外参数：

```bash
python3 -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50_Opset17_bs128.onnx \
    --output_dir ./resnet50_remote_autotuned \
    --schemes_per_region 50 \
    --use_trtexec \
    --trtexec_benchmark_args "--remoteAutoTuningConfig=\"<remote autotuning config>\" --safe --skipInference"
```

**要求：**

- TensorRT 10.15 或更高版本
- 有效的远程自动调谐配置
- `--use_trtexec` 必须设置（基准测试用途） `trtexec` （而不是 TensorRT Python API）
- `--safe --skipInference` 必须通过以下方式启用 `--trtexec_benchmark_args`

代替 `<remote autotuning config>` 使用实际的远程自动调谐配置字符串（见 `trtexec --help` 更多详情）。
 其他 TensorRT 基准测试选项（例如） `--timing_cache`， `--warmup_runs`， `--timing_runs`， `--plugin_libraries`）也可用；运行 `--help` 详情请见下文。

## 程序化 API 使用

以上所有示例均使用命令行界面。若要在 Python 代码中进行**底层程序化控制**，请直接使用 Python API。这样用户可以：

- 将自动调优集成到自定义管道中
- 实现自定义评估函数
- 控制状态管理和检查点
- 构建自定义优化工作流程

**有关底层用法，请参阅 API 参考文档：**

- [`docs/source/reference/2_qdq_placement.rst`](../../docs/source/reference/2_qdq_placement.rst)

API 文档包含以下方面的详细示例：

- 使用 `QDQAutotuner` 阶级和 `region_pattern_autotuning_workflow`
- 自定义区域发现和方案生成
- 以编程方式管理优化状态和模式缓存
- 实现自定义性能评估器（例如通过 `init_benchmark_instance` 和 `benchmark_onnx_model`）

## 文档

有关 QDQ 布局优化的完整文档，请参阅：

- **用户指南**： [`docs/source/guides/9_qdq_placement.rst`](../../docs/source/guides/9_qdq_placement.rst)
  - 自动调谐器工作原理的详细说明
  - 高级使用模式和最佳实践
  - 配置选项和性能调优
  - 常见问题排查

- **API 参考**： [`docs/source/reference/2_qdq_placement.rst`](../../docs/source/reference/2_qdq_placement.rst)
  - 所有类和函数的完整 API 文档
  - 底层使用示例
  - 状态管理和模式缓存详情

有关命令行帮助和所有选项（例如 `--state_file`， `--node_filter_list`， `--default_dq_dtype`， `--verbose`）：

```bash
python3 -m modelopt.onnx.quantization.autotune --help
```
