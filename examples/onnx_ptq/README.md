# ONNX 训练后量化 (PTQ)

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
这款 ONNX PTQ 工具包提供了一套全面的工具，旨在通过量化来优化 ONNX（开放神经网络交换）模型。我们的工具包面向希望在部署于 TensorRT 时提升性能、减小模型大小并加快推理速度，同时又不影响神经网络准确性的开发者。

量化是一种有效的模型优化技术，可以压缩模型。结合模型优化器进行量化可以将模型大小压缩 2 到 4 倍，从而在保持模型质量的同时加快推理速度。

模型优化器支持高性能量化格式，包括 NVFP4、FP8、INT8、INT4，并支持 AWQ 和双量化等高级算法，同时提供易于使用的 Python API。

<div align="center">

|    **章节**     |                        **描述**                        |          **链接**          |                           **文档**                           |
| :-------------: | :----------------------------------------------------: | :------------------------: | :----------------------------------------------------------: |
|    先决条件     |            使用此技术所需的必需和可选软件包            |  [Link](#pre-requisites)   |                                                              |
|    入门指南     |  了解如何使用 PTQ 优化模型，以降低精度并提高推理效率   |  [Link](#getting-started)  | [docs](https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html) |
| PyTorch 到 ONNX | 示例脚本演示如何使用 PyTorch 进行量化，然后转换为 ONNX |   [Link](../torch_onnx/)   |                                                              |
|    高级功能     |          演示如何使用 ONNX 高级量化功能的示例          | [Link](#advanced-features) |                                                              |
|      资源       |                   相关资源的更多链接                   |     [Link](#resources)     |                                                              |


</div>

## 先决条件

### Docker

请使用 TensorRT docker 镜像（例如， `nvcr.io/nvidia/tensorrt:26.02-py3`）或访问我们的网站 [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) 了解更多信息。

> **注意：**如果您正在使用 `onnxruntime-gpu`我们建议使用 `nvcr.io/nvidia/tensorrt:25.06-py3` 因为它采用 CUDA 12 构建，而稳定版需要 CUDA 12。 `onnxruntime-gpu` 包裹。

在 TensorRT docker 容器内设置以下环境变量。

```bash
export CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"
```

此外，请按照以下安装步骤升级到最新版本的模型优化器并安装示例特定的依赖项。

### 本地安装

安装模型优化器 `onnx` 使用依赖项 `pip` 从 [PyPI](https://pypi.org/project/nvidia-modelopt/) 并安装示例所需的依赖项：

```bash
pip install -U nvidia-modelopt[onnx]
pip install -r requirements.txt
```

对于 TensorRT 编译器框架工作负载：

安装最新版本 [TensorRT](https://developer.nvidia.com/tensorrt) 从 [here](https://developer.nvidia.com/tensorrt/download)。

## 入门

### 准备示例模型

本文档中的大多数示例都使用 `vit_base_patch16_224.onnx` 作为输入模型。该模型可通过以下脚本下载：

```bash
python download_example_onnx.py \
    --timm_model_name=vit_base_patch16_224 \
    --onnx_save_path=vit_base_patch16_224.onnx \
    --fp16 # <Optional, if the desired output ONNX precision is FP16>
```

### 准备校准数据

校准数据是训练集或验证集的一个代表性子集，用于量化过程中确定将浮点值转换为低精度格式（INT8、FP8、INT4）的最佳比例因子。该数据通过分析整个网络中的激活分布，有助于在量化后保持模型精度。

首先，准备一些校准数据。TensorRT 建议 CNN 和 ViT 模型的校准数据大小至少为 500 张。以下命令从数据集中选取 500 张图像： [tiny-imagenet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) 将数据集转换为 numpy 格式的校准数组。在资源受限的环境下，减小校准数据的大小。

```bash
python image_prep.py \
    --calibration_data_size=500 \
    --output_path=calib.npy \
    --fp16 # <Optional, if the input ONNX is in FP16 precision>
```

> *对于 Int4 量化，建议设置 `--calibration_data_size=64`.*

### 将 ONNX 模型量化为 FP8、INT8 或 INT4 格式。

该模型可以使用 CLI 或 Python API 量化为 FP8、INT8 或 INT4 模型。对于 FP8 和 INT8 量化，您可以选择以下两种方式： `max` 和 `entropy` 校准算法。对于 INT4 量化， [awq_clip](https://arxiv.org/abs/2306.00978) 或者 [rtn_dq](https://ar5iv.labs.arxiv.org/html/2301.12017) 可以选择算法。

> *有关 NVFP4 和 MXFP8 ONNX 的信息，请参阅 [PyTorch to ONNX example](../torch_onnx/).*

> *最低操作集要求：int8（版本 13+）、fp8（版本 21+）、int4（版本 21+）。ModelOpt 会自动升级较低版本的操作集以满足这些要求。*

#### 选项 1：命令行界面

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224.onnx \
    --quantize_mode=<fp8|int8|int4> \
    --calibration_data=calib.npy \
    --calibration_method=<max|entropy|awq_clip|rtn_dq> \
    --output_path=vit_base_patch16_224.quant.onnx
```

#### 选项 2：Python API

```python
from modelopt.onnx.quantization import quantize

quantize(
    onnx_path="vit_base_patch16_224.onnx",
    quantize_mode="int8",       # fp8, int8, int4 etc.
    calibration_data="calib.npy",
    calibration_method="max",   # max, entropy, awq_clip, rtn_dq etc.
    output_path="vit_base_patch16_224.quant.onnx",
)
```

### 评估量化的 ONNX 模型

评估脚本会自动下载并使用 [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) 数据集来自 Hugging Face。此受限存储库需要通过 Hugging Face 访问令牌进行身份验证。详情请参阅 <https://huggingface.co/docs/hub/en/security-tokens>。量化后的 ONNX ViT 模型可在 ImageNet 数据集上按如下方式进行评估：

```bash
python evaluate.py \
    --onnx_path=<path to classification model> \
    --imagenet_path=<HF dataset card or local path to the ImageNet dataset> \
    --engine_precision=stronglyTyped \
    --model_name=vit_base_patch16_224
```

该脚本将量化的 ONNX 模型转换为 TensorRT 引擎，并使用该引擎进行评估。最终，评估结果将按如下方式报告：

```bash
The top1 accuracy of the model is <accuracy score between 0-100%>
The top5 accuracy of the model is <accuracy score between 0-100%>
Inference latency of the model is <X> ms
```

## 高级功能

### ONNX模型的节点校准

逐节点校准是一种内存优化功能，旨在降低大型 ONNX 模型量化过程中的内存消耗。该功能并非一次性对整个网络进行推理，而是逐个节点地处理模型，从而显著降低峰值内存使用量并防止内存溢出 (OOM) 错误。

#### 工作原理

启用逐节点校准后，量化过程如下：

1. **模型分解**：将原始 ONNX 模型拆分为多个单节点子模型。
1. **管理依赖关系**：跟踪节点间的输入/输出依赖关系，以确保正确的执行顺序。
1. **按顺序处理**：使用拓扑处理顺序对每个节点分别运行校准。
1. **内存管理**：自动清理中间结果并管理引用计数，以最大限度地减少内存使用量
1. **汇总结果**：合并所有节点的校准数据，生成最终的量化模型。

#### 何时使用逐节点校准

每个节点的校准尤其有利于：

- **大型模型**会在标准校准过程中导致OOM错误
- **内存受限环境**，即GPU内存有限的环境。
- 具有复杂架构且中间内存需求高的模型

#### 用法

要启用逐节点校准，请添加 `--calibrate_per_node` 量化命令的标志：

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224.onnx \
    --quantize_mode=<int8/fp8> \
    --calibration_data=calib.npy \
    --calibrate_per_node \
    --output_path=vit_base_patch16_224.quant.onnx
```

> **注意**：INT4 量化方法不支持逐节点校准（`awq_clip`， `rtn_dq`）

### 使用自定义操作量化 ONNX 模型

此功能需要 `TensorRT 10+` 和 `ORT>=1.20`为了正确使用，请确保路径正确。 `libcudnn*.so` 以及 TensorRT `lib/` 位于 `LD_LIBRARY_PATH` 环境变量以及 `tensorrt` 已安装python包。

文中提供了一个独立完整的示例。 [`custom_op_plugin/`](./custom_op_plugin/) 子文件夹，基于 [leimao/TensorRT-Custom-Plugin-Example](https://github.com/leimao/TensorRT-Custom-Plugin-Example)请参考以下步骤。

**步骤 1**：构建 TensorRT 插件并创建示例 ONNX 模型。

1.1. 编译 TensorRT 插件：

```bash
cmake -S custom_op_plugin/plugin -B /tmp/plugin_build
cmake --build /tmp/plugin_build --config Release --parallel
```

这将生成 `/tmp/plugin_build/libidentity_conv_plugin.so`。

1.2. 使用自定义创建 ONNX 模型 `IdentityConv` 操作员：

```bash
python custom_op_plugin/create_identity_neural_network.py \
    --output_path=/tmp/identity_neural_network.onnx
```

**步骤 2**：使用编译后的插件量化 ONNX 模型。

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=/tmp/identity_neural_network.onnx \
    --trt_plugins=/tmp/plugin_build/libidentity_conv_plugin.so
```

**步骤 3**：使用 TensorRT 部署量化模型。

```bash
trtexec --onnx=/tmp/identity_neural_network.quant.onnx \
    --staticPlugins=/tmp/plugin_build/libidentity_conv_plugin.so
```

### 利用自动调谐优化 Q/DQ 节点位置

此功能使用 TensorRT 性能测量来自动优化 ONNX 模型的 Q/DQ（量化/反量化）节点放置。
有关独立工具包的更多信息，请参阅： [autotune](./autotune)。

要在 ONNX 量化工作流程中使用此功能，只需添加 `--autotune` 在您的命令行界面中：

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224.onnx \
    --quantize_mode=<fp8|int8|int4> \
    --calibration_data=calib.npy \
    --calibration_method=<max|entropy|awq_clip|rtn_dq> \
    --output_path=vit_base_patch16_224.quant.onnx \
    --autotune=<quick,default,extensive>
```

如需更精细的自动调优参数，请参阅 [API guide](https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html)。

## 资源

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/1699)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

### 技术资源

示例脚本支持多种量化方案：

1. 这 [FP8 format](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) 可在 Hopper 和 Ada GPU 上使用 [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) 大于或等于 8.9。

1. 这 [INT4 AWQ](https://arxiv.org/abs/2306.00978) INT4 AWQ 是一种仅使用 INT4 权重的量化和校准方法。INT4 AWQ 在小批量推理中尤为有效，因为此时推理延迟主要取决于权重加载时间而非计算时间本身。对于小批量推理，INT4 AWQ 的延迟可能低于 FP8/INT8，且精度损失小于 INT8。

1. 这 [NVFP4](https://blogs.nvidia.com/blog/generative-ai-studio-ces-geforce-rtx-50-series/) NVFP4 是 NVIDIA Blackwell GPU 支持的一种新型 FP4 格式，与其他 4 位格式相比，它展现出良好的精度。NVFP4 可应用于模型权重和激活值，与 Blackwell 上的 FP8 数据格式相比，它有望显著提高数学运算吞吐量，并降低内存占用和内存带宽使用。
