# Torch 量化到 ONNX 导出

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)

本示例演示了如何量化 PyTorch 模型并将其导出为 ONNX 格式。脚本利用 ModelOpt 工具包进行量化和 ONNX 导出。

对于**视觉模型**而言， `torch_quant_to_onnx.py` 此目录中的脚本直接处理量化和 ONNX 导出。

对于**LLM 和 VLM**，请使用 [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) 它提供了一个完整的管道，用于使用 ModelOpt 对模型进行量化，并将其导出为优化的 ONNX，以便在边缘平台（Jetson、DRIVE）上部署。

<div align="center">


|    **章节**    |                 **描述**                  |                          **链接**                           |
| :------------: | :---------------------------------------: | :---------------------------------------------------------: |
|    前提条件    |          使用此示例所需的软件包           |                   [关联](#pre-requisites)                   |
|    视觉模型    |     量化 timm 模型并导出为 ONNX 格式      |                   [关联](#vision-models)                    |
| LLM 量化和导出 | 通过 TensorRT-Edge-LLM 量化和导出 LLM/VLM | [关联](#llm-quantization-and-export-with-tensorrt-edge-llm) |
|   支持的模型   | TensorRT-Edge-LLM 支持的 LLM 和 VLM 模型  |                  [关联](#supported-models)                  |
|    混合精度    |        自动模式，实现最佳逐层量化         |       [关联](#mixed-precision-quantization-auto-mode)       |
|      资源      |            相关资源的更多链接             |                     [关联](#resources)                      |

</div>

## 先决条件

### Docker

请使用 TensorRT docker 镜像（例如， `nvcr.io/nvidia/tensorrt:26.02-py3`）或访问我们的网站 [安装文档](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) 了解更多信息。

在 TensorRT docker 容器内设置以下环境变量。

```bash
export CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"
```

### 本地安装

安装模型优化器 `onnx` 使用依赖项 `pip` 从 [PyPI](https://pypi.org/project/nvidia-modelopt/) 并安装示例所需的依赖项：

```bash
pip install -U "nvidia-modelopt[onnx]"
pip install -r requirements.txt
```

对于 TensorRT 编译器框架工作负载：

安装最新版本 [TensorRT](https://developer.nvidia.com/tensorrt) 从 [这里](https://developer.nvidia.com/tensorrt/download)。

## 视觉模型

这 `torch_quant_to_onnx.py` 脚本量化 [蒂姆](https://github.com/huggingface/pytorch-image-models) 建立视觉模型并将其导出为 ONNX 格式。

### 它的作用

- 加载预训练的 timm torch 模型（默认值：ViT-Base）。
- 使用 ModelOpt 将 torch 模型量化为 FP8、MXFP8、INT8、NVFP4 或 INT4_AWQ。
- 对于具有 Conv2d 层的模型（例如 SwinTransformer），自动将 Conv2d 量化覆盖为 FP8（对于 MXFP8/NVFP4 模式）或 INT8（对于 INT4_AWQ 模式），以实现与 TensorRT 的兼容性。
- 将量化模型导出为 ONNX 格式。
- 对 ONNX 模型进行后处理，使其与 TensorRT 兼容。
- 保存最终的 ONNX 模型。

> *操作集 20 用于将火炬模型导出为 ONNX 格式。*

### 用法

```bash
# 下载示例模型
hf download \
  timm/vit_base_patch16_224.augreg2_in21k_ft_in1k \
  --local-dir ./vit_base_patch16_224
  
python torch_quant_to_onnx.py \
    --timm_model_name=./vit_base_patch16_224 \
    --quantize_mode=<fp8|mxfp8|int8|nvfp4|int4_awq> \
    --onnx_save_path=<path to save the exported ONNX model>
```

### Conv2d 量化覆盖

TensorRT 仅支持 FP8 和 INT8 卷积运算。当量化带有 Conv2d 层（例如 SwinTransformer）的模型时，脚本会自动应用以下覆盖：

|   量化模式   |   Conv2d 覆盖   | 原因         |
| :----------: | :-------------: | :----------- |
|  FP8、INT8   |  无（已兼容）   | 原生支持 TRT |
| MXFP8、NVFP4 | 二维卷积 -> FP8 | TRT 卷积限制 |
|   INT4_AWQ   | Conv2d -> INT8  | TRT 转化限制 |

### 评估

如果输入模型是图像分类类型，请使用以下脚本对其进行评估。默认使用 Hugging Face 上的 [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) 数据集；此受限存储库需要通过 Hugging Face 访问令牌进行身份验证。详情请参阅 <https://huggingface.co/docs/hub/en/security-tokens>。如果只需快速评估，可使用本地标准目录格式的 Tiny ImageNet；脚本会将其 WordNet 类别映射到 ImageNet-1K 标签。

> *注：评估 MXFP8 或 NVFP4 ONNX 模型需要 TensorRT 10.11 或更高版本。*

```bash
python ../onnx_ptq/evaluate.py \
    --onnx_path=<path to the exported ONNX model> \
    --dataset_path=<HF dataset card or local path to the ImageNet dataset> \
    --engine_precision=stronglyTyped \
    --model_name=<timm model name>
# 数据集准备
hf download \
  zh-plus/tiny-imagenet \
  --repo-type dataset \
  --local-dir ./tiny-imagenet 

python ../onnx_ptq/evaluate.py \
    --onnx_path=./onnx/VisionTransformer.onnx \
    --dataset_path=./tiny-imagenet \
    --engine_precision=stronglyTyped \
    --model_name=./vit_base_patch16_224

```

## 使用 TensorRT-Edge-LLM 进行 LLM 量化和导出

[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) 提供使用 NVIDIA ModelOpt 对 LLM 和 VLM 进行量化的完整管道，并将它们导出为优化的 ONNX，以便在 NVIDIA Jetson 和 DRIVE 等边缘平台上部署。

### 概述

该流程遵循以下步骤：

1. **量化**（带 GPU 的 x86 主机）— 使用 ModelOpt 降低模型精度（FP8、INT4 AWQ、NVFP4）
2. **导出**（带 GPU 的 x86 主机）— 将量化模型转换为 ONNX 格式
3. **构建**（边缘设备）— 将 ONNX 编译成 TensorRT 引擎
4. 推理（边缘设备）— 运行已编译的引擎

### 安装

```bash
# Use the PyTorch Docker image (recommended)
docker pull nvcr.io/nvidia/pytorch:25.12-py3
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.12-py3 bash

# Clone and install TensorRT-Edge-LLM
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
python3 -m venv venv
source venv/bin/activate
pip3 install .

# Verify installation
tensorrt-edgellm-quantize --help
tensorrt-edgellm-export --help
```

系统要求：

- x86-64 Linux（推荐使用 Ubuntu 22.04 或 24.04）
- NVIDIA GPU，计算能力 8.0+（Ampere 或更新版本）
- CUDA 12.x 或 13.x，Python 3.10+
- GPU显存：3B及以下型号为16GB，4B及以下型号为40GB，8B及以下型号为80GB。

### 命令行工具

工具 | 用途 |
| :--- | :--- |
| `tensorrt-edgellm-quantize` 使用 ModelOpt (FP8、INT4 AWQ、NVFP4) 对模型进行量化；子命令： `llm`， `draft` |
| `tensorrt-edgellm-export` | 将量化或 FP16/BF16 检查点导出为 ONNX；自动检测 VLM 和音频组件 |
| `tensorrt-edgellm-insert-lora` | 将 LoRa 模式插入到现有的 ONNX 模型中 |
| `tensorrt-edgellm-process-lora` | 处理 LoRA 适配器权重以进行运行时加载 |

### 示例：量化并导出LLM

```bash
# Step 1: Quantize with ModelOpt
tensorrt-edgellm-quantize llm \
    --model_dir Qwen/Qwen2.5-3B-Instruct \
    --quantization fp8 \
    --output_dir quantized/qwen2.5-3b-fp8

# Step 2: Export to ONNX
tensorrt-edgellm-export \
    quantized/qwen2.5-3b-fp8 \
    onnx_models/qwen2.5-3b
```

### 示例：量化并导出 VLM

```bash
# Quantize with ModelOpt (handles both LLM and visual components)
tensorrt-edgellm-quantize llm \
    --model_dir Qwen/Qwen2.5-VL-3B-Instruct \
    --quantization fp8 \
    --output_dir quantized/qwen2.5-vl-3b

# Export to ONNX (auto-detects VLM and exports LLM + visual encoder to separate subdirs)
tensorrt-edgellm-export \
    quantized/qwen2.5-vl-3b \
    onnx_models/qwen2.5-vl-3b
```

### 示例：EAGLE 推测解码

```bash
# Quantize base model
tensorrt-edgellm-quantize llm \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --output_dir quantized/llama3.1-8b-base

# Export base model with EAGLE flag
tensorrt-edgellm-export \
    quantized/llama3.1-8b-base \
    onnx_models/llama3.1-8b/base \
    --eagle-base

# Quantize EAGLE draft model
tensorrt-edgellm-quantize draft \
    --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
    --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
    --quantization fp8 \
    --output_dir quantized/llama3.1-8b-draft

# Export draft model
tensorrt-edgellm-export \
    quantized/llama3.1-8b-draft \
    onnx_models/llama3.1-8b/draft
```

### 量化方法

方法 | 描述 |
| :--- | :--- |
| FP8 | 在SM89+硬件（Hopper、Ada）上实现最佳精度与内存平衡 |
| INT4 AWQ | 仅权重量化；适用于内存受限平台和小批量推理 |
| NVFP4 | 适用于 NVIDIA Blackwell 和 Thor 硬​​件的 4 位格式；同时适用于权重和激活值 |
| MXFP8 | 实验性；适用于 SM89+ 硬件的微缩 FP8 格式 |
| INT8 SmoothQuant | 实验性；使用 SmoothQuant 进行 INT8 权重和激活量化 |
| INT4 GPTQ | 可直接从 HuggingFace Hub 加载（无需额外量化） |

### 支持的型号

有关最新的支持矩阵，请参阅 [TensorRT-Edge-LLM支持的模型](https://nvidia.github.io/TensorRT-Edge-LLM/developer_guide/getting-started/supported-models.html) 页。

#### LLM

| 型号                                                         | FP16 | FP8  | INT4 | NVFP4 |
| :----------------------------------------------------------- | :--: | :--: | :--: | :---: |
| [Llama-3-8B-指导](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Llama-3.1-8B-指令](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Llama-3.2-3B-指导](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2-0.5B-指导](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2-1.5B-指导](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2-7B-指导](https://huggingface.co/Qwen/Qwen2-7B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2.5-0.5B-指导](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2.5-1.5B-指导](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2.5-3B-指导](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2.5-7B-指导](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)         |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)         |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen3-4B-指令-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)             |  ✅   |  ✅   |  ✅   |   ✅   |
| [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |  ✅   |  ✅   |  ✅   |   ✅   |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |  ✅   |  ✅   |  ✅   |   ✅   |

#### VLM

| 型号                                                         | FP16 | FP8  | INT4 | NVFP4 |
| :----------------------------------------------------------- | :--: | :--: | :--: | :---: |
| [Qwen2-VL-2B-指令](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2-VL-7B-指令](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2.5-VL-3B-指令](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen2.5-VL-7B-指令](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen3-VL-2B-指令](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen3-VL-4B-指令](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Qwen3-VL-8B-指令](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |  ✅   |  ✅   |  ✅   |   ✅   |
| [实习生VL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B) |  ✅   |  ✅   |  ✅   |   ✅   |
| [实习生VL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) |  ✅   |  ✅   |  ✅   |   ✅   |
| [Phi-4-多模态指导](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) |  ✅   |  ✅   |  ✅   |   ✅   |

### 故障排除

- **GPU 显存不足**：请使用更大显存的 GPU（4B 及以下型号为 40 GB，8B 及以下型号为 80 GB），或尝试 `--device cpu` （精度支持有限）。
- **校准数据集问题**：请手动下载数据集并提供本地路径。 `--calib_dataset ./path/to/dataset`。
- **精度下降**：尝试使用 FP8 代替 INT4/NVFP4，或增加校准样本量。

完整文档请参见 [TensorRT-Edge-LLM 开发人员指南](https://nvidia.github.io/TensorRT-Edge-LLM/)。

## 混合精度量化（自动模式）

这 `auto` 该模式通过搜索每一层的最佳量化格式来实现混合精度量化。这种方法根据各层的敏感性，为不同的层分配不同的精度格式（例如，NVFP4、FP8），从而平衡模型精度和压缩率。

### 工作原理

1. **灵敏度分析**：使用基于梯度的分析方法计算每一层的灵敏度得分
2. **格式搜索**：针对每一层，搜索指定的量化格式。
3. 约束优化：寻找满足有效比特约束并最大限度减少精度损失的最佳格式分配方案。

### 关键参数

| 参数                      | 默认值 | 描述                                                         |
| :------------------------ | :----: | :----------------------------------------------------------- |
| `--effective_bits`        |  4.8   | 模型中每个权重的平均目标比特数。数值越低，压缩率越高，但精度可能越低。搜索算法会找到满足此约束条件并最大限度减少精度损失的最佳逐层格式分配方案。例如，4.8 表示每个权重的平均比特数为 4.8（FP4 和 FP8 层的混合）。 |
| `--num_score_steps`       |  128   | 用于通过基于梯度的分析计算每层灵敏度得分的前向/后向传播次数。值越高，灵敏度估计越准确，但搜索时间越长。建议范围：64-256。 |
| `--calibration_data_size` |  512   | 用于灵敏度评分和校准的校准样本数量。自动模式下，损失计算需要标签。 |

### 用法

```bash
python torch_quant_to_onnx.py \
    --timm_model_name=vit_base_patch16_224 \
    --quantize_mode=auto \
    --auto_quantization_formats NVFP4_AWQ_LITE_CFG FP8_DEFAULT_CFG \
    --effective_bits=4.8 \
    --num_score_steps=128 \
    --calibration_data_size=512 \
    --evaluate \
    --onnx_save_path=vit_base_patch16_224.auto_quant.onnx
```

## ONNX导出支持的视觉模型

| Model | FP8 | INT8 | MXFP8 | NVFP4 | INT4_AWQ | Auto |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [vit_base_patch16_224](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [swin_tiny_patch4_window7_224](https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [swinv2_tiny_window8_256](https://huggingface.co/timm/swinv2_tiny_window8_256.ms_in1k) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [resnet50](https://huggingface.co/timm/resnet50.a1_in1k) | ✅ | ✅ | ✅ | ✅ | | ✅ |

## 资源

- 📅 [路线图](https://github.com/NVIDIA/Model-Optimizer/issues/1699)
- 📖 [文档](https://nvidia.github.io/Model-Optimizer)
- 🎯 [基准](../benchmark.md)
- 💡 [发行说明](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [提交错误报告](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [提交功能请求](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

### 技术资源

示例脚本支持多种量化方案：

1. 这 [FP8格式](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) 可在 Hopper 和 Ada GPU 上使用 [CUDA 计算能力](https://developer.nvidia.com/cuda-gpus) 大于或等于 8.9。

1. 这 [INT4 AWQ](https://arxiv.org/abs/2306.00978) INT4 AWQ 是一种仅使用 INT4 权重的量化和校准方法。INT4 AWQ 在小批量推理中尤为有效，因为此时推理延迟主要取决于权重加载时间而非计算时间本身。对于小批量推理，INT4 AWQ 的延迟可能低于 FP8/INT8，且精度损失小于 INT8。

1. 这 [NVFP4](https://blogs.nvidia.com/blog/generative-ai-studio-ces-geforce-rtx-50-series/) NVFP4 是 NVIDIA Blackwell GPU 支持的一种新型 FP4 格式，与其他 4 位格式相比，它展现出良好的精度。NVFP4 可应用于模型权重和激活值，与 Blackwell 上的 FP8 数据格式相比，它有望显著提高数学运算吞吐量，并降低内存占用和内存带宽使用。

