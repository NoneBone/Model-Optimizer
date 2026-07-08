# 训练后量化（PTQ）

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
量化是一种有效的模型优化技术，可以压缩模型。结合模型优化器进行量化可以将模型大小压缩 2 到 4 倍，从而在保持模型质量的同时加快推理速度。

模型优化器支持高性能量化格式，包括 NVFP4、FP8、INT8、INT4，并支持 SmoothQuant、AWQ、SVDQuant 和双重量化等高级算法，并提供易于使用的 Python API。

本节重点介绍训练后量化技术，该技术在训练后降低模型精度，以提高推理效率，而无需重新训练。

<div align="center">

| **章节** | **描述** | **链接** | **文档** |
| :------------: | :------------: | :------------: | :------------: |
| 先决条件 | 使用此技术所需的必需和可选软件包 | \[[Link](#pre-requisites)\] | |
| 入门指南 | 了解如何使用 PTQ 优化模型，以降低精度并提高推理效率 |[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
| 支持矩阵 | 查看支持矩阵，了解不同型号的量化兼容性和功能可用性 | \[[Link](#support-matrix)\] | |
| 自动量化 | 自动选择混合精度量化的层/精度，以增强推理性能和准确性之间的权衡 | \[[Link](#autoquantize)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#optimal-partial-quantization-using-auto-quantize)\] |
| Real Quant | Real Quant 将模型权重压缩为低精度格式，以降低量化所需的内存。 | \[[Link](https://nvidia.github.io/Model-Optimizer/guides/_compress_quantized_models.html)\] | |
| 框架脚本 | 示例脚本演示了用于优化 Hugging Face / Megatron-Bridge / Megatron-LM 模型的量化技术 | \[[Link](#framework-scripts)\] | |
| 评估准确率 | 评估您的模型准确率！ | \[[Link](#evaluate-accuracy)\] | |
| 导出检查点 | 导出到 Hugging Face 统一检查点并部署到 TRT-LLM/vLLM/SGLang | \[[Link](#exporting-checkpoints)\] | \[[docs](https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html)\] |
| 预量化检查点 | 准备部署 Hugging Face 预量化检查点 | \[[Link](#pre-quantized-checkpoints)\] | |
| 资源 | 相关资源的更多链接 | \[[Link](#resources)\] | |

</div>

## 先决条件

### Docker

对于拥抱脸模型，请使用 TensorRT-LLM docker 镜像（例如， `nvcr.io/nvidia/tensorrt-llm/release:1.2.0`）。
访问我们的网站 [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) 了解更多信息。

此外，请按照以下安装步骤升级到最新版本的模型优化器并安装示例特定的依赖项。

### 本地安装

对于拥抱脸模型，请安装模型优化器。 `hf` 使用依赖项 `pip` 从 [PyPI](https://pypi.org/project/nvidia-modelopt/) 并安装示例所需的依赖项：

```bash
pip install -U nvidia-modelopt[hf]
pip install -r requirements.txt
```

对于 TensorRT-LLM 的部署，请使用 TensorRT-LLM Docker 镜像或按照他们的说明进行操作。 [installation docs](https://nvidia.github.io/TensorRT-LLM/installation/index.html)。
同样，对于 vLLM 或 SGLang 的部署，请使用其安装文档。

## 入门

### 1. 量化（训练后量化）

借助下方简洁的 API，您可以轻松使用模型优化器对模型进行量化。模型优化器通过将模型的精度转换为所需的精度，然后使用小型数据集（通常为 128-512 个样本）来实现这一点。 [calibrate](https://nvidia.github.io/Model-Optimizer/guides/_basic_quantization.html) 量化缩放因子。PTQ 的精度通常在不同的校准数据选择下都具有鲁棒性，默认情况下，模型优化器使用混合数据。 [`cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail) 和 [`nemotron-post-training-dataset-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)用户可以通过轻松修改参数来尝试其他数据集。 `calib_set`。

```python
import modelopt.torch.quantization as mtq

# Setup the model
model = AutoModelForCausalLM.from_pretrained("...")

# Simplified example set up a calibration data loader with the desired calib_size
calib_set = get_dataloader(num_samples=calib_size)

# Prepare the calibration set and define a forward loop
def forward_loop(model):
    for batch in calib_set:
        model(batch)

# PTQ with in-place replacement to quantized modules
model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)
```

> 为了获得更高的 NVFP4 PTQ 准确率，我们建议使用 `mtq.NVFP4_MLP_ONLY_CFG`， `mtq.NVFP4_EXPERTS_ONLY_CFG`， 或者 `mtq.NVFP4_OMLP_ONLY_CFG` 而不是 `mtq.NVFP4_DEFAULT_CFG`。 `NVFP4_MLP_ONLY_CFG` 对 MLP（和 MoE）层应用 NVFP4 量化，而注意力层则不进行量化。 `NVFP4_EXPERTS_ONLY_CFG` 仅量化专家层（`*mlp.experts*` 和 `*block_sparse_moe*`），对于 MoE 模型很有用，其中密集 MLP 和注意力机制保持较高的精度。 `NVFP4_OMLP_ONLY_CFG` 此外，还对 `o_proj` 层。所有方法都能在保持敏感注意力 QKV 投影精度的同时，提供显著的压缩效果。*

### 2. 出口量化模型

模型量化完成后，即可将其导出到检查点，以便轻松部署。
我们提供两种 API 来导出量化模型：

- 统一的拥抱面部检查点，可部署在 TensorRT-LLM（Pytorch 和 C++ 后端）上， [vLLM](https://github.com/vllm-project/vllm) 和 [SGLang](https://github.com/sgl-project/sglang)。
- （旧版）TensorRT-LLM 检查点，一种仅适用于 TensorRT-LLM C++ 后端的格式。

#### 统一拥抱面部检查点

```python
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir,  # The directory where the exported files will be stored.
    )
```

请参考我们的 [framework scripts](#framework-scripts) 以及我们的 [docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html) 更多详情请见下文。

## 支持矩阵

### Hugging Face Supported Models

| Model | fp8 | int8_sq | int4_awq | w4a8_awq<sup>1</sup> | nvfp4<sup>5</sup> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| LLAMA 3.x | ✅ | ❌ | ✅ | ✅<sup>3</sup> | ✅ |
| LLAMA 4 <sup>6</sup> | ✅ | ❌ | ❌ | ❌ | ✅ |
| Mixtral | ✅ | ❌ | ✅<sup>2</sup> | ❌ | ✅ |
| Phi-3,4 | ✅ | ✅ | ✅ | ✅<sup>3</sup> | - |
| Phi-3.5 MOE | ✅ | ❌ | ❌ | ❌ | - |
| Llama-Nemotron Super | ✅ | ❌ | ❌ | ❌ | ✅ |
| Llama-Nemotron Ultra | ✅ | ❌ | ❌ | ❌ | ❌ |
| Gemma 3 | ✅<sup>2</sup> | - | ✅ | - | - |
| QWen 2, 2.5 <sup>4</sup> | ✅ | ✅ | ✅ | ✅ | ✅ |
| QWen3, 3.5 MOE, Next <sup>6</sup> | ✅ | - | - | - | ✅ |
| QwQ | ✅ | - | - | - | ✅ |
| DeepSeek V3, R1, V3.1, V3.2<sup>7</sup> | - | - | - | - | ✅ |
| GLM-4.7<sup>8</sup> | ✅ | - | - | - | ✅ |
| Kimi K2 | - | - | - | - | ✅ |
| MiniMax M2.1 | - | - | - | - | ✅ |
| GPT-OSS<sup>10</sup> | - | - | - | - | ✅ |
| T5 | ✅ | ✅ | ✅ | ✅ | - |
| Whisper<sup>9</sup> | ✅ | ❌ | ❌ | ❌ | - |
| Nemotron-3 | ✅ | ❌ | ❌ | ❌ | ✅ |
| Llava (VLM)<sup>11</sup> | ✅ | ✅<sup>12</sup> | ✅ | ✅ | - |
| Phi-3-vision, Phi-4-multimodal (VLM)<sup>11</sup> | ✅ | ✅<sup>12</sup> | ✅ | ✅ | ✅ |
| Qwen2, 2.5-VL (VLM)<sup>11</sup> | ✅ | ✅<sup>12</sup> | ✅ | ✅ | ✅ |
| Gemma 3 (VLM)<sup>11</sup> | ✅ | - | - | - | - |
| Nemotron VL (VLM)<sup>11,13</sup> | ✅ | - | - | - | ✅ |

> *这只是部分支持的型号。完整列表请查看[此处应插入链接]。 [TensorRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/reference/precision.html#support-matrix)*

> *<sup>1.</sup>w4a8_awq 是一种实验性的量化方案，可能会导致更高的精度损失。*
> *<sup>2.</sup>某些型号仅支持导出量化检查点。*
> *<sup>3.</sup>W4A8_AWQ 仅适用于部分型号，并非所有型号都适用*
> *<sup>4.</sup>对于某些模型，KV缓存量化可能会导致更高的精度损失。*
> *<sup>5.</sup>内部测试仅包含部分常用模型。实际支持的模型列表可能更长。NVFP4 推理需要 Blackwell GPU 和 TensorRT-LLM v0.17 或更高版本*
> *<sup>6.</sup>部分型号目前仅支持导出为 HF 格式。*
> *<sup>7.</sup>[PTQ for DeepSeek](../deepseek/README.md)* \
> *<sup>8.</sup>GLM-4.7 具有 MTP（多标记预测）层，这些层会自动加载并排除在量化之外。*
> *<sup>9.</sup>运行 Whisper 模型时，Transformers 版本必须大于等于 5.0。 [torchcodec](https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-cuda-enabled-torchcodec) 以及其他系统软件包（例如 ffmpeg）。*
> *<sup>10.</sup>GPT-OSS 自带原生 MXFP4 权重；NVFP4 导出是通过闭式函数生成的。 `--cast_mxfp4_to_nvfp4` 演员表（见） [MXFP4 → NVFP4 cast](#mxfp4--nvfp4-cast-for-gpt-oss)).* \
> *<sup>11.</sup>视觉语言模型（VLM）：仅对语言模型进行量化，而视觉编码器保持高精度。通过 `--vlm` 到 shell 脚本（见 [VLM quantization](#vlm-quantization)).* \
> *<sup>12.</sup>对于 VLM， `int8_sq` 仅支持 TensorRT-LLM 检查点导出，与 TensorRT-LLM torch 后端不兼容。*
> *<sup>13.</sup>Nemotron VL 可使用图像-文本对自动校准；参见 [VLM calibration with image-text pairs](#vlm-calibration-with-image-text-pairs-eg-nemotron-vl).*

> *PTQ后的精度损失可能因实际模型和量化方法而异。不同模型的精度损失可能不同，通常基模型较小时精度损失更为显著。如果PTQ后的精度不符合要求，请尝试修改…… [hf_ptq.py](./hf_ptq.py) 并禁用 KV 缓存量化或使用 [QAT](./../llm_qat/README.md) 相反。具体来说，对于 NVFP4 量化，我们建议 `nvfp4_mlp_only`， `nvfp4_experts_only`， 或者 `nvfp4_omlp_only` 通过将量化限制在 MLP/专家层（以及可选的层）来实现更高的精度 `o_proj` 层）同时保持对未量化 QKV 投影的关注。*

> 您还可以使用以下方式创建自己的自定义配置 [this](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#custom-calibration-algorithm) 指导。

> *视觉语言模型（VLM）列于上方的支持矩阵中（标记为*的行）。 `(VLM)`PTQ 的
> VLM由同一方处理。 `hf_ptq.py` 入口点和 shell 脚本作为语言模型——语言模型是
> 在保持视觉编码器高精度的同时进行量化。通过 `--vlm` 到 shell 脚本（见
> [VLM quantization](#vlm-quantization)有关 TensorRT-LLM torch 后端多模态支持的详细内容，
> 请参考 [this doc](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md#multimodal-feature-support-matrix-pytorch-backend).*

## 框架脚本

### 拥抱脸部示例 [Script](./scripts/huggingface_example.sh)

对于 LLM 模型，例如 [Llama-3](https://huggingface.co/meta-llama)：

```bash
# Install model specific pip dependencies if needed

export HF_PATH=<the downloaded LLaMA checkpoint from the Hugging Face hub, or simply the model card>
scripts/huggingface_example.sh --model $HF_PATH --quant <QFORMAT> --tp [1|2|4|8]
```

支持 `QFORMAT` 值： `fp8`， `fp8_pc_pt`， `fp8_pb_wo`， `int8`， `int8_sq`， `int8_wo`， `int4_awq`， `w4a8_awq`， `nvfp4`， `nvfp4_awq`， `nvfp4_mse`， `nvfp4_mlp_only`， `nvfp4_experts_only`， `nvfp4_omlp_only`， `nvfp4_svdquant`， `nvfp4_local_hessian`， `w4a8_nvfp4_fp8`， `w4a8_mxfp4_fp8`， `mxfp8`。

> *默认情况下 `trust_remote_code` 已设置为 false。如果模型校准和评估需要，请使用以下命令将其打开： `--trust_remote_code`.*

> *如果由于张量放置不匹配导致多GPU系统上的Huggingface模型校准失败，请尝试将CUDA_VISIBLE_DEVICES设置为较小的数值。*

> *不建议在GPU内存有限的大型模型上进行FP8校准，但理论上是可以的。 [accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling) 软件包。请调整 device_map 设置。 [`example_utils.py`](./example_utils.py) 如果需要进行模型加载和校准，过程可能会比较慢。*

> *用 Huggingface 模型训练 `modelopt.torch.speculative` 可以像普通 Huggingface 模型一样在 PTQ 中使用。注意：已知 Huggingface 模型在多个 GPU 上加载进行推理时存在一个问题（例如，“预期所有张量都在同一设备上，但发​​现至少有两个设备……”）。如果在 PTQ 中使用推测性解码模型时遇到此错误，请尝试减少使用的 GPU 数量。*

> *默认情况下，Huggingface 分词器校准时使用左侧 padding_side，因为这样通常能降低准确率损失。导出的分词器文件会恢复默认的 padding_side。*

> *如果在模型量化过程中，即使内存充足，仍然出现 GPU 内存溢出 (OOM) 错误，设置 `--use_seq_device_map` 标志可以解决问题。这将强制执行顺序设备映射，把模型分布到多个 GPU 上，并利用每个 GPU 高达 80% 的内存。*

> 您可以添加 `--low_memory_mode` 此命令用于降低 PTQ 进程的内存需求。在此模式下，脚本会在校准前将模型权重压缩至低精度。此模式仅支持 FP8 和 NVFP4 格式，且仅支持最大校准精度。*

#### 基于配方的量化

而不是指定 `--qformat` 和 `--kv_cache_qformat` 此外，您还可以使用**配方**——一个声明式的 YAML 文件，其中包含完整的量化配置。配方通过以下方式加载： `--recipe` 并优先于 `--qformat`。

```bash
# Using a built-in recipe name (without .yaml suffix)
python hf_ptq.py \
  --pyt_ckpt_path <huggingface_model_card> \
  --recipe general/ptq/nvfp4_default-kv_fp8_cast \
  --export_path <quantized_ckpt_path>

# Using a custom recipe YAML file path
python hf_ptq.py \
  --pyt_ckpt_path <huggingface_model_card> \
  --recipe /path/to/my_ptq.yaml \
  --export_path <quantized_ckpt_path>
```

内置食谱位于 `modelopt_recipes/general/ptq/` 对于与模型无关的配方和 `modelopt_recipes/huggingface/<model_type>/ptq/` 针对特定拥抱脸的食谱 `model_type` （看 [`modelopt_recipes/huggingface/README.md`](../../modelopt_recipes/huggingface/README.md)您还可以提供自定义 YAML 配方文件或目录的路径。请参阅 [recipe documentation](https://nvidia.github.io/Model-Optimizer) 有关 YAML 架构和可用配方的详细信息。

> *什么时候 `--recipe` 已指定， `--qformat` 被忽略。KV 缓存处理取决于配方类型：**PTQ** 配方会将 KV 缓存嵌入到其配置中并忽略。 `--kv_cache_qformat`；**自动量化**配方会回退到 `--kv_cache_qformat` 除非它明确地设置了 `kv_cache` 场地。*

#### KV缓存量化

键值缓存量化通过量化键值缓存来降低推理期间的内存使用量。这是通过以下方式控制的： `--kv_cache_qformat` 标志（默认值：） `fp8_cast`）。

```bash
# FP8 KV cache with cast (no calibration needed, fast)
python hf_ptq.py --pyt_ckpt_path <model> --qformat fp8 --kv_cache_qformat fp8_cast --export_path <path>

# NVFP4 KV cache with data-driven calibration
python hf_ptq.py --pyt_ckpt_path <model> --qformat nvfp4 --kv_cache_qformat nvfp4 --export_path <path>

# Disable KV cache quantization
python hf_ptq.py --pyt_ckpt_path <model> --qformat fp8 --kv_cache_qformat none --export_path <path>
```

通过 shell 脚本：

```bash
scripts/huggingface_example.sh --model $HF_PATH --quant fp8 --kv_cache_quant nvfp4
```

可用的KV缓存格式：

| 格式 | 描述 |
| :---: | :--- |
| `fp8_cast` （默认）| FP8 KV 缓存，无数据驱动校准（最大设置为 FP8 范围）|
| `fp8` | 采用数据驱动校准的FP8 KV缓存 |
| `fp8_affine` | 采用仿射量化的FP8键值缓存 |
| `nvfp4_cast` | NVFP4 KV 缓存无需数据驱动校准 |
| `nvfp4` | 采用数据驱动校准的 NVFP4 KV 缓存 |
| `nvfp4_affine` | 采用仿射量化的 NVFP4 键值缓存 |
| `nvfp4_rotate` | 带旋转功能的 NVFP4 KV 缓存 |
| `none` | 禁用 KV 缓存量化 |

> *格式以...结尾 `_cast` (fp8_cast、nvfp4_cast) 速度很快——它们无需数据驱动校准即可将 amax 设置为格式的完整范围。其他格式则使用数据驱动校准，以获得可能更高的精度。*

#### MXFP4 → NVFP4 转换（用于 GPT-OSS）

GPT-OSS 检查点（`openai/gpt-oss-20b`， `openai/gpt-oss-120b`) 附带原生 MXFP4 权重 (`*_blocks` + `*_scales` 在检查站， `quantization_config.quant_method == "mxfp4"`传递 `--cast_mxfp4_to_nvfp4` 讲述 `hf_ptq.py` 读取源 MXFP4 比例并生成闭式、位精确的 NVFP4 权重导出——无需对权重进行 GEMM 级别的重新校准。

```bash
python hf_ptq.py \
  --pyt_ckpt_path openai/gpt-oss-20b \
  --qformat nvfp4_mlp_only \
  --cast_mxfp4_to_nvfp4 \
  --export_path <quantized_ckpt_path>
```

每个 NVFP4 模块的铸造销钉 `scale_2 = 2^(k_max - 8)` 和 `_amax = 6 * 2^k_j`两者均源自 MXFP4 E8M0 源尺度。对于其 `k_j` 落在 E4M3 的可表示窗口中（`k_max - k_j ≤ 17`),NVFP4 减量与 MXFP4 减量逐位匹配；超出范围的块回退到数据导出的每个块 amax。

> *`--cast_mxfp4_to_nvfp4` 需要 NVFP4 系列 `--qformat` （例如。 `nvfp4_mlp_only`， `nvfp4_experts_only`， `nvfp4`）并且与 AutoQuantize 配方（多格式搜索）不兼容。*

#### Deepseek R1

[PTQ for DeepSeek](../deepseek/README.md) 展示了如何使用 FP4 对 DeepSeek 模型进行量化，并导出到 TensorRT-LLM。

#### VLM量化

视觉语言模型通过同一脚本进行量化。添加 `--vlm` 所以脚本运行了
使用 TensorRT-LLM 多模态快速入门作为部署冒烟测试，而不是仅使用文本测试：

```bash
scripts/huggingface_example.sh --model <Hugging Face model card or checkpoint> --quant fp8 --vlm
```

支持 `--quant` VLM 的值为 `fp8`， `nvfp4`， `int8_sq`， `int4_awq`， 和 `w4a8_awq` （看
这 `(VLM)` 行 [Support Matrix](#hugging-face-supported-models)）。

> *这巩固了前者 `examples/vlm_ptq` 例如，现在转发到这里。*

#### 使用图像-文本对进行 VLM 校准（例如，Nemotron VL）

对于视觉语言模型而言，使用图像-文本对而不是纯文本数据可能会提高校准质量，尤其是在视觉理解任务中：

```bash
python hf_ptq.py \
  --pyt_ckpt_path <huggingface_model_card> \
  --qformat nvfp4 \
  --export_path <quantized_ckpt_path> \
  --trust_remote_code \
  --calib_with_images \
  --calib_size 512
```

shell 脚本也暴露了相同的标志：

```bash
scripts/huggingface_example.sh --model <model> --quant nvfp4 --vlm --calib_with_images --trust_remote_code
```

> 注意：当 `--calib_with_images` 已设置， `--calib_size` 必须是单个值，校准数据集为 nvidia/nemotron_vlm_dataset_v2。
此功能目前处于测试阶段，已在以下情况下进行过测试： `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`。

### Megatron-Bridge 示例脚本

请参考 [examples/megatron_bridge/README.md](../megatron_bridge/README.md) 例如，PTQ / QAD 与 Megatron-Bridge 的脚本通常比 Hugging Face 脚本性能更高。

### Megatron-LM 示例脚本

Megatron-LM 框架的 PTQ 和 TensorRT-LLM 部署示例维护在 Megatron-LM GitHub 代码库中。请参阅这些示例。 [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt)。

## 自动量化

[AutoQuantize (`mtq.auto_quantize`)](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize) 是一种 PTQ 算法，它通过搜索每一层的最佳量化格式来量化模型，同时满足用户指定的性能约束。 `AutoQuantize` 简化了模型精度和性能之间的权衡。

`AutoQuantize` 使用有效位目标（`effective_bits`作为性能约束（对于两者
仅权重量化和权重与激活量化）—量化模型的有效比特数。

您可以指定一个 `effective_bits` 目标值例如 5.4，用于混合精度量化 `NVFP4_DEFAULT_CFG` & `FP8_DEFAULT_CFG`。
`AutoQuantize` 将自动量化高敏感层 `FP8_DEFAULT_CFG` 同时保留不太敏感的层 `NVFP4_DEFAULT_CFG` （甚至可以对任何极其敏感的层跳过量化）
最终的混合精度量化模型具有 5.4 位有效量化比特。该模型比使用原始方法量化的模型具有更高的精度。 `NVFP4_DEFAULT_CFG` 由于采用了更激进的配置，因此更具侵略性 `NVFP4_DEFAULT_CFG` 对于高度敏感的层，未采用量化方法。

以下是一个使用示例： `AutoQuantize` 算法（请参见） [auto_quantize](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize) （更多详情请参见 API）

```python

    import modelopt.torch.quantization as mtq

    # Define the model & calibration dataloader
    model = ...
    calib_dataloader = ...

    # Define forward_step function.
    # forward_step should take the model and data as input and return the output
    def forward_step(model, data):
        output =  model(data)
        return output

    # Define loss function which takes the model output and data as input and returns the loss
    def loss_func(output, data):
        loss = ...
        return loss


    # Perform AutoQuantize
    model, search_state_dict = mtq.auto_quantize(
        model,
        constraints = {"effective_bits": 5.4},
        # supported quantization formats are listed in `modelopt.torch.quantization.config.choices`
        quantization_formats = ["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"]
        data_loader = calib_dataloader,
        forward_step=forward_step,
        loss_func=loss_func,
        ...
        )
```

### 拥抱脸模型的自动量化

`AutoQuantize` 可以对 Huggingface LLM 模型执行类似操作。 [Qwen](https://huggingface.co/Qwen/Qwen3-8B) / [Nemotron](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) 如下所示：

`AutoQuantize` 由传递的**AutoQuantize配方**驱动 `--recipe`食谱定义了
候选人格式 `effective_bits` 目标、成本模型、评分方法、搜索禁用层和
成本排除层——参见 [`AutoQuantizeConfig`](../../modelopt/recipe/config.py)已寄出的食谱存在于
[`modelopt_recipes/general/auto_quantize/`](../../modelopt_recipes/general/auto_quantize)特定型号
配方（包含特定于架构的禁用层，例如 VL 视觉塔）位于
`modelopt_recipes/huggingface/<model>/auto_quantize/`。

> *迁移：优先选择自动量化 `--recipe`。 这 `--auto_quantize_bits`， `--auto_quantize_method`，
> `--auto_quantize_score_size`， `--auto_quantize_cost_model`， 和 `--auto_quantize_active_moe_expert_ratio`
> CLI 标志已被**弃用但仍有效**——它们会被转换为 `AutoQuantizeConfig` 即兴
> （带着一个 `DeprecationWarning`）并将于未来的版本中移除。它们对应于配方字段：
> `--auto_quantize_bits` → `constraints.effective_bits`， `--auto_quantize_method` → `auto_quantize_method`，
> `--auto_quantize_score_size` → `score_size`， `--auto_quantize_cost_model` → `constraints.cost_model`，
> `--auto_quantize_active_moe_expert_ratio` → `constraints.cost.active_moe_expert_ratio`以及
> `--qformat fp8,nvfp4` 候选人名单 → `candidate_formats`转换后，共享基
> `disabled_layers` 和 `cost_excluded_layers` 图案会自动添加。 `--auto_quantize_checkpoint`
> 未更改。从已发布的配方开始。 `modelopt_recipes/general/auto_quantize/`.*

[Script](./scripts/huggingface_example.sh)

```bash
export HF_PATH=<the downloaded checkpoint from the Hugging Face hub, or simply the model card>
# --recipe selects an AutoQuantize recipe; the recipe defines the candidate formats and the
# effective-bits target (here NVFP4 + FP8 at 5.4 effective bits).
scripts/huggingface_example.sh --model $HF_PATH --recipe general/auto_quantize/nvfp4_fp8_at_5p4bits --calib_batch_size 4
```

该方案使用更激进的格式（例如 NVFP4）对精度要求较低的层进行量化，
对更敏感的成分保持较高的精度（或未量化），因此该模型符合配方要求。
`effective_bits` 目标。要创建自己的配方，请复制已发布的配方并进行调整 `candidate_formats`，
`constraints.effective_bits`， `auto_quantize_method` （`gradient` / `kl_div`）， `score_size`，
`disabled_layers` （已排除在搜索范围之外），以及 `cost_excluded_layers` （未计入预算）
会计（例如 VL 视觉塔）。配方可以拼接共享基础 `disabled_layers` 通过设置
`$import` （看 `modelopt_recipes/configs/auto_quantize/units/base_disabled_layers`）。

bf16（无量化）始终是每层的一个隐式选择，因此 `candidate_formats` 只需列出
量化选项——单一格式（例如） `[fp8]`) 给出了一个 `{fp8, bf16}` 逐层搜索。

对于不支持反向传播的模型（例如 Llama-4），请使用 `kl_div` 评分方法——参见已发货
`general/auto_quantize/nvfp4_fp8_kl_div_at_5p4bits` 食谱。

KV 缓存作为统一的后置步骤应用，而非逐层搜索的一部分。自动量化配方
回退到 `--kv_cache_qformat` （默认 `fp8_cast`除非它明确设置了 `kv_cache` 场地。

唯一的运行时标志是 `--auto_quantize_checkpoint` — 保存/恢复搜索状态以继续搜索
中断搜索（跳过重新评分）：

```bash
scripts/huggingface_example.sh --model $HF_PATH --recipe general/auto_quantize/nvfp4_fp8_at_5p4bits \
  --auto_quantize_checkpoint /path/to/auto_quantize.pth --calib_batch_size 4
```

上面的示例脚本还有一个额外的标志。 `--tasks`其中，脚本中实际运行的任务可以自定义。允许的任务有： `quant,mmlu,lm_eval,livecodebench,simple_eval` 脚本中指定的 [parser](./scripts/parser.sh)可以使用逗号分隔的任务列表来指定任务组合。某些任务（例如 mmlu）可能需要很长时间才能运行。要运行 lm_eval 任务，请同时指定 `--lm_eval_tasks` 带有逗号分隔的 lm_eval 任务的标志 [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)。

> *如果运行脚本时出现GPU内存不足错误，请尝试编辑脚本并减小最大批处理大小以节省GPU内存。*

> *注意：AutoQuantize 需要模型进行反向传播。不支持反向传播的模型（例如 Llama-4）在使用 AutoQuantize 时将无法正常工作。 `gradient` 方法。 `kl_div` 该方法不需要反向传播。*

## 真实量化

处理大型语言模型时，内存限制可能是一个重大挑战。ModelOpt 提供了一种工作流程，用于在多个 GPU 上使用压缩权重初始化 HF 模型，从而显著降低内存使用量。 `--low_memory_mode` 有关更多详细信息，请参阅 hf_ptq.py 中的选项。

```python
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins import init_quantized_weights
from transformers import AutoModelForCausalLM, AutoConfig

# Step 1: Initialize the model with compressed weights
with init_quantized_weights(mtq.NVFP4_DEFAULT_CFG):
    model = AutoModelForCausalLM.from_pretrained(ckpt_path)

# Step 2: Calibrate the model
mtq.calibrate(model, algorithm="max", forward_loop=calibrate_loop)
```

## 基于FSDP2的多节点后训练量化

ModelOpt 能够使用各种量化格式，在多个 GPU 节点上对 LLM 进行量化。它利用 HuggingFace 的 Accelerate 库和 FSDP2 进行分布式模型分片和校准。

### 用法

对于跨多个节点的分布式执行，请使用 `accelerate` 库。模板配置文件（`fsdp2.yaml`）提供，并可根据用户特定需求进行定制。

在每个节点上运行以下命令：

```bash
accelerate launch --config_file fsdp2.yaml \
    --num_machines=<num_nodes> \
    --machine_rank=<current_node_rank> \
    --main_process_ip=<node0_ip_addr> \
    --main_process_port=<port> \
    --fsdp_transformer_layer_cls_to_wrap=<decoder_layer_name>
     multinode_ptq.py \
    --pyt_ckpt_path <path_to_model> \
    --qformat <fp8/nvfp4/nvfp4_mlp_only/nvfp4_experts_only/nvfp4_omlp_only/nvfp4_awq/int8> \
    --kv_cache_qformat <fp8/nvfp4/nvfp4_affine/none> \
    --batch_size <calib_batch_size> \
    --calib_size <num_calib_samples> \
    --dataset <dataset> \
    --export_path <export_path> \
    --trust_remote_code
```

导出的检查点可以使用 TensorRT-LLM/ vLLM/ SGLang 进行部署。更多详情请参阅…… [deployment section](#deployment) 本文档。

> *性能提示：FSDP2 专为训练工作负载而设计，可能会导致校准和导出时间延长。为了加快校准速度，请根据可用 GPU 内存最大化批处理大小，并选择合适的 GPU 数量以避免不必要的通信。*

## 评估准确性

### TensorRT-LLM 验证

文中提供了一系列准确性验证基准。 [llm_eval](../llm_eval/README.md) 目录。目前，本示例通过指定目录来支持 MMLU。 `--tasks` 标记正在运行上述脚本。

这 `benchmark_suite.py` 该脚本用作快速性能基准测试。详情请参阅…… [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/)

这个例子也涵盖了 [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness)MMLU 和人工评估准确率基准，其详细信息可在此处找到。 [here](../llm_eval/README.md)支持的 lm_eval 评估任务如下所示。 [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)

## 导出检查点

模型优化器支持提供两种导出量化模型的途径：

- 统一的拥抱面部检查点，可部署在 TensorRT-LLM（Pytorch 和 C++ 后端）上， [vLLM](https://github.com/vllm-project/vllm) 和 [SGLang](https://github.com/sgl-project/sglang)。
- （旧版）TensorRT-LLM 检查点，一种仅适用于 TensorRT-LLM C++ 后端的格式。

统一检查站<sup>1</sup> 格式设计体现了两个关键特征：1. 层结构和张量名称与原始 Hugging Face 检查点保持一致；2. 同一个检查点无需修改即可部署到多个推理框架中。可以使用以下命令导出统一的检查点：

> *<sup>1.</sup>统一检查点导出目前不支持稀疏性。推测性解码仅在统一检查点导出中受支持。对于旧版部署，导出的统一检查点需要使用 TensorRT-LLM 检查点转换器（例如， [this](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/eagle/convert_checkpoint.py)）用于转换和构建 TensorRT 引擎以进行部署。或者，调用 TensorRT-LLM LLM-API 来部署统一检查点，例如，请查看示例。 [here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/README.md).*

### API

```python
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir,  # The directory where the exported files will be stored.
    )
```

### 量化和出口

```bash
python hf_ptq.py --pyt_ckpt_path <huggingface_model_card> --qformat fp8 --export_path <quantized_ckpt_path> --trust_remote_code
```

> *要导出用于 vLLM 服务的伪量化模型（例如，用于研究或 real-quant 尚不支持的内核），请使用 `--vllm_fakequant_export` 旗帜。见 [vllm_serve/README.md](../vllm_serve/README.md) 详情请见*

### 拥抱脸框架 [Script](./scripts/huggingface_example.sh)

或者，框架脚本 `huggingface_example.sh` 还支持量化和导出：

```bash
scripts/huggingface_example.sh --model <huggingface_model_card> --quant fp8
```

### 部署

______________________________________________________________________

#### TRT-LLM

```python
from tensorrt_llm import LLM

llm_fp8 = LLM(model="<the exported model path>")
print(llm_fp8.generate(["What's the age of the earth? "]))
```

#### vLLM

```python
from vllm import LLM

llm_fp8 = LLM(model="<the exported model path>", quantization="modelopt")
print(llm_fp8.generate(["What's the age of the earth? "]))
```

#### SGLang

```python
import sglang as sgl

llm_fp8 = sgl.Engine(model_path="<the exported model path>", quantization="modelopt")
print(llm_fp8.generate(["What's the age of the earth? "]))
```

### 统一高频检查点部署模型支持矩阵

| 模型 | 量化格式 | TRT-LLM | vLLM | SGLang |
| :---: | :---: | :---: | :---: | :---: |
| LLAMA 3.x | FP8 | ✅ | ✅ | ✅ |
| LLAMA 3.x | FP4 | ✅ | ✅ | ✅ |
|骆驼 4 | FP8 | ✅ | - | ✅ |
|骆驼 4 | FP4 | ✅ | - | - |
| DS-R1 | FP8 | ✅ | ✅ | ✅ |
| DS-R1 | FP4 | ✅ | ✅ | ✅ |
| DS-V3 | FP8 | ✅ | ✅ | ✅ |
| DS-V3 | FP4 | ✅ | ✅ | ✅ |
| QWen3 | FP8 | ✅ | ✅ | ✅ |
| QWen3 | FP4 | ✅ | ✅ | - |
| QWen3 教育部 | FP8 | ✅ | ✅ | ✅ |
| QWen3 MoE | FP4 | ✅ | - | - |
| QWen3.5 MoE | FP4 | - | - | ✅ |
| QWen2.5 | FP8 | ✅ | ✅ | ✅ |
| QWen2.5 | FP4 | ✅ | ✅ | - |
| QwQ-32B | FP8 | ✅ | ✅ | ✅ |
| QwQ-32B | FP4 | ✅ | ✅ | - |
|混合 8x7B | FP8 | ✅ | ✅ | ✅ |
|混合 8x7B | FP4 | ✅ | - | - |

### （传统）TensorRT-LLM 检查点

用户可以指定推理时间 TP 和 PP 大小，导出 API 将调整权重以适应目标 GPU。

```python
from modelopt.torch.export import export_tensorrt_llm_checkpoint

with torch.inference_mode():
    export_tensorrt_llm_checkpoint(
        model,  # The quantized model.
        decoder_type,  # The type of the model, e.g gpt, gptj, or llama.
        dtype,  # The exported weights data type.
        export_dir,  # The directory where the exported files will be stored.
        inference_tensor_parallel,  # The number of GPUs used in the inference time tensor parallel.
        inference_pipeline_parallel,  # The number of GPUs used in the inference time pipeline parallel.
        use_nfs_workspace,  # If exporting in a multi-node setup, please specify a shared directory like NFS for cross-node communication.
    )
```

### 构建 TensorRT-LLM 引擎

导出 TensorRT-LLM 检查点后，您可以使用 `trtllm-build` 使用构建命令从导出的检查点构建引擎。请检查。 [TensorRT-LLM Build API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/architecture/workflow.md#build-apis) 供参考的文档。

## 预先量化的检查点

- 准备部署的检查点[[🤗 Hugging Face - Nvidia Model Optimizer Collection](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)\]
- 可部署于 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)， [vLLM](https://github.com/vllm-project/vllm) 和 [SGLang](https://github.com/sgl-project/sglang)
- 更多车型即将推出！

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

1. 这 [INT8 SmoothQuant](https://arxiv.org/abs/2211.10438)由 MIT HAN 实验室和 NVIDIA 开发的，旨在减少 LLM 推理的 GPU 内存占用和推理延迟。

1. 这 [INT4 AWQ](https://arxiv.org/abs/2306.00978) INT4 AWQ 是一种仅使用 INT4 权重的量化和校准方法。INT4 AWQ 在小批量推理中尤为有效，因为此时推理延迟主要取决于权重加载时间而非计算时间本身。对于小批量推理，INT4 AWQ 的延迟可能低于 FP8/INT8，且精度损失小于 INT8。

1. W4A8 AWQ 是 INT4 AWQ 量化的扩展，它也使用 FP8 进行激活，以获得更快的速度和更高的加速度。

1. 这 [NVFP4](https://blogs.nvidia.com/blog/generative-ai-studio-ces-geforce-rtx-50-series/) NVFP4 是 NVIDIA Blackwell GPU 支持的一种新型 FP4 格式，与其他 4 位格式相比，它展现出良好的精度。NVFP4 可应用于模型权重和激活值，与 Blackwell 上的 FP8 数据格式相比，它有望显著提高数学吞吐量，并减少内存占用和内存带宽使用。为了获得更高的 NVFP4 PTQ 精度，我们建议 `nvfp4_mlp_only`， `nvfp4_experts_only`， 或者 `nvfp4_omlp_only`。 `nvfp4_mlp_only` 将 NVFP4 量化限制在 MLP（和 MoE）层，使注意力层保持更高的精度。 `nvfp4_experts_only` 仅量化专家层（`*mlp.experts*` 和 `*block_sparse_moe*`），非常适合 MoE 模型。 `nvfp4_omlp_only` 通过量化，扩展了 MLP 的功能 `o_proj` 该层提供了介于完全 NVFP4 量化和仅 MLP 量化之间的中间方案。
