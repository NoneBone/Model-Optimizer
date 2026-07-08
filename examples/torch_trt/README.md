# Torch-TensorRT 量化

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
[Torch-TensorRT](https://docs.pytorch.org/TensorRT/) 将 PyTorch 模型编译成优化的 TensorRT 引擎，无需单独导出或运行时环境。此示例使用 NVIDIA 模型优化器量化 PyTorch/HuggingFace 模型，然后使用 Torch-TensorRT 在框架内编译量化后的图以进行部署。

量化是一种有效的模型优化技术，可以压缩模型。模型优化器会将 Q/DQ 节点插入到 PyTorch 的 eager 图中； `torch_tensorrt.compile(ir="dynamo")` 然后按照以下方式将这些 Q/DQ 节点转换为原生 TensorRT FP8 精度层： [Torch-TensorRT quantization guide](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html)。

本节重点介绍框架内的 Torch-TensorRT 路径：PyTorch 前端（`mtq.quantize`）为 Dynamo 编译的 TensorRT 引擎提供输入，并在 HuggingFace ViT 图像分类器上进行了端到端演示。如果您需要可移植的 ONNX → TensorRT 组件，或者您从 ONNX 模型开始，请参阅同系列文档。 [`torch_onnx`](../torch_onnx/) 和 [`onnx_ptq`](../onnx_ptq/) 例如（在……中进行比较） [Support Matrix](#support-matrix)）。

<div align="center">

| **章节** | **描述** | **链接** | **文档** |
| :------------: | :------------: | :------------: | :------------: |
| 先决条件 | 所需软件包和安装 | \[[Link](#pre-requisites)\] | |
| 入门指南 | 几行代码即可量化并编译 ViT | \[[Link](#getting-started)\] | \[[docs](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html)\] |
| 支持矩阵 | 此路径与 ONNX 示例的比较 | \[[Link](#support-matrix)\] | |
| ViT 配方 | 示例中附带的 FP8 配方 | \[[Link](#vit-recipes)\] | |
| 用法 | 量化和准确度脚本的 CLI 标志 | \[[Link](#usage)\] | |
| 评估准确率 | 测量 ImageNet top-1 / top-5 准确率 | \[[Link](#evaluate-accuracy)\] | |
| 自定义配方 | 插入您自己的配方/模型 | \[[Link](#custom-recipes)\] | |
| 资源 | 路线图、文档、基准测试和支持 | \[[Link](#resources)\] | |

</div>

## 先决条件

### Docker

请使用 TensorRT docker 镜像（例如， `nvcr.io/nvidia/tensorrt:26.02-py3`）或访问我们的网站 [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) 了解更多信息。

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/tensorrt:26.02-py3 bash
```

此外，请按照以下安装步骤升级到最新版本的模型优化器并安装示例特定的依赖项。

### 本地安装

```bash
pip install -U "nvidia-modelopt[hf]"
pip install -r requirements.txt
```

### 硬件要求

Torch-TensorRT 生成的低精度内核需要支持目标格式的 GPU：

<div align="center">

| 配方 | 最低显卡要求 |
| :---: | :---: |
| `fp8` | Ada / Hopper — 计算能力 8.9+ |

</div>

> [！笔记]
> 老款GPU仍然允许 `mtq.quantize` 成功——它在 PyTorch 中生成伪量化节点——但是 `torch_tensorrt.compile` 对于不支持的格式，将找不到真正的低精度内核。

## 入门

对 HuggingFace ViT 进行量化，然后使用 Torch-TensorRT 将 Q/DQ 图编译成单个 `torch.nn.Module` 你从 PyTorch 调用：

```python
import torch
import torch_tensorrt

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.utils import export_torch_mode

# 1. Quantize the eager PyTorch model with a Model Optimizer PTQ recipe.
recipe = load_recipe("huggingface/vit/ptq/fp8")
mtq.quantize(model, recipe.quantize.model_dump(), forward_loop=calibrate)

# 2. Compile the quantized (Q/DQ) graph with Torch-TensorRT.
#    export_torch_mode() makes Model Optimizer emit Q/DQ in the TRT-friendly form,
#    and min_block_size=1 lets single-node Q/DQ + matmul subgraphs become TRT
#    precision layers (per the Torch-TensorRT quantization guide).
with export_torch_mode():
    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        min_block_size=1,
        truncate_double=True,
        inputs=[torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(128, 3, 224, 224),
            max_shape=(1024, 3, 224, 224),
            dtype=torch.float16,
        )],
    )

logits = trt_model(pixel_values)  # call it like any nn.Module
```

可运行脚本 [`torch_tensorrt_ptq.py`](./torch_tensorrt_ptq.py) 它完整​​地封装了整个流程。它：

1. 加载 HuggingFace ViT 分类器（默认） `google/vit-large-patch16-224`）。
1. 用……构建一个微型校准加载器 `zh-plus/tiny-imagenet` （避免使用门控） `ILSVRC/imagenet-1k` 仓库，因此示例在未经身份验证的情况下运行）。
1. 跑 `mtq.quantize` 其中一道菜谱如下 [`modelopt_recipes/`](../../modelopt_recipes/) （看 [ViT Recipes](#vit-recipes)）。
1. 将量化后的模型优化器状态（FP16权重+Q/DQ元数据）保存到 `<save_dir>/vit_modelopt_state.pt` 无需重新校准即可重复使用（见 [Custom Recipes](#custom-recipes)）。
1. 使用以下方式编译量化模型 `torch_tensorrt.compile` 并验证编译模型 argmax 与样本输入上的伪量化 argmax 是否匹配。

```bash
# Default model is google/vit-large-patch16-224, default recipe is the ViT FP8 recipe.
python torch_tensorrt_ptq.py --calib_samples 1024 --batch_size 128

# Quantize but don't TRT-compile (handy on a non-TRT host).
python torch_tensorrt_ptq.py --skip_trt
```

> [！笔记]
> 两个都 `torch_tensorrt_ptq.py` 以及准确性脚本（[`torch_tensorrt_accuracy.py`](./torch_tensorrt_accuracy.py)）运行模型 `float16`。

## 支持矩阵

这三个例子都到达了同一个目的地——一个低精度的 TensorRT 引擎——但它们在管道中的量化点不同，并且会生成不同的工件，因此它们适用于不同的部署堆栈：

<div align="center">

| | Torch-TensorRT（本例） | [`torch_onnx`](../torch_onnx/) | [`onnx_ptq`](../onnx_ptq/) |
| :---: | :---: | :---: | :---: |
| 起点 | PyTorch / HF 模型 | PyTorch / timm 模型 | 已导出的 ONNX 模型 |
| 量化 | 渴望的 PyTorch 图（`mtq.quantize`) | 渴望的 PyTorch 图（`mtq.quantize`) | 直接使用 ONNX 图（ONNX PTQ） |
| 导出步骤 | 无 — FX/Dynamo 图表保持处理中 | `torch.onnx.export` Q/DQ 图，经 TRT 后处理 | 无 — Q/DQ 直接插入 ONNX 图 |
| 中间制品 | 无 | Q/DQ ONNX 文件 | Q/DQ ONNX 文件 |
| 编译器 + 运行时 | `torch_tensorrt.compile(ir="dynamo")` → a `torch.nn.Module` 您从 PyTorch 调用 | TensorRT 从 ONNX 构建独立引擎 | TensorRT 从 ONNX 构建独立引擎 |
| 最佳应用场景 | PyTorch 原生服务；需要即插即用的编译模块 | 在 PyTorch 中进行量化，但通过可移植的 ONNX → TRT 引擎进行部署 | 只有 ONNX 模型，从不使用 PyTorch |

</div>

这个例子和 [`torch_onnx`](../torch_onnx/) 共享同一个 PyTorch 前端（`mtq.quantize`因此，数值计算结果完全相同——它们的区别仅在于后端：前者将图保存在进程中并将其交给 Torch-TensorRT，而后者则不然。 `torch_onnx` 导出可移植的 ONNX 工件，供独立的 TensorRT 运行时使用。 [`onnx_ptq`](../onnx_ptq/) 它直接量化 ONNX 图，适用于从 ONNX 模型而非 PyTorch 模型开始的情况。如果您的服务栈是 PyTorch 原生的，并且您希望避免导出 ONNX 的步骤，请选择此示例。

## ViT食谱

这是 CLI 默认选择的配方 `--model_id` 指向高频ViT分类器。它针对高频ViT模块布局进行了调整，并由共享组件构成。 `$import` 构建模块 [`modelopt_recipes/configs/`](../../modelopt_recipes/configs/) （`ptq/units/{w8a8_fp8_fp8,attention_qkv_fp8}`而不是逐字逐句地解释。 `quant_cfg` 入口。

<div align="center">

| `--recipe` 值 | 校准 | 它量化的是什么 |
| :---: | :---: | :--- |
| `huggingface/vit/ptq/fp8` （默认）| `max` | 每个权重上的每个张量 FP8 (E4M3) + 输入量化器匹配 `*weight_quantizer` / `*input_quantizer` globs — 编码器 Linears，补丁嵌入 `nn.Conv2d` 投影，以及 `classifier` 头部——加上 FP8 的注意力 Q/K/V BMM 和 softmax。所有输出量化器均已禁用。

</div>

## 用法

### `torch_tensorrt_ptq.py`

[Script](./torch_tensorrt_ptq.py) — 量化和（可选地）Torch-TensorRT 编译 ViT。

<div align="center">

| 标志 | 默认值 | 描述 |
| :---: | :---: | :--- |
| `--model_id` | `google/vit-large-patch16-224` | 要量化的 ViT 分类器的 HuggingFace 模型 ID。 |
| `--recipe` | `huggingface/vit/ptq/fp8` | 食谱路径（相对于 `modelopt_recipes/` 或者一个完全 YAML 文件）。
| `--calib_samples` | `1024` 用于校准的 TinyImageNet 样本数量。
| `--batch_size` | `128` | 校准/TRT编译的批次大小。 |
| `--save_dir` | `./modelopt_quantized` | 量化模型优化器状态字典（FP16 权重 + Q/DQ 元数据）始终保存到以下目录： `vit_modelopt_state.pt` —无需重新校准即可在多次运行中重复使用。
| `--skip_trt` | 关闭 | 量化 + 仅运行伪量化模型；跳过 `torch_tensorrt.compile`适用于未安装 Torch-TensorRT 的环境。
| `--layer_info_path` | 取消设置 | 如果设置，则写入已编译的 TRT 引擎的逐层信息（`get_layer_info()`）到此文件。 |

</div>

```bash
# Custom model + custom recipe, saving the quantized state elsewhere.
python torch_tensorrt_ptq.py \
    --model_id <huggingface/model-id> \
    --recipe <recipe-path-relative-to-modelopt_recipes-or-absolute-yaml> \
    --save_dir ./my_quantized

# Dump the compiled engine's per-layer info to inspect FP8 fusion.
python torch_tensorrt_ptq.py --layer_info_path ./vit_fp8_layers.txt
```

### `torch_tensorrt_accuracy.py`

[Script](./torch_tensorrt_accuracy.py) — 对 ImageNet 进行量化、编译和评分（参见 [Evaluate Accuracy](#evaluate-accuracy)）。

<div align="center">

| 标志 | 默认值 | 描述 |
| :---: | :---: | :--- |
| `--model_id` | `google/vit-large-patch16-224` | 用于量化和评分的 ViT 分类器 HuggingFace 模型 ID。 |
| `--recipe` | `huggingface/vit/ptq/fp8` | 食谱路径（相对于 `modelopt_recipes/` 或者一个完全 YAML 文件）。
| `--calib_samples` | `1024` 用于校准的 TinyImageNet 样本数量。
| `--batch_size` | `128` | 校准/编译/评估批处理大小。Torch-TRT 引擎是动态的（`min=1`， `opt=max(--batch_size, 2)`， `max=1024`）并处理任何批次，包括末尾的部分批次。
| `--eval_data_size` | 总计 50k | 待评分的 ImageNet 验证图像数量。 |
| `--imagenet_path` | `ILSVRC/imagenet-1k` | HF 数据集卡或指向 ImageNet 验证集（已设置门控）的本地路径。 |
| `--baseline` | 关闭 | 同时对未量化模型进行评分作为参考。它与量化模型一样，都是使用 Torch-TensorRT 编译的（或者在 eager 模式下运行）。 `--skip_trt`因此，这样的比较是公平的。
| `--skip_trt` | 关闭 | 对 fake-quant（模型优化器）模型进行评分；跳过 `torch_tensorrt.compile`适用于未安装 Torch-TensorRT 的环境。
| `--results_path` | 取消设置 | 如果已设置，则将准确率结果写入此 CSV 路径。 |

</div>

## 评估准确性

[`torch_tensorrt_accuracy.py`](./torch_tensorrt_accuracy.py) 重用上述量化→编译流程，并通过以下方式报告 ImageNet-1k top-1 / top-5 准确率： `onnx_ptq` 例子 `evaluate()` 马具 （[`examples/onnx_ptq/evaluation.py`](../onnx_ptq/evaluation.py)）：

```bash
python torch_tensorrt_accuracy.py \
    --recipe huggingface/vit/ptq/fp8 \
    --batch_size 128 \
    --baseline \
    --eval_data_size 5000 \
    --results_path results.csv
```

- `--baseline` 同时对未量化模型进行评分。它与量化模型一样，都是使用 Torch-TensorRT 编译的，因此每个报告的数字都来自相同的 TRT 运行时（通过）。 `--skip_trt` （改为对积极/虚假量化模型进行评分）。
- 该评估使用**动态**引擎（默认）。 `--batch_size 128`对于两种精度，它都能处理任何批次大小的尾随部分批次。
- `--results_path results.csv` 写入指标表（`Metric`， `Top1 (%)`， `Top5 (%)`) 转换为 CSV。

> [！笔记]
> 验证使用门控 `ILSVRC/imagenet-1k` 拆分：接受其许可/设置 `HF_TOKEN`或点 `--imagenet_path` 在本地副本上。 `evaluate()` 重新分配，因此部分 `--eval_data_size` 每次运行都会抽取不同的随机子集——为了获得稳定、可比较的分数，可以省略它（完整的 50k 数据集）。

## 定制食谱

使用 `--recipe <path>` 插入不同的配方——要么是相对于某个配方的路径 `modelopt_recipes/` （根据内置配方库解析）或 YAML 文件的绝对文件系统路径。配方通过以下方式加载： `modelopt.recipe.load_recipe`必须声明 `metadata.recipe_type: ptq` 和 `quantize:` 部分及其 `quantize` 配置直接传递给 `mtq.quantize`参见现有内容 [`modelopt_recipes/huggingface/vit/ptq/*.yaml`](../../modelopt_recipes/huggingface/vit/ptq/) 这里使用的图案。

### 从已保存的检查点恢复

`torch_tensorrt_ptq.py` 始终将量化的模型优化器状态保存到 `<save_dir>/vit_modelopt_state.pt` （默认 `--save_dir ./modelopt_quantized`） 通过 `mto.save`. 要在不重新校准的情况下重新加载，请在 TRT 编译步骤之前将其恢复到新加载的模型上：

```python
import modelopt.torch.opt as mto

mto.restore(model, "./modelopt_quantized/vit_modelopt_state.pt")
```

> [！笔记]
> 参见 [save / restore guide](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html) 完整版 `mto.save` / `mto.restore` 工作流程。

## 资源

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)
