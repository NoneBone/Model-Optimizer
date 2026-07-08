# 扩散器模型优化

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
模型优化器支持缓存扩散和扩散模型量化等技术。

训练后量化 (PTQ) 是一种有效的模型优化技术，它可以将模型压缩到较低的精度，例如 INT8、FP8、NVFP4 等。使用模型优化器进行量化可以将模型大小压缩 2 到 4 倍，从而在保持模型质量的同时加快推理速度。量化感知训练 (QAT) 是一种强大的模型优化技术，尤其适用于 PTQ 方法无法满足任务需求的情况。

缓存扩散是一种技术，它重用先前扩散步骤中缓存的输出，而不是重新计算它们。这种无需训练的缓存方法与多种模型兼容，例如 DiT 和 UNet，能够在不牺牲质量的前提下显著提高速度。

<div align="center">

|          **章节**          |                           **描述**                           |                          **链接**                          |                           **文档**                           |
| :------------------------: | :----------------------------------------------------------: | :--------------------------------------------------------: | :----------------------------------------------------------: |
|          先决条件          |               使用此技术所需的必需和可选软件包               |                \[[Link](#pre-requisites)\]                 |                                                              |
|          入门指南          | 了解如何使用量化/缓存扩散来优化模型，以降低精度并提高推理效率 |                [[Link](#getting-started)\]                 | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
|          支持矩阵          | 查看支持矩阵，了解不同模型的量化/缓存扩散兼容性和功能可用性  |                \[[Link](#support-matrix)\]                 | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
| 稀疏注意力（Skip-softmax） |            用于扩散模型的 Skip-softmax 稀疏注意力            |         \[[Link](#sparse-attention-skip-softmax)\]         |                                                              |
|          缓存扩散          |          一种在不影响质量的前提下加速推理的缓存技术          |                [[Link](#cache-diffusion)\]                 |                                                              |
|      训练后量化 (PTQ)      |             如何在扩散模型上运行 PTQ 的示例脚本              |        \[[Link](#post-training-quantization-ptq)\]         | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
|     量化感知训练 (QAT)     |             如何在扩散模型上运行 QAT 的示例脚本              |        \[[Link](#quantization-aware-training-qat)\]        | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
|     量化感知蒸馏 (QAD)     |             如何在扩散模型上运行 QAD 的示例脚本              |      \[[Link](#quantization-aware-distillation-qad)\]      | \[[docs](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)\] |
|  使用 TensorRT 构建和运行  |             如何使用 TensorRT 构建和运行量化模型             | [[Link](#build-and-run-with-tensorrt-compiler-framework)\] |                                                              |
|            LoRa            |                 在量化之前融合您的 LoRa 权重                 |                     \[[Link](#lora)\]                      |                                                              |
|        预量化检查点        |              准备部署 Hugging Face 预量化检查点              |           \[[Link](#pre-quantized-checkpoints)\]           |                                                              |
|            资源            |                      相关资源的更多链接                      |                   \[[Link](#resources)\]                   |                                                              |



</div>

## 先决条件

### Docker

请使用 TensorRT docker 镜像（例如， `nvcr.io/nvidia/tensorrt:26.02-py3`）或访问我们的网站 [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) 了解更多信息。

此外，请按照以下安装步骤升级到最新版本的模型优化器并安装示例特定的依赖项。

### 本地安装

安装模型优化器 `onnx` 和 `hf` 使用依赖项 `pip` 从 [PyPI](https://pypi.org/project/nvidia-modelopt/)：

```bash
pip install nvidia-modelopt[onnx,hf]
pip install -r requirements.txt
```

每个子部分（快速生成、蒸馏等）可能都有自己的 `requirements.txt` 需要单独安装的文件。

您可以找到最新的 TensorRT [here](https://developer.nvidia.com/tensorrt/download)。

访问我们的网站 [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html) 了解更多信息。

## 入门

### 量化

借助下方简洁的 API，您可以轻松使用模型优化器对模型进行量化。模型优化器通过将模型的精度转换为所需的精度，然后使用小型数据集（通常为 128-512 个样本）来实现这一点。 [calibrate](https://nvidia.github.io/Model-Optimizer/guides/_basic_quantization.html) 量化缩放因子。

```python
import modelopt.torch.quantization as mtq

def forward_pass(model):
    for prompt in prompts:
        _ = model(prompt)

mtq.quantize(model=transformer, config=quant_config, forward_func=forward_pass)
```

## 支持矩阵

### TensorRT 编译器框架

| Model | fp8 | int8_sq | int4_awq | w4a8_awq<sup>1</sup> | nvfp4<sup>2</sup> | nvfp4_svdquant<sup>3</sup> | Cache Diffusion |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Stable Diffusion XL](https://huggingface.co/papers/2307.01952) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |

> *<sup>1.</sup>w4a8_awq 是一种实验性的量化方案，可能会导致更高的精度损失。*

> *<sup>2.</sup>内部测试仅包含部分常用模型。实际支持的模型列表可能更长。NVFP4 推理需要 Blackwell GPU 和 TensorRT-LLM v0.17 或更高版本*

> *<sup>3.</sup>TRT 中的 SVDQuant 性能可能不如…… [Nunchaku: MIT-Nvidia](https://github.com/nunchaku-tech/nunchaku) 此时此刻。*

## 训练后量化（PTQ）

我们支持 INT8、FP8 和 FP4 精度的校准，并支持权重和激活值的校准。以下示例展示了如何使用模型优化器来校准和量化扩散模型的主干部分。主干部分通常会消耗超过 95% 的端到端扩散延迟。

我们还提供关于如何部署和运行基于模型优化器量化 INT8 和 FP8 主干的端到端扩散流水线的说明，以便在目标 GPU 上生成图像并测量延迟。请注意，由于软件不兼容，目前不支持 Jetson 设备。

> [！笔记]
> 模型校准所需的GPU计算能力相对高于部署所需的GPU计算能力。它不需要与部署目标GPU使用相同的GPU。ONNX导出和TensorRT引擎指令位于[此处]。 [`quantization/ONNX-TRT-Deployment.md`](./quantization/ONNX-TRT-Deployment.md)。

### 量化脚本

#### FLUX|SD3|SDXL INT8 [Script](./quantization/quantize.py)

```sh
python quantize.py \
    --model {flux-dev|flux-schnell|sdxl-1.0|sdxl-turbo|sd3-medium|sd3.5-medium} \
    --format int8 --batch-size 2 \
    --calib-size 32 --alpha 0.8 --n-steps 20 \
    --model-dtype {Half/BFloat16} \
    --quantized-torch-ckpt-save-path ./{MODEL_NAME}.pt \
    --hf-ckpt-dir ./hf_ckpt
```

#### FLUX|SD3|SDXL|LTX|WAN2.2 FP8/FP4 [Script](./quantization/quantize.py)

```sh
python quantize.py \
    --model {flux-dev|flux-schnell|sdxl-1.0|sdxl-turbo|sd3-medium|sd3.5-medium|ltx-video-dev|wan2.2-t2v-14b|wan2.2-t2v-5b} \
    --model-dtype {Half|BFloat16} \
    --format {fp8|fp4} --batch-size 2 --calib-size {128|256} --quantize-mha \
    --n-steps 20 --quantized-torch-ckpt-save-path ./{MODEL_NAME}.pt --collect-method default \
    --hf-ckpt-dir ./hf_ckpt
```

#### Wan 2.2 VAE NVFP4（Conv3D隐式GEMM）

Wan 2.2 VAE（`AutoencoderKLWan`（在 5B 和 14B 流水线之间共享）由 3D 卷积构成。当使用 NVFP4 对 VAE 进行量化时， `Conv3d` 各层通过自定义的 BF16 WMMA 隐式 GEMM 内核自动分发，该内核采用融合的 FP4 激活量化。需要 SM80+（Ampere 或更新版本）。参见 [`modelopt/torch/kernels/quantization/conv/README.md`](../../modelopt/torch/kernels/quantization/conv/README.md) 有关内核详细信息。

```sh
python quantize.py \
    --model {wan2.2-t2v-14b|wan2.2-t2v-5b} \
    --backbone vae \
    --format fp4 --quant-algo max --collect-method default \
    --model-dtype BFloat16 --trt-high-precision-dtype BFloat16 \
    --batch-size 1 --calib-size 32 --n-steps 30 \
    --quantized-torch-ckpt-save-path ./wan22_vae_fp4.pt
```

#### [LTX-2](https://github.com/Lightricks/LTX-2) FP4

> [！警告]
> **第三方许可声明 — LTX-2**
>
> LTX-2 是由 Lightricks 开发和提供的第三方模型及软件包。
> **不**受 NVIDIA Model Optimizer 所适用的 Apache 2.0 许可证的约束。
>
> 通过安装和使用 LTX-2 软件包（`ltx-core`， `ltx-pipelines`， `ltx-trainer`） 和
> 使用 NVIDIA 模型优化器时，您**必须**遵守以下规定：
> [LTX Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE)。
>
> 使用 NVIDIA 模型优化器从 LTX-2 生成任何衍生模型或微调权重
> （包括量化或提炼的检查点）仍受 LTX 社区许可的约束。
> 协议不包含在 Apache 2.0 中。

此示例产生三个输出：一个 PyTorch 检查点（`--quantized-torch-ckpt-save-path`），一个拥抱脸检查点（`--hf-ckpt-dir`），以及与 ComfyUI 兼容的合并安全张量（`--extra-param merged_base_safetensor_path`）。

```sh
python quantize.py \
    --model ltx-2 --format fp4 --batch-size 1 --calib-size 32 --n-steps 40 \
    --extra-param checkpoint_path=./ltx-2-19b-dev-fp8.safetensors \
    --extra-param distilled_lora_path=./ltx-2-19b-distilled-lora-384.safetensors \
    --extra-param spatial_upsampler_path=./ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --extra-param gemma_root=./gemma-3-12b-it-qat-q4_0-unquantized \
    --extra-param fp8transformer=true \
    --quantized-torch-ckpt-save-path ./ltx-2-transformer.pt \
    --hf-ckpt-dir ./LTX2-NVFP4/ \
    --extra-param merged_base_safetensor_path=./ltx-2-19b-dev-fp8.safetensors
```

要进一步应用 NVFP4 标度调整和填充，请添加：

```sh
    --extra-param enable_swizzle_layout=true \
    --extra-param padding_strategy=row_col
```

#### 重要参数

- `percentile`：控制量化缩放因子（amax）的采集范围，这意味着我们将采集所选amax值范围内的数据。 `(n_steps * percentile)` 步骤。建议：1.0
- `alpha`SmoothQuant 中的一个参数，仅用于线性层。建议值：SDXL 为 0.8
- `calib-size`对于 SDXL INT8，我们建议使用 32 或 64 位色深；对于 SDXL FP8，我们建议使用 128 位色深。
- `n_steps`推荐型号：SD/SDXL 20 或 30，SDXL-Turbo 4。

**您可以直接在 PyTorch 中使用生成的检查点，导出 Hugging Face 检查点（`--hf-ckpt-dir`) 将模型部署在 SGLang/vLLM/TRTLLM 上，或者遵循 ONNX/TensorRT 工作流程 [`quantization/ONNX-TRT-Deployment.md`](./quantization/ONNX-TRT-Deployment.md).**

## 量化感知训练（QAT）

量化感知训练 (QAT) 是一种强大的模型优化技术，尤其适用于训练后量化 (PTQ) 方法无法满足任务需求的情况。通过在训练过程中模拟量化的影响，QAT 使模型能够学习如何最小化量化误差，最终提高模型的准确率。

以下示例为简化起见使用了 Hugging Face Accelerate。您可以使用自己偏好的训练设置将 QAT 集成到工作流程中。

### ModelOPT 中的 QAT 工作原理

ModelOPT通过前向传播模拟量化过程，使模型能够调整权重，从而最大限度地减少训练损失并降低量化误差。这使得模型能够更好地应对量化硬件的限制，而不会造成显著的性能损失。

```python
import modelopt.torch.opt as mto

# Restore the model in its quantized state using ModelOPT's API
mto.restore(transformer_model, args.restore_quantized_ckpt)

# Move the model to the appropriate device and set the desired weight precision
transformer_model.to(accelerator.device, dtype=weight_dtype)
transformer_model.requires_grad_(True)

transformer_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    transformer_model, optimizer, train_dataloader, lr_scheduler
)

```

模型通过 ModelOPT 加载到量化状态后，即可进行常规训练。QAT 过程将在前向传播过程中自动执行。

## 量化感知蒸馏（QAD）

蒸馏法是一种强大的方法，它利用高精度模型（教师）指导量化模型（学生）的训练。ModelOPT 通过处理大部分复杂性，简化了将蒸馏法与 QAT 相结合的过程。

有关蒸馏的更多详细信息，请参阅此链接。 [link](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html)。

```diff
import modelopt.torch.opt as mto
import modelopt.torch.distill as mtd

# Restore the model in its quantized state using ModelOPT's API
mto.restore(transformer, args.restore_quantized_ckpt)

'''
After mtd.convert, the model structure becomes:

model:
    transformer_0
    transformer_1
    teacher_model:
        transformer_0
        transformer_1

And the forward pass is automatically monkey-patched to:

def forward(input):
    student_output = model(input)
    _ = teacher_model(input)
    return student_output
'''

+ # Configuration for knowledge distillation (KD)
+ kd_config = {
+     "teacher_model": teacher_model,
+     "criterion": distill_config["criterion"],
+     "loss_balancer": distill_config["loss_balancer"],
+ }
+ transformer = mtd.convert(transformer, mode=[("kd_loss", kd_config)])

# Move the model to the appropriate device and set the desired weight precision
transformer.to(accelerator.device, dtype=weight_dtype)
transformer.requires_grad_(True)

# Making sure to freeze the weights from model._teacher_model
transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    transformer, optimizer, train_dataloader, lr_scheduler
)

# Compute the distillation loss using ModelOPT's compute_kd_loss
+ ...
+ loss = transformer.compute_kd_loss(...)
+ ...

```

## 使用 TensorRT 编译器框架构建和运行

ONNX导出和TensorRT引擎指令的文档位于[此处应填写文档内容]。 [`quantization/ONNX-TRT-Deployment.md`](./quantization/ONNX-TRT-Deployment.md)。

### 罗拉

为了获得最佳的 INT8/FP8 量化模型性能，我们强烈建议在量化之前融合 LoRA 权重。否则，在将 LoRA 层与 INT8/FP8 量化-反量化 (QDQ) 节点集成时，可能会干扰 TensorRT 内核的融合，从而导致性能损失。

有关如何融合 LoRa 权重的详细指南，请参阅 Hugging Face。 [PEFT documentation](https://github.com/huggingface/peft)：

融合权重后，继续进行校准，您可以按照我们的代码进行量化。

```python
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")
pipe.load_lora_weights(
    "CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy"
)
pipe.fuse_lora(lora_scale=0.9)
...
# All the LoRA layers should be fused
check_lora(pipe.unet)

mtq.quantize(pipe.unet, quant_config, forward_loop)
mto.save(pipe.unet, ...)
```

当需要将模型导出为 ONNX 格式时，请确保先加载 PEFT 修改后的 LoRA 模型。

```python
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.load_lora_weights(
    "CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy"
)
pipe.fuse_lora(lora_scale=0.9)
mto.restore(pipe.unet, your_quantized_ckpt)
...
# Export the onnx model
```

按照这些步骤，您的 PEFT LoRA 模型应该能够使用 ModelOpt 进行高效量化，从而可以部署并最大限度地提高性能。

## 稀疏注意力（Skip-Softmax）

Skip-softmax稀疏注意力机制在softmax计算过程中跳过注意力得分可忽略不计的KV图块，从而在不重新训练的情况下减少FLOPs。指数模型（`scale_factor = a * exp(b * target_sparsity)`只需校准一次，即可在运行时调整目标稀疏度而无需重新校准。校准后的系数可以导出为 Hugging Face 检查点（嵌入在每个组件中）。 `config.json` 在下面 `sparse_attention_config`）直接被TRT-LLM消耗 `SkipSoftmaxAttentionConfig.resolve_for_target_sparsity` — 下游无需额外转换。

### 入门

```python
import modelopt.torch.sparsity.attention_sparsity as mtsa

# 1. Define config with calibration
config = {
    "sparse_cfg": {
        "calibration": {
            "target_sparse_ratio": {"prefill": 0.5},
        },
        "*.attn1": {
            "method": "triton_skip_softmax",
            "backend": "triton",
            "is_causal": False,
            "collect_stats": True,
            "enable": True,
        },
        "*.attn2": {"enable": False},
        "default": {"enable": False},
    },
}

# 2. Provide a calibration forward loop
def forward_loop(model):
    pipeline(prompt="a cat", num_frames=81, num_inference_steps=40, ...)

# 3. Sparsify + calibrate
mtsa.sparsify(transformer, config, forward_loop=forward_loop)

# 4. Generate as usual — sparsity is applied automatically
output = pipeline(prompt="a dog on the beach", ...)
```

### 示例脚本

#### 他们 2.2 [Script](./sparsity/wan22_skip_softmax.py)

14B 模型会自动稀疏化两者 `transformer` 和 `transformer_2`。

```bash

# 5B/14B model — calibrate and export a TRT-LLM-ready checkpoint
python sparsity/wan22_skip_softmax.py \
    --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers|Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --calibrate --target-sparsity 0.5 --calib-size 4 \
    --export-dir /path/to/wan22-skip-softmax-ckpt \
    --prompt "A sunset over mountains" --output out.mp4
```

## 缓存扩散

缓存扩散方法，例如 [DeepCache](https://arxiv.org/abs/2312.00858)， [Block Caching](https://arxiv.org/abs/2312.03209) 和 [T-Gate](https://arxiv.org/abs/2404.02747)通过重用先前步骤的缓存输出而非重新计算，可以优化性能。这种**无需训练**的缓存方法兼容多种模型，例如**DiT**和**UNet**，能够在不牺牲质量的前提下显著提升性能。

<div align="center">
  <img src="./cache_diffusion/assets/sdxl_cache.png" width="900" alt="SDXL Cache"/>
  此图显示了本示例中的默认 SDXL 缓存计算图。
  通过跳过特定步骤中的某些代码块，可以显著提高速度。
</div>

### 入门

借助下方简洁的 API，您可以轻松地使用模型优化器将缓存扩散应用于您的模型。模型优化器通过将不同的模块分解成单个 TensorRT 引擎来实现这一点，并在推理时，以类似于 PyTorch 模块的方式组合多个 TensorRT 引擎。

```python
import torch
from cache_diffusion import cachify
from cache_diffusion.utils import SDXL_DEFAULT_CONFIG
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("cuda")

num_inference_steps = 20
prompt = "beautiful lady, (freckles), big smile, blue eyes, short hair, dark makeup, hyperdetailed photography, soft light, head and shoulders portrait, cover"

cachify.prepare(pipe, SDXL_DEFAULT_CONFIG)
cachify.enable(pipe)

generator = torch.Generator(device="cuda").manual_seed(2946901)

with cachify.infer(pipe) as cached_pipe:
    img = cached_pipe(
        prompt=prompt, num_inference_steps=num_inference_steps, generator=generator
    ).images[0]

img
```

### PyTorch框架

请参考 [example.ipynb](./cache_diffusion/example.ipynb) 有关如何应用缓存扩散的更多详细信息。

### TensorRT 编译器框架

要在TensorRT中执行缓存扩散，请按照以下步骤操作：

```python
# Load the model

compile(
    pipe.unet,
    model_id="sdxl",
    onnx_path=Path("./onnx"),
    engine_path=Path("./engine"),
)

cachify.prepare(pipe, num_inference_steps, SDXL_DEFAULT_CONFIG)
```

之后，将其用作标准缓存扩散管道来生成图像。

请注意，只有 UNET 组件在 TensorRT 中运行，而其他部分仍然在 PyTorch 中运行。

### 定制

模型优化器还提供了一个 API，只需调整参数即可创建各种计算图。例如，SDXL 的默认参数为：

```python
SDXL_DEFAULT_CONFIG = [
    {
        "wildcard_or_filter_func": lambda name: "up_blocks.2" not in name,
        "select_cache_step_func": lambda step: (step % 2) != 0,
    }
]

cachify.prepare(pipe, num_inference_steps, SDXL_DEFAULT_CONFIG)
```

两个参数至关重要： `wildcard_or_filter_func` 和 `select_cache_step_func`。

`wildcard_or_filter_func`这可以是字符串或函数。如果模块与给定的字符串或 filter_func 匹配，则会执行缓存操作。例如，如果您的输入是字符串 `*up_blocks*`它将匹配所有包含的名称 `up_blocks` 将来，当您使用时，它将执行缓存操作。 `fnmatch` 要匹配字符串。如果改用函数，模块名称将传递给您提供的函数，如果该函数返回 True，则会执行缓存操作。

`select_cache_step_func`在推理过程中，代码会在每个步骤检查是否要根据以下情况执行缓存操作： `select_cache_step_func` 您已提供。如果 `select_cache_step_func(current_step)` 如果返回 True，则模块将被缓存；否则，不会缓存。

可以设置多种配置，但请确保 `wildcard_or_filter_func` 运行正常。如果您输入多对参数且参数值相同，则无法正常工作。 `wildcard_or_filter_func`列表中后一项会覆盖前一项。

### 演示

以下演示图像是使用以下方法生成的 `torch==2.3.0` 采用单块 RTX 6000 Ada GPU。

与简单地减少生成步骤相比，缓存扩散不仅可以实现相同的加速，还能显著提升图像质量，甚至接近参考图像。如果图像质量无法满足您的需求或产品要求，您可以将我们的默认配置替换为自定义设置。

#### 稳定扩散 - XL

![SDXL 缓存扩散](./cache_diffusion/assets/SDXL_Cache_Diffusion_Img.png)

### 关于随机性的笔记

稳定扩散流程严重依赖于随机采样操作，包括创建高斯噪声张量进行去噪以及在调度步骤中添加噪声。在量化过程中，我们没有固定随机种子。因此，每次运行校准流程时，您都可能得到不同的量化器 amax 值。这可能会导致生成的图像与使用原始模型生成的图像有所不同。我们建议您多运行几次，并选择最佳结果。

## 预先量化的检查点

- 准备部署的检查点[[🤗 Hugging Face - Black Forest Labs](https://huggingface.co/black-forest-labs)\]
- 可部署于 [TensorRT](https://developer.nvidia.com/tensorrt) 和 [PyTorch](https://github.com/pytorch/pytorch)
- 更多车型即将推出！

## 资源

- 📅 [Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/1699)
- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)
