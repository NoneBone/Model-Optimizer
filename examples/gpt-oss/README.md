# OpenAI GPT-OSS 量化感知训练 (QAT) 和量化部署

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
此文件夹展示了 OpenAI 的 GPT-OSS 模型（200 亿和 1200 亿参数）的量化感知训练 (QAT) 和部署示例。GPT-OSS 模型本身就已使用量化技术进行量化。 [MXFP4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) （微缩 FP4），一种 4 位浮点格式（E2M1）。得益于 MXFP4，20 位型号可以装入 16 GB 的 GPU，而 120 位型号可以装入单个 80 GB 的 GPU。

MXFP4 是一种开放权重模型，开发者可以对其进行微调，以添加特殊功能或领域知识。原生 MXFP4 的微调具有挑战性，因为其动态范围和精度可能不足以处理反向传播过程中的梯度变化。将 MXFP4 模型反量化为 BF16 并进行 BF16 训练是一种可行的方案。然而，这会导致 BF16 权重模型的大小约为原始模型的 4 倍。因此，下一个方案是对微调后的模型执行 MXFP4 训练后量化 (PTQ)。但是，PTQ 会降低微调后模型的精度。

使用量化感知训练 (QAT) 进行微调可以解决这些问题。经过 QAT 后的模型精度为 MXFP4，可以像原始 GPT-OSS 模型一样，以更小的内存占用部署到高性能服务框架中，例如
[TensorRTLLM](https://github.com/NVIDIA/TensorRT-LLM)， [vLLM](https://github.com/vllm-project/vllm) 或者 [SGLang](https://github.com/sgl-project/sglang)。

## 目录

1. [Setup](#setup)
1. [Quantization Aware Training from ModelOpt](#quantization-aware-training-from-modelopt)
1. [Deployment](#deployment)
1. [LoRA QAT: low memory footprint alternative for full parameter QAT](#lora-qat-low-memory-footprint-alternative-for-full-parameter-qat)
1. [Quantization Aware Training & Deployment for models beyond GPT-OSS](#quantization-aware-training--deployment-for-models-beyond-gpt-oss)

## 设置

安装必要的依赖项：

```bash
pip install -U nvidia-modelopt[hf]
pip install -r requirements.txt
```

## 来自 ModelOpt 的量化感知训练

在量化感知训练（Quantization Aware Training，QAT）中，前向计算使用“伪量化”值，而反向计算则使用高精度数据类型。在“伪量化”中，量化值的数值等效值使用高精度数据类型（例如 BF16）表示。因此，QAT 可以集成到标准训练流程中，例如常规的 BF16 混合精度训练。

在量化评估训练 (QAT) 过程中，模型会学习如何在量化后恢复精度。要执行 QAT，请先使用 ModelOpt 对模型进行量化。 [`mtq.quantize`](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.quantize) API。然后，您可以使用现有的训练流程来训练这个量化模型。

以下是一个代码示例：

```python
import modelopt.torch.quantization as mtq

# Specify quantization config;
config = mtq.MXFP4_MLP_WEIGHT_ONLY_CFG

# Define forward loop for calibration
def forward_loop(model):
    for data in calib_set:
        model(data)

# quantize the model and prepare for QAT
model = mtq.quantize(model, config, forward_loop)

# QAT with your regular finetuning pipeline
train(model, train_loader, optimizer, scheduler, ...)
```

如需查看展示上述工作流程的完整示例，请查看 [qat-finetune-transformers.ipynb](./qat-finetune-transformers.ipynb)。

如果您正在使用 Huggingface 提供的训练器课程来训练 Huggingface 模型，例如 [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) 进行QAT测试就更简单了——只需将训练器替换为同等型号的训练器即可。 `QATSFTTrainer` 从 ModelOpt 中指定额外的量化参数。 `QATSFTTrainer` 将在后端执行必要的量化步骤，并像训练原始模型一样训练模型。 `SFTTrainer`。

一个完整的端到端示例是： `sft.py` 在此文件夹中。要对 GPT-OSS 20B 模型执行包含完整参数 SFT 的 QAT，请运行：

```sh
# Other supported quantization configs include NVFP4_MLP_WEIGHT_ONLY_CFG, NVFP4_MLP_ONLY_CFG etc.
# [Optional] For faster FlashAttention3, add '--attn_implementation kernels-community/vllm-flash-attn3'
accelerate launch --config_file configs/zero3.yaml sft.py \
    --config configs/sft_full.yaml --model_name_or_path openai/gpt-oss-20b \
    --quant_cfg MXFP4_MLP_WEIGHT_ONLY_CFG \
    --output_dir gpt-oss-20b-qat
```

GPT-OSS 20B 全参数 SFT 需要一个配备 8 个 80 GB GPU 的节点。要更改数据集或训练超参数，请修改以下配置： `configs/sft_full.yaml` 或者将它们作为命令行参数传递。

### 推荐的QAT食谱

为了提高准确性，我们推荐以下QAT配方：

- **第一步：高精度微调模型**

- **步骤 2：对微调后的模型（来自步骤 1）应用 QAT**

  - 对于高精度训练后的 QAT，使用 Adam 优化器，较小的学习率（例如 1e-5）效果很好。
  - QAT通常可在几百万到几十亿个代币内恢复准确性。请评估您的检查点以确定准确性是否已恢复。

要执行此推荐的 QAT 方案，请运行：

```sh
# Step 1: Perform high precision SFT without quantization
accelerate launch --config_file configs/zero3.yaml sft.py \
  --config configs/sft_full.yaml --model_name_or_path openai/gpt-oss-20b \
  --output_dir gpt-oss-20b-sft

# Step 2: Perform QAT on the high precision SFT checkpoint
accelerate launch --config_file configs/zero3.yaml sft.py \
    --config configs/sft_full.yaml --model_name_or_path gpt-oss-20b-sft \
    --quant_cfg MXFP4_MLP_WEIGHT_ONLY_CFG \
    --output_dir gpt-oss-20b-qat \
```

最终的 QAT 检查点采用伪量化形式。低内存占用和加速效果随后实现。 [deployment](#deployment) 加速运行时。

注意：要恢复 PyTorch 原生评估的模型检查点，请参阅 [ModelOpt Restore using Huggingface APIs](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html#modelopt-save-restore-using-huggingface-checkpointing-apis)。

## 部署

上述 GPT-OSS QAT 模型可以以 MXFP4 格式部署到 TensorRT-LLM、vLLM 和 SGLang 等高性能服务引擎上。为此，我们提供了一个自定义转换脚本，可以将 Hugging Face 兼容的 BF16 检查点转换为与原始 GPT-OSS 版本相同的 MXFP4 权重格​​式。这种真正的 MXFP4 量化检查点可以像原始 GPT-OSS MXFP4 模型一样部署到受支持的运行时环境中。

要将 QAT 检查点导出为实际量化的 MXFP4，请运行：

```bash
python convert_oai_mxfp4_weight_only.py  \
    --model_path gpt-oss-20b-qat \
    --output_path gpt-oss-20b-qat-real-mxfp4
```

注意：模型优化器目前导出的量化检查点格式并非 MXFP4。我们计划并正在积极开发对 vLLM、SGLang 和 TensorRT-LLM 中由模型优化器生成的 MXFP4 检查点的支持。

<details>
<summary><strong>在 TensorRT-LLM 上部署</strong></summary>

要设置 TensorRT-LLM，请按照官方指南操作： [Deploying GPT-OSS on TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog9_Deploying_GPT_OSS_on_TRTLLM.md)
安装完成后，使用以下命令启动与 OpenAI 兼容的端点：

```bash
trtllm-serve path/to/quantized/checkpoint --tokenizer /path/to/tokenizer --max_batch_size <max_batch_size> --max_num_tokens <max_num_tokens> --max_seq_len <max_seq_len> --tp_size <tp_size> --pp_size <pp_size> --host <host_ip_address> --port <port> --kv_cache_free_gpu_memory_fraction 0.95

```

</details>

<details>
<summary><strong>在 SGLang 上部署</strong></summary>

要设置 SGLang，请参考设置问题： [SGLang Setup Guide](https://github.com/sgl-project/sglang/issues/8833)
然后使用以下命令启动服务器：

```bash
python3 -m sglang.launch_server --model <model-path> --tp <tp_size>

```

</details>

<details>
<summary><strong>在 vLLM 上部署</strong></summary>

要使用 vLLM 进行部署，请按照以下步骤操作： [OpenAI Cookbook instructions](https://cookbook.openai.com/articles/gpt-oss/run-vllm)
然后使用以下命令启动服务器：

```bash
vllm serve <model_path>

```

</details>
<br>

## LoRA QAT：低内存占用的全参数 QAT 替代方案

您可以使用 LoRa 运行 QAT 以降低训练所需的 GPU 内存。使用一个配备 8 个 80 GB GPU 的节点，您可以在 GPT OSS 120B 模型上执行基于 LoRa 的 QAT。

以下是如何为 GPT OSS 120B 型号运行 LoRA QAT：

```bash
python sft.py --config configs/sft_lora.yaml \
    --model_name_or_path openai/gpt-oss-120b \
    --quant_cfg MXFP4_MLP_WEIGHT_ONLY_CFG \
    --output_dir gpt-oss-120b-lora-qat
```

上述 QAT 过程中得到的 LoRA-QAT 适配器权重需要与部署所需的基准权重合并。
上述自定义转换脚本在导出 MXFP4 权重之前会执行 Lora 适配器合并。为此，请指定： `lora_path` 和 `base_model_path` 到自定义转换脚本：

```sh
python convert_oai_mxfp4_weight_only.py  \
    --lora_path gpt-oss-120b-lora-qat \
    --base_path openai/gpt-oss-120b \
    --output_path gpt-oss-120b-lora-qat-merged-real-mxfp4
```

您可以像部署原始 GPT-OSS MXFP4 模型一样部署这个真正的量化 MXFP4 检查点。

## 面向GPT-OSS以外模型的量化感知训练与部署

### 使用 LLaMA-Factory 从 ModelOpt 轻松进行 QAT

ModelOpt 通过以下方式提供简便的端到端快速评估测试 (QAT) [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，一个用于 LLM/VLM 微调的开源代码库。请参阅 [LLaMa-Factory QAT example](../llm_qat/llama_factory) 用于对您喜爱的模型执行 QAT。

### ModelOpt QAT/PTQ 模型在 GPT-OSS 之外的部署

ModelOpt支持将QAT/PTQ处理后的多种模型导出为TensorRT-LLM、vLLM、SGLang等格式。详情请参阅…… [hf_ptq](../hf_ptq)。
