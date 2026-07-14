<div align="center">


![Banner image](docs/source/assets/model-optimizer-banner.png)

# NVIDIA Model Optimizer

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)

[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/Model-Optimizer)
[![version](https://img.shields.io/pypi/v/nvidia-modelopt?label=Release)](https://pypi.org/project/nvidia-modelopt/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue)](./LICENSE)

[Documentation](https://nvidia.github.io/Model-Optimizer) |
[Roadmap](https://github.com/NVIDIA/Model-Optimizer/issues/1699)

</div>

______________________________________________________________________

NVIDIA 模型优化器（简称模型优化器或 ModelOpt）是一个包含最先进模型优化功能的库。 [techniques](#techniques) 包括量化、剪枝、神经架构搜索（NAS）、蒸馏、推测性解码和稀疏性，以加速模型。

**[Input]** Model Optimizer 目前支持输入 [Hugging Face](https://huggingface.co/)， [PyTorch](https://github.com/pytorch/pytorch) 或者 [ONNX](https://github.com/onnx/onnx) 模型。

**[优化]** Model Optimizer 提供 Python API，方便用户组合上述模型优化技术并导出优化后的量化检查点。Model Optimizer 还与 [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)， [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 和 [Hugging Face Accelerate](https://github.com/huggingface/accelerate) 集成，支持需要训练的推理优化技术。

**[Export for deployment]** 与 NVIDIA AI 软件生态系统无缝集成，Model Optimizer 生成的量化检查点可直接部署到下游推理框架中，如 [SGLang](https://github.com/sgl-project/sglang)， [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization)， [TensorRT](https://github.com/NVIDIA/TensorRT)， 或者 [vLLM](https://github.com/vllm-project/vllm)统一的 Hugging Face 导出 API 现在同时支持变换器和扩散器模型。

## 最新动态

- [2026/05/27] [**Nemotron-3-Nano-30B-A3B 端到端优化教程**](./examples/megatron_bridge/tutorials/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)：剪枝 + 两阶段蒸馏 + FP8 量化，实现 2.6× vLLM 吞吐量和 2.6× 内存缩减。

- [2026/05/13] [**Puzzletron**](./examples/puzzletron)：一种用于 LLM 和 VLM 模型异构剪枝及神经架构搜索（NAS）的新算法。

- [2026/04/15] 客户案例：[Domyn 使用 ModelOpt 的 Minitron 剪枝 + 蒸馏将 Colosseum-355B 压缩至 260B](https://www.domyn.com/blog/domyn-large-the-journey-of-a-european-sovereign-ai-model-for-regulated-industries)

- [2026/03/17] 客户案例：[Bielik.AI 使用 ModelOpt 的 Minitron 剪枝 + 蒸馏构建 Bielik Minitron 7B（体积缩小 33%、速度提升 50%、保留 90% 质量）](https://bielik.ai/en/nvidia-gtc-bielik-minitron-premiere/)

- [2026/03/11] Model Optimizer 量化的 Nemotron-3-Super 检查点已在 Hugging Face 发布可供下载：[FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)、[NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)。了解更多请查看 [Nemotron 3 Super 发布博客](https://blogs.nvidia.com/blog/nemotron-3-super-agentic-ai/)。点击[此处]()了解如何量化 Nemotron 3 模型以加速部署

- [2026/03/11] [NeMo Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)现已支持 Nemotron-3-Super 量化（PTQ 和 QAT）及导出工作流，使用 Model Optimizer 库。参见 [量化（PTQ 和 QAT）指南](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/super-v3/docs/models/llm/nemotron3-super.md#quantization-ptq-and-qat)获取 FP8/NVFP4 量化及 HF 导出说明。

- [2025/12/11] [博客：面向更快速、更智能推理的五大 AI 模型优化技术](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/)

- [2025/12/08] NVIDIA TensorRT Model Optimizer 正式更名为 NVIDIA Model Optimizer。

- [2025/10/07] [博客：使用 NVIDIA Model Optimizer 对 LLM 进行剪枝与蒸馏](https://developer.nvidia.com/blog/pruning-and-distilling-llms-using-nvidia-tensorrt-model-optimizer/)

- [2025/09/17] [博客：降低 AI 推理延迟的推测解码（Speculative Decoding）入门](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

- [2025/09/11] [博客：量化感知训练（QAT）如何实现低精度准确率恢复](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/)

- [2025/08/29] [博客：使用量化感知训练微调 gpt-oss 以提升准确率与性能](https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/)

- [2025/08/01] [博客：使用后训练量化（PTQ）优化 LLM 的性能与准确率](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)

- [2025/06/24] [博客：介绍 NVFP4——面向高效且准确的低精度推理](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

- [2025/05/14] [NVIDIA TensorRT 为 NVIDIA Blackwell GeForce RTX 50 系列 GPU 解锁 FP4 图像生成能力](https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)

- [2025/04/21] [Adobe 使用 Model Optimizer + TensorRT 优化部署，扩散模型延迟降低 60%、总拥有成本（TCO）降低 40%](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/)

- [2025/04/05] [NVIDIA 加速 Meta Llama 4 Scout 和 Maverick 推理](https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/)。了解如何量化 Llama4 以加速部署[在此](#support-matrix)

- [2025/03/18] [基于 Blackwell FP4 实现全球最快 DeepSeek-R1 推理 & 提升 Blackwell 上图像生成效率](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)

- [2025/02/25] Model Optimizer 量化的 NVFP4 模型已在 Hugging Face 发布可供下载：[DeepSeek-R1-FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4)、[Llama-3.3-70B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4)、[Llama-3.1-405B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP4)

- [2025/01/28] Model Optimizer 新增 NVFP4 支持。查看 NVFP4 PTQ 示例[此处](./examples/hf_ptq/README.md#getting-started)。

- [2025/01/28] Model Optimizer 现已开源！

<details close>
<summary>Previous News</summary>

- [2024/10/23] Model Optimizer 量化后的 FP8 Llama-3.1 Instruct 模型已在 Hugging Face 发布可供下载：[8B](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8)、[70B](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8)、[405B](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP8)。

- [2024/09/10] [使用 NVIDIA NeMo 和 Model Optimizer 对 LLM 进行后训练量化（PTQ）](https://developer.nvidia.com/blog/post-training-quantization-of-llms-with-nvidia-nemo-and-nvidia-tensorrt-model-optimizer/)。

- [2024/08/28] [在 NVIDIA H200 GPU 上使用 Model Optimizer 将 Llama 3.1 405B 推理性能提升高达 44%](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-performance-by-up-to-44-with-nvidia-tensorrt-model-optimizer-on-nvidia-h200-gpus/)

- [2024/08/28] [借助 Medusa 实现最高 1.9 倍 Llama 3.1 性能提升](https://developer.nvidia.com/blog/low-latency-inference-chapter-1-up-to-1-9x-higher-llama-3-1-performance-with-medusa-on-nvidia-hgx-h200-with-nvlink-switch/)

- [2024/08/15] 近期版本新功能：缓存扩散（Cache Diffusion）、[配合 NVIDIA NeMo 的 QLoRA 工作流](https://docs.nvidia.com/nemo-framework/user-guide/24.09/sft_peft/qlora.html)等。详见[我们的博客](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/)。

- [2024/06/03] Model Optimizer 现提供实验性特性可部署至 vLLM，作为支持主流部署框架的一部分。查看工作流[此处](#vllm)

- [2024/05/08] [官宣：Model Optimizer 正式发布，进一步加速生成式 AI 推理性能](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)

- [2024/03/27] [Model Optimizer 助力 TensorRT-LLM 在 MLPerf LLM 推理基准测试中刷新纪录](https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/)

- [2024/03/18] [GTC 会议：使用 TensorRT-LLM 和 TensorRT 中的量化技术优化生成式 AI 推理](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63213/)

- [2024/03/07] [Model Optimizer 的 8-bit 后训练量化使 TensorRT 将 Stable Diffusion 推理加速近 2 倍](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/)

- [2024/02/01] [使用 Model Optimizer 量化技术在 TRT-LLM 中加速推理](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md)

</details>

## 安装

通过 [PyPI](https://pypi.org/project/nvidia-modelopt/)用 `pip`安装 Model Optimizer 稳定版：

```
pip install -U nvidia-modelopt[all]
```

Model Optimizer 会下载并安装额外的第三方开源软件包，使用前请查阅其许可证条款。

从源码以可编辑模式安装含所有开发依赖或体验最新特性，运行：

```
# 克隆 Model Optimizer 仓库
git clone git@github.com:NVIDIA/Model-Optimizer.git
cd Model-Optimizer

pip install -e .[dev]
```

也可直接使用预装 Model Optimizer 的 NVIDIA 容器镜像：

- `nvcr.io/nvidia/pytorch:<version>-py3`

- `nvcr.io/nvidia/nemo:<version>`

- `nvcr.io/nvidia/tensorrt-llm/release:<version>`

拉取和使用容器镜像前请查阅各自的许可证条款。确保按上述说明将 Model Optimizer 升级至最新版本。更多关于依赖精细控制、替代 Docker 镜像及环境变量设置，请参阅[安装指南](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html)。

## Techniques

<div align="center">

|             **技术**             |                       **描述**                        |                           **示例**                           |                           **文档**                           |
| :------------------------------: | :---------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|        后训练量化（PTQ）         | 将模型尺寸压缩 2-4 倍，在保持模型质量的同时加速推理！ | [HFLLMs/VLMs](./examples/hf_ptq/README.md);[Megatron−BridgeLLMs/VLMs](./examples/megatron_bridge/README.md); [Diffusers](./examples/diffusers/README.md); [ONNX](./examples/onnx_ptq/README.md); [Windows](./examples/windows/README.md) | [文档](https://nvidia.github.io/Model−Optimizer/guides/1quantization.html) |
|  量化感知训练 / 蒸馏（QAT/QAD）  |      通过少量训练步骤进一步修正量化模型的精度！       | [HuggingFace](./examples/llmqat/)[Megatron−Bridge](./examples/megatronbridge) | [文档](https://nvidia.github.io/Model−Optimizer/guides/1quantization.html) |
|         剪枝（Pruning）          |  通过移除不必要权重减小参数量或内存占用，加速推理！   | [通用](./examples/pruning/)[Megatron−Bridge](./examples/megatronbridge/) |                                                              |
|     知识蒸馏（Distillation）     |      让小模型模仿大模型行为，缩减部署模型尺寸！       | [HuggingFace](./examples/llmdistill/)[Megatron−Bridge](./examples/megatronbridge/)[Megatron-LM](./examples/llm_distill/README.md#knowledge-distillation-kd-in-nvidia-megatron-lm-framework) | [文档](https://nvidia.github.io/Model−Optimizer/guides/4distillation.html) |
| 推测解码（Speculative Decoding） |        训练 Draft 模块在推理时预测额外 Token！        | [HuggingFace](./examples/speculativedecoding/)[Megatron-LM](./examples/speculative_decoding#mlm-example) | [文档](https://nvidia.github.io/Model−Optimizer/guides/5speculativedecoding.html) |
|        稀疏化（Sparsity）        |        仅存储非零参数值及其位置，高效压缩模型         |            [HuggingFace](./examples/llmsparsity/)            | [文档](https://nvidia.github.io/Model−Optimizer/guides/6sparsity.html) |

</div>

## 预量化检查点

- 开箱即用的可部署检查点 [🤗HuggingFace−NVIDIAModelOptimizer合集](https://huggingface.co/collections/nvidia/inference−optimized−checkpoints−with−model−optimizer)

- 可部署于 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[vLLM](https://github.com/vllm-project/vllm)和 [SGLang](https://github.com/sgl-project/sglang)

- 更多模型即将推出！

## 资源

- 📅 [路线图](https://github.com/NVIDIA/Model-Optimizer/issues/1699)

- 📖 [文档](https://nvidia.github.io/Model-Optimizer)

- 🎯 [基准测试](./examples/benchmark.md)

- 💡 [发行说明](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)

- 🐛 [提交 Bug](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=1_bug_report.md)

- ✨ [功能需求](https://github.com/NVIDIA/Model-Optimizer/issues/new?template=2_feature_request.md)

## 模型支持矩阵

| 模型类型                         | 支持矩阵                                                     |
| -------------------------------- | ------------------------------------------------------------ |
| LLM / VLM 量化                   | [查看支持矩阵](#support-matrix) |
| Diffusers 量化                   | [查看支持矩阵](./examples/diffusers/README.md#support-matrix) |
| ONNX 量化                        | [查看支持矩阵](./examples/torch_onnx/README.md#onnx-export-supported-llm-models) |
| Windows 量化                     | [查看支持矩阵](./examples/windows/README.md#support-matrix) |
| 量化感知训练（QAT/QAD）          | [查看支持矩阵](./examples/llm_qat/README.md#support-matrix) |
| 剪枝（Pruning）                  | [查看支持矩阵](./examples/pruning/README.md#support-matrix) |
| 知识蒸馏（Distillation）         | [查看支持矩阵](./examples/llm_distill/README.md#support-matrix) |
| 推测解码（Speculative Decoding） | [查看支持矩阵](./examples/speculative_decoding/README.md#support-matrix) |

## 弃用策略

Model Optimizer 对弃用功能采用结构化管理办法：

- **通知**：弃用说明记录于[变更日志](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)。被弃用项含源码标注注明弃用时机，运行时若被调用将发出警告。

- **迁移期**：由于 Model Optimizer 尚处于 pre-1.0 阶段，自弃用起提供 1 个版本（约 1 个月）的迁移窗口。期间弃用功能仍可正常工作并发出警告。

- **范围**：适用于完整 API 移除及部分参数移除但方法保留的情形。

- **移除**：迁移期结束后，弃用元素将在次版本更新时按语义化版本规范移除，0.x 阶段可能包含破坏性变更。

## 引用

若您在研究中使用了 NVIDIA Model Optimizer，请按如下方式引用：

```
@misc{nvidia-modelopt,
  author       = {{NVIDIA Corporation}},
  title        = {{NVIDIA Model Optimizer}},
  howpublished = {\url{https://github.com/NVIDIA/Model-Optimizer}},
  year         = {2024--2026},
  note         = {GitHub repository}
}
```

## 贡献

Model Optimizer 现已开源！欢迎反馈、功能请求及 PR。请先阅读[贡献指南](./CONTRIBUTING.md)了解参与方式。

## AI Agents

关于 AI 辅助开发环境配置，参见 [Agent 工具说明](./.agents/TOOLING.md)。

### 主要贡献者

[![Contributors](https://contrib.rocks/image?repo=NVIDIA/Model-Optimizer)](https://github.com/NVIDIA/Model-Optimizer/graphs/contributors)

Happy optimizing!