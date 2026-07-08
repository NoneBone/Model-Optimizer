# 模型优化器基准测试参考

本文档总结了[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)在几种流行模型上的性能和精度测量结果。下表中的基准测试数据仅供参考，**不应被视为** Model Optimizer 所能达到的峰值性能。所有性能数据均使用[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)或[TensorRT](https://developer.nvidia.com/tensorrt-getting-started)进行测试。

## 1. 大语言模型的训练后量化

### 1.1 性能

配置：H200，nvidia-modelopt v0.21.1，TensorRT-LLM v0.15，延迟通过[trtllm-bench](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-overview.md#for-non-gh200-systems-1)测量。推理加速比与 BF16 基线进行比较。**加速比已按 GPU 数量归一化**。

> 基准测试场景：输入 token 数 2048，输出 token 数 128。实际性能可能因目标用例和构建 TensorRT-LLM 引擎所使用的标志而异。

> 此处未报告内存节省情况，因为 TensorRT-LLM 会占用所有剩余可用 GPU 内存用于 KV 缓存。

> 如果 GPU 内存受限，则较低比特的量化可能在更少的 TP 下获得更好的 GPU 数量归一化吞吐量增益。

|              |        | BF16 (8B:TP1, 70B:TP2) |      | FP8 (TP1)  |        |      | INT4 AWQ (TP1) |        |      | W4A8 AWQ (TP1) |        |
| :----------: | :----: | :--------------------: | :--: | :--------: | :----: | :--: | :------------: | :----: | :--: | :------------: | :----: |
|     模型     | 批大小 |       Token数/秒       |      | Token数/秒 | 加速比 |      |   Token数/秒   | 加速比 |      |   Token数/秒   | 加速比 |
| Llama3.1-8B  |   1    |         173.80         |      |   245.03   | 1.41x  |      |     231.75     | 1.33x  |      |     239.70     | 1.38x  |
|              |   8    |         803.11         |      |  1,051.17  | 1.31x  |      |     599.72     | 0.75x  |      |     801.72     | 1.00x  |
|              |   64   |        1,679.74        |      |  2,190.93  | 1.30x  |      |    1,392.78    | 0.83x  |      |    1,930.86    | 1.15x  |
| Llama3.1-70B |   1    |         45.81          |      |   43.46    | 1.90x  |      |     44.10      | 1.93x  |      |     46.31      | 2.02x  |
|              |   8    |         182.61         |      |   182.07   | 1.99x  |      |     93.98      | 1.03x  |      |     140.02     | 1.53x  |
|              |   64   |         401.50         |      |   420.64   | 2.10x  |      |     176.68     | 0.88x  |      |     345.43     | 1.72x  |

### 1.2 精度

下表显示了与 BF16 基线相比的 MMLU 损失百分比。

配置：H100，nvidia-modelopt v0.21.1，TensorRT-LLM v0.15。

请注意，对于 H100，通常 FP8 是首选方案。当 GPU 内存受限时，推荐使用 4 位 AWQ 方法。更多使用早期版本 Model Optimizer 的基准测试可参见此[TensorRT-LLM README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md#benchmark)。

|          模型           | MMLU 损失 FP8 | MMLU 损失 INT4 AWQ | MMLU 损失 W4A8 AWQ |
| :---------------------: | :-----------: | :----------------: | :----------------: |
| Llama3.1-8B (instruct)  |     1.50%     |       5.66%        |       6.00%        |
| Llama3.1-70B (instruct) |     0.38%     |       1.07%        |       1.20%        |

## 2. Stable Diffusion 的训练后量化

下表显示了在 Stable Diffusion XL 1.0 基础模型上，与 FP16 基线相比，INT8 和 FP8 的推理加速比。

配置：图像分辨率=1024×1024，30 步。TensorRT v9.3。预热运行次数=1。批大小=1。

|     GPU      | INT8 延迟 (ms) | FP8 延迟 (ms) | 加速比 (INT8 vs FP16) | 加速比 (FP8 vs FP16) |
| :----------: | :------------: | :-----------: | :-------------------: | :------------------: |
| RTX 6000 Ada |    2,479.19    |   2,441.16    |         1.43x         |        1.45x         |
|   RTX 4090   |    2,058.11    |   2,161.38    |         1.20x         |        1.14x         |
|     L40S     |    2,338.88    |   2,167.82    |         1.25x         |        1.35x         |

## 3. 量化感知训练

下表展示了使用 nvidia-modelopt v0.11.0 对 Llama 2 7B 模型进行量化感知训练与训练后量化的验证损失对比。基线是在目标数据集上微调后的模型。请注意，我们使用 INT4 来展示 QAT 在低精度下能更好地保持模型精度。这表明 QAT 可以以较低的训练成本应用，使得对精度下降敏感的生成式 AI 应用即使在[NVIDIA Blackwell 平台](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)的权重和激活均为 4 位的超低精度下也能保持精度。

|         方法         |        数据集        | 验证损失 - BF16 基线 | 验证损失 - PTQ | 验证损失 - QAT (越低越好) |
| :------------------: | :------------------: | :------------------: | :------------: | :-----------------------: |
| INT4 权重, FP16 激活 |        samsum        |        1.036         |     1.059      |         **1.044**         |
| INT4 权重, INT8 激活 |        samsum        |        1.036         |     3.321      |         **1.294**         |
| INT4 权重, FP16 激活 | databricks-dolly-15k |        1.151         |     1.305      |         **1.172**         |
| INT4 权重, INT8 激活 | databricks-dolly-15k |        1.151         |     2.313      |         **1.640**         |

## 4. 稀疏性

### 4.1 性能

下表显示了在不同批大小下，稀疏化的 Llama 2 70B 模型与基线稠密模型相比的推理加速比。

batch_size=896 的基准测试是[MLPerf Inference v4.0](https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/)的一部分。

配置：NVIDIA H100 80GB GPU。所有稀疏化模型使用 FP8，TP=1，PP=1。由于权重尺寸较大，稠密模型需要 TP=2。

| 批大小 | 推理加速比 (与 FP8 稠密模型相比) |
| :----: | :------------------------------: |
|   32   |              1.62x               |
|   64   |              1.52x               |
|  128   |              1.35x               |
|  896   |              1.30x               |

### 4.2 精度

我们建议结合微调使用稀疏性以避免精度下降。

下表展示了使用和不使用微调的 Llama 2 70B 稀疏化模型的验证损失对比。微调和验证均在 Open-Orca 数据集上进行。

|          方法           | 验证损失 (越低越好) |
| :---------------------: | :-----------------: |
|       FP8 (基线)        |        0.721        |
| FP8 + SparseGPT, 无微调 |        2.724        |
|  FP8 + 稀疏性, 有微调   |      **1.01**       |