# 使用 vLLM 为 fakequant 模型提供服务

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
这是一个简单的示例，用于演示如何在 vLLM 中校准和提供 ModelOpt fakequant 模型。

与 RealQuant 相比，FakeQuant 的速度慢 2-5 倍，但不需要专门的内核支持，并且有利于研究。

本示例已使用 vllm 0.9.0 和 0.19.1 版本进行测试。

## 准备环境

请按照以下说明构建 docker 环境，或使用 pip 安装 vllm。

```bash
docker build -f examples/vllm_serve/Dockerfile -t vllm-modelopt .
```

## 在 vLLM 中校准和部署虚假量化模型

步骤 1：配置量化设置。  
你可以编辑 `quant_config` 字典 `vllm_serve_fakequant.py`或者设置以下环境变量来控制量化行为：

| 变量                | 描述                                                         | 默认值        |
| ------------------- | ------------------------------------------------------------ | ------------- |
| QUANT_DATASET       | 用于校准的数据集名称                                         | cnn_dailymail |
| QUANT_CALIB_SIZE    | 用于校准的样本数量                                           | 512           |
| QUANT_CFG           | 量化配置                                                     | 无            |
| KV_QUANT_CFG        | KV缓存量化配置                                               | 无            |
| QUANT_FILE_PATH     | 导出的量化器状态字典的可选路径 `quantizer_state.pth`         | 无            |
| MODELOPT_STATE_PATH | 可选的导出路径 `vllm_fq_modelopt_state.pth` （恢复量化器状态和参数） | 无            |
| CALIB_BATCH_SIZE    | 校准批次大小                                                 | 1             |
| RECIPE_PATH         | ModelOpt PTQ 配方 YAML 的可选路径                            | 无            |

根据需要，在您的 shell 或 Docker 环境中设置这些变量以自定义校准。

步骤 2：运行以下命令，并启用所有支持的标志。 `vllm serve`：

```bash
python vllm_serve_fakequant.py <model_path> -tp 8 --host 0.0.0.0 --port 8000
```

对于公开的 vLLM 版本 `--moe-backend`该启动器默认设置为 `--moe-backend triton`。
ModelOpt 专家 fakequant 需要一个分解的 MoE 后端，以便在过程中两个专家 GEMM 都可见。
校准。

步骤 3：使用 curl 测试 API 服务器：

```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions"     -H "Content-Type: application/json"     -d '{
          "model": "<model_path>",
          "messages": [
              {"role": "user", "content": "Hi, what is your name"}
          ],
          "max_tokens": 8
        }'

```

步骤 4（可选）：使用 lm_eval 运行评估

```bash
lm_eval --model local-completions --tasks gsm8k --model_args model=<model_name>,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=128,tokenizer_backend=None
```

## 加载 QAT/PTQ 模型并在 vLLM 中运行（进行中）

步骤 1：导出包含 BF16 权重和量化器状态的模型。导出模型的方法如下：

- 对于 **HF** 型号，请使用 `examples/hf_ptq/hf_ptq.py` 和 `--vllm_fakequant_export`：

```bash
python ../hf_ptq/hf_ptq.py \
  --pyt_ckpt_path <MODEL_PATH> \
  --recipe <PATH_TO_RECIPE> \
  --calib_size 512 \
  --export_path <EXPORT_DIR> \
  --vllm_fakequant_export \
  --trust_remote_code
```

  这将产生 `<EXPORT_DIR>/vllm_fq_modelopt_state.pth` （用于 vLLM 伪量化重载的 ModelOpt 量化器状态）并将 HF 导出的模型保存到 `<EXPORT_DIR>` （config/tokenizer/weights）。

  笔记： `--pyt_ckpt_path` 可以指向 HF 检查点或 ModelOpt 保存的检查点（例如，由 QAT/QAD 生成的检查点）。 `examples/llm_qat/train.py`如果输入检查点已经量化，则脚本将**跳过重新量化**，仅导出用于 vLLM fakequant 重新加载的工件。

- 对于 **MCore** 模型，请使用标志导出模型。 `--export-vllm-fq` 如上文所述 [Megatron-LM README](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt#-nvfp4-quantization-qauntization-aware-training-and-model-export)这将生成 `quantizer_state.pth`其中包含用于通过 vLLM 重新加载的量化器张量 `QUANT_FILE_PATH`。

步骤 2：在提供服务时使用导出的工件：

- **HF导出**：传递导出的数据 `vllm_fq_modelopt_state.pth` 通过 `MODELOPT_STATE_PATH`

```bash
# HF
MODELOPT_STATE_PATH=<vllm_fq_modelopt_state.pth> python vllm_serve_fakequant.py <model_path> -tp 8 --host 0.0.0.0 --port 8000
```

- **MCore 导出**：传递导出的内容 `quantizer_state.pth` 通过 `QUANT_FILE_PATH` 并设置 `QUANT_CFG` 为了与 MCore 量化方案相匹配

```bash
# MCore
QUANT_CFG=<quant_cfg> QUANT_FILE_PATH=<quantizer_state.pth> python vllm_serve_fakequant.py <model_path> -tp 8 --host 0.0.0.0 --port 8000
```

## 在 vLLM 中应用稀疏注意力模型

在服务时应用 ModelOpt 稀疏注意力机制。启动器替换了 vLLM 的 `FlashAttentionImpl` 和 `ModelOptSparseAttentionImpl` 模型加载后，在每个注意力层上立即应用（支持分页KV缓存的Triton内核）。

配置信息是从检查点读取的。 `config.json` `sparse_attention_config` 该模块由 ModelOpt 的 HF 导出写入。启动器恢复校准后的 skip-softmax 元数据和 N:M sparse-softmax 元数据（`sparsity_n`， `sparsity_m`， `dense_sink_tokens`， `dense_recent_tokens`）。使用两个元数据条目导出的检查点使用 ModelOpt Triton 进行稀疏预填充启动；仅解码启动和没有活动稀疏工作的启动委托回 vLLM FlashAttention。

工作流程：

1. 使用以下方式校准并导出模型 `examples/llm_sparsity/attention_sparsity/hf_sa.py`这段文字 `sparse_attention_config` 导入到导出的检查点中 `config.json`。
2. 使用以下命令提供导出的检查点 `--enforce-eager` （CUDA 图捕获尚未通过稀疏注意力核的验证——请参阅已知问题）：

   ```bash
   python vllm_serve_sparse_attn.py <EXPORT_DIR> --enforce-eager -tp 8 --host 0.0.0.0 --port 8000
   ```

如果检查点没有 `sparse_attention_config`工作进程记录一条消息并继续传递——vLLM 保持不变。仅限量化交易的流程由……处理。 `vllm_serve_fakequant.py`; 稀疏 + 量化相结合的方法将在后续的 PR 中实现。

局限性：

- 通过将查询位置偏移到更长的 KV 跨度中，支持 vLLM V1 分块预填充和前缀缓存后缀关注。
- CUDA 图捕获尚未经过验证——使用 `--enforce-eager`。

## 已知问题

1. **MCore 重载不使用 `MODELOPT_STATE_PATH`**; 使用 `QUANT_FILE_PATH` 并确保 `QUANT_CFG` 与原始 MCore 模型使用的量化配方相匹配（否则量化器键/配置将无法对齐）。
2. MCore 目前还不支持 KV 缓存量化导出和重新加载。
3. **`NVFP4_KV_CFG` 和 `NVFP4_AFFINE_KV_CFG` 要求 `--enforce-eager`**；这些配置使用动态块 Triton 内核进行 KV 缓存量化，这与 CUDA 图捕获不兼容（内核网格由 Python 级张量形状计算得出，这些形状在捕获时被嵌入）。 `--enforce-eager`对于不同的批次大小，捕获的网格将不正确，从而产生不正确的输出。
