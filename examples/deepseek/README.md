# 将 Deepseek 模型量化为 FP4

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
本示例将演示将 DeepSeek 模型量化为 FP4 并导出可与 TRT-LLM 一起部署的统一检查点的步骤。

## 设置

由于模型大小，目前量化 FP8 模型需要 8xH200 或 16xH100，我们将以 8xH200 为例。

## 目录布局

- `deepseek_v3/`DeepSeek V3、R1、V3.1 和 V3.2 FP4 量化。
- `deepseek_v4/`DeepSeek V4 路由专家 NVFP4 量化。

## DeepSeek V3 FP4

### 转换 DeepSeek FP8 推理的 HF 检查点

```bash
# set up variables to run the example
export HF_FP8_CKPT={path_to_downloaded_hf_checkpoint}
export DS_CKPT={path_to_save_converted_checkpoint}
export FP4_QUANT_PATH={path_to_save_quantization_results}
export HF_FP4_PATH={path_to_save_the_final_FP4_checkpoint}
```

### DeepSeek V3 R1 V3.1

```bash
# download the FP8 checkpoint from Hugginface. This is an example of DeepSeek-R1
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir $HF_FP8_CKPT

# clone DeepSeek-V3 (base model of R1) Github repository for FP8 inference,
git clone https://github.com/deepseek-ai/DeepSeek-V3.git && cd DeepSeek-V3 && git checkout 9b4e978
```

### 【实验版】DeepSeek V3.2

```bash
# download the FP8 checkpoint from Hugginface.
huggingface-cli download deepseek-ai/DeepSeek-V3.2-Exp --local-dir $HF_FP8_CKPT

# clone DeepSeek-V3.2 Github repository for FP8 inference,
git clone https://github.com/deepseek-ai/DeepSeek-V3.2-Exp.git && cd DeepSeek-V3.2-Exp && git checkout 87e509a

# Install requirements
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git
pip install -r inference/requirements.txt
```

### 转换检查点

```bash
# convert the HF checkpoint to a specific format for Deepseek
python inference/convert.py --hf-ckpt-path $HF_FP8_CKPT --save-path $DS_CKPT --n-experts 256 --model-parallel 8
```

## 训练后量化

### 运行校准脚本

DeepSeek V3、R1、V3.1

```bash
torchrun --nproc-per-node 8 --master_port=12346 deepseek_v3/ptq.py --model_path $DS_CKPT --config DeepSeek-V3/inference/configs/config_671B.json --quant_cfg NVFP4_DEFAULT_CFG --output_path $FP4_QUANT_PATH
```

DeepSeek V3.2

```bash
torchrun --nproc-per-node 8 --master_port=12346 deepseek_v3/ptq.py --model_path $DS_CKPT --config DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json --quant_cfg NVFP4_DEFAULT_CFG --output_path $FP4_QUANT_PATH
```

#### 教育部专家校准

默认情况下，校准使用模型的原生 top-k 路由，然后运行
校准后同步，设置每位专家的 `input_quantizer.amax` （w1/w2/w3）
到每层全局对等最大值（跨 EP 等级全部减少）。
`weight_quantizer.amax` 始终以专家为单位；任何未经校准的专家都只能退而求其次。
一条基于去量化FP8权重的计算路径。这与……相呼应
`layer_sync_moe_local_experts_amax` mtq 自动运行的流程
QuantSequentialMLP 衍生的 MoE。

要恢复原始行为——强制每个令牌都经过每个专家。
校准期间（速度较慢，正向扫描速度约为原来的 2 倍，无校准后同步）——通过
`--calib_all_experts`：

```bash
torchrun --nproc-per-node 8 --master_port=12346 deepseek_v3/ptq.py --model_path $DS_CKPT --config DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json --quant_cfg NVFP4_DEFAULT_CFG --output_path $FP4_QUANT_PATH --calib_all_experts
```

每个张量量化器的摘要都会被写入到 `$FP4_QUANT_PATH/.quant_summary.txt`。

### 将 FP8 高频检查点量化为 FP4

我们提供一个一步到位的脚本，它将：

- 将权重量化为 NVFP4
- 将杂项文件复制到量化检查点

```bash
./deepseek_v3/quantize_fp8_to_nvfp4.sh --amax_path $FP4_QUANT_PATH --fp4_output_path $HF_FP4_PATH --fp8_hf_path $HF_FP8_CKPT --world_size 8
```

## DeepSeek V4 路由专家 NVFP4

DeepSeek V4 采用混合原生检查点布局。V4 版本对检查点进行了量化。
只有被路由到 NVFP4 W4A4 的专家才会关注预测，
路由器门、共享专家、嵌入以及 LM 头在其原始状态下
格式。

### 准备 MP 检查点

在使用 DeepSeek 的 MXFP4 进行重新分片时，请保留专家级配置。 `convert.py`：

```bash
export DS_V4=/path/to/DeepSeek-V4-Pro
export MP=8
export MP_CKPT=/path/to/DeepSeek-V4-Pro-mp${MP}-mxfp4
export AMAX=/path/to/amax-nvfp4-experts
export HF_NVFP4_PATH=/path/to/DeepSeek-V4-Pro-nvfp4-experts

python ${DS_V4}/inference/convert.py \
    --hf-ckpt-path ${DS_V4} \
    --save-path ${MP_CKPT} \
    --n-experts 384 \
    --model-parallel ${MP}
```

### 校准路由专家

单节点：

```bash
torchrun --nproc-per-node ${MP} --master_port 12346 deepseek_v4/ptq.py \
    --model_path ${MP_CKPT} \
    --config ${DS_V4}/inference/config.json \
    --dsv4_inference_dir ${DS_V4}/inference \
    --output_path ${AMAX}
```

两个配备 4 个 GPU 的节点 `MP=8`：

```bash
# node 0
torchrun --nnodes=2 --node_rank=0 --master_addr=<ip> --master_port=12346 \
    --nproc-per-node 4 deepseek_v4/ptq.py \
    --model_path ${MP_CKPT} \
    --config ${DS_V4}/inference/config.json \
    --dsv4_inference_dir ${DS_V4}/inference \
    --output_path ${AMAX}

# node 1
torchrun --nnodes=2 --node_rank=1 --master_addr=<ip> --master_port=12346 \
    --nproc-per-node 4 deepseek_v4/ptq.py \
    --model_path ${MP_CKPT} \
    --config ${DS_V4}/inference/config.json \
    --dsv4_inference_dir ${DS_V4}/inference \
    --output_path ${AMAX}
```

### 导出回 HF 分片布局

`deepseek_v4/quantize_to_nvfp4.py` 在原有的HF型V4检查点上运行，
生成一个新的 HF 风格检查点，其中路由的专家权重被替换为
NVFP4 张量加 `weight_scale`， `weight_scale_2`， 和 `input_scale`。

```bash
python deepseek_v4/quantize_to_nvfp4.py \
    --amax_path ${AMAX} \
    --source_ckpt ${DS_V4} \
    --output_ckpt ${HF_NVFP4_PATH}
```

输出结果包含更新后的内容 `model.safetensors.index.json`，一个 `config.json`
和 `quantization_config.moe_quant_algo = "NVFP4"`， 和 `hf_quant_config.json`
描述混合的 NVFP4 专家层。

当源路由专家为 MXFP4 时（如 V4 版本中），添加
`--cast_mxfp4_to_nvfp4` 为了实现无损重量转换——建议使用……
默认的有损去量化/再量化路径。请参见下文。

#### 无损 MXFP4 → NVFP4 加权转换 (`--cast_mxfp4_to_nvfp4`）

源检查点中的路由专家已经是 MXFP4（E2M1 nibbles +
每个 32 元素块采用 2 的幂次方 E8M0 比例）。如果没有该标志，则导出
使用校准后的BF16对其进行去定量，然后重新定量为NVFP4。
每个张量权重 amax，它从数据中重新导出每个块的尺度，
因此，它是有损的。 `--cast_mxfp4_to_nvfp4`每个张量 `scale_2` 是
固定到 `2^(k_max - 8)` 每个模块的 E4M3 比例 `2^(k_j - m)` 直的
来自 E8M0 标尺的来源，所以 `per_block_scale * scale_2 = 2^k_j` 以及 NVFP4
半字节与源 MXFP4 半字节逐位相等（对于每个块， `k_j`
落在 E4M3 的可表示窗口中；罕见的超出范围的块回退到 a
（数据衍生尺度）。该标志仅影响路由专家**权重**——激活
`input_scale` 仍然来自 `${AMAX}` 校准——运行结果会打印出来
`[cast] lossless MXFP4->NVFP4 blocks: …` 摘要。这与 GPTOSS 的演员阵容相呼应。
[`examples/hf_ptq/cast_mxfp4_to_nvfp4.py`](../hf_ptq/cast_mxfp4_to_nvfp4.py); 这
V4 的变化在于 w1/w3 共享一个 `scale_2` （融合的GEMM1），所以 `k_max` 被接管
两种预测。
