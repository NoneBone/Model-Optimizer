# 【已弃用】视觉语言模型的训练后量化（PTQ）

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
> **此示例已合并为 [`examples/hf_ptq`](../hf_ptq/README.md) 并且是
> 已弃用。** 它将在未来的版本中移除。VLM PTQ 现在共享同一个入口点。
> （`hf_ptq.py`）和 shell 脚本作为 LLM PTQ。

## 移民

使用 `hf_ptq` 脚本 `--vlm` 旗帜：

```bash
cd examples/hf_ptq
scripts/huggingface_example.sh --model <Hugging Face model card or checkpoint> --quant [fp8|nvfp4|int8_sq|int4_awq|w4a8_awq] --vlm
```

前 `examples/vlm_ptq/scripts/huggingface_example.sh` 入口点仍然有效：现在
打印弃用警告并转发到上面的命令。

## 事物的流动

| 主题 | 新地点 |
| :--- | :--- |
| 支持的 VLM / 支持矩阵 | [hf_ptq/README.md#hugging-face-supported-models](../hf_ptq/README.md#hugging-face-supported-models) |
| VLM 量化工作流程（`--vlm`）| [hf_ptq/README.md#vlm-quantization](../hf_ptq/README.md#vlm-quantization) |
| 图像-文本校准（`--calib_with_images`）| [hf_ptq/README.md#vlm-calibration-with-image-text-pairs-eg-nemotron-vl](../hf_ptq/README.md#vlm-calibration-with-image-text-pairs-eg-nemotron-vl) |
| Megatron-Bridge VLM PTQ | [examples/megatron_bridge/](../megatron_bridge/README.md) |

## 资源

- 📖 [Documentation](https://nvidia.github.io/Model-Optimizer)
- 💡 [Release Notes](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html)
