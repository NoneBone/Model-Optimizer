# 面向卷积神经网络的量化感知训练（QAT）

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
NVIDIA ModelOpt 的量化感知训练 (QAT) 在训练过程中注入模拟量化噪声，以恢复因训练后量化 (PTQ) 而损失的精度。一个通过量化的 CNN 模型 `mtq.quantize()` 可以使用现有的训练循环进行微调。在 QAT 期间，量化器尺度会被冻结，而模型权重则会进行调整。

了解更多信息。 [ModelOpt QAT guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#quantization-aware-training-qat)。

> **_注意：_** 本示例使用 TorchVision ResNet-50 和 ImageNet 风格的数据集，但您可以将相同的步骤扩展到任何 CNN 和计算机视觉数据集。

## 系统要求

- GPU：≥1 个支持 CUDA 的 NVIDIA GPU
- 内存和性能：因型号、批处理大小和图像分辨率而异

## QAT 工作流程

1. 在目标任务上加载并评估您的全精度（FP32/FP16）模型。

1. 通过以下方式量化 FP32/FP16 模型

```python
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, calibrate_fn)
```

然后重新评估以建立量化基线。

1. 使用较小的学习率对量化模型进行微调，以恢复准确率。

> **_笔记：_**
>
> - 最优超参数（学习率、迭代次数等）取决于你的模型和数据。
> - 如果您已经有一个 PTQ 量化模型，您可以直接跳到步骤 3。

以下是使用 CNN 执行 QAT 的示例代码结构：

```python
from modelopt.torch.quantization import mtq
from modelopt.torch.opt import mto

# ... build model, loaders, optimizer, scheduler ...

def calibrate_fn(m):
    m.eval()
    seen = 0
    for x, _ in calib_loader:
        m(x.to(device))
        seen += x.size(0)
        if seen >= 512:
            break

# 1. PTQ quantization + calibration
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, calibrate_fn)

# 2. QAT fine-tuning
for epoch in range(1, epochs + 1):
    train(model, train_loader, ...)
    scheduler.step()

# 3. Save final QAT model (weights + quantizer state)
mto.save(model, "cnn_qat_best.pth")

# 4. To reload for inference or further training:
model = build_model()
mto.restore(model, "cnn_qat_best.pth")
model.to(device)
```

查看完整剧本 [torchvision_qat.py](./torchvision_qat.py) 对所有样板代码（参数解析、DDP 设置、日志记录等）进行处理。

> **_注意：_** 以上示例使用 [mto.save](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html#saving-modelopt-models) 和 [mto.restore](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html#restoring-modelopt-models) 用于保存和恢复 ModelOpt 修改后的模型。这些函数处理模型权重以及量化器状态。请参阅 [saving & restoring](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html) 了解更多信息。

### 端到端 QAT 示例

此文件夹包含一个端到端的可运行 QAT 流程，用于在 ImageNet 风格的数据集上使用 ResNet50 模型。 [torchvision_qat.py](./torchvision_qat.py) 脚本。

该脚本执行以下步骤：

- 从 TorchVision 加载预训练的 ResNet50 模型。
- 在验证集上评估其 FP32 准确率。
- 使用验证数据的校准子集执行 PTQ（默认 INT8 量化）。
- 评估PTQ模型的准确性。
- 执行指定次数的 QAT 训练。
- 在每个训练周期结束后评估 QAT 模型准确率，并保存性能最佳的模型。

以下是使用多 GPU QAT 的示例命令 `torchrun`：

```sh
torchrun --nproc_per_node <num_gpus> torchvision_qat.py \
    --train-data-path /path/to/your/imagenet/train \
    --val-data-path /path/to/your/imagenet/val \
    --batch-size 64 \
    --num-workers 8 \
    --epochs 5 \
    --lr 1e-4 \
    --print-freq 50 \
    --output-dir ./resnet50_qat_output
```

对于单GPU训练，您可以运行：

```sh
python torchvision_qat.py \
    --train-data-path /path/to/your/imagenet/train \
    --val-data-path /path/to/your/imagenet/val \
    --batch-size 64 \
    --num-workers 8 \
    --epochs 5 \
    --lr 1e-4 \
    --print-freq 50 \
    --output-dir ./resnet50_qat_output
    --gpu 0 # Specify the GPU ID
```

> **提示：** 对于单 GPU 运行，您还可以使用 `CUDA_VISIBLE_DEVICES` 使用环境变量控制 GPU 可见性。例如： `CUDA_VISIBLE_DEVICES=1 python torchvision_qat.py ... --gpu 0` 脚本将在物理 GPU 1 上运行，因为 PyTorch 会将其识别为 GPU 1。 `cuda:0`。

自定义旗帜— `--epochs`， `--lr`， `--batch-size`等等，以适应您的硬件和数据。您还可以使用 **ModelOpt** 提供的其他量化格式。有关支持的量化格式及其使用方法的更多详细信息，请参见下文：

```python
import modelopt.torch.quantization as mtq

# Learn about quantization formats and configs
help(mtq.config)
```

然后您可以修改 `quant_cfg` 在 `torchvision_qat.py` 因此。

> **_笔记：_**
>
> - 由于需要存储量化参数和可能不同的优化器状态，​​QAT 有时可能需要比全精度微调更多的内存。
> - 与其他模型训练一样，QAT 模型的准确率可以通过优化训练超参数（如学习率、训练持续时间、权重衰减以及优化器和调度器的选择）来进一步提高。

## 示例结果

| 模型阶段 | 准确率（前1名） |
|-----------------|------------------|
| FP32 ResNet50 | 约 76.1% |
| PTQ INT8 | ~75.5% |
| INT8层 | 约75.9% |

实际结果会因数据集、具体超参数和训练时长而异。通常情况下，您应该观察到：

- PTQ 准确率可能略低于 FP32 准确率。
- QAT 应该有助于恢复 PTQ 期间损失的部分或全部准确性，并且在某些情况下甚至有可能超过 FP32 基线，或者非常接近它。

## 使用 TensorRT 进行部署

QAT 后的最终模型，已保存 `mto.save()`该文件包含模型权重和量化元数据。导出 ONNX 格式后，该模型可以部署到 TensorRT 进行推理。该过程与上述过程大致类似。 [deploying an ONNX PTQ](../onnx_ptq/README.md#evaluate-the-quantized-onnx-model) 来自 ModelOpt 的模型。
