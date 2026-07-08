# PTQ 食谱和方案

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
本文档分两部分详细介绍了**PTQ量化方案**：
与模型无关的配方 [`general/ptq/`](general/ptq/) （推荐的）
（任何模型的起点），然后是
[model-specific recipes](#model-specific-recipes-huggingface-and-models) 在下面
`huggingface/` 和 `models/` — 将每一项与其总体基线进行比较，
解释其偏差原因。

---

## 通用配方

通用配方与模型无关。每个文件名都包含一个
**格式 + 范围**（量化哪些内容，以及量化为什么格式）
**KV缓存模式**，可选配**算法**（校准变体）：

```text
<formats-scope>-<kv-mode>[-<algorithm>].yaml
            nvfp4_experts_only - kv_fp8_cast
```

选择一种车身模型方案和一种键值缓存方案；提供的文件名为：
支持的组合。

---

### 运送的食谱

<details>
<summary>全部19 <code>一般/ptq/</code> 食谱（点击展开）</summary>

| 配方 | 模型主体 | KV缓存 | 校准 |
|--------|-----------|----------|-------------|
| `fp8_default-kv_fp8` | FP8 W8A8，所有线性 | FP8（已校准） | 最大值 |
| `fp8_default-kv_fp8_cast` | FP8 W8A8，所有线性 | FP8（恒定 amax） | 最大值 |
| `nvfp4_default-kv_fp8` | NVFP4 W4A4，所有线性 | FP8（已校准） | 最大值 |
| `nvfp4_default-kv_fp8_cast` | NVFP4 W4A4，所有线性 | FP8（恒定 amax） | 最大值 |
| `nvfp4_default-kv_nvfp4_cast` | NVFP4 W4A4，所有线性| NVFP4（恒定 amax）|最大|
| `nvfp4_default-kv_none-gptq` | NVFP4 W4A4（静态 W），全部线性 | 无 | GPTQ（逐层） |
| `nvfp4_mlp_only-kv_fp8` | NVFP4 W4A4、MLP + MoE 专家 | FP8（已校准） | 最大值 |
| `nvfp4_mlp_only-novit-kv_fp8` | NVFP4 W4A4，MLP + MoE 专家（不含 VL 视觉塔） | FP8（已校准） | 最大值 |
| `nvfp4_mlp_only-kv_fp8_cast` | NVFP4 W4A4、MLP + MoE 专家 | FP8（恒定 amax） | 最大值 |
| `nvfp4_mlp_only_mse-kv_fp8_cast` | NVFP4 W4A4、MLP + MoE 专家 | FP8（恒定 amax） | MSE + FP8 扫描 |
| `nvfp4_experts_only-kv_fp8` | NVFP4 W4A4，仅限 MoE 专家 | FP8（已校准） | 最大值 |
| `nvfp4_experts_only-kv_fp8_cast` | NVFP4 W4A4，仅限教育部专家 | FP8（恒定最大加速度） | 最大值 |
| `nvfp4_experts_only-kv_fp8_layerwise` | NVFP4 W4A4，仅限 MoE 专家 | FP8（已校准） | 最大，逐层 |
| `nvfp4_experts_only_mse-kv_fp8_cast` | NVFP4 W4A4，仅限 MoE 专家 | FP8（恒定 amax） | MSE + FP8 扫描 |
| `nvfp4_omlp_only-kv_fp8` | NVFP4 W4A4，o_proj + MLP/MoE | FP8（已校准） | 最大值 |
| `nvfp4_omlp_only-kv_fp8_cast` | NVFP4 W4A4，o_proj + MLP/MoE | FP8（常量 amax） | 最大值 |
| `nvfp4_weight_only-kv_fp16` | NVFP4 W4A16，仅权重 | 无 (BF16/FP16) | 最大 |
| `nvfp4_weight_only-kv_fp8_cast` | NVFP4 W4A16，仅权重 | FP8（恒定 amax） | 最大值 |
| `int4_blockwise_weight_only` | INT4 W4A16，第 128 块，仅权重 | 无 | 最大值 |

</details>

---

### 模型车身方案

机身设计是主要杠杆：它在精度和内存/吞吐量之间进行权衡。
通过选择**模型的哪些部分精度降低**以及**是否
激活值也进行了量化**（W4A4/W8A8 与仅权重 W4A16 相比）。

#### 全模型方案（将所有事物量化）

- **`fp8_default`** — **每个张量** FP8 E4M3 **W8A8** 在每个线性（注意力）
  q/k/v/o + MLP/MoE）——每个权重/激活张量对应一个尺度。最安全。
  激进方案：FP8 具有宽广的动态范围，因此精度损失通常
  可忽略不计。FP8 内核需要 Hopper+。目标平台的良好默认值。
  硬件是 FP8 级别的，你想要获得最大的加速。
- **`nvfp4_default`** — NVFP4（E2M1，block-16，FP8 块缩放）**W4A4** 在每个
  线性。最激进的方案——4 位权重*并且*所有位置都激活。
  — 适用于 Blackwell+ 的最大内存/吞吐量。准确率损失风险最高；
  如果出现倒退，则回退到以下指定范围的方案之一，而不是：
  放弃NVFP4。

#### 范围方案（量化模型的一部分）

- **`nvfp4_experts_only`** — NVFP4 W4A4 仅限教育部路由专家使用**
  （`*.experts.*`， `*block_sparse_moe*`）密集层、共享专家和
  注意 BF16。**最推荐的 MoE 模型 NVFP4 配方**：是
  这是最窄、精度保持性最好的NVFP4瞄准镜，因此它能恢复最多的图像。
  准确性——同时还能很好地压缩数据，因为路由专家通常是
  在模型总权重中占比最大。
- **`nvfp4_mlp_only`** — NVFP4 W4A4 应用于 **所有 MLP/FFN 计算**：密集 MLP 层，
  教育部召集了专家，并且 `block_sparse_moe` 阻塞。注意力集中在 BF16。
  **推荐用于密集模型**：大多数浮点运算/参数都位于多层感知器（MLP）中，因此
  赢得了大部分胜利，同时又没有触及敏感的注意力路径。
  为了准确起见。
- **`nvfp4_omlp_only`** — NVFP4 W4A4 在 **MLP/MoE 加上注意力输出
  投影**（`o_proj`但不是 q/k/v。介于两者之间的折中方案 `mlp_only`
  和 `default`：添加 o_proj GEMM（通常是安全的），而不进行量化。
  敏感的 q/k/v 投影。

> **范围与压缩。** 这些方案保持了对精度敏感的关注度。
> 以模型原始精度（大多数情况下为 BF16）计算路径（或整个密集路径）。
> 检查点——并且仅量化 FFN/专家权重，这些权重主导参数和
> 计算。这有利于提高精度，但被排除在外的层将保持未压缩状态。如何实现？
> 许多重要因素都**取决于模型**：在 MoE 模型中，路由专家是
> 绝大多数权重，因此在 BF16 中忽略其他因素几乎不会造成任何损失。
> 磁盘；在某些稠密模型中，注意力投射足够大，以至于它们
> 明显限制了检查点的大小。*如果*注意力权重很大，并且你
> 如果要压缩它们，我们建议添加一条 **FP8** 规则以引起注意
> 预测（将 NVFP4 保留在 MLP/专家中），而不是将其保留在 BF16 — FP8 中。
> 与 NVFP4 相比，它能以更高的精度保持敏感路径的安全性能，同时还能将那些
> 重量与 BF16 相比。

#### 仅重量方案（W4A16 — 激活保持 BF16）

仅量化权重；激活函数采用 BF16 精度。这会缩小模型尺寸。
（内存密集型解码获胜）准确率风险远低于 W4A4，并且**不需要
校准前向传递**。

这些通常推荐用于**低并发部署**——边缘和
设备端/客户端使用场景——工作负载受限于内存带宽；
降低权重是主要优势。对于高并发数据中心而言，
服务**，更倾向于采用能够量化激活的方案（W4A4/W8A8 主体）。
上述方案）：在大批量处理时，GEMM 会受到计算能力的限制，因此比特率较低。
激活函数和张量核心数学运算是实现吞吐量的关键。

- **`nvfp4_weight_only`** — NVFP4 权重，BF16 激活。节省内存
  4 位权重，无激活量化风险。
- **`int4_blockwise_weight_only`** — INT4 权重，块大小 128，BF16
  激活。经典的 W4A16 重量压缩；无需 NVFP4 类即可使用。
  硬件。

---

### KV缓存方案

这 `kv_*` 后缀控制注意力键值缓存的量化方式——独立
主体方案。量化键值缓存可以减少长时间上下文的内存占用。

- **`kv_fp8_cast`** — FP8 KV，**恒定 amax**（投射模式）：跳过 KV
  完全校准。生产成本更低，是千伏 (KV) 的安全默认值。对于大多数情况而言。
  模型与校准后的模型一样精确 `kv_fp8`** 下方，所以最好选择它
  除非您有特殊原因需要校准 KV 秤。Hopper+。
- **`kv_nvfp4_cast`** — NVFP4 KV 缓存，具有恒定的 amax。更激进的 KV
  压缩（4 位）；可与任何主体方案结合使用。Blackwell+。
- **`kv_fp8`** — FP8 E4M3 KV 缓存，具有**校准过的**每个张量 amax。KV
  在校准过程中测量秤的精度。料斗+。

> **`kv_fp8_cast` 对比 `kv_fp8`两者都产生 FP8 KV 缓存。 `_cast` 使用
> 采用固定比例尺，并跳过KV校准步骤（速度更快，无需额外数据）。
> 依赖）；普通 `kv_fp8` 根据数据校准刻度。铸造版本
> 通常与校准精度相匹配，所以从……开始 `kv_fp8_cast`。

---

### 校准变体

量化尺度的搜索方式。默认值（无后缀）是 `max`。

- **`max`**（默认）— amax/max 校准。快速，只需一次校准；
  基准选择。
- **`mse`**（例如） `nvfp4_mlp_only_mse`， `nvfp4_experts_only_mse`— MSE 搜索
  适用于**静态** NVFP4 称重秤，FP8 秤盘扫描范围覆盖 e4m3 秤盘。
  值。MSE 搜索适用于权重；激活值仍然是最大值。
  （最大）按照默认配方进行校准。需要更多校准时间，但
  恢复精度 NVFP4 W4A4 在普通最大值下可能会丢失。当需要时，请尝试使用它。
  `max` 配方退步了。
- **`gptq`**（`nvfp4_default-kv_none-gptq`）— GPTQ 逐层校准
  权重标尺；逐层写入检查点。GPTQ 最适合用于
  **INT4 仅权重**量化；其对 **NVFP4** 权重的有效性
  量化方法因模型而异——当其他因素影响量化结果时，它往往最有帮助。
  食谱显示出更大的准确率损失。将 GPTQ 应用于 **MoE** 模型仍然
  这是一个开放的研究课题，还需要对配方进行进一步调整。
- **`layerwise`**（`nvfp4_experts_only-kv_fp8_layerwise`) — 最大校准完成
  一次只解码一层解码器，以**降低峰值内存占用**；数值计算与……相同
  非分层变体。

当单个方法不足以满足需求时，这些方法也可以**堆叠**使用——例如 `mse` +
`gptq` 结合了 MSE 搜索的权重尺度和 GPTQ 的逐层更新。

---

### 选择通用食谱

1. **请根据您的硬件/目标设备选择合适的格式。** FP8（`fp8_default`) 在 Hopper+ 上；
   NVFP4（`nvfp4_*`) 在 Blackwell+ 上；仅重量（`*_weight_only`当你想的时候
   压缩时风险极低，或者缺少 NVFP4 类内核。
2. **从最精确的范围开始，然后逐步量化以适应你的
   内存/性能目标。** 对于**低并发**部署（边缘，
   （设备端/客户端），从**仅重量**配方开始（`nvfp4_weight_only` /
   `int4_blockwise_weight_only`——减轻体重是主要优势。
   高并发服务，从激活量化最窄的开始
   范围 - `nvfp4_experts_only` 教育部 `nvfp4_mlp_only` 对于密集区域——然后扩大
   （`mlp_only` → `omlp_only` → `default`仅限于你的内存/吞吐量
   目标要求，并随时检查准确性。
3. **在撤回瞄准镜之前，请先通过校准恢复精度。** 如果
   更广泛的配方倒退，切换其 `max` 到 `mse` 变体之前
   缩小范围。
4. 按部署方式选择关键值。 `kv_fp8_cast` 是安全的默认值（通常为
   经校准后准确无误 `kv_fp8`）; 使用 `kv_nvfp4_cast` 最大千伏
   压缩。

---

## 特定型号配方（`huggingface/` 和 `models/`）

以上通用配方与模型无关：它们通过通配符选择图层。
（`*mlp*`， `*self_attn*`， `*[kv]_bmm_quantizer`）并依靠共享资源
`default_disabled_quantizers` 排除项，因此同一个文件可以用于任何情况。
架构的模块名称遵循常规约定。仅供参考。
获得以下名额 `huggingface/<model_type>/` 或者 `models/<checkpoint>/` 当
模型必须**偏离**该基准线。偏差分为四种类型：

| 种类 | 与常规配方相比有哪些变化 | 示例 |
|------|-------------------------------------|----------|
| **架构感知型 `quant_cfg`** | 单个通配符方案无法表达的子模块格式选择 | `qwen3_5`， `qwen3_5_moe` |
| **算法覆盖** | 数值和范围相同，但*校准算法*已进行调整，因为默认算法出现故障或退步 | `gemma`， `mpt` |
| **额外排除项** | 添加禁用量化器模式，以便非语言分支保持完整精度 | `nemotron_vl`， `phi4mm` |
| **检查点镜像** | 精确还原已发布检查点的混合精度地图 | `models/Nemotron-3-Super-120B-A12B` |

数值和标准排除项仍然继承自以前。 `configs/`
尽可能地——模型文件夹*只*捕获增量。每个 `<task>/`
文件夹里装着一个 `README.md` 把那个三角洲拼出来。

### 架构感知 `quant_cfg` — `qwen3_5`， `qwen3_5_moe`

`huggingface/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast` （及其教育部姊妹篇，
它具有相同的 `quant_cfg` （片段）是一个混合方案，没有单一的通用方案
身体覆盖物**：NVFP4 **W4A16** 在 MLP / 专家投影权重上和 `lm_head`，
**FP8** 关于自我注意力和大型线性注意力投影
（`in_proj_qkv`， `in_proj_z`， `out_proj`），加上 FP8 KV 投屏。它还会禁用
参考配方中未包含的架构特定子模块
（`linear_attn.in_proj_a/b`， `conv1d`以及任何 `visual`/`mtp` 兄弟姐妹）。

*特别之处：*这些是混合的**线性注意力+softmax注意力**模型。
通用方案是为每个通配符类别应用一种格式；这种架构
对于大型线性注意力投影需要 FP8，但对于 MLP 权重需要 NVFP4，并且
需要保留线性注意力卷积/门子模块。密集型和 MoE
家族共享相同的通配符规则，因此一个代码片段可以驱动两者。

在 Qwen3.5 / Qwen3.6 上，此 W4A16 配方**通常不会导致精度下降**。
与官方检查站相比，它的设计旨在实现最佳性能
低并发用例**（仅对 MLP 使用权重可保持内存密集型解码）
路径快速（无需量化激活）。

打火机：**`step3p5/Step3.5-Flash/ptq/nvfp4-mlp-only`** 接近
`general/ptq/nvfp4_mlp_only` （NVFP4 在 MoE/MLP 权重+输入上，FP8 KV）但已固定
到已发布的检查点并携带特定于实例的禁用项
（`share_expert`， `moe.gate`（conv1d 分支）。

### 算法覆盖 — `gemma`， `mpt`

这些量化步骤与通用配方**相同**；只是
`quantize.algorithm` 模块有所不同，以规避特定模型的数值问题：

- **`gemma/ptq/w4a8_awq-kv_fp8_cast`**（INT4 块权重 + FP8 输入 + FP8 KV）
  演员）和**`mpt/ptq/w4a8_awq-kv_fp8_cast`** 使用 `awq_lite` 和 `alpha_step: 1`
  使用默认的 AWQ 搜索，而不是默认的 AWQ 搜索。默认搜索会导致 TRT-LLM 溢出。
  这些模型上的核函数；较粗的扫描可以避免这种情况，而且不会造成明显的损害。
  准确性。
- **`gemma/ptq/int8_sq-kv_fp8_cast`**（INT8 每通道权重 + INT8 输入 +
  FP8 KV cast) 设置 SmoothQuant `alpha: 0.5` 而不是默认值 `1.0` —
  Gemma 7B 回归 `1.0`， 和 `0.5` 恢复它。

*特殊之处：* 与通用方案的范围/数值相同，但通用
此处的默认算法会溢出或倒退。

### 额外排除项（多模式）— `nemotron_vl`， `phi4mm`

两者在数值上完全相同 `general/ptq/nvfp4_default-kv_fp8_cast`
（NVFP4 W4A4 + FP8 KV 铸造）。它们的特别之处在于其模型本地化。
`disabled_quantizers.yaml` 该单元*扩展*了标准除外条款，因此仅限
语言解码器是量化的：

- **`nemotron_vl`**（视觉语言，包括 Nemotron-Parse）添加
  `*vision*`， `*image*`， `*radio*`， `*visual*`， `*encoder*`， `*model_encoder*`。
- **`phi4mm`**（Phi-4-多模态）添加 `*speech*`， `*audio*`， `*image*`，
  `*vision*`。

*特殊之处：*一般的配方很容易就能量化视觉/听觉效果。
编码器，回归这些模态。额外的模式使它们保持完整。
精准度很高；其他方面都符合一般配方。

### 检查点镜像 — `models/Nemotron-3-Super-120B-A12B`

这 `models/` 层级复现**单个已发布检查点的**量化配置
逐字。 `super-nvfp4.yaml` 镜子
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` 没错——混合型**曼巴-MoE**
采用手工映射的、**逐组件**精度方案：

- 教育部路由专家 → NVFP4 W4A4， `group_size 16`静态体重秤
- 共享专家和曼巴 `in/out_proj` → FP8 每个张量
- KV缓存 → FP8
- 注意 q/k/v，MTP 头部， `lm_head`，潜在MoE，Mamba conv1d → **BF16**

*特别之处：*与任何通用配方不同，它**混合了FP8和NVFP4**
不同的组件类型**，并硬编码精确的已发布布局（匹配于
HF 和 Megatron-Core 模块名称）而不是便携式通配符方案。
`super-nvfp4.yaml` 使用 MSE 校准和 FP8 标度扫描（匹配
发布）; `super-nvfp4-max-calib.yaml` 平面下的图层地图是相同的。
`max` 校准数据，保留以供比较。

---

如需查看完整目录以及如何为特定型号选择起始配方，请参阅
[`README.md`](README.md)。
