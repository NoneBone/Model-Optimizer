# 模型优化配方

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
此文件夹是 **ModelOpt 优化配方** 的库——声明式
描述完整模型优化工作流程（训练后）的 YAML 文件
量化、推测解码训练、扩散蒸馏）。

**目的：** 食谱是唯一受版本控制的、关于“如何”烹饪的权威来源。
模型经过优化——算法、逐层数值格式和校准——
以数据而非代码的形式表达。这使得优化运行结果可复现。
可进行差异比较，无需手动编写 Python 配置即可共享，并且允许进行调优。
可以通过名称查找配置。相同的 YAML 文件驱动 Python API。
（`load_recipe`），示例 CLI（`--recipe`），以及——对于以下预设
`configs/` — 内置的 `*_CFG` 常数。

配方由小型、可重复使用的结构单元组成。 `$import`
然后通过相对于此文件夹的路径加载系统，例如：

```python
# PTQ recipe -> mtq.quantize()
from modelopt.recipe import load_recipe
cfg = load_recipe("general/ptq/nvfp4_default-kv_fp8_cast")

# distillation recipe -> DMDConfig
from modelopt.torch.fastgen import load_dmd_config
cfg = load_dmd_config("general/distillation/dmd2_qwen_image")
```

或者从脚本/命令行标志中选择，例如 `hf_ptq.py --recipe`
huggingface/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast`。

> 📖 **PTQ配方调优必读 → [`ptq.md`](ptq.md)这是
> PTQ方案指南——身体范围（NVFP4/FP8，仅限专家/仅限MLP/
> 仅称重模式）、KV缓存模式和校准变体——附带具体指南
> 关于如何为您的模型和部署选择和调整配方。从这里开始。
> 在选择食谱之前。
>
> 此README文件是所有食谱系列的**目录**； `ptq.md` 是
> PTQ操作指南。

## 布局

| 目录 | 这里住着什么 |
|-----------|-----------------|
| `general/` | **与模型无关**的配方——适用于任何模型的良好起点。PTQ 组合、推测解码训练和蒸馏。 |
| `huggingface/<model_type>/` | 由HF提供的**特定型号**配方 `model_type`可以按已发布的检查点进行嵌套。如果您的模型有条目，请先使用这些检查点。
| `models/<model_name>/` | **实例特定的**配方，与特定已发布检查点的量化配置相对应。 |
| `configs/` | 共享构建模块（`numerics/`， `ptq/units/`， `ptq/presets/`）这些食谱由……组成 `$import`不要直接运行。

**选择查看位置：** 检查 `huggingface/<model_type>/` （然后任何嵌套的
`<checkpoint>/`首先针对您的模型；如果没有条目，则回退到
`general/`模型文件夹的存在表明存在推荐的、经过调整的配方。

---

## 通用配方

与模型无关的配方存在于以下环境中 `general/`对于 **PTQ**，配方是
格式、范围、KV缓存模式和校准的混合搭配——
**[`ptq.md`](ptq.md) 这是指南；阅读它以了解方案并选择。
一。**

其他通用食谱类别都记录在各自的文件夹中：
`general/speculative_decoding/` （EAGLE3 / DFlash 草稿头训练）
`general/distillation/` （例如扩散蒸馏，如DMD2）。

---

## `huggingface/` — 特定型号配方

每个生命都生活在各自的HF之下。 `model_type`模型文件夹的目的是捕获
**与通用预设的不同之处**——通常是算法调整或
非文本分支的禁用量化器模式。数值和标准
排除项仍然继承自以前 `configs/`浏览
[`huggingface/`](huggingface/) 可用的 `model_type`s；每个 `<task>/`
文件夹包含一个 `README.md` 描述确切的增量。见 [`ptq.md`](ptq.md) 为了
特定型号的配方与通用配方有何异同？为什么会有这些差异？

## `models/` — 检查点特定食谱

这些配置与单个已发布的检查点的量化配置完全一致——
一种针对特定版本进行调整的组件混合精度方案。浏览
[`models/`](models/) 针对可用的检查点。

---

## 添加食谱

- **适用于所有型号的全新组合** → 添加到 `general/ptq/` 通过组合现有
  `configs/` 单位；遵循 `<formats-scope>-<kv-mode>[-<algorithm>]` 我们的。
- **针对高频架构进行了优化** → `huggingface/<model_type>/<task>/`，带着一个
  `README.md` 记录与通用预设值的差异。验证其准确性。
  `model_type` 针对检查站的 `config.json` 放置之前。
- **镜像特定已发布的检查点** → `models/<model_name>/`。
- 通过以下方式共享可重复使用的尸体 `# modelopt-schema:`带标签的片段和 `$import`
  它；保持食谱包装纸薄。
