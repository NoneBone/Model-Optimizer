# 实验优化技术

 en [English](./README_en.md) ｜ zh_CN [简体中文](./README.md)
 
正在积极开发实验性优化算法和研究原型。

## 目的

对于以下新的优化技术（量化、剪枝、稀疏性等）：

- 新型或研究阶段算法
- 尚未达到量产标准。
- 可能存在不稳定的API。

**⚠️警告**：实验性功能无法保证在不同版本中都能正常运行。API 可能会更改，功能可能会被移除，恕不另行通知。使用风险自负。

## 要求

每项实验技术必须包括：

- **README.md** - 解释了该技术的功能、使用方法、当前状态、模型支持和参考资料。
- **可运行的代码** - 清晰易读的实现
- **全面测试** - 良好的测试覆盖率证明正确性
- **详细文档** - 清晰的文档，涵盖用法、API 和行为说明。
- **示例** - 用法演示
- **模型支持列表** - 支持哪些模型/框架
- **部署信息** - 支持的部署框架（TensorRT-LLM、vLLM、SGLang 等）以及是否需要自定义内核
- **requirements.txt** - 除基础模型之外的其他依赖项
- **许可证头文件** - 所有 Python 文件都包含 Apache 2.0 许可证头文件

## 示例结构

按逻辑顺序组织代码。以下是一些示例：

**简单的扁平结构：**

```text
experimental/my_technique/
├── README.md
├── requirements.txt
├── my_technique.py
├── test_my_technique.py
└── example.py
```

**包结构：**

```text
experimental/my_technique/
├── README.md
├── requirements.txt
├── my_technique/
│   ├── __init__.py
│   ├── core.py
│   └── config.py
├── tests/
│   └── test_core.py
└── examples/
    └── example_usage.py
```

## 质量标准

实验代码必须符合质量标准：

- 需要全面的测试覆盖率
- 需要清晰的文件。
- 通过所有提交前检查

## 公关指南

保持 PR 内容简洁明了且易于审核：

- **拆分大型功能**：如有需要，将复杂技术拆分为多个 PR。
- **合理的范围**：包含数万行代码的 PR 难以审核。
- **增量式开发**：考虑先提交核心功能，然后再进行增强。
- 如果你的技术规模很大，请先在一份问题报告中讨论实施计划。

## 示例文档模板

您的技术文档中的 README.md 文件应包含以下内容：

```markdown
# Your Technique Name

Brief description of the optimization technique.

## Model Support

| Model/Framework | Supported | Notes |
|-----------------|-----------|-------|
| LLMs (Llama, GPT, etc.) | ✅ | Tested on Llama 3.1 |
| Diffusion Models | ❌ | Not yet supported |
| Vision Models | ✅ | Experimental |

## Deployment

| Framework | Supported | Notes |
|-----------|-----------|-------|
| TensorRT-LLM | ✅ | Requires custom kernel |
| vLLM | ❌ | Not yet supported |
| SGLang | ✅ | Uses standard ops |

## Usage

\`\`\`python
from experimental.my_technique import my_optimize
...
\`\`\`

## Status

Current state: Prototype

Known issues:
- Issue 1
- Issue 2

## References

- [Paper](link)
- [Code repository](link)
- [Project page](link)
- [Related work](link)
```

## 生产路径

当一项技术准备好投入生产环境（经过验证有效、API稳定、测试完整、文档详尽）时，就可以将其提升为主要技术。 `modelopt` 包裹。

**贡献者**：提出毕业提案，并提供有效性和稳定性的证据。

用户：如果您发现某个实验性功能很有价值，请在 GitHub 上提交 issue，请求将其升级到正式版。用户需求是衡量产品是否已准备好上线的重要指标。

## 问题？

在 GitHub 上创建一个 issue，内容如下： `[experimental]` 前缀。
