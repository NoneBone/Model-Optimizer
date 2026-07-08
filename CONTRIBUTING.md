# 贡献指南：Model Optimizer

感谢您有兴趣为 Model Optimizer（ModelOpt）做出贡献！

> [!NOTE]
>
> 对本仓库的任何贡献仅接受 Apache 2.0 许可协议。

## 🛠️ 搭建开发环境

确保 Model Optimizer（ModelOpt）以可编辑模式安装，并且所有 `dev`可选依赖也已安装：

```
pip install -e ".[dev]"
```

如果您正在开发需要 TensorRT-LLM 或 Megatron-Core 等依赖的功能，建议使用 Docker 容器简化环境搭建流程。更多信息请访问我们的[安装文档](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html)。

## 🧹 代码风格与格式化

- 所有代码（Python、C++、Markdown 等）在提交时会自动检查是否符合编码规范（详情见下文）。

- 各工具的详细信息请参见 [`.pre-commit-config.yaml`](https://yuanbao.tencent.com/chat/naQivTmsDa/.pre-commit-config.yaml)。

- 对于 VSCode 或 Cursor，我们提供了默认工作区设置，可将代码检查工具集成到您的 IDE 中：参见[工作区设置](https://yuanbao.tencent.com/chat/naQivTmsDa/.vscode/settings.json)。

### Pre-commit 钩子

启用 pre-commit 钩子可在提交前自动检查和修复代码质量问题：

```
pre-commit install
```

如需临时跳过检查进行提交，请在提交时使用 `-n`标志：

```
git commit -m "临时提交" -n
```

如需在不提交的情况下运行 pre-commit 钩子，请使用：

```
pre-commit run --all-files
```

## 📐 编码规范

ModelOpt 生产代码的指导原则：简洁、模块化和精炼。

### 基本原则

- **优先选择简单精准的改动。** 只改动任务所需的部分。避免推测性重构、大规模重写以及"顺手清理"。

- **追求简洁易读的设计。** 选择最容易理解和维护的设计方案。代码自上而下阅读：将高层行为放在前面，将底层细节隐藏在命名良好的辅助函数之后，并将复杂的分支视为重新考虑设计的信号。

- **优先采用模块化、可组合的方案。** 避免针对特定输入或特定情况的硬编码。当现有扩展点适用时优先使用。若无合适扩展点，则添加一个简单、专注的辅助函数、类或插件来清晰封装新行为。将范围限制在已知场景内。

- **尊重继承边界。** 父类抽象应定义共享契约和共享行为，而非子类特定的特殊情况。

- **避免重复，保持单一事实来源。** 当这样做能使设计更简洁时，使用共享辅助函数、API 或抽象来整合重复的逻辑或意图。避免可能因不同步而产生偏差的重复。

- **谨慎添加注释。** 注释应补充上下文，而非将代码翻译成英文。首先尽量让代码本身具有自解释性。仅在对非显而易见的意图或约束进行说明时使用注释。此指南仅适用于新注释；不要仅为风格而重写或删除现有注释。

- **为公共 API 编写文档。** 公开和高层 API 应有文档字符串，必要时包含示例。内部辅助函数通常应通过清晰的命名和结构实现自文档化。

- **修复 Bug 的根本原因，而非表面症状。** 修复 Bug 时，找到根本原因，而非修补其副作用。

- **对外部输入进行一次校验。** 在接口边界检查类型和值。内部代码可以信任这些检查，避免冗余断言。

- **移除死代码。** 删除未使用的导入、不可达的分支和过时的辅助函数。

- **将导入语句置于文件顶部。** 将所有导入放在源文件和测试文件的模块顶部，以便导入错误在模块加载时暴露，而非在运行时或特定测试中。仅在存在具体原因时才将导入放入函数内部：解决无法重构的循环导入、保护可选依赖（如 TensorRT-LLM、Megatron-Core），或在有明确理由时推迟异常沉重的导入。在这些情况下添加简短注释说明原因。

- **使用 `__all__`定义公共 API 并通过 `from .module import \*`重新导出。** 每个模块在文件顶部通过 `__all__ = [...]`声明其公共接口。包的 `__init__.py`文件通过 `from .module import *`重新导出子模块。这使得公共 API 在其定义处显式可见，避免了 `__init__.py`中手工维护的导入列表不同步的问题，并通过将星号导入限制在精心策划的 `__all__`名称中使其安全可靠。

### 高性能 AI 代码

- **将张量操作保持在 GPU 上，避免不必要的 CPU-GPU 同步。** 读取 `tensor.shape`等元数据没有问题。避免提取 Python 标量和使用诸如 `tensor.item()`、`float(tensor)`或 `min(tensor)`等操作符，因为它们可能触发 CPU-GPU 同步。默认使用 PyTorch 张量操作（如 `tensor.min()`），仅在 CPU 确实需要该值时再提取 Python 标量。基于张量值的 Python 分支也可能破坏 CUDA Graphs。

- **开发时考虑分布式处理。** 例如：尽可能使用 `print_rank_0`或 `warn_rank_0`以避免日志过多。对共享副作用（如文件写入或共享状态更新）进行防护，防止多 Rank 间的竞态条件。

### 兼容性

- **保持配置和检查点的向后兼容性。** ModelOpt 检查点包含序列化的 `ModeloptBaseConfig`实例（如 `QuantizeConfig`）。如果这些基于 Pydantic 的配置发生变更而未做向后兼容处理，旧检查点可能无法加载。请确保破坏性变更是明确且有意的。

## 添加新的 PIP 依赖

目前我们在两个地方管理 PIP 依赖：[pyproject.toml](https://yuanbao.tencent.com/chat/naQivTmsDa/pyproject.toml)用于 ModelOpt 库所需的依赖，以及 `examples/<example-name>/requirements.txt`用于特定示例所需的依赖。

如果在其中任一位置添加新的 PIP 依赖，请务必验证该依赖的许可证。如果其并非宽松许可证（如 MIT、Apache 2），您需要在 PR 中提供使用该依赖的理由，并与 `@NVIDIA/modelopt-setup-codeowners`确认是否允许。

## 🔒 安全编码实践

所有贡献者必须遵循 [SECURITY.md](https://yuanbao.tencent.com/chat/naQivTmsDa/SECURITY.md#security-coding-practices-for-contributors)页面中 *Security Coding Practices for Contributors*部分所述的安全编码实践。

任何安全敏感性的例外情况都需要 `@NVIDIA/modelopt-setup-codeowners`的审查和批准。

## 📋 从其他来源复制代码

使用第三方代码需获得开源审查委员会（OSRB）团队的授权，并遵循适当的代码贡献指南。

如果您是外部贡献者，请向 `@NVIDIA/modelopt-setup-codeowners`寻求下一步指导。对于内部贡献者，请遵循以下步骤：

- **更新 NVBug 以记录开源代码的使用详情：** 重新打开 NVBug 6046893，并在表格中添加您的用例。合并包含从宽松许可仓库（如 MIT、Apache 2）复制的代码的 PR 通常是可行的，但对于其他许可证，在合并 PR 之前有必要获得专家指导。

- **许可证头部格式：** 包含从其他第三方 GitHub 仓库复制代码的文件应按以下顺序排列：

  1. 指向代码来源的引用链接（包含提交哈希）。

  2. 原始仓库的版权 / 许可证。

  3. NVIDIA Apache 2.0 版权 / 许可证头部。

- **更新 `SPDX-License-Identifier`：** 如果第三方代码使用不同于 Apache 2.0 的许可证，请更新 NVIDIA 头部中的 `SPDX-License-Identifier`，使用 SPDX 表达式语法反映两种许可证。例如，对于 MIT 许可的源代码：

  ```
  # SPDX-License-Identifier: Apache-2.0 AND MIT
  ```

  如果第三方代码也是 Apache 2.0，则无需更改（`SPDX-License-Identifier: Apache-2.0`保持不变）。

- **更新 `LICENSE`：** 将第三方版权持有人添加到 [`LICENSE`](https://yuanbao.tencent.com/chat/naQivTmsDa/LICENSE)文件中相应的许可证章节（位于 *Third-Party Software Notices*下）。如果该第三方许可证尚未列出，请添加包含完整许可证文本的新章节。

- **从 license pre-commit 钩子中排除：** 将复制的文件从 license pre-commit 钩子中排除，以防止其自动在文件顶部添加 NVIDIA Apache 2.0 许可证。将文件路径添加到 [`.pre-commit-config.yaml`](https://yuanbao.tencent.com/chat/naQivTmsDa/.pre-commit-config.yaml)中 `insert-license`钩子的 `exclude`列表中。

正确的许可证头部格式示例请参见 [`modelopt/torch/quantization/utils/calib_utils.py`](https://yuanbao.tencent.com/chat/naQivTmsDa/modelopt/torch/quantization/utils/calib_utils.py)。

## 📝 编写和运行测试

我们使用 [pytest](https://docs.pytest.org/)进行所有测试。对于任何新功能 / 示例，请确保添加测试并且您的 PR 通过覆盖率检查。测试组织在以下目录中：

- `tests/unit`：核心 ModelOpt 库的快速基于 CPU 的单元测试。运行时间不应超过几秒钟。

- `tests/gpu`：核心 ModelOpt 库的快速基于 GPU 的单元测试。大多数情况下，运行时间不应超过几秒钟。

- `tests/gpu_megatron`：核心 ModelOpt 库中 Megatron-Core 功能的快速基于 GPU 的单元测试。大多数情况下，运行时间不应超过几秒钟。

- `tests/gpu_trtllm`：核心 ModelOpt 库中 TensorRT-LLM 功能的快速基于 GPU 的单元测试。大多数情况下，运行时间不应超过几秒钟。

- `tests/gpu_vllm`：核心 ModelOpt 库中 vLLM 功能的快速基于 GPU 的单元测试。大多数情况下，运行时间不应超过几秒钟。

- `tests/examples`：ModelOpt 示例的集成测试。运行时间不应超过几分钟。更多详情请参考[示例测试 README](https://yuanbao.tencent.com/chat/naQivTmsDa/tests/examples/README.md)。

对于轻量级本地验证，直接在相关测试路径上运行 `pytest`。例如：

```
pytest tests/unit/torch/quantization
```

对于更广泛的仓库验证和依赖设置，请使用 [noxfile.py](https://yuanbao.tencent.com/chat/naQivTmsDa/noxfile.py)。运行 `nox -l`列出可用会话，然后使用 `nox -s <session>`运行匹配的会话。`unit-3.12(torch_211, tf_latest)`会话使用特定的 Torch 和 Transformers 组合运行 `tests/unit`：

```
nox -s "unit-3.12(torch_211, tf_latest)"
```

### 测试设计原则

- **开发时编写聚焦的测试。** 在开发过程中，根据需要编写尽可能多的聚焦测试，包括底层单元测试或内部探测，以理解和固化行为。

- **精选生产测试并保持精简。** 在暂存或提交之前，决定哪些测试应该被检入。检入的测试应记录预期行为、防止回归或标记向后不兼容的行为变更。当更高层次的测试已经覆盖相同行为时，移除冗余的底层测试，保持 CI/CD 快速精简。

- **保持 `tests/unit`离线——不访问 HuggingFace Hub。** 单元测试必须是封闭的，以免因网络/超时问题而出现不稳定。不要使用 Hub ID 调用 `from_pretrained("<org>/<model>")`、`load_dataset("<hub-id>")`、`snapshot_download(...)`等。而是本地构建虚拟模型、分词器、配置和数据集——例如使用 `tests/_test_utils/`中的 `create_tiny_*`辅助函数和 `get_tiny_tokenizer()`，或使用 `datasets.Dataset.from_dict(...).to_parquet(...)`编写小型磁盘数据集目录。

- **遵守每项测试的超时时间。** `tests/conftest.py`按目录应用默认的每测试调用超时；对单个慢速测试使用 `@pytest.mark.timeout(<seconds>)`覆盖，并在该映射中注册任何新的顶层 `tests/<group>/`目录（否则会导致收集错误）。

## ✍️ 签署您的工作

- 我们要求所有贡献者对其提交进行"签名"。这证明该贡献是您的原创作品，或者您有权在相同许可证或兼容许可证下提交它。

- 您还需要使用 SSH/GPG 密钥对提交进行加密签名，该密钥不同于用于身份验证的密钥。更多详情请参见 [GitHub 文档](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)。请注意，设置 SSH 密钥比 GPG 密钥简单得多，因此建议按照以下步骤使用 SSH 签名密钥（需要 `git>=2.34`）。

  -   按照[本文档](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)中的步骤生成新的 SSH 密钥。例如：

    ```
    ssh-keygen -t ed25519 -f "${HOME}/.ssh/id_ed25519_git_signing" -P ""
    ```

  -   将公钥（`cat "${HOME}/.ssh/id_ed25519_git_signing.pub"`）作为新的 SSH 密钥上传到您的 [GitHub 设置](https://github.com/settings/ssh/new)，使用适当的标题并选择密钥类型为 `Signing Key`。

  -   配置本地 `git`使用新的 SSH 密钥进行提交签名：

    ```
    git config --global user.signingkey "${HOME}/.ssh/id_ed25519_git_signing.pub"
    git config --global gpg.format ssh
    git config --global commit.gpgsign true
    ```

- **任何包含未经签名的提交的贡献将不被接受。**

- 一旦您设置了 SSH/GPG 密钥，要签署提交只需在提交更改时使用 `--signoff --gpg-sign`（或 `-s -S`）选项：

  ```
  git commit -s -S -m "添加酷炫功能。"
  ```

  > *提示：要在 VSCode 中启用此功能，可以在 VSCode 设置（`Ctrl/Cmd + ,`）中启用 `git.alwaysSignOff`和 `git.enableCommitSigning`。*

  这将把以下内容附加到您的提交消息中：

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- 开发者原创证书（DCO）全文：

  ```
  Developer Certificate of Origin
    Version 1.1
  
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
  
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  
  
    Developer's Certificate of Origin 1.1
  
    By making a contribution to this project, I certify that:
  
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
  
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
  
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
  
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## 提交您的代码

- 提交 Pull Request，让自动分配的审阅者（基于 [CODEOWNERS](https://yuanbao.tencent.com/chat/naQivTmsDa/.github/CODEOWNERS)）审查您的 PR。

- 如果有任何 CI/CD 检查失败，请修复问题并再次推送。

- 一旦您的 PR 获得批准且所有检查通过，其中一位审阅者将合并该 PR。