# ModelOpt 的 Agent 指令

这些指令适用于本仓库中由 AI 辅助的工作。

## 仓库导航

- 从 `README.md`开始了解项目概览和安装方式。

- 源代码位于 `modelopt/`，聚焦的测试覆盖在 `tests/`中，

  使用模式见 `examples/`或 `docs/`。

- **Agent 技能和共享配置位于 `.agents/`下**——这是规范的、

  与 Agent 无关的权威来源（`.agents/skills/<名称>/SKILL.md`、

  `.agents/scripts/`、`.agents/clusters.yaml.example`）。Claude Code 的

  `.claude/skills`、`.claude/scripts`和 `.claude/clusters.yaml.example`是指向

  `.agents/`的相对符号链接。始终编辑 `.agents/`下的文件，而不是符号链接路径。

  参见 `.agents/README.md`了解此约定。

## 编码指南

- **编码指南：** 代码开发和审查需要阅读并遵循

  [CONTRIBUTING.md 中的编码标准](#-coding-standards)；

  不要跳过此步骤。

- **在命令和文件引用中使用相对于仓库根目录的相对路径。**

## 迭代开发

- **运行测试：** 遵循

  [编写和运行测试](#-writing-and-running-tests)

  的说明。为了快速初始迭代，从 `tests/`中选择针对变更区域的聚焦测试。

- **运行 pre-commit：** 遵循

  [pre-commit 钩子说明](#pre-commit-hooks)。钩子可能会修改文件；

  在提交前审查并重新暂存这些更改。

- **签名提交：** 使用 `git commit -s -S -m "<消息>"`进行提交，以满足

  [签署您的工作](#-signing-your-work)的要求。

- **在当前轮次中，未经明确批准切勿 `git push`。** 本地提交没问题；但不要发布到远程。

- 在 `git commit`之后，等待用户说 "push"、"publish"、

  "ship" 或类似词语后，再运行 `git push`、`gh pr create`

  或任何推送选项标志如 `-o merge_request.create`。

## 贡献和 PR 就绪状态

- 在打开 PR 或将其标记为准备审查之前，请阅读

  [提交您的代码](#submitting-your-code)指南。

- 阅读 `.github/PULL_REQUEST_TEMPLATE.md`并满足清单中的要求。