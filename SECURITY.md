# 安全

NVIDIA 致力于保护我们软件产品和服务的安全与信任，包括通过我们的组织管理的所有源代码仓库。

如果您需要报告安全问题，请使用下面列出的适当联系方式。**请不要通过 GitHub 报告安全漏洞。**

## 报告 NVIDIA 产品中的潜在安全漏洞

要报告任何 NVIDIA 产品中的潜在安全漏洞：

- 网页：[安全漏洞提交表单](https://www.nvidia.com/object/submit-security-vulnerability.html)

- 电子邮件：[psirt@nvidia.com](mailto:psirt@nvidia.com)

  -   我们鼓励您使用以下 PGP 密钥进行安全的电子邮件通信：[NVIDIA 公共 PGP 通信密钥](https://www.nvidia.com/en-us/security/pgp-key)

  -   请包含以下信息：

    -  存在漏洞的产品/驱动程序名称及版本/分支

    -  漏洞类型（代码执行、拒绝服务、缓冲区溢出等）

    -  复现漏洞的步骤说明

    -  概念验证或利用代码

    -  漏洞的潜在影响，包括攻击者如何利用该漏洞

虽然 NVIDIA 目前没有漏洞赏金计划，但我们在协调披露政策下处理外部报告的安全问题时，会予以致谢。请访问我们的[产品安全事件响应团队 (PSIRT)](https://www.nvidia.com/en-us/security/psirt-policies/)政策页面了解更多信息。

## NVIDIA 产品安全

有关所有安全相关问题，请访问 NVIDIA 的[产品安全门户](https://www.nvidia.com/en-us/security)。

------

## 安全考量

### 概述

NVIDIA Model Optimizer (ModelOpt) 是一个用于优化机器学习模型的库，可能会加载和处理用户提供的工件（模型、权重、配置、校准数据）及其依赖项。安全部署取决于您如何获取工件、验证输入以及加固运行 ModelOpt 的环境。

### 需要注意的事项

#### 不受信任的模型和数据输入

- 模型、权重、配置和数据可能是恶意的或被篡改的。

#### 反序列化和代码执行风险

- 不安全的反序列化可能导致在提供不受信任输入时执行任意代码。

- 避免使用可以反序列化任意对象的序列化格式/设置。

#### 输入验证和资源耗尽

- 大型或格式错误的输入可能触发崩溃或过度使用 CPU/GPU/内存。

- 缺少大小/类型检查会增加拒绝服务风险。

#### 传输中和静态数据

- 如果通过网络获取模型或依赖项，不安全的传输可能导致篡改。

- 存储的工件、日志和缓存可能包含敏感数据。

#### 日志记录和可观测性

- 日志可能无意中包含敏感输入、路径、令牌或专有模型细节。

- 过于详细的日志可能泄露操作和安全相关信息。

#### 供应链和第三方组件

- 依赖项可能包含已知漏洞或被攻破。

- 运行时加载的第三方插件/组件可能不具有相同的安全保障。

### 示例安全方法

#### 工件完整性

- 仅从可信来源加载工件。

- 优先使用签名工件；加载前验证签名。

#### 安全解析和反序列化

- 优先使用更安全的存储格式（避免对不受信任输入进行对象反序列化）。

- 避免使用 `pickle`、对不受信任权重使用 `torch.load()`或 YAML `unsafe_load`。

- 将任何未经验证的工件视为不受信任，并阻止/防护其加载。

#### 加固和最小权限

- 以最小权限运行并隔离工作负载。

#### 数据保护

- 加密静态敏感数据；传输中使用 TLS 1.3。

- 切勿硬编码或记录凭据。

#### 弹性

- 验证输入并强制执行限制（文件大小、超时、配额等）。

- 保持操作系统、容器和依赖项的补丁更新；扫描已知漏洞。

------

## 贡献者的安全编码实践

ModelOpt 处理来自各种来源的模型检查点和权重。贡献者必须避免可能引入安全漏洞的模式。这些规则适用于除测试外的所有代码。这些规则涵盖以下几个关键安全考量：

### 反序列化不受信任的数据

**除非提供记录的例外情况，否则不要使用 `torch.load(..., weights_only=False)`。** 它在底层使用 pickle，可以从恶意检查点执行任意代码。

```
# 不好——允许从检查点文件执行任意代码
state = torch.load(path, weights_only=False)

# 好
state = torch.load(path, weights_only=True, map_location="cpu")

# 仅在有内联注释解释为什么需要 weights_only=False
# 并确认文件是内部生成/可信的情况下才可接受
state = torch.load(
    path,
    weights_only=False,  # 加载的文件由 ModelOpt 内部生成，非用户提供
    map_location="cpu",
)
```

**除非提供记录的例外情况，否则不要使用 `numpy.load(..., allow_pickle=True)`。** 它在底层使用 pickle，可以从恶意检查点执行任意代码。

```
# 不好——允许从检查点文件执行任意代码
state = numpy.load(path, allow_pickle=True)

# 好 - 让调用者决定；默认为 False
def load_data(path: str, trust_data: bool = False):
    return numpy.load(path, allow_pickle=trust_data)
```

**不要使用 `yaml.load()`** — 始终使用 `yaml.safe_load()`。默认加载器可以执行嵌入在 YAML 中的任意 Python 对象。

### 使用 `trust_remote_code`加载 transformers 模型

**不要硬编码 `trust_remote_code=True`。** 此标志告诉 Transformers 执行随检查点附带的任意 Python，如果模型来源不受信任，这是一个远程代码执行向量。

```
# 不好——静默地让每个用户都执行远程代码
model = AutoModel.from_pretrained(name, trust_remote_code=True)

# 好——让调用者决定；默认为 False
def load_model(name: str, trust_remote_code: bool = False):
    return AutoModel.from_pretrained(name, trust_remote_code=trust_remote_code)
```

### 子进程和 shell 命令

**永远不要在使用字符串插值或用户提供输入时使用 `shell=True`。** 这是一个命令注入向量。

```
# 不好——如果 model_name 包含 shell 元字符，则存在命令注入
subprocess.run(f"python convert.py --model {model_name}", shell=True)

# 好——将参数作为列表传递
subprocess.run(["python", "convert.py", "--model", model_name])
```

### 其他应避免的模式

- **对外部输入派生的字符串使用 `eval()`/ `exec()`**。如果您必须动态生成和执行代码，请对照安全模式的白名单验证输入。

- **硬编码的秘密或凭据** — 切勿提交令牌、密码或 API 密钥。使用环境变量或列在 `.gitignore`中的配置文件。

### Bandit 安全检查

Bandit 用作预提交钩子，用于检查代码中的安全敏感模式。**不允许使用 `# nosec`注释**来绕过安全检查。

### 创建安全例外

如果确实需要安全敏感模式（例如 `pickle`、`subprocess`），贡献者必须：

1. **添加内联注释**解释*为什么*需要该模式以及*为什么*在此特定上下文中是安全的（例如"加载的文件由 ModelOpt 内部生成"）。

2. **请求 [@NVIDIA/modelopt-setup-codeowners](https://github.com/orgs/NVIDIA/teams/modelopt-setup-codeowners)的审核**并在拉取请求描述中包含明确的理由。