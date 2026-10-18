# 基于 Context Folding 的长程 Coding Agent Harness

**围绕 Skill 经验蒸馏、长程上下文折叠与 Trace 评测，提升持续代码任务的执行能力。**

[English](README.md) · [架构](docs/architecture.md) · [Skill 机制](docs/skills.md) · [Context Folding](docs/context-folding.md) · [评测与证据](docs/evaluation.md)

项目在固定模型下保留跨阶段任务约束与执行证据，将任务反馈沉淀为可复用 Skill，支持长程开发中的持续执行与策略迭代。

## 核心机制

- **RSI 经验蒸馏与策略演进：** 通过执行 ID、测试指纹关联失败与修复轨迹，提炼包含适用条件、操作策略及验证规则的版本化 Skill；以增量合并、冻结回放与保留集晋级检查迭代策略，按需加载相关正文并校验来源哈希。
- **长程上下文折叠：** 复刻并调优 [ICML 2026 Context Folding](https://proceedings.mlr.press/v306/sun26x.html) 推理机制，结合分支隔离、事实检查点与绑定源码版本的历史观测恢复，在完整模型请求预算内保留任务约束和下一步操作。
- **Trace 与分层评测：** 通过 ContextVar 关联异步调用，以 SHA-256 内容寻址和原子落盘保存执行证据；将轨迹中的问题沉淀为 [7 类专项回归](tests/INDEX.md)，结合 API 回放与独立任务评分，指导 Skill 和折叠策略更新。

技术栈：**Python 3.12 / asyncio / 自研 Agent 运行时 / OpenAI 与 Anthropic 客户端 SDK / 类型化执行回执 / JSONL / SHA-256 / pytest / Git / Docker 适配器**。终端使用 Rich。

## 真实任务成绩

| 实验 | 已记录成绩 | 范围 |
|---|---|---|
| SCB `circuit_eval` 连续 8 阶段 | **527/566（93.1%）** 检查项通过 | 同一仓库跨阶段保留代码与对话，逐阶段执行独立官方检查 |
| SkillLearnBench 四类场景 | **12/16（75%）** 任务通过官方检查 | DBSCAN、地震板块计算、Offer 生成和排程，共 16 个测试实例 |

SCB 普通摘要参考配置为 **500/566（88.3%）**，与当前配置相差 **4.8 个百分点**。这是跨配置比较：当前／参考配置的 token 预算分别为 **15M／5M**，阶段响应上限和请求超时也有差异。[评测协议与阶段记录](docs/evaluation.md)

评测覆盖 SWE-bench Verified／Pro 代码修复、Terminal-Bench 终端操作、SCB 连续开发和 SkillLearnBench 技能迁移，通过 Trace 将任务结果关联到可复现的问题及候选策略评估。

[公开日志与评分回执](experiment/published/20261016/README.md) · [Skill 场景清单](benchmarks/cohorts/skilllearn-v1.json) · [公开方法对比](docs/skills.md#comparison-with-published-methods)

## 使用

```powershell
git clone https://github.com/Sanssssssssssssssss/long-horizon-coding-harness.git
cd long-horizon-coding-harness
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
.venv/Scripts/python.exe -m pip install --no-deps -e .
Copy-Item .env.example .env
# 配置一个模型提供方后启动：
.venv/Scripts/sans.exe --thinking
```

CLI 名称为 `sans`，支持 `/plan`、`/skills`、`/skill-candidates`、`/compact`、`/trace`、`/exit`。用 `sans --version` 查看安装版本，`--max-model-calls`、`--max-tool-calls` 和 `--request-timeout` 控制运行上限。[运行配置](docs/running.md)

CLI 使用原生 Agent；benchmark 入口选择折叠策略与任务评分器。[Skill 配置](docs/skills.md)说明候选暂存和晋级方式，[安全边界](SECURITY.md)说明工具权限与进程隔离。

## 仓库结构

源码放在 `agents/`，实验适配器在 `benchmarks/`，回归在 `tests/`，开发脚本在 `scripts/`，维护文档在 `docs/`，研究证据及运行产物统一在 `experiment/`。

[测试](TESTING.md) · [变更记录](CHANGELOG.md) · [发布流程](docs/releases.md) · [贡献](CONTRIBUTING.md) · [安全边界](SECURITY.md) · [来源与许可状态](SNAPSHOT_NOTICE.md)
