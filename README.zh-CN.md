# 面向持续代码长程的 Agent Harness

**固定模型下，研究程序性记忆与上下文折叠如何支持长程执行。**

[English](README.md) · [架构](docs/architecture.md) · [Skill 机制](docs/skills.md) · [Context Folding](docs/context-folding.md) · [评测与证据](docs/evaluation.md)

长任务会积累过期判断、反复读取和失去适用范围的验证结果。项目围绕两条主线组织运行时与实验：

- **程序性记忆：** 从用户反馈和执行回执提炼操作规则，将来源证据与规则正文分开持久化；局部合并、版本快照、源码哈希校验与有界正文加载支持经验复用。候选通过冻结样本划分和新鲜基线回放评估。
- **上下文折叠：** 保留任务更新和完整工具交换，分别管理模型摘要与执行事实；按完整请求成本决定压缩是否可接纳，提供可恢复观测和随源码变化失效的读取视图。研究策略进一步保留入口覆盖缺口与下一步编辑位置。

技术栈：**Python 3.12 / asyncio / OpenAI 与 Anthropic SDK / 类型化执行回执 / JSONL Trace / SHA-256 / pytest / Git / Docker 适配器**。终端使用 Rich。

| 实验 | 已记录成绩 | 范围 |
|---|---|---|
| SCB `circuit_eval` 连续 8 阶段 | 最佳记录 **561/566（99.1%）** 检查项通过 | 一个仓库持续开发，逐阶段执行官方检查 |
| SkillLearnBench 四类场景 | **12/16（75%）** 任务通过官方检查 | DBSCAN、地震板块计算、Offer 生成和排程，共 16 个测试实例 |
| 摘要截断恢复 | 原失败输入在 8192 上限下完整返回 5888 输出 tokens | 一次真实 API 回放及 35 项专项回归，独立于完整任务分数 |

表中成绩对应上述任务范围。[评测协议、阶段曲线与运行记录](docs/evaluation.md) · [可核对的公开日志](experiment/published/20261016/README.md)

[当前场景清单](benchmarks/cohorts/skilllearn-v1.json)保留 DBSCAN、地震板块计算、Offer 生成和排程；[完整 benchmark 预测](experiment/published/20261017-scope/README.md)列出按任务类别推算的估计与条件。

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

源码放在 `agents/`，实验适配器在 `benchmarks/`，回归在 `tests/`，维护文档在 `docs/`，研究证据及运行产物统一在 `experiment/`。默认 CLI 使用原生运行时；`ContinuationFoldingLoop` 是实验策略。候选晋级的执行评分依赖研究适配器。

[测试](TESTING.md) · [变更记录](CHANGELOG.md) · [发布流程](docs/releases.md) · [贡献](CONTRIBUTING.md) · [安全边界](SECURITY.md) · [来源与许可状态](SNAPSHOT_NOTICE.md)
