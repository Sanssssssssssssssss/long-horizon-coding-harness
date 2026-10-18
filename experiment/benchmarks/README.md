# Benchmark 固定入口

新小题统一使用 `python -B -m benchmarks.local`，复用现有 Harbor 和 SansAgent。求解逻辑仍在 `agents/`，本次不改 harness。旧 frozen 实验保持原样，不再复制里面的 `run_live.py` 给新题使用。

## 目录

| 位置 | 内容 |
| --- | --- |
| `benchmarks/local.py` | `prepare` 与 `run` 两个入口，环境放行检查 |
| `experiment/benchmarks/<case>/` | 固定题面、版本、环境配方、隐藏测试、参考补丁；标准 Harbor 任务 |
| `experiment/benchmarks/python311/` | 共享轻量 Python 工具链配方 |
| `experiment/cache/benchmark/` | 已下载源码、依赖、有效环境验收回执；可重建 |
| `experiment/runs/<run>/` | 每次运行的配置、日志、trace、patch、验收报告及 token；禁止覆盖 |

Docker 镜像缓存在现有 WSL Docker 存储中。容器每次新建、结束删除；镜像继续复用。历史大镜像和冻结结果未删除。

## 使用

在 WSL Ubuntu、仓库目录下运行，解释器为 `/opt/harness-runtime-study/venv/bin/python`：

```bash
python -B -m benchmarks.local prepare \
  --case experiment/benchmarks/python-slugify-148 \
  --output experiment/runs/slugify148-prepare-NEW

python -B -m benchmarks.local run \
  --case experiment/benchmarks/python-slugify-148 \
  --output experiment/runs/slugify148-live-NEW
```

`run` 使用宿主机环境变量 `SANS_BENCHMARK_API_KEY`。沿用 `api.commandcode.ai/provider/v1`、`deepseek/deepseek-v4-flash`、thinking=true；没有增加调用次数限制。每个输出目录必须是新的。密钥不写入任务配置，宿主机代理通过 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量配置，不写死机器地址。

`prepare` 不调用模型。任务镜像已存在时不构建、不下载；缺少时只从已校验的本地源码归档和工具链镜像离线构建。随后用 Harbor 原生 `nop` 和 `oracle`，在关闭外网的全新容器里验收：

- Python、pytest、Git、依赖完整性和实际导入路径可用；Git 历史只有基线提交，源码树匹配固定版本。
- 原版现有测试全过；独立缺陷测试按预期失败，收集错误、跳过与异常不能算缺陷复现。
- 参考补丁通过全部独立测试，证明题目可评测。

`run` 必须具有上述 `READY` 回执。它先核对题面与原始 UTF-8 文本的固定哈希，再核对镜像 ID、全部题目文件 SHA256，以及预检证据文件；任一变化或证据缺失都会在调用模型前拒绝运行。正式容器禁止拉镜像、外网关闭，求解器只看到基线源码和公开题面；参考补丁仅上传给 oracle，隐藏测试在求解结束后才上传。

中断与失败也自动保存 token 回执；未返回用量的调用单列为未知。环境预检失败与解题结果分开保存。模型未修好是这次题目的结果；容器、依赖或 provider 异常属于不完整运行，不能算解题失败分数。

## 当前小题

`python-slugify-148` 是另一个公开缺陷，19 个文件，处理预置字符替换表中缺失的大写字符。来源：[公开 PR #148](https://github.com/un33k/python-slugify/pull/148)，固定源码 `26b81c2e224ebb65c7fba40d37d17d762be3782f`。测试来自上游合入版本；这是自建小题，不是 SWE-bench 官方成绩，也不是另一种项目规模的泛化验证。

环境校准：原版 81 项现有测试通过，独立 82 项测试中 1 项失败；参考补丁 82/82 通过。镜像 `harness-benchmark-slugify148:v2` 约 368 MB，取代这类小题此前使用的 4.45 GB 环境。

## 缓存缺失时如何准备

首次准备新项目/版本仍可能需要下载。这一步只发生在准备阶段，不在正式解题中临时安装。以下配方用于重建当前小题；已有镜像时不用执行。

1. 在 `experiment/cache/benchmark/` 缓存对应源码和依赖。源码可从公开仓库取得，但向镜像提供的是归档，不能把完整历史复制给求解器。**Windows 导出时必须关闭换行转换**：

   ```bash
   git -C /path/to/cached/python-slugify -c core.autocrlf=false archive \
     26b81c2e224ebb65c7fba40d37d17d762be3782f \
     --output=/mnt/e/GPTProject2/harness-runtime-study/experiment/cache/benchmark/python-slugify-148/source.tar
   ```

   输出路径应使用本仓库的绝对路径；与 `case.json` 的 `source_sha256` 匹配后才能构建。

2. 工具链基于已固定 digest 的 Python 3.11 镜像，安装 Git，使用缓存 wheel 安装 pytest 8.3.3、text-unidecode 1.3、setuptools 75.8.0、wheel 0.45.1 及其依赖。配方在 `python311/Dockerfile`；构建上下文为 `experiment/cache/benchmark/python311/`，里面只有 `wheels/`。准备阶段可传代理构建，运行阶段不需要容器联网。
3. 工具链重建后的 ID 可能改变。更新 `case.json` 中工具链 ID，明确升级任务镜像标签，再重新 `prepare`，不能沿用旧 READY。

Windows 本地 runner 回归若遇到 Git 深层路径错误，可仅给测试进程设置 `core.longpaths=true`；本次未修改全局 Git 设置。
