# Running and configuration

Use Python 3.12 and Git. Install `requirements.lock.txt` in an isolated virtual environment. `requirements.txt` describes the direct dependencies; the lock file records the tested environment.

Install the runtime entrypoint with `python -m pip install --no-deps -e .` after installing the lock file. Run `sans --version` and `sans --help` using the same environment. On Windows without activation, use `.venv/Scripts/sans.exe`; on Linux use `.venv/bin/sans`.

Copy `.env.example` to `.env` and enable one provider configuration. The runtime supports OpenAI-compatible and Anthropic-compatible endpoints. Supply the model identifier accepted by your provider. `.env` and `.mcp.json` are ignored.

```text
python -B -m agents.main --help
python -B -m agents.main --thinking
```

| Command | Purpose |
|---|---|
| `/plan` | Enter the planning workflow |
| `/skills` | List discovered Skills |
| `/skill-stats` | Inspect recorded Skill usage |
| `/skill-candidates` | Inspect staged candidate identities and current activation state without API calls |
| `/compact` | Request context compaction |
| `/trace` | Locate the current trace |
| `/exit` | Exit the session |

On Windows, invoke `.venv/Scripts/python.exe` directly. On Linux/macOS, use `.venv/bin/python`. VS Code tasks use `${workspaceFolder}` and the project interpreter.

Sessions live under `experiment/runtime/sessions/`; CLI traces and usage summaries live under `experiment/runs/cli/`. Automatic Skill evolution is off in the example configuration. Enabling it allows extraction and staging; newly extracted candidates require a task evaluator before promotion. The standard CLI leaves them pending. Local user/project Skills are discovered by the runtime; inspect [skills.py](../agents/skills.py) for source precedence.

For bounded execution, use `sans --max-model-calls 40 --max-tool-calls 120 --max-output-tokens 8192 --request-timeout 120 "your task"`. The model-call limit applies to the solve loop; optional memory, learning and evaluation side queries have their own accounting. `/skill-eval` invokes response replay and may use the model API; it does not replace task-level candidate evaluation.

Exit codes are 0 for normal completion, 1 for a runtime error, 2 for an incomplete/budget-limited run, and 130 for interruption. Incomplete provider responses do not authorize tool execution or successful completion. Output truncation permits at most two recovery attempts per Agent instance, including resumed phases on that instance; original provider trace payloads and measured usage remain recorded.

MCP is optional: copy `.mcp.example.json` to `.mcp.json` and configure the intended servers. Git, ripgrep and task-specific programs must be available to the selected tools. Container benchmarks require their own prepared images and grader dependencies.

Offline verification uses [TESTING.md](../TESTING.md). A source checkout does not automatically download benchmark workspaces or launch paid evaluations. Research runners live in `benchmarks/`; historical comparison fixtures are in `experiment/fixtures/`. Legacy research commands can require local frozen task data. Active Skill task selection is recorded in `benchmarks/cohorts/skilllearn-v1.json`; `python -B scripts/skill_cohort.py` summarizes its recorded evidence without model calls.

Tool permissions are application policies. Run untrusted tasks in a separately isolated operating-system/container environment appropriate to the workload.
