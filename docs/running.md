# Running and configuration

Use Python 3.12 and Git. Install `requirements.lock.txt` in an isolated virtual environment. `requirements.txt` describes the direct dependencies; the lock file records the tested environment.

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
| `/compact` | Request context compaction |
| `/trace` | Locate the current trace |
| `/exit` | Exit the session |

On Windows, invoke `.venv/Scripts/python.exe` directly. On Linux/macOS, use `.venv/bin/python`. VS Code tasks use `${workspaceFolder}` and the project interpreter.

Sessions live under `experiment/runtime/sessions/`; CLI traces and usage summaries live under `experiment/runs/cli/`. Automatic Skill evolution is off in the example configuration. Enabling it allows local Skill updates through the configured maintenance policy. Local user/project Skills are discovered by the runtime; inspect [skills.py](../agents/skills.py) for source precedence.

MCP is optional: copy `.mcp.example.json` to `.mcp.json` and configure the intended servers. Git, ripgrep and task-specific programs must be available to the selected tools. Container benchmarks require their own prepared images and grader dependencies.

Offline verification uses [TESTING.md](../TESTING.md). A source checkout does not automatically download benchmark workspaces or launch paid evaluations. Research runners live in `benchmarks/`; retained experiment support files are listed in `experiment/README.md`.

Tool permissions are application policies. Run untrusted tasks in a separately isolated operating-system/container environment appropriate to the workload.
