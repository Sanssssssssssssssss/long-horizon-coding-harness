# Security and execution boundaries

This harness can edit files and execute shell commands with the account that runs it. Permission modes control runtime approval; they do not provide OS isolation. Run unfamiliar tasks in a dedicated checkout or prepared container with suitable host permissions.

Keep API keys, local MCP settings, cookies and credentials outside Git. `.env` and `.mcp.json` are ignored. CI uses offline fixtures and receives no model credentials. Inspect tool outputs and model request contexts before publishing a trace.

Report a suspected vulnerability privately to the repository owner through an existing private channel. Do not put credentials or private context into a GitHub issue. Include the source revision, trigger, affected boundary and a small reproduction.

Reviewed source provenance and the publication scope are documented in [SNAPSHOT_NOTICE.md](SNAPSHOT_NOTICE.md).
