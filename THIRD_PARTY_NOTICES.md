# Sources and attribution

The runtime was adapted from a reviewed public snapshot; later work adds execution evidence, benchmark controls, context reliability, procedural memory and CLI changes. It is not presented as a runtime written entirely from scratch.

| Component | Source and boundary |
| --- | --- |
| Baseline runtime | [Public mirror](https://github.com/susan-zhou622/harness-demo), revision `335b293b9f91c9a24b0018bf31a15df79b205f91`; no license found in the reviewed snapshot |
| Windows compatibility baseline | [Recorded patch](https://github.com/y99805961-ctrl/BearCode/commit/015f561892db2c873c0627ec366446f3e0a0fc79) |
| Research fixture | python-slugify task/repository revisions are recorded in the experiment protocols; fixture copies stay local |
| Sans terminal character | Sans is from Toby Fox's Undertale; the terminal design references the [dialogue portrait](https://www.spriters-resource.com/pc_computer/undertale/asset/76325/) and does not imply endorsement |
| Dependencies | OpenAI and Anthropic SDKs, python-dotenv and Rich; installed versions are pinned in `requirements.lock.txt` and retain their own package notices |
| Research ideas | Papers and mature-agent references are linked from the research reports; their results are not attributed to this harness |
| SkillsBench task evidence | [SkillsBench](https://github.com/benchflow-ai/skillsbench), revision `5544ce7c2708c231099d974089e7c9e81e67ee7a`; original task environments stay local, published score metadata retains task identity |
| SkillEvolver training variant | [SkillEvolver](https://github.com/THU-AICosmos/skillevolver), revision `16f3cc6392a5bb296231fc352ed0654312caab4e`; historical dialogue-parser training work is attributed to its authors |
| SCB continuous development | [SCB problems](https://github.com/gabeorlanski/scb-problems), revision `9cd9ca3a51c3d3e2a99d2488a25baf73a2204451`; public receipts retain official case identities and original grading hashes |

The repository is publicly viewable under the scope described in the [snapshot notice](SNAPSHOT_NOTICE.md). No project-wide redistribution license has been added. Frozen historical records retain their original attribution and names.
