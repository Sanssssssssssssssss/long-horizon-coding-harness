from __future__ import annotations

from agents.prompt import SOLVER_SYSTEM_PROMPT
from agents.tools import get_solver_tool_definitions, tool_definitions
from benchmarks.live import B0_SYSTEM_PROMPT, LiveAgentAdapter


REPO_TOOLS = {
    "read_file",
    "write_file",
    "edit_file",
    "list_files",
    "grep_search",
    "run_shell",
}


def _names(profile: str) -> set[str]:
    return {tool["name"] for tool in get_solver_tool_definitions(profile)}


def test_solver_profiles_share_one_identity_and_narrow_tool_surface() -> None:
    assert B0_SYSTEM_PROMPT == SOLVER_SYSTEM_PROMPT
    assert "Solve only the supplied repository task" in SOLVER_SYSTEM_PROMPT
    assert "skill evolution" not in SOLVER_SYSTEM_PROMPT.lower()
    assert "subagent" not in SOLVER_SYSTEM_PROMPT.lower()

    assert _names("B0") == REPO_TOOLS | {"update_work_state", "read_observation"}
    assert _names("B1") == REPO_TOOLS | {"update_work_state", "read_observation", "skill"}
    assert "update_work_state" not in {tool["name"] for tool in tool_definitions}
    assert "update_plan" in {tool["name"] for tool in tool_definitions}

    state_tool = next(
        tool for tool in get_solver_tool_definitions("B0")
        if tool["name"] == "update_work_state"
    )
    properties = state_tool["input_schema"]["properties"]
    case_properties = properties["acceptance_cases"]["items"]["properties"]
    hypothesis_properties = properties["hypotheses"]["items"]["properties"]
    assert "status" not in case_properties
    assert hypothesis_properties["status"]["enum"] == ["active", "supported", "rejected"]


def test_live_solver_defaults_resolve_without_changing_explicit_overrides() -> None:
    common = {
        "api_base": "https://example.invalid/v1",
        "api_key": "secret",
        "model": "test-model",
    }
    for profile in ("B0", "B1"):
        adapter = LiveAgentAdapter(profile=profile, **common)
        config = adapter.benchmark_config()["live_agent"]
        assert config["system_prompt_mode"] == "solver"
        assert config["tool_schema_hash"]
        assert adapter._resolved_system_prompt() == SOLVER_SYSTEM_PROMPT
        assert {tool["name"] for tool in adapter._resolved_tools()} == _names(profile)

    custom_tools = ({"name": "read_file"},)
    custom = LiveAgentAdapter(
        profile="B1",
        custom_system_prompt="custom prompt",
        custom_tools=custom_tools,
        **common,
    )
    assert custom.custom_system_prompt == "custom prompt"
    assert custom.custom_tools == custom_tools
    assert custom.benchmark_config()["live_agent"]["system_prompt_mode"] == "fixed"
