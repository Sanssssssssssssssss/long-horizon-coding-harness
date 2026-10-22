"""Durable learning handoff, isolated resumption and explicit retries without model calls."""
import asyncio
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from agents import session
from agents.agent import Agent


def make_agent():
    agent = Agent(api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
                  custom_tools=[], is_sub_agent=True, skills_enabled=True)
    agent._session_persistence_enabled = True
    return agent


def set_window(agent, request="Preserve document headers."):
    agent._set_pending_skill_extraction_window(original_user_message=request,
        assistant_text="Verified document fields.", retrieved_reference=None)
    return agent._pending_skill_extraction_window


def test_last_window_survives_explicit_resume_without_model_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async def run():
        original, restored, unrelated = make_agent(), make_agent(), make_agent()
        try:
            window = set_window(original)
            window["execution_evidence"] = [{"id": "task:execution:24", "check_fingerprint": "check-v1"}]
            original._auto_save()
            restored.restore_session(session.load_session(original.session_id))
            assert restored._pending_skill_extraction_window == window
            assert restored.session_id == original.session_id
            assert unrelated._pending_skill_extraction_window is None
            assert not restored._background_skill_tasks
        finally:
            await asyncio.gather(*(agent.shutdown() for agent in (original, restored, unrelated)))
    asyncio.run(run())


def test_failed_extraction_is_retried_after_restart_with_same_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async def run():
        first, restored = make_agent(), make_agent()
        calls = []
        async def extract(window, **kwargs):
            calls.append(deepcopy(window))
            return {"ok": len(calls) > 1, "action": "failed" if len(calls) == 1 else "staged", "error": "provider unavailable"}
        monkeypatch.setattr(first, "_run_online_skill_evolution", extract)
        monkeypatch.setattr(restored, "_run_online_skill_evolution", extract)
        try:
            set_window(first)
            assert not (await first.extract_now("Verified on the saved document."))["ok"]
            saved = session.load_session(first.session_id)
            assert saved is not None
            restored.restore_session(saved)
            assert len(calls) == 1
            assert (await restored.extract_now())["ok"]
            assert calls[0] == calls[1]
            records = session.load_session(restored.session_id)["skillLearning"]["records"]
            assert len(records) == 1 and records[0]["status"] == "completed"
            assert records[0]["attempts"] == 2 and records[0]["failures"]
            assert restored._pending_skill_extraction_window is None
        finally:
            await first.shutdown()
            await restored.shutdown()
    asyncio.run(run())


def test_disk_failure_prevents_learning_request_and_keeps_evidence(tmp_path, monkeypatch):
    import agents.agent as runtime
    monkeypatch.chdir(tmp_path)
    async def run():
        agent = make_agent()
        calls = []
        async def extract(window, **kwargs):
            calls.append(window)
            return {"ok": True, "action": "staged"}
        monkeypatch.setattr(agent, "_run_online_skill_evolution", extract)
        try:
            previous = set_window(agent)
            with monkeypatch.context() as blocked:
                blocked.setattr(runtime, "save_session", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
                result = await agent.extract_now()
                assert not result["ok"] and not calls
                assert agent._pending_skill_extraction_window is previous
            assert (await agent.extract_now())["ok"]
            assert len(calls) == 1
        finally:
            await agent.shutdown()
    asyncio.run(run())


def test_clear_cancels_old_learning_without_losing_failed_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async def run():
        agent = make_agent()
        started = asyncio.Event()
        async def late_result(window, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return {"ok": True, "action": "staged"}
        monkeypatch.setattr(agent, "_run_online_skill_evolution", late_result)
        try:
            previous = set_window(agent)
            task = asyncio.create_task(agent.extract_now())
            await started.wait()
            agent.clear_history()
            current = set_window(agent, "Next independent task.")
            result = await asyncio.wait_for(task, timeout=1)
            assert not result["ok"]
            assert agent._pending_skill_extraction_window is current
            saved = session.load_session(agent.session_id)["skillLearning"]
            assert saved["pending"] == current
            record = next(item for item in saved["records"] if item["id"] == previous["window_id"])
            assert record["status"] == "failed" and record["failures"]
        finally:
            await agent.shutdown()
    asyncio.run(run())


def test_workspace_mismatch_rejects_learning_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async def run():
        first, restored = make_agent(), make_agent()
        try:
            set_window(first)
            first._auto_save()
            data = session.load_session(first.session_id)
            data["metadata"]["cwd"] = str(tmp_path / "other-project")
            with pytest.raises(ValueError, match="workspace"):
                restored.restore_session(data)
            assert restored._pending_skill_extraction_window is None
        finally:
            await first.shutdown()
            await restored.shutdown()
    asyncio.run(run())


def test_failed_atomic_session_replace_keeps_previous_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = {"metadata": {"id": "saved", "cwd": str(tmp_path)}, "generation": 1}
    session.save_session("saved", data)
    path = tmp_path / "experiment/runtime/sessions/saved.json"
    before = path.read_bytes()
    import os
    monkeypatch.setattr(os, "replace", lambda *args: (_ for _ in ()).throw(OSError("checkpoint locked")))
    with pytest.raises(OSError, match="checkpoint locked"):
        session.save_session("saved", {**data, "generation": 2})
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_background_failure_is_preserved_alongside_newer_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async def run():
        agent = make_agent()
        async def chat(text):
            agent._turn_output_buffer.append("Current task completed.")
        async def failed(window, **kwargs):
            return {"ok": False, "action": "failed", "error": "temporary timeout"}
        async def usage(*args):
            pass
        monkeypatch.setattr(agent, "_chat_openai", chat)
        monkeypatch.setattr(agent, "_augment_user_message_with_skill_context", lambda text: (text, None))
        monkeypatch.setattr(agent, "_run_online_skill_evolution", failed)
        monkeypatch.setattr(agent, "_run_skill_usage_tracking", usage)
        try:
            previous = set_window(agent)
            await agent._chat_impl("Now validate the footer.")
            await agent.drain_background_skill_tasks()
            data = session.load_session(agent.session_id)["skillLearning"]
            assert data["pending"]["latest_user"] == "Now validate the footer."
            record = next(item for item in data["records"] if item["id"] == previous["window_id"])
            assert record["status"] == "failed"
            assert record["window"]["next_user_feedback"] == "Now validate the footer."
        finally:
            await agent.shutdown()
    asyncio.run(run())


def test_completed_learning_retries_checkpoint_without_repeating_model(tmp_path, monkeypatch):
    import agents.agent as runtime
    monkeypatch.chdir(tmp_path)
    async def run():
        agent = make_agent()
        calls = []
        async def extract(window, **kwargs):
            calls.append(window)
            return {"ok": True, "action": "staged"}
        monkeypatch.setattr(agent, "_run_online_skill_evolution", extract)
        save = runtime.save_session
        writes = []
        def interrupted_save(*args):
            writes.append(args)
            if len(writes) == 2:
                raise OSError("checkpoint locked after model completed")
            return save(*args)
        monkeypatch.setattr(runtime, "save_session", interrupted_save)
        try:
            previous = set_window(agent)
            assert not (await agent.extract_now())["ok"]
            assert agent._pending_skill_extraction_window is previous
            assert (await agent.extract_now())["ok"]
            assert len(calls) == 1
            assert session.load_session(agent.session_id)["skillLearning"]["pending"] is None
        finally:
            await agent.shutdown()
    asyncio.run(run())


def test_clear_cancels_learning_before_background_coroutine_starts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async def run():
        agent = make_agent()
        called = []
        async def forbidden(*args, **kwargs):
            called.append(True)
            return {"ok": True, "action": "staged"}
        monkeypatch.setattr(agent, "_run_online_skill_evolution", forbidden)
        try:
            record = agent._queue_skill_learning(set_window(agent))
            agent._schedule_background_skill_task(agent._process_skill_learning(record))
            agent.clear_history()
            await agent.drain_background_skill_tasks()
            assert not called
            assert session.load_session(agent.session_id)["skillLearning"]["records"][0]["status"] == "pending"
        finally:
            await agent.shutdown()
    asyncio.run(run())


def test_three_process_learning_resume_and_retry(tmp_path):
    script = r'''
import asyncio,json,sys
from agents.agent import Agent
from agents.session import get_latest_session_id,load_session
async def main():
    phase=sys.argv[1]
    agent=Agent(api_base='http://unused.invalid',api_key='test',custom_system_prompt='fixed',
                custom_tools=[],is_sub_agent=True,skills_enabled=True)
    agent._session_persistence_enabled=True
    calls=[]
    async def extract(window,**kwargs):
        calls.append(window['window_id'])
        return {'ok':phase=='retry','action':'staged' if phase=='retry' else 'failed','error':'temporary provider failure'}
    agent._run_online_skill_evolution=extract
    try:
        if phase=='save':
            agent._set_pending_skill_extraction_window(original_user_message='Preserve headers and footers.',
                assistant_text='Verified the generated document.',retrieved_reference=None)
            agent._auto_save(strict=True)
        else:
            agent.restore_session(load_session(get_latest_session_id()))
            assert not calls
            result=await agent.extract_now()
            assert result['ok']==(phase=='retry')
            if phase=='retry':
                assert not (await agent.extract_now())['ok']
            assert len(calls)==1
        snapshot=load_session(agent.session_id)
        print('RECEIPT:'+json.dumps({'session_id':agent.session_id,'calls':calls,'learning':snapshot['skillLearning']}))
    finally:
        await agent.shutdown()
asyncio.run(main())
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONDONTWRITEBYTECODE": "1"}
    receipts = []
    for phase in ("save", "fail", "retry"):
        result = subprocess.run([sys.executable, "-B", "-c", script, phase], cwd=tmp_path, env=env,
                                text=True, encoding="utf-8", capture_output=True, timeout=40)
        assert result.returncode == 0, result.stdout + result.stderr
        receipts.append(json.loads(next(line[8:] for line in result.stdout.splitlines() if line.startswith("RECEIPT:"))))
    assert len({row["session_id"] for row in receipts}) == 1
    assert receipts[0]["calls"] == []
    assert receipts[1]["calls"] == receipts[2]["calls"]
    record = receipts[2]["learning"]["records"][0]
    assert record["status"] == "completed" and record["attempts"] == 2 and len(record["failures"]) == 1
    assert receipts[2]["learning"]["pending"] is None


@pytest.mark.parametrize("disabled", ["environment", "plan"])
def test_disabled_learning_keeps_only_latest_window_without_fake_failures(tmp_path, monkeypatch, disabled):
    monkeypatch.chdir(tmp_path)
    async def run():
        agent = make_agent()
        if disabled == "environment":
            monkeypatch.setenv("SANS_AUTO_SKILL_EVOLUTION", "0")
        else:
            agent.permission_mode = "plan"
        async def chat(text):
            agent._turn_output_buffer.append("Completed response.")
        monkeypatch.setattr(agent, "_chat_openai", chat)
        monkeypatch.setattr(agent, "_augment_user_message_with_skill_context", lambda text: (text, None))
        try:
            set_window(agent)
            for task in ("Second task.", "Third task."):
                await agent._chat_impl(task)
            await agent.drain_background_skill_tasks()
            result = await agent.extract_now()
            assert result["action"] == "disabled"
            snapshot = session.load_session(agent.session_id)["skillLearning"]
            assert snapshot["records"] == []
            assert snapshot["pending"]["latest_user"] == "Third task."
        finally:
            await agent.shutdown()
    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["side_query", "confirmation", "task_evaluator"])
def test_clear_rejects_late_results_before_learning_mutation(tmp_path, monkeypatch, boundary):
    import agents.online_skill_evolution as evolution
    monkeypatch.chdir(tmp_path)
    async def run():
        agent = make_agent()
        started = asyncio.Event()
        mutated = []
        async def delayed(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return {"completed": True}
        async def ingest(**kwargs):
            callback = kwargs[boundary if boundary != "confirmation" else "confirm_write"]
            await callback({})
            mutated.append(True)
            return {"ok": True, "action": "staged"}
        monkeypatch.setattr(evolution, "online_ingest", ingest)
        monkeypatch.setattr(agent, "_build_side_query", lambda **kwargs: delayed)
        monkeypatch.setattr(agent, "_confirm_online_skill_write", delayed)
        agent._skill_task_evaluator = delayed
        try:
            set_window(agent)
            task = asyncio.create_task(agent.extract_now())
            await started.wait()
            agent.clear_history()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not mutated
            assert session.load_session(agent.session_id)["skillLearning"]["records"][0]["status"] == "failed"
        finally:
            await agent.shutdown()
    asyncio.run(run())
