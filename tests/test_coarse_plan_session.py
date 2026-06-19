from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_plan_queue_class():
    path = Path(__file__).resolve().parents[1] / "himem_bridge_vla" / "model" / "planner" / "session.py"
    spec = importlib.util.spec_from_file_location("plan_token_queue_for_tests", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PlanTokenQueue


def test_plan_token_queue_consumes_by_cumulative_executed_steps():
    torch = __import__("torch")
    PlanTokenQueue = _load_plan_queue_class()
    queue = PlanTokenQueue(planning_horizon_steps=64, token_span_steps=8)
    plan = torch.arange(8).reshape(1, 8, 1)

    assert queue.should_refresh("episode", requested_execute_steps=16)
    queue.put("episode", plan)
    queue.record_executed_steps("episode", 4)
    assert queue.state("episode").consumed_tokens == 0
    queue.record_executed_steps("episode", 4)

    state = queue.state("episode")
    assert state.consumed_tokens == 1
    assert state.residual_steps == 0
    assert queue.active_plan_tokens("episode").reshape(-1).tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_plan_token_queue_refreshes_when_next_chunk_exceeds_horizon():
    torch = __import__("torch")
    PlanTokenQueue = _load_plan_queue_class()
    queue = PlanTokenQueue(planning_horizon_steps=64, token_span_steps=8)

    queue.put("episode", torch.zeros(1, 8, 2))
    queue.record_executed_steps("episode", 56)

    assert not queue.should_refresh("episode", requested_execute_steps=8)
    assert queue.should_refresh("episode", requested_execute_steps=9)


def test_plan_token_queue_refresh_request_overrides_remaining_suffix():
    torch = __import__("torch")
    PlanTokenQueue = _load_plan_queue_class()
    queue = PlanTokenQueue(planning_horizon_steps=64, token_span_steps=8)

    queue.put("episode", torch.zeros(1, 8, 2))

    assert queue.should_refresh("episode", refresh_requested=True, requested_execute_steps=1)


def test_plan_token_queue_reset_clears_one_key():
    torch = __import__("torch")
    PlanTokenQueue = _load_plan_queue_class()
    queue = PlanTokenQueue(planning_horizon_steps=64, token_span_steps=8)

    queue.put("a", torch.zeros(1, 8, 2))
    queue.put("b", torch.zeros(1, 8, 2))
    queue.reset("a")

    assert queue.should_refresh("a", requested_execute_steps=1)
    assert not queue.should_refresh("b", requested_execute_steps=1)
