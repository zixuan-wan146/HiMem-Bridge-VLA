from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_session_cache_class():
    path = Path(__file__).resolve().parents[1] / "himem_bridge_vla" / "model" / "planner" / "session.py"
    spec = importlib.util.spec_from_file_location("coarse_plan_session_for_tests", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CoarsePlanSessionCache


def test_coarse_plan_session_reuses_until_expired():
    CoarsePlanSessionCache = _load_session_cache_class()
    cache = CoarsePlanSessionCache(max_age_steps=2)

    assert cache.should_refresh("episode")
    cache.put("episode", "plan-a")
    assert not cache.should_refresh("episode")
    assert cache.get("episode") == "plan-a"
    assert not cache.should_refresh("episode")
    assert cache.get("episode") == "plan-a"
    assert cache.should_refresh("episode")


def test_coarse_plan_session_refresh_request_overrides_age():
    CoarsePlanSessionCache = _load_session_cache_class()
    cache = CoarsePlanSessionCache(max_age_steps=5)

    cache.put("episode", "plan-a")

    assert cache.should_refresh("episode", refresh_requested=True)


def test_coarse_plan_session_reset_clears_one_key():
    CoarsePlanSessionCache = _load_session_cache_class()
    cache = CoarsePlanSessionCache(max_age_steps=2)

    cache.put("a", "plan-a")
    cache.put("b", "plan-b")
    cache.reset("a")

    assert cache.should_refresh("a")
    assert not cache.should_refresh("b")
