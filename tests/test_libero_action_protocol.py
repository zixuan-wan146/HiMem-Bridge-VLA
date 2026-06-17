from __future__ import annotations

import json

import pytest

from evaluations.libero.libero_action_protocol import (
    parse_action_response,
    parse_action_response_with_metadata,
    select_actions_for_transition_policy,
    to_libero_action,
)


def test_parse_action_response_returns_horizon_prefix():
    message = json.dumps(
        [
            [0, 1, 2, 3, 4, 5, 0.6, 7],
            [8, 9, 10, 11, 12, 13, 0.4, 15],
            [16, 17, 18, 19, 20, 21, 0.2, 23],
        ]
    )

    actions = parse_action_response(message, horizon=2)

    assert actions == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.6, 7.0],
        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 0.4, 15.0],
    ]


def test_parse_action_response_accepts_debug_payload():
    message = json.dumps(
        {
            "actions": [[0, 1, 2, 3, 4, 5, 0.6]],
            "transition_trigger": {
                "ready": True,
                "score": 0.91,
                "memory_write": True,
                "soft_plan": False,
                "hard_plan": True,
                "should_plan": True,
            },
        }
    )

    parsed = parse_action_response_with_metadata(message, horizon=1)

    assert parsed.actions == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.6]]
    assert parsed.transition_trigger == {
        "ready": True,
        "score": 0.91,
        "memory_write": True,
        "soft_plan": False,
        "hard_plan": True,
        "should_plan": True,
    }
    assert parse_action_response(message, horizon=1) == parsed.actions


def test_parse_action_response_rejects_server_error_payload():
    with pytest.raises(RuntimeError, match="server returned error"):
        parse_action_response(json.dumps({"error": "bad request"}), horizon=1)


def test_parse_action_response_rejects_short_horizon():
    with pytest.raises(ValueError, match="expected at least horizon"):
        parse_action_response(json.dumps([[0, 1, 2, 3, 4, 5, 6]]), horizon=2)


def test_parse_action_response_rejects_debug_payload_without_actions():
    with pytest.raises(ValueError, match="must contain 'actions'"):
        parse_action_response(json.dumps({"transition_trigger": {"ready": False}}), horizon=1)


def test_parse_action_response_rejects_short_action_dim():
    with pytest.raises(ValueError, match="expected at least 7"):
        parse_action_response(json.dumps([[0, 1, 2, 3, 4, 5]]), horizon=1)


def test_parse_action_response_rejects_non_numeric_value():
    with pytest.raises(ValueError, match="not numeric"):
        parse_action_response(json.dumps([[0, 1, 2, 3, 4, 5, "closed"]]), horizon=1)


def test_to_libero_action_converts_gripper_sign():
    assert to_libero_action([0, 1, 2, 3, 4, 5, 0.6, 99]) == [0, 1, 2, 3, 4, 5, -1.0]
    assert to_libero_action([0, 1, 2, 3, 4, 5, 0.5, 99]) == [0, 1, 2, 3, 4, 5, 1.0]


def test_select_actions_for_transition_policy_keeps_default_chunk():
    actions = [[float(step)] * 7 for step in range(3)]

    selected = select_actions_for_transition_policy(
        actions,
        {"ready": True, "should_plan": True},
        replan_action_limit=0,
    )

    assert selected == actions


def test_select_actions_for_transition_policy_shortens_triggered_chunk():
    actions = [[float(step)] * 7 for step in range(3)]

    selected = select_actions_for_transition_policy(
        actions,
        {"ready": True, "soft_plan": True, "should_plan": True},
        replan_action_limit=1,
    )

    assert selected == [actions[0]]


def test_select_actions_for_transition_policy_keeps_untriggered_chunk():
    actions = [[float(step)] * 7 for step in range(3)]

    selected = select_actions_for_transition_policy(
        actions,
        {"ready": True, "soft_plan": False, "hard_plan": False, "should_plan": False},
        replan_action_limit=1,
    )

    assert selected == actions
