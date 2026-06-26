import pytest

torch = pytest.importorskip("torch")

from himem_bridge_vla.model.planner import ProgressPretrainHeads
from himem_bridge_vla.model.planner import ProgressStateConfig
from himem_bridge_vla.model.planner import ProgressStatePlanner
from himem_bridge_vla.model.planner import progress_diagnostics
from himem_bridge_vla.model.planner import progress_warmup_loss


def test_progress_state_planner_shapes_and_loss():
    config = ProgressStateConfig(
        hidden_dim=16,
        state_dim=5,
        action_dim=3,
        replan_stride=4,
        latent_dim=6,
        action_summary_hidden_dim=8,
        state_hidden_dim=8,
        updater_hidden_dim=32,
        planner_ffn_dim=32,
        planner_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    model = ProgressStatePlanner(config)
    heads = ProgressPretrainHeads(config)
    batch_size = 2
    state = model.initial_state(batch_size)

    output = model.forward_step(
        state,
        vl_summary=torch.randn(batch_size, 16),
        robot_state=torch.randn(batch_size, 5),
        executed_actions=torch.randn(batch_size, 4, 3),
        executed_mask=torch.ones(batch_size, 4, dtype=torch.bool),
    )
    head_output = heads(output.planner_token, output.progress_state)
    target = torch.randn(batch_size, 6)
    loss, metrics = progress_warmup_loss(head_output, target, use_order_loss=False)
    diagnostics = progress_diagnostics(
        output.planner_token,
        output.progress_state.current_stage,
        head_output.planner_intent,
        head_output.stage_intent,
    )

    assert tuple(output.progress_state.tokens.shape) == (batch_size, 2, 16)
    assert tuple(output.planner_token.shape) == (batch_size, 1, 16)
    assert tuple(head_output.planner_intent.shape) == (batch_size, 6)
    assert loss.item() >= 0.0
    assert set(metrics) == {"plan_loss", "stage_loss", "mem_pool_loss", "order_loss"}
    assert "cos_g_p" in diagnostics


def test_action_summary_zero_mask_outputs_zero():
    config = ProgressStateConfig(
        hidden_dim=8,
        state_dim=4,
        action_dim=2,
        replan_stride=3,
        latent_dim=5,
        action_summary_hidden_dim=8,
        state_hidden_dim=8,
        updater_hidden_dim=16,
        planner_ffn_dim=16,
        planner_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = ProgressStatePlanner(config)
    summary = model.action_summary(
        torch.randn(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    assert torch.allclose(summary, torch.zeros_like(summary))
