from __future__ import annotations

import torch

from himem_bridge_vla.model.himem_bridge_vla import HiMemBridgeVLA
from himem_bridge_vla.model.internvl3.internvl3_embedder import InternVL3EmbeddingOutput


def test_run_inference_prefers_raw_visual_tokens_for_current_context():
    model = HiMemBridgeVLA.__new__(HiMemBridgeVLA)
    model.use_bridge = True
    captured = {}

    lm_tokens = torch.ones(1, 4, 3)
    visual_tokens = torch.full((1, 2, 3), 2.0)
    hidden_states = [torch.full((1, 4, 3), 3.0)]

    def fake_get_vl_embeddings(**_kwargs):
        return InternVL3EmbeddingOutput(
            fused_tokens=lm_tokens,
            hidden_states=hidden_states,
            attention_mask=torch.ones(1, 4),
            visual_tokens=visual_tokens,
        )

    def fake_prepare_state(state_input):
        return torch.as_tensor(state_input, dtype=torch.float32).unsqueeze(0)

    def fake_predict_action(fused_tokens, state, **kwargs):
        captured["fused_tokens"] = fused_tokens
        captured["state"] = state
        captured["hidden_states"] = kwargs["hidden_states"]
        return torch.zeros(1, 1, 3)

    model.get_vl_embeddings = fake_get_vl_embeddings
    model.prepare_state = fake_prepare_state
    model.predict_action = fake_predict_action

    output = model.run_inference(
        images=[torch.zeros(3, 2, 2)],
        image_mask=torch.ones(1, dtype=torch.int32),
        prompt="pick",
        state_input=[0.1, 0.2],
    )

    assert output.shape == (1, 1, 3)
    assert captured["fused_tokens"] is visual_tokens
    assert captured["hidden_states"] is hidden_states
