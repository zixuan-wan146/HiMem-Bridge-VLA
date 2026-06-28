from __future__ import annotations

import torch

from himem_bridge_vla.model.internvl3 import internvl3_embedder
from himem_bridge_vla.model.internvl3.internvl3_embedder import InternVL3Embedder


class FakeInternVL3Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(16)])
        self.language_model.lm_head = torch.nn.Identity()

    def to(self, device):
        self.loaded_device = device
        return self


def test_internvl3_embedder_loads_tokenizer_and_model_from_local_files(monkeypatch):
    calls = {}

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        calls["tokenizer"] = (model_name, kwargs)
        return object()

    def fake_model_from_pretrained(model_name, **kwargs):
        calls["model"] = (model_name, kwargs)
        return FakeInternVL3Model()

    monkeypatch.setattr(internvl3_embedder.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr(internvl3_embedder.AutoModel, "from_pretrained", fake_model_from_pretrained)

    embedder = InternVL3Embedder(
        model_name="/models/InternVL3-1B",
        device="cpu",
        local_files_only=True,
    )

    assert embedder.local_files_only is True
    assert calls["tokenizer"][0] == "/models/InternVL3-1B"
    assert calls["tokenizer"][1]["local_files_only"] is True
    assert calls["tokenizer"][1]["trust_remote_code"] is True
    assert calls["model"][0] == "/models/InternVL3-1B"
    assert calls["model"][1]["local_files_only"] is True
    assert calls["model"][1]["trust_remote_code"] is True
