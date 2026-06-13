import logging
from types import SimpleNamespace
from typing import List, Tuple, Union

from PIL import Image
import torch
import torch.nn as nn

try:
    from ..experiment_config import resolve_experiment_config
    from .action_head.flow_matching import FlowmatchingActionHead
    from .bridge import BridgeAdapter, BridgeAdapterConfig, BridgeAdapterOutput
    from .himem import EpisodeMemoryBank, HierarchicalEpisodeMemory, HiMemTokenWriter
    from .internvl3.internvl3_embedder import InternVL3Embedder, InternVL3EmbeddingOutput
except ImportError:
    from himem_bridge_vla.experiment_config import resolve_experiment_config
    from himem_bridge_vla.model.action_head.flow_matching import FlowmatchingActionHead
    from himem_bridge_vla.model.bridge import BridgeAdapter, BridgeAdapterConfig, BridgeAdapterOutput
    from himem_bridge_vla.model.himem import EpisodeMemoryBank, HierarchicalEpisodeMemory, HiMemTokenWriter
    from himem_bridge_vla.model.internvl3.internvl3_embedder import InternVL3Embedder, InternVL3EmbeddingOutput


class HiMemBridgeVLA(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        config = resolve_experiment_config(config)
        self.config = config
        self._device = config.get("device", "cuda")
        self.return_cls_only = config.get("return_cls_only", False)

        vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
        self.embedder = InternVL3Embedder(
            model_name=vlm_name,
            device=self._device,
            allow_image_token_truncation=bool(config.get("allow_image_token_truncation", False)),
        )

        action_head_type = config.get("action_head", "flowmatching").lower()
        if action_head_type != "flowmatching":
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

        horizon = config.get("action_horizon", config.get("horizon", 16))
        per_action_dim = config.get("per_action_dim", 7)
        action_dim = horizon * per_action_dim

        config["horizon"] = horizon
        config["per_action_dim"] = per_action_dim
        config["action_dim"] = action_dim

        if action_dim != horizon * per_action_dim:
            raise ValueError(
                f"action_dim ({action_dim}) must equal horizon ({horizon}) * "
                f"per_action_dim ({per_action_dim})"
            )

        self.horizon = horizon
        self.per_action_dim = per_action_dim

        action_head_config = SimpleNamespace(
            embed_dim=config.get("embed_dim", 896),
            hidden_dim=config.get("hidden_dim", 1024),
            action_dim=action_dim,
            horizon=horizon,
            per_action_dim=per_action_dim,
            state_dim=config.get("state_dim", 7),
            state_hidden_dim=config.get("state_hidden_dim", 1024),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 8),
            dropout=config.get("dropout", 0.0),
            num_inference_timesteps=config.get("num_inference_timesteps", 50),
            num_categories=config.get("num_categories", 1),
        )
        self.action_head = FlowmatchingActionHead(config=action_head_config).to(self._device)
        self.use_bridge = bool(config.get("use_bridge", False))
        self.use_himem = bool(config.get("use_himem", False)) and self.use_bridge
        self.bridge_variant = str(config.get("bridge_variant", "crosskv"))
        self.bridge_context_mode = str(
            config.get("bridge_context_mode", "bridge_residual" if self.use_bridge else "fused_only")
        )
        self.memory_placement = str(config.get("memory_placement", "crosskv"))
        self.bridge_adapter = None
        self.memory_bank = None
        self.memory_runtime = None
        self.memory_writer = None
        self.skill_tokens = None
        self.fused_residual_gate = None
        self.last_bridge_output: BridgeAdapterOutput | None = None

        if self.use_bridge:
            if self.bridge_context_mode not in {
                "fused_only",
                "bridge_clean",
                "bridge_residual",
                "bridge_gated_residual",
            }:
                raise ValueError(f"Unknown bridge_context_mode: {self.bridge_context_mode}")
            bridge_config = BridgeAdapterConfig(
                embed_dim=config.get("bridge_hidden_dim", config.get("embed_dim", 896)),
                raw_dim=config.get("bridge_raw_dim", config.get("embed_dim", 896)),
                state_dim=config.get("state_dim", 7),
                num_layers=config.get("bridge_num_layers", 2),
                num_heads=config.get("bridge_num_heads", 8),
                num_bridge_tokens=config.get("bridge_num_tokens", 16),
                num_action_queries=config.get("bridge_num_action_queries", 64),
                dropout=config.get("bridge_dropout", config.get("dropout", 0.0)),
                raw_gate_init=config.get("bridge_raw_gate_init", 0.0),
                ffn_mult=config.get("bridge_ffn_mult", 4),
            )
            self.bridge_adapter = BridgeAdapter(bridge_config).to(self._device)
            if self.bridge_context_mode == "bridge_gated_residual":
                gate_init = float(config.get("bridge_fused_gate_init", 0.0))
                self.fused_residual_gate = nn.Parameter(torch.tensor(gate_init))

            if bool(config.get("skill_tokens_enabled", False)):
                skill_count = int(config.get("skill_num_tokens", 4))
                if skill_count <= 0:
                    raise ValueError(f"skill_num_tokens must be positive, got {skill_count}")
                self.skill_tokens = nn.Parameter(torch.empty(skill_count, bridge_config.embed_dim))
                nn.init.normal_(self.skill_tokens, mean=0.0, std=0.02)

        if self.use_himem:
            if self.memory_placement not in {"crosskv", "mixed_latent"}:
                raise ValueError(f"Unknown memory_placement: {self.memory_placement}")
            self.memory_bank = EpisodeMemoryBank(
                max_tokens=config.get("memory_max_tokens", 32),
                token_dim=config.get("bridge_hidden_dim", config.get("embed_dim", 896)),
            )
            self.memory_writer = HiMemTokenWriter(
                hidden_dim=config.get("bridge_hidden_dim", config.get("embed_dim", 896)),
                num_tokens=config.get("memory_write_tokens", 4),
                num_heads=config.get("memory_writer_num_heads", config.get("bridge_num_heads", 8)),
                dropout=config.get("memory_writer_dropout", config.get("dropout", 0.0)),
            ).to(self._device)
            self.memory_runtime = HierarchicalEpisodeMemory(
                bank=self.memory_bank,
                read_top_k=config.get("memory_read_top_k", 8),
                write_threshold=config.get("memory_write_threshold", 0.5),
                segment_accumulator=config.get("memory_segment_accumulator", "ema"),
                segment_ema_decay=config.get("memory_segment_ema_decay", 0.9),
                write_policy=config.get("memory_write_policy", "boundary"),
            )

    def get_vl_embeddings(
        self,
        images: List[Image.Image],
        image_mask: torch.Tensor,
        prompt: str = "",
        return_cls_only: Union[bool, None] = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | InternVL3EmbeddingOutput:
        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image tensor.")

        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
            return_hidden_states=return_hidden_states,
            selected_layers=self.config.get("bridge_raw_layers", None),
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError(f"Unsupported state input type: {type(state_input)!r}")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
        hidden_states: list[torch.Tensor] | None = None,
        memory_context: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        fused_tokens = self._augment_context_with_bridge(
            fused_tokens,
            state=state,
            hidden_states=hidden_states,
            memory_context=memory_context,
        )
        if actions_gt is None:
            return self.action_head.get_action(
                fused_tokens,
                state=state,
                action_mask=action_mask,
                embodiment_id=embodiment_ids,
            )

        return self.action_head(
            fused_tokens,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_ids,
        )

    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None,
        episode_id: str | None = None,
        session_id: str | None = None,
        reset_memory: bool = False,
    ) -> torch.Tensor:
        embedding_output = self.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=return_cls_only,
            return_hidden_states=self.use_bridge,
        )
        if isinstance(embedding_output, InternVL3EmbeddingOutput):
            fused_tokens = embedding_output.fused_tokens
            hidden_states = embedding_output.hidden_states
        else:
            fused_tokens = embedding_output
            hidden_states = None
        state_tensor = self.prepare_state(state_input)
        memory_episode_id = self._memory_episode_id(episode_id, session_id)
        memory_context = self._read_memory(memory_episode_id, reset_memory, fused_tokens)
        action = self.predict_action(
            fused_tokens,
            state_tensor,
            action_mask=action_mask,
            hidden_states=hidden_states,
            memory_context=memory_context,
        )
        self._maybe_write_memory(memory_episode_id)
        return action

    def forward(
        self,
        fused_tokens,
        state=None,
        actions_gt=None,
        action_mask=None,
        embodiment_ids=None,
        hidden_states=None,
        memory_context=None,
    ):
        return self.predict_action(
            fused_tokens,
            state,
            actions_gt,
            action_mask,
            embodiment_ids,
            hidden_states=hidden_states,
            memory_context=memory_context,
        )

    def _augment_context_with_bridge(
        self,
        fused_tokens: torch.Tensor,
        *,
        state: torch.Tensor | None,
        hidden_states: list[torch.Tensor] | None,
        memory_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.bridge_adapter is None:
            self.last_bridge_output = None
            return fused_tokens

        bridge_output = self.bridge_adapter(
            fused_tokens,
            hidden_states=hidden_states,
            state=state,
            memory_context=memory_context if self.memory_placement == "crosskv" else None,
        )
        self.last_bridge_output = bridge_output
        return self._build_action_context(fused_tokens, bridge_output.bridge_tokens, memory_context)

    def _build_action_context(
        self,
        fused_tokens: torch.Tensor,
        bridge_tokens: torch.Tensor,
        memory_context: torch.Tensor | None,
    ) -> torch.Tensor:
        fused_tokens = _ensure_rank3(fused_tokens, "fused_tokens")
        if self.bridge_context_mode == "fused_only":
            context_tokens = fused_tokens
        elif self.bridge_context_mode == "bridge_clean":
            context_tokens = bridge_tokens
        elif self.bridge_context_mode == "bridge_residual":
            context_tokens = torch.cat([fused_tokens, bridge_tokens], dim=1)
        elif self.bridge_context_mode == "bridge_gated_residual":
            if self.fused_residual_gate is None:
                raise RuntimeError("fused_residual_gate was not initialized")
            gate = torch.tanh(self.fused_residual_gate).to(device=fused_tokens.device, dtype=fused_tokens.dtype)
            context_tokens = torch.cat([gate * fused_tokens, bridge_tokens], dim=1)
        else:
            raise ValueError(f"Unknown bridge_context_mode: {self.bridge_context_mode}")

        if self.memory_placement == "mixed_latent" and memory_context is not None:
            memory_context = _ensure_rank3(memory_context, "memory_context").to(
                device=context_tokens.device,
                dtype=context_tokens.dtype,
            )
            if memory_context.shape[1] > 0:
                context_tokens = torch.cat([context_tokens, memory_context], dim=1)

        if self.skill_tokens is not None:
            skill_tokens = self.skill_tokens.to(device=context_tokens.device, dtype=context_tokens.dtype)
            skill_tokens = skill_tokens.unsqueeze(0).expand(context_tokens.shape[0], -1, -1)
            context_tokens = torch.cat([context_tokens, skill_tokens], dim=1)

        return context_tokens

    def _read_memory(
        self,
        episode_id: str | None,
        reset_memory: bool,
        fused_tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.memory_bank is None or not episode_id:
            return None
        if reset_memory:
            self.memory_runtime.reset(episode_id)
        return self.memory_runtime.read(episode_id, fused_tokens)

    def _maybe_write_memory(self, episode_id: str | None) -> None:
        if self.memory_runtime is None or not episode_id or self.last_bridge_output is None:
            return
        boundary_prob = torch.sigmoid(self.last_bridge_output.boundary_logits.detach()).reshape(-1)
        if self.memory_writer is None:
            memory_tokens = self.last_bridge_output.bridge_tokens.detach().mean(dim=1)
        else:
            memory_tokens = self.memory_writer(self.last_bridge_output.bridge_tokens.detach())
        self.memory_runtime.write(episode_id, memory_tokens, gate=boundary_prob)

    def _memory_episode_id(self, episode_id: str | None, session_id: str | None = None) -> str | None:
        if not episode_id:
            return None
        if not session_id:
            return str(episode_id)
        return f"{session_id}:{episode_id}"

    def _freeze_module(self, module: nn.Module, name: str):
        logging.info(f"Freezing {name} parameters...")
        for param in module.parameters():
            param.requires_grad = False

    def set_finetune_flags(self):
        if not self.config.get("finetune_vlm", False):
            self._freeze_module(self.embedder, "VLM (InternVL3)")
        else:
            logging.info("Finetuning VLM (InternVL3)...")

        if not self.config.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
        else:
            logging.info("Finetuning Action Head...")


def _ensure_rank3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D] or [B, D], got {tuple(tensor.shape)}")
    return tensor
