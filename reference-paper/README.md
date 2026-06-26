# Reference Papers

This folder stores local paper copies for memory-related VLA / embodied-AI design discussion.

## Primary Reading Queue

- `2511.18112_EchoVLA_synergistic_declarative_memory.pdf`
  - Use for: long-horizon mobile manipulation memory decomposition.
  - Core takeaway: long memory is not a bare visual-token cache; EchoVLA separates scene memory as spatial-semantic maps and episodic memory as task-level experience.
- `2603.03596_MEM_multi_scale_embodied_memory.pdf`
  - Use for: multi-scale memory carrier design.
  - Core takeaway: short-term memory can stay video/perceptual, while long-term memory should be textual/semantic task progress memory.
- `2510.20328_MemER_experience_retrieval.pdf`
  - Use for: keyframe-based long-term memory without directly feeding action expert raw visual history.
  - Core takeaway: selected keyframes are consumed by a high-level VLM to produce semantic subtasks; the low-level policy/action expert consumes the subtask, not the raw long-term visual memory.
- `2603.04639_RoboMME_memory_robotic_generalist_policies.pdf`
  - Use for: systematic comparison of memory representations and integration mechanisms.
  - Core takeaway: symbolic, perceptual, and recurrent memory each fit different task types; direct memory-as-context is not always the best integration path.

## Secondary / Parked

These papers are kept locally for possible later checks, but should not drive the current HiMem-Bridge-VLA memory design unless a specific detail becomes useful.

- `2604.18791_HELM_long_horizon_memory_vla.pdf`
- `2511.11478_LIBERO_Mem_Embodied_SlotSSM.pdf`
- `2606.02775_AURA_action_gated_memory.pdf`
- `2606.03784_ERVLA_embodied_cot.pdf`
- `2606.09740_ProbeAct_failure_recovery.pdf`

Current design stance:

```text
short-term memory: visual/video tokens are acceptable and useful
long-term memory : semantic/task-state/object-state carrier, not raw visual tokens
```

Naming note:

```text
MEM   = Multi-Scale Embodied Memory from Physical Intelligence
MemER = Memory via Experience Retrieval from Stanford
```
