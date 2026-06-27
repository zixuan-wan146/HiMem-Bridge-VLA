# Runtime Contract

Runtime receives benchmark-neutral `PolicyRequest` objects and returns policy action chunks. Benchmark-specific observation parsing and action decoding belong under `himem_bridge_vla.benchmarks.*`, not the runtime layer.

The runtime path is:

```text
PolicyRequest
-> feature extraction
-> short-memory construction
-> HiMemBridgeVLA.predict_action
-> PolicyResponse / action chunk
```

Current and future implementations must keep `planner_vl_summary`, short-memory offsets, state vectors, executed actions, and executed-action masks explicit. Do not reuse ambiguous names such as `fused_tokens` for different tensor contracts.

For active Stage1-compatible runtime, `planner_vl_summary` is the VLM last-valid-token hidden state, matching the progress warm-up cache. `executed_actions` must represent the previous chunk of actions actually sent to the benchmark environment and are normalized by the server before they enter the progress planner.
