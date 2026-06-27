# LIBERO Benchmark Contract

LIBERO protocol code belongs in `himem_bridge_vla.benchmarks.libero`. The active contract uses two views, an 8-dimensional state, a 7-dimensional action, short-memory offsets `(16, 8)`, and a replan stride of 16.

The benchmark adapter is responsible for converting environment observations and history into runtime requests, and for decoding model action chunks back to LIBERO actions, including gripper sign handling. LIBERO requests carry current observation images, offset-indexed short-memory images, and the previous chunk of actual actions sent to `env.step`.

Stage1 and runtime progress-planner evidence must use the same `planner_vl_summary` definition as progress warm-up: the InternVL3 language-model hidden state at the last valid token. Mean pooling raw visual tokens is only a fallback for non-Stage1 smoke paths.
