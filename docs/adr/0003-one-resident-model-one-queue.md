# ADR-0003: One resident model, one queue, a small KV cache

## Status
Accepted (2026-11-16)

## Context
The Go app needed memory heuristics (`ClampConcurrency`, a 12 GB per-job estimate) because
each concurrent job was a whole process with its own weights.

## Decision
- One thread owns the model; requests run one at a time from a queue (`crates/oido/src/worker.rs`).
- The context is sized for one chunk (`n_ctx = 2048`), not the model's training context.
- The helper's code-predictor RNG is reseeded at every generation (local llama.cpp patch,
  `patches/llama.cpp/0001-*.patch`) so a resident model is as reproducible as a fresh process.

## Measurements (CPU, 1.7B Q4_K_M, same benchmark text)
| | Peak RSS | Wall for 10.9 s audio |
|---|---|---|
| `llama-tts`, default context | 7.9 GB | 19.4 s (~3.9 s of it load) |
| `llama-tts -c 2048` | 4.6 GB | 17.0 s |
| `oido` resident engine | 4.5 GB | 0.6 s load once, then 16.2–17.7 s per run |

Audio was byte-identical across all three.

Same benchmark with the vendored llama.cpp built with `GGML_VULKAN=ON` (Radeon 780M, RADV):

| | Process RSS | Per run (10.96 s audio) | RTF |
|---|---|---|---|
| CPU backend | 4.5 GB | 13.6–16.9 s | 1.25–1.55 |
| Vulkan, backbone on CPU (`--gpu-layers 0`) | 1.7 GB | ~11.2 s | ~0.96 |
| Vulkan, everything on GPU (default) | 1.5 GB | 6.6 s | 0.60 |

End to end, a 4-chunk `oido say` (19 s of audio) took 30.5 s on CPU and 12.6 s on Vulkan,
including model load. The iGPU shares system RAM, so the lower process RSS moves memory
into GPU allocations rather than freeing it. Vulkan output differs slightly from CPU output
(floating-point differences) but is reproducible run to run for the same seed.

## Consequences
- Memory no longer scales with request count or text length.
- Throughput is one generation at a time. llama.cpp can decode several sequences in one
  context; that is the path to parallel podcast turns if needed.
