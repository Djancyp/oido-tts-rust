# ADR-0001: Run inference in-process through libllama + libmtmd

## Status
Accepted (2026-11-16)

## Context
The Go app (oido-tts-v3) spawned a `llama-tts` process per chunk, so every chunk reloaded
the model, and each process held ~6–8 GB of RAM, which ruled out running chunks in parallel.
The dev machine (Ryzen 7 8845HS, Radeon 780M iGPU, 25 GB RAM, no NVIDIA) can only accelerate
through ggml (CPU or Vulkan).

## Decision
Bind llama.cpp's C API with bindgen (`crates/llama-sys`) and drive the audio-generation
helper (`mtmd_helper_gen_audio_*`) from Rust (`crates/oido-engine`). The model loads once
per process.

## Alternatives considered
- **`llama-tts` subprocess**: the status quo; pays model load every chunk.
- **`qts` crate (GGUF + GGML)**: v0.1.0, unknown maintenance; upstream llama.cpp already
  exposes the whole pipeline through a plain C API.
- **Candle (`qwen_tts`, `any-tts`)**: pure Rust but no Vulkan backend (CPU only on this
  machine), and voice-cloning problems have been reported in a Rust port.

## Consequences
- Output is byte-identical to `llama-tts` for the same seed (verified).
- The mtmd audio API is marked experimental upstream; llama.cpp is vendored at a pinned
  commit (`third_party/llama.cpp`) and upgraded deliberately.
- A crash in C++ code kills the process. MCP clients restart servers; acceptable.
- Pocket TTS works through the same API with no extra code.
- The Vulkan backend runs the whole pipeline on the 780M iGPU at RTF 0.60, 2.5x the CPU
  backend (see ADR-0003). It is the default build.
