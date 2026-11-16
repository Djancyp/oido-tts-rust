# ADR-0002: Qwen3-TTS Base as the default model

## Status
Accepted (2026-11-16)

## Context
Needs zero-shot voice cloning from a short clip, multilingual output, and good local speed.

## Decision
Qwen3-TTS-12Hz Base (GGUF via ggml-org) through libmtmd. Cloning uses the model's x-vector
speaker encoder on the reference clip.

## Alternatives considered
- **Pocket TTS** (Kyutai, ~100M): far smaller and supported by the same libmtmd API, but
  one language per weight set, and the cloning weights are gated on Hugging Face. Strong
  candidate for a fast CPU tier; not yet benchmarked.
- **Chatterbox, F5-TTS, VibeVoice**: PyTorch-only or GPU-oriented; slow on this machine.
- **XTTS-v2**: non-commercial license.

## Consequences
- Upstream llama.cpp has no `--tts-instruct` equivalent, so the Go app's style tags and
  `instruct` cannot be applied. They are parsed, stripped from spoken text, and reported as
  ignored rather than read aloud.
- Only a 1.7B GGUF is published by ggml-org; a 0.6B would need converting from the HF weights.
