# ADR-0002: Qwen3-TTS Base as the default model

## Status
Accepted (2026-11-16)

## Context
Needs zero-shot voice cloning from a short clip, multilingual output, and good local speed.

## Decision
Qwen3-TTS-12Hz Base (GGUF via ggml-org) through libmtmd. Cloning uses the model's x-vector
speaker encoder on the reference clip.

## Alternatives considered
- **Pocket TTS** (Kyutai, ~100M, CC-BY-4.0, gated): runs through the same libmtmd API with
  no code changes. Measured 2026-11-17 against Qwen3-TTS 1.7B, same text and cloned voice
  (`languages/english`, f16, 159 MB backbone + 60 MB mmproj):

  | | RTF (780M Vulkan) | RTF (CPU) | Whisper WER | Speaker similarity (ECAPA) |
  |---|---|---|---|---|
  | Pocket TTS | 0.13 | 0.29 | 0.04 (one inserted word) | 0.789 |
  | Qwen3-TTS 1.7B | 0.68 | ~1.5 | 0.00 | 0.793 |

  About 5x faster with the same voice fidelity, but one language per model (en/fr/de/pt/it/es),
  a reference clip is mandatory, and the language argument is ignored. The 2026-10 tiny
  variants (7M, 18M) do not work with the pinned llama.cpp: 7M fails conversion, 18M's
  mmproj fails to load (`a.gen.flow.time.0.freqs` missing).
- **Chatterbox, F5-TTS, VibeVoice**: PyTorch-only or GPU-oriented; slow on this machine.
- **XTTS-v2**: non-commercial license.

## Consequences
- Upstream llama.cpp has no `--tts-instruct` equivalent, so the Go app's style tags and
  `instruct` cannot be applied. They are parsed, stripped from spoken text, and reported as
  ignored rather than read aloud.
- Only a 1.7B GGUF is published by ggml-org; a 0.6B would need converting from the HF weights.
