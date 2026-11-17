# Pocket TTS

[Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts) is a ~100M-parameter TTS model
that runs through the same llama.cpp audio API as Qwen3-TTS. On a Radeon 780M it generates
speech about 7x faster than real time (Qwen3-TTS 1.7B: ~1.5x) with the same voice-cloning
fidelity; see `adr/0002-model-qwen3-tts.md` for measurements.

Trade-offs: one language per model (en, fr, de, pt, it, es), a reference voice clip is
required on every request, and `lang` is ignored.

## Get the model

The weights are gated (CC-BY-4.0; the terms forbid cloning a voice without consent). Accept
them at https://huggingface.co/kyutai/pocket-tts while logged in, then download one language
and convert it with the vendored llama.cpp:

```sh
uv venv /tmp/pocket-venv
uv pip install -p /tmp/pocket-venv/bin/python \
  -r third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  'protobuf>=5.29' huggingface_hub
/tmp/pocket-venv/bin/hf download kyutai/pocket-tts \
  --include 'languages/english/*' --exclude 'languages/english/embeddings/*'

SRC=$(echo ~/.cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/*/languages/english)
OUT=~/.local/share/oido/models/pocket-tts-english
mkdir -p "$OUT"
cd third_party/llama.cpp
/tmp/pocket-venv/bin/python convert_hf_to_gguf.py "$SRC" --outtype f16 \
  --outfile "$OUT/pocket-tts-english-f16.gguf"
/tmp/pocket-venv/bin/python convert_hf_to_gguf.py "$SRC" --mmproj --outtype f16 \
  --outfile "$OUT/mmproj-pocket-tts-english-f16.gguf"
```

`protobuf>=5.29` overrides llama.cpp's `<5` pin, which breaks with current sentencepiece.
The 2026-10 tiny variants (`english_7m_*`, `english_18m_*`) do not work with the pinned
llama.cpp.

## Use

oido finds models in `~/.local/share/oido/models/` automatically:

```sh
oido --engine pocket say "Hello." --voice ~/Music/voice.mp3 -o hello.wav
OIDO_ENGINE=pocket oido say "Hello." --voice alice -o hello.wav
oido --engine pocket mcp      # MCP server running Pocket TTS
```
