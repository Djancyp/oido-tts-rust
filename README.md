# oido

Local, private text-to-speech with voice cloning, as a CLI and an MCP server. A Rust rebuild
of oido-tts-v3 (Go). Inference runs in-process on llama.cpp (Qwen3-TTS via libmtmd); nothing
leaves the machine.

The model loads once per process and stays resident, instead of once per chunk. On a
Radeon 780M iGPU (Vulkan) it generates speech ~1.7x faster than real time with the 1.7B
model. See [docs/adr](docs/adr) for the design and measurements.

## Build

llama.cpp is a git submodule pinned to an upstream commit, with one local patch
(`patches/llama.cpp/`). `scripts/build-llama.sh` fetches it, applies the patch, and builds
its shared libraries; then build the workspace. Vulkan runs the model on the GPU (including
integrated AMD/Intel GPUs) and falls back to CPU when no device is found; on Arch it needs
`vulkan-headers spirv-headers shaderc` plus a Vulkan driver (`vulkan-radeon` for AMD).

```sh
git clone --recurse-submodules git@github.com:Djancyp/oido-tts-rust.git
cd oido-tts-rust
scripts/build-llama.sh          # or --cpu for a CPU-only build
cargo build --release
```

To use a different llama.cpp build, set `LLAMA_CPP_DIR` (source tree) and optionally
`LLAMA_LIB_DIR` (directory holding `libmtmd.so` etc.). The binary's rpath points at that
directory and at `lib/` next to the executable.

## Model

Download a Qwen3-TTS Base GGUF and its mmproj, e.g. from
[ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF),
and point oido at them:

```sh
export TTS_MODEL_PATH=/path/to/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf
export TTS_MMPROJ_PATH=/path/to/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
```

(`--model` / `--mmproj` flags work too.)

## Use

```sh
oido say "Hello there." -o hello.wav
oido say -f chapter.txt --voice alice --lang en -o chapter.wav
echo "Piped text works too." | oido say -o piped.wav

oido voice add alice ~/clips/alice.wav     # 3-20 s of clean speech
oido voice ls

oido podcast episode.txt --voice Host=alice --voice Guest=~/clips/bob.mp3 -o episode.wav
```

Text is split into sentence-sized chunks, synthesized, and stitched with short gaps. Inline
`[pause:2s]` adds silence before the text that follows; `[pause:-300ms]` overlaps it into
the previous audio. Scripts use one `Speaker: line` paragraph per turn.

The same text, voice, language, and `--seed` always produce the same audio, and each chunk
is cached in `~/.cache/oido/synth` (500 MB, least recently used first), so repeats are
instant. `--no-cache` skips it.

Without `--voice`, the first chunk's voice is cloned for the remaining chunks so one
request stays one speaker.

### MCP

```json
{
  "mcpServers": {
    "oido-tts": {
      "command": "/path/to/oido",
      "args": ["mcp"],
      "env": { "TTS_MODEL_PATH": "...", "TTS_MMPROJ_PATH": "..." }
    }
  }
}
```

Tools: `synthesize_speech` (`text`, `lang`, `voice`, `speaker_file`, `output_dir`),
`build_podcast` (`script`, `voices`, `lang`, `output_dir`), `list_voices`. They return the
path of the written wav. The model loads on the first call and unloads after 10 idle
minutes (`oido mcp --idle-unload SECS`, 0 = never).

## Not supported

Style tags (`[excited]`) and `instruct` from the Go app need a llama.cpp patch that is not
upstream. They are stripped from the spoken text and reported as ignored.

## Develop

```sh
cargo test                                  # includes parity with the Go implementation
cargo clippy --all-targets -- -D warnings
oido bench "Some text." --runs 3            # load time, per-run RTF, peak RSS
```

`crates/oido-core/tests/fixtures/go_golden.json` holds the Go chunker, tag, and script
parser outputs for 368 inputs; the Rust port must match them exactly.

## Layout

```
crates/llama-sys     bindgen bindings to libllama + libmtmd
crates/oido-engine   resident model: load once, synthesize(text, voice, seed) -> PCM
crates/oido-core     chunking, tags, scripts, job plans, stitching, cache
crates/oido          CLI, MCP server, voice library, engine worker thread
```
