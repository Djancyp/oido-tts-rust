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

(`--model` / `--mmproj` flags work too.) With neither set, oido uses a Qwen3-TTS (or Pocket
TTS) GGUF found in the Hugging Face cache (`$HF_HUB_CACHE`, `$HF_HOME/hub`, or
`~/.cache/huggingface/hub`), so after `hf download ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`
nothing needs configuring.

`--engine qwen|pocket` (or `OIDO_ENGINE`) picks the model family to find. Pocket TTS is about
5x faster at the same voice fidelity, but each model speaks one language and it always
needs a voice clip; see [docs/pocket-tts.md](docs/pocket-tts.md).

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

`oido mcp` serves the tools over stdio. Run one server per engine: `oido-tts` for Qwen3-TTS
(multilingual, voice optional) and `oido-tts-pocket` for Pocket TTS (about 5x faster, one
language, voice required). Both find their model automatically (see [Model](#model)); Pocket
needs the converted model from [docs/pocket-tts.md](docs/pocket-tts.md) first.

**Claude Code** (`-s user` = every project, `-s local` = current project only):

```sh
claude mcp add oido-tts -s user -- ~/.local/bin/oido mcp
claude mcp add oido-tts-pocket -s user -- ~/.local/bin/oido --engine pocket mcp
claude mcp get oido-tts-pocket          # status
claude mcp remove oido-tts-pocket -s user
```

Open sessions pick up a new server after `/mcp` → reconnect, or a restart.

**Other clients** (Claude Desktop, Cursor, …), in the client's MCP config:

```json
{
  "mcpServers": {
    "oido-tts": {
      "command": "/home/you/.local/bin/oido",
      "args": ["mcp"]
    },
    "oido-tts-pocket": {
      "command": "/home/you/.local/bin/oido",
      "args": ["--engine", "pocket", "mcp"]
    }
  }
}
```

To pin exact files instead of auto-discovery, add
`"env": { "TTS_MODEL_PATH": "...", "TTS_MMPROJ_PATH": "..." }` to a server.

**Tools** (same on both servers):

| Tool | Arguments |
|---|---|
| `synthesize_speech` | `text`, `voice` (saved name) or `speaker_file` (clip path), `lang`, `output_dir` (absolute) |
| `build_podcast` | `script` (`Speaker: line` paragraphs), `voices` (speaker → name or path), `lang`, `output_dir` |
| `list_voices` | none |

Each call writes a wav and returns its path, as text and as structured `{"path": ...}`;
without `output_dir` files go to `~/.cache/oido-tts/mcp/`. The model loads on the first call
and unloads after 10 idle minutes (`oido mcp --idle-unload SECS`, 0 = never).

**Using them:** ask in plain language, e.g. *"Using oido-tts-pocket, say 'Welcome back to
the channel' with ~/Music/voice.mp3"*, or *"with voice alice"* after
`oido voice add alice ~/Music/voice.mp3`. With Pocket, always name a voice (every speaker
in a podcast needs one); `lang` is ignored because the model's language is fixed.

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
