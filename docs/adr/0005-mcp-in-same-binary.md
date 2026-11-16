# ADR-0005: MCP server in the same binary, lazy load, idle unload

## Status
Accepted (2026-11-16)

## Decision
`oido mcp` serves `synthesize_speech`, `build_podcast` (same arguments as the Go server),
and `list_voices` over stdio with `rmcp`. The model loads on the first call and is dropped
after `--idle-unload` seconds (default 600).

## Consequences
- Existing MCP client configs only change the command (`oido mcp` instead of `oido-tts-mcp`).
- The Go `concurrency` argument is gone (ignored if sent): one resident model serves all calls.
- Output files go to a caller-given absolute directory or `~/.cache/oido-tts/mcp`, always
  under a generated file name, so a call cannot overwrite an arbitrary file.
