# ADR-0004: Named voice library

## Status
Accepted (2026-11-16)

## Decision
`oido voice add NAME CLIP` copies a reference clip to `$XDG_DATA_HOME/oido/voices/`. Every
`--voice` / MCP `voice` argument accepts a saved name or a file path. Names are restricted
to `[A-Za-z0-9_-]{1,64}`, so they cannot escape the directory.

## Consequences
- Agents can discover voices with the `list_voices` MCP tool.
- The x-vector is still computed from the clip on every chunk; libmtmd does not expose a way
  to pass a precomputed embedding through the helper. Cloned runs measured no slower per
  frame than uncloned ones, so this is not worth patching yet.
