//! MCP server over stdio, exposing the same tools as the Go `oido-tts-mcp`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use oido_core::jobs::{self, Plan};
use oido_core::script::parse_script;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use serde::Deserialize;

use crate::synth::{self, DEFAULT_SEED, RenderOptions, Rendered};
use crate::voices::Voices;
use crate::worker::Worker;

/// Largest accepted text or script, in bytes (several hours of speech).
const MAX_INPUT_BYTES: usize = 200_000;
const SPEECH_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const PODCAST_TIMEOUT: Duration = Duration::from_secs(20 * 60);

/// Arguments for `synthesize_speech`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SynthesizeInput {
    /// The text to speak. Inline `[pause:2s]` / `[pause:-300ms]` tags set timing.
    text: String,
    /// Language code: en, zh, ja, ko, de, fr, es, it, pt, ru. Omit for English.
    lang: Option<String>,
    /// Name of a saved voice (see `list_voices`) to clone.
    voice: Option<String>,
    /// Local path to a wav/mp3/flac reference clip to clone. Ignored if `voice` is set.
    speaker_file: Option<String>,
    /// Style instruction. Not supported by the current engine; accepted and ignored.
    instruct: Option<String>,
    /// Absolute directory to save the wav into; omit for a default cache directory.
    output_dir: Option<String>,
}

/// Arguments for `build_podcast`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct PodcastInput {
    /// The script: one "Speaker: line" paragraph per turn, separated by blank lines, e.g.
    /// "HOST: Welcome back.\n\nGUEST: [pause:-300ms] Thanks for having me."
    script: String,
    /// Maps a speaker name as written in the script to a saved voice name or a local clip
    /// path. Unlisted speakers use the default voice.
    voices: Option<BTreeMap<String, String>>,
    /// Language code: en, zh, ja, ko, de, fr, es, it, pt, ru. Omit for English.
    lang: Option<String>,
    /// Absolute directory to save the wav into; omit for a default cache directory.
    output_dir: Option<String>,
}

#[derive(Debug, Clone)]
struct OidoServer {
    worker: Arc<Worker>,
    voices: Voices,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl OidoServer {
    #[tool(
        description = "Convert text to speech with a local model, optionally cloning a voice. \
        Writes a wav file and returns its path. Runs fully on-device."
    )]
    async fn synthesize_speech(
        &self,
        Parameters(input): Parameters<SynthesizeInput>,
    ) -> Result<CallToolResult, McpError> {
        Ok(saved_result(self.synthesize(input).await))
    }

    #[tool(
        description = "Turn a \"Speaker: line\" script into one stitched multi-speaker episode wav, \
        each speaker with its own voice. Returns the file path. Runs fully on-device."
    )]
    async fn build_podcast(
        &self,
        Parameters(input): Parameters<PodcastInput>,
    ) -> Result<CallToolResult, McpError> {
        Ok(saved_result(self.podcast(input).await))
    }

    #[tool(
        description = "List saved voice names usable as `voice` in synthesize_speech or in build_podcast's `voices`."
    )]
    async fn list_voices(&self) -> Result<CallToolResult, McpError> {
        let listing = self.voices.list().map(|v| {
            if v.is_empty() {
                "No saved voices. Add one with `oido voice add NAME CLIP`.".to_owned()
            } else {
                v.into_iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        });
        Ok(tool_result(listing))
    }
}

impl OidoServer {
    async fn synthesize(&self, input: SynthesizeInput) -> Result<(String, PathBuf)> {
        check_size(&input.text)?;
        let voice = match (input.voice.as_deref(), input.speaker_file.as_deref()) {
            (Some(spec), _) | (None, Some(spec)) => Some(self.voices.resolve(spec)?),
            (None, None) => None,
        };
        let plan = jobs::compose(&input.text, voice);
        anyhow::ensure!(!plan.jobs.is_empty(), "text must not be empty");
        let out = synth::timestamped_path(&output_dir(input.output_dir.as_deref())?, "oido-tts");
        let opts = RenderOptions {
            lang: input.lang,
            seed: DEFAULT_SEED,
            anchor_default_voice: true,
        };
        let rendered = self
            .worker
            .render(plan.clone(), opts, out.clone(), SPEECH_TIMEOUT)
            .await?;
        let summary = report("speech", &out, &rendered, &plan, input.instruct.is_some());
        Ok((summary, out))
    }

    async fn podcast(&self, input: PodcastInput) -> Result<(String, PathBuf)> {
        check_size(&input.script)?;
        let turns = parse_script(&input.script);
        anyhow::ensure!(
            !turns.is_empty(),
            "no speaker turns found; use \"Speaker: line\" per paragraph"
        );
        let voices = input
            .voices
            .unwrap_or_default()
            .into_iter()
            .map(|(speaker, spec)| Ok((speaker, self.voices.resolve(&spec)?)))
            .collect::<Result<BTreeMap<String, PathBuf>>>()?;
        let plan = jobs::podcast(&turns, &voices);
        let out = synth::timestamped_path(&output_dir(input.output_dir.as_deref())?, "podcast");
        let opts = RenderOptions {
            lang: input.lang,
            seed: DEFAULT_SEED,
            anchor_default_voice: false,
        };
        let rendered = self
            .worker
            .render(plan.clone(), opts, out.clone(), PODCAST_TIMEOUT)
            .await?;
        Ok((report("podcast", &out, &rendered, &plan, false), out))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for OidoServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("oido-tts", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Local text-to-speech with voice cloning. synthesize_speech for single-voice text, \
                 build_podcast for multi-speaker scripts, list_voices for saved voice names. \
                 Both return the path of a wav file on this machine.",
            )
    }
}

/// Serves MCP on stdin/stdout until the client disconnects.
///
/// # Errors
///
/// Fails if the transport fails.
pub async fn serve(worker: Worker, voices: Voices) -> Result<()> {
    let server = OidoServer {
        worker: Arc::new(worker),
        voices,
        tool_router: OidoServer::tool_router(),
    };
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// A saved-file result: the summary as text, and `{"path": ...}` as structured content, as the
/// Go server returned it.
fn saved_result(result: Result<(String, PathBuf)>) -> CallToolResult {
    match result {
        Ok((summary, path)) => {
            let mut r = CallToolResult::success(vec![ContentBlock::text(summary)]);
            r.structured_content = Some(serde_json::json!({ "path": path }));
            r
        }
        Err(e) => tool_result(Err(e)),
    }
}

fn tool_result(result: Result<String>) -> CallToolResult {
    match result {
        Ok(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
        Err(e) => CallToolResult::error(vec![ContentBlock::text(format!("{e:#}"))]),
    }
}

fn check_size(text: &str) -> Result<()> {
    if text.len() > MAX_INPUT_BYTES {
        bail!(
            "input is {} bytes; the limit is {MAX_INPUT_BYTES}",
            text.len()
        );
    }
    Ok(())
}

/// The caller's directory (must be absolute) or the default `~/.cache/oido-tts/mcp`.
fn output_dir(requested: Option<&str>) -> Result<PathBuf> {
    let dir = match requested.filter(|d| !d.is_empty()) {
        Some(d) => {
            let p = PathBuf::from(d);
            anyhow::ensure!(p.is_absolute(), "output_dir must be an absolute path");
            p
        }
        None => dirs::cache_dir()
            .context("no user cache directory")?
            .join("oido-tts")
            .join("mcp"),
    };
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

fn report(kind: &str, out: &Path, r: &Rendered, plan: &Plan, instruct_given: bool) -> String {
    let mut s = format!(
        "Saved {kind} to {} ({:.1}s, {} chunks, {} from cache)",
        out.display(),
        r.clip.duration_secs(),
        r.jobs,
        r.cached
    );
    if r.truncated > 0 {
        let _ = write!(
            s,
            "\nWarning: {} chunk(s) hit the length cap and may be cut off.",
            r.truncated
        );
    }
    if !plan.ignored_styles.is_empty() || instruct_given {
        s.push_str(
            "\nNote: style tags and `instruct` are not supported by this engine and were ignored.",
        );
    }
    s
}
