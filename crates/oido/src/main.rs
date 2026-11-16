//! `oido`: local voice-cloning text-to-speech, as a CLI and an MCP server.

mod mcp;
mod synth;
mod voices;
mod worker;

use std::collections::BTreeMap;
use std::io::{self, IsTerminal, Read, Write};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use oido_core::audio::Clip;
use oido_core::jobs::{self, Plan};
use oido_core::script::parse_script;
use oido_engine::{Engine, EngineConfig, Request};

use crate::synth::{DEFAULT_SEED, RenderOptions, Synth};
use crate::voices::Voices;
use crate::worker::Worker;

#[derive(Debug, Parser)]
#[command(version, about = "Local voice-cloning text-to-speech")]
struct Cli {
    #[command(flatten)]
    model: ModelArgs,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Args)]
struct ModelArgs {
    /// Backbone gguf.
    #[arg(long, env = "TTS_MODEL_PATH", global = true)]
    model: Option<PathBuf>,
    /// Matching mmproj gguf.
    #[arg(long, env = "TTS_MMPROJ_PATH", global = true)]
    mmproj: Option<PathBuf>,
    /// CPU threads (default: physical cores).
    #[arg(long, global = true)]
    threads: Option<u32>,
    /// KV-cache size in positions.
    #[arg(long, default_value_t = oido_engine::DEFAULT_N_CTX, global = true)]
    ctx: u32,
    /// Backbone layers to offload to the GPU backend.
    #[arg(long, global = true)]
    gpu_layers: Option<i32>,
}

impl ModelArgs {
    fn config(&self) -> Result<EngineConfig> {
        let model = self
            .model
            .clone()
            .context("no model: pass --model or set TTS_MODEL_PATH")?;
        let mmproj = self
            .mmproj
            .clone()
            .context("no mmproj: pass --mmproj or set TTS_MMPROJ_PATH")?;
        let mut config = EngineConfig::new(model, mmproj);
        config.n_ctx = self.ctx;
        config.threads = self.threads;
        config.gpu_layers = self.gpu_layers;
        Ok(config)
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Speak text into a wav file.
    Say {
        /// Text to speak; reads --file or stdin when omitted.
        text: Option<String>,
        /// Read the text from a file.
        #[arg(short, long, conflicts_with = "text")]
        file: Option<PathBuf>,
        /// Saved voice name or reference clip path to clone.
        #[arg(long)]
        voice: Option<String>,
        #[command(flatten)]
        out: OutputArgs,
    },
    /// Render a "Speaker: line" script into one multi-speaker wav.
    Podcast {
        /// Script file.
        script: PathBuf,
        /// Voice for a speaker, as SPEAKER=VOICE (saved name or clip path); repeatable.
        #[arg(long = "voice", value_parser = parse_speaker_voice)]
        voices: Vec<(String, String)>,
        #[command(flatten)]
        out: OutputArgs,
    },
    /// Manage saved voices.
    #[command(subcommand)]
    Voice(VoiceCommand),
    /// Serve the MCP tools over stdio.
    Mcp {
        /// Unload the model after this many idle seconds (0 keeps it loaded).
        #[arg(long, default_value_t = 600)]
        idle_unload: u64,
    },
    /// Load the model once and time repeated synthesis of the same text.
    Bench {
        /// Text to speak.
        text: String,
        /// Synthesis runs after loading.
        #[arg(long, default_value_t = 3)]
        runs: u32,
        /// Reference clip to clone.
        #[arg(long)]
        voice: Option<PathBuf>,
        /// Language.
        #[arg(long)]
        lang: Option<String>,
    },
}

#[derive(Debug, Args)]
struct OutputArgs {
    /// Output wav path.
    #[arg(short, long, default_value = "out.wav")]
    output: PathBuf,
    /// Language (en, zh, ja, ko, de, fr, es, it, pt, ru).
    #[arg(long)]
    lang: Option<String>,
    /// RNG seed; the same seed and input give the same audio.
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u32,
    /// Skip the synthesis cache.
    #[arg(long)]
    no_cache: bool,
}

#[derive(Debug, Subcommand)]
enum VoiceCommand {
    /// Save a reference clip (wav/mp3/flac, ideally 3-20 s of clean speech) under a name.
    Add {
        /// Voice name.
        name: String,
        /// Reference clip.
        clip: PathBuf,
    },
    /// List saved voices.
    #[command(alias = "ls")]
    List,
    /// Delete a saved voice.
    #[command(alias = "rm")]
    Remove {
        /// Voice name.
        name: String,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("OIDO_LOG")
                .unwrap_or_else(|_| "warn,llama=error".into()),
        )
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Say {
            text,
            file,
            voice,
            out,
        } => {
            let text = read_text(text, file)?;
            let voice = voice
                .map(|v| Voices::open_default()?.resolve(&v))
                .transpose()?;
            let plan = jobs::compose(&text, voice);
            render_to_file(&cli.model, &plan, &out, true)
        }
        Command::Podcast {
            script,
            voices,
            out,
        } => {
            let text = std::fs::read_to_string(&script)
                .with_context(|| format!("reading {}", script.display()))?;
            let turns = parse_script(&text);
            anyhow::ensure!(
                !turns.is_empty(),
                "no speaker turns found; use \"Speaker: line\" per paragraph"
            );
            let library = Voices::open_default()?;
            let voices = voices
                .into_iter()
                .map(|(speaker, spec)| Ok((speaker, library.resolve(&spec)?)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            for speaker in voices.keys() {
                if !turns.iter().any(|t| &t.speaker == speaker) {
                    eprintln!("warning: --voice {speaker}=... matches no speaker in the script");
                }
            }
            render_to_file(&cli.model, &jobs::podcast(&turns, &voices), &out, false)
        }
        Command::Voice(cmd) => voice_command(cmd),
        Command::Mcp { idle_unload } => {
            let config = cli.model.config()?;
            let idle = (idle_unload > 0).then(|| Duration::from_secs(idle_unload));
            let worker = Worker::spawn(config, idle);
            let voices = Voices::open_default()?;
            tokio::runtime::Runtime::new()?.block_on(mcp::serve(worker, voices))
        }
        Command::Bench {
            text,
            runs,
            voice,
            lang,
        } => bench(&cli.model, &text, runs, voice.as_deref(), lang.as_deref()),
    }
}

fn render_to_file(
    model: &ModelArgs,
    plan: &Plan,
    out: &OutputArgs,
    anchor_default_voice: bool,
) -> Result<()> {
    anyhow::ensure!(!plan.jobs.is_empty(), "no text to speak");
    if !plan.ignored_styles.is_empty() {
        let styles: Vec<&str> = plan.ignored_styles.iter().map(String::as_str).collect();
        eprintln!(
            "warning: style tags are not supported by this engine; ignored: {}",
            styles.join(", ")
        );
    }
    let cache = if out.no_cache {
        None
    } else {
        synth::default_cache()
    };
    let mut synth = Synth::load(&model.config()?, cache)?;
    let opts = RenderOptions {
        lang: out.lang.clone(),
        seed: out.seed,
        anchor_default_voice,
    };
    let tty = io::stderr().is_terminal();
    let start = Instant::now();
    let rendered = synth.render(plan, &opts, &AtomicBool::new(false), |done, total| {
        if tty {
            eprint!("\rchunk {done}/{total}");
            let _ = io::stderr().flush();
        }
    })?;
    if tty {
        eprintln!();
    }
    synth::save(&out.output, &rendered.clip)?;
    eprintln!(
        "wrote {} ({:.1}s audio in {:.1}s, {} chunks, {} cached)",
        out.output.display(),
        rendered.clip.duration_secs(),
        start.elapsed().as_secs_f64(),
        rendered.jobs,
        rendered.cached
    );
    if rendered.truncated > 0 {
        eprintln!(
            "warning: {} chunk(s) hit the length cap and may be cut off",
            rendered.truncated
        );
    }
    Ok(())
}

fn voice_command(cmd: VoiceCommand) -> Result<()> {
    let voices = Voices::open_default()?;
    match cmd {
        VoiceCommand::Add { name, clip } => {
            let path = voices.add(&name, &clip)?;
            println!("saved {name} -> {}", path.display());
        }
        VoiceCommand::List => {
            for (name, path) in voices.list()? {
                println!("{name}\t{}", path.display());
            }
        }
        VoiceCommand::Remove { name } => voices.remove(&name)?,
    }
    Ok(())
}

fn bench(
    model: &ModelArgs,
    text: &str,
    runs: u32,
    voice: Option<&std::path::Path>,
    lang: Option<&str>,
) -> Result<()> {
    let start = Instant::now();
    let mut engine = Engine::load(&model.config()?)?;
    println!(
        "load {:.2}s  kind={} rss={}MB",
        start.elapsed().as_secs_f64(),
        engine.kind(),
        rss_mb("VmRSS")
    );
    for run in 1..=runs {
        let mut req = Request::new(text);
        req.speaker = voice;
        req.lang = lang;
        req.seed = Some(DEFAULT_SEED);
        let start = Instant::now();
        let audio = engine.synthesize(&req, |_| ControlFlow::Continue(()))?;
        let secs = start.elapsed().as_secs_f64();
        println!(
            "run {run}: {secs:.2}s for {:.2}s audio ({} frames)  RTF {:.2}",
            audio.duration_secs(),
            audio.frames,
            secs / audio.duration_secs()
        );
        if run == 1 {
            synth::save(
                std::path::Path::new("bench.wav"),
                &Clip {
                    sample_rate: audio.sample_rate,
                    samples: audio.samples,
                },
            )?;
        }
    }
    println!("peak rss={}MB", rss_mb("VmHWM"));
    Ok(())
}

fn read_text(arg: Option<String>, file: Option<PathBuf>) -> Result<String> {
    let text = match (arg, file) {
        (Some(t), _) => t,
        (None, Some(path)) => {
            std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?
        }
        (None, None) => {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .context("reading text from stdin")?;
            buf
        }
    };
    anyhow::ensure!(!text.trim().is_empty(), "no text to speak");
    Ok(text)
}

fn parse_speaker_voice(s: &str) -> Result<(String, String), String> {
    let (speaker, voice) = s.split_once('=').ok_or("expected SPEAKER=VOICE")?;
    let (speaker, voice) = (speaker.trim(), voice.trim());
    if speaker.is_empty() || voice.is_empty() {
        return Err("expected SPEAKER=VOICE".into());
    }
    Ok((speaker.to_owned(), voice.to_owned()))
}

/// A `/proc/self/status` memory field in MB (0 if unavailable).
fn rss_mb(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map_or(0, |kb| kb / 1024)
}
