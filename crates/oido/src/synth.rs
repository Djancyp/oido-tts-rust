//! Running a [`Plan`] through the engine: cache lookups, voice anchoring, stitching.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use oido_core::audio::{self, Clip};
use oido_core::cache::{Cache, CacheKey, CacheKeyBuilder};
use oido_core::jobs::Plan;
use oido_engine::{Engine, EngineConfig, EngineError, Request};

/// Seed used when none is given. Qwen3-TTS Base has no voice identity without a reference
/// clip, so the RNG picks the speaker; a fixed seed keeps that speaker stable across runs.
pub const DEFAULT_SEED: u32 = 42;

/// Per-request synthesis options.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Language code or name; `None` uses the model default.
    pub lang: Option<String>,
    /// RNG seed.
    pub seed: u32,
    /// Clone the first default-voice chunk's audio for later default-voice chunks, so one
    /// speaker stays one voice across chunks. Off for podcasts, where default-voice speakers
    /// are distinct people.
    pub anchor_default_voice: bool,
}

/// What a render produced.
#[derive(Debug, Clone)]
pub struct Rendered {
    /// Stitched audio.
    pub clip: Clip,
    /// Jobs synthesized.
    pub jobs: usize,
    /// Jobs served from the cache.
    pub cached: usize,
    /// Jobs that hit the frame cap and may be cut off.
    pub truncated: usize,
}

/// A loaded engine plus the cache.
#[derive(Debug)]
pub struct Synth {
    engine: Engine,
    cache: Option<Cache>,
    /// Hash state over the model files and build; every cache key starts from it.
    key_base: CacheKeyBuilder,
    max_frames: u32,
}

impl Synth {
    /// Loads the model described by `config`.
    ///
    /// # Errors
    ///
    /// Fails if the model cannot be loaded or its files cannot be stat'ed.
    pub fn load(config: &EngineConfig, cache: Option<Cache>) -> Result<Self> {
        let key_base = CacheKey::builder()
            .str("oido", env!("CARGO_PKG_VERSION"))
            .file_identity("model", &config.model)
            .with_context(|| format!("reading {}", config.model.display()))?
            .file_identity("mmproj", &config.mmproj)
            .with_context(|| format!("reading {}", config.mmproj.display()))?;
        let engine = Engine::load(config)?;
        Ok(Self {
            engine,
            cache,
            key_base,
            max_frames: oido_engine::DEFAULT_MAX_FRAMES,
        })
    }

    /// Synthesizes every job in `plan` and stitches the results. `progress` receives
    /// `(jobs done, jobs total)`. Setting `cancel` stops at the next frame.
    ///
    /// # Errors
    ///
    /// Fails if the plan is empty, a voice clip cannot be read, generation fails, or
    /// `cancel` is set.
    pub fn render(
        &mut self,
        plan: &Plan,
        opts: &RenderOptions,
        cancel: &AtomicBool,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Rendered> {
        anyhow::ensure!(!plan.jobs.is_empty(), "nothing to synthesize");
        let total = plan.jobs.len();
        let mut clips = Vec::with_capacity(total);
        let mut anchor: Option<tempfile::TempPath> = None;
        let (mut cached, mut truncated) = (0, 0);

        for (i, job) in plan.jobs.iter().enumerate() {
            let voice: Option<&Path> = match (&job.voice, &anchor) {
                (Some(v), _) => Some(v),
                (None, Some(a)) if opts.anchor_default_voice => Some(a),
                _ => None,
            };
            let key = self.key(&job.text, opts, voice)?;
            let clip = match self.cache.as_ref().and_then(|c| c.get(&key)) {
                Some(clip) => {
                    cached += 1;
                    clip
                }
                None => {
                    let mut req = Request::new(&job.text);
                    req.lang = opts.lang.as_deref();
                    req.speaker = voice;
                    req.seed = Some(opts.seed);
                    req.max_frames = self.max_frames;
                    let audio =
                        self.engine
                            .synthesize(&req, |_| {
                                if cancel.load(Ordering::Relaxed) {
                                    ControlFlow::Break(())
                                } else {
                                    ControlFlow::Continue(())
                                }
                            })
                            .map_err(|e| match e {
                                EngineError::Cancelled => anyhow::anyhow!("cancelled"),
                                e => anyhow::Error::new(e)
                                    .context(format!("chunk {} of {total}", i + 1)),
                            })?;
                    if audio.frames >= self.max_frames {
                        truncated += 1;
                        tracing::warn!(chunk = i + 1, "hit the frame cap; audio may be cut off");
                    }
                    let clip = Clip {
                        sample_rate: audio.sample_rate,
                        samples: audio.samples,
                    };
                    if let Some(cache) = &self.cache
                        && let Err(e) = cache.put(&key, &clip)
                    {
                        tracing::warn!("cache write failed: {e}");
                    }
                    clip
                }
            };
            if opts.anchor_default_voice && job.voice.is_none() && anchor.is_none() {
                anchor = Some(write_anchor(&clip)?);
            }
            clips.push(clip);
            progress(i + 1, total);
        }

        let clip = audio::concat(&clips, &plan.gaps_ms())?;
        Ok(Rendered {
            clip,
            jobs: total,
            cached,
            truncated,
        })
    }

    fn key(&self, text: &str, opts: &RenderOptions, voice: Option<&Path>) -> Result<CacheKey> {
        let b = self
            .key_base
            .clone()
            .str("text", text)
            .opt_str("lang", opts.lang.as_deref())
            .u64("seed", u64::from(opts.seed))
            .u64("max_frames", u64::from(self.max_frames));
        let b = match voice {
            Some(v) => b
                .file_content("voice", v)
                .with_context(|| format!("reading voice clip {}", v.display()))?,
            None => b.bytes("voice", &[]),
        };
        Ok(b.build())
    }
}

/// Writes the anchor clip to a temp wav the engine can read as a speaker reference.
fn write_anchor(clip: &Clip) -> Result<tempfile::TempPath> {
    let file = tempfile::Builder::new()
        .prefix("oido-anchor-")
        .suffix(".wav")
        .tempfile()?;
    let path = file.into_temp_path();
    audio::write_wav16(&path, clip)?;
    Ok(path)
}

/// Default cache location.
pub fn default_cache() -> Option<Cache> {
    dirs::cache_dir().map(|d| {
        Cache::new(
            d.join("oido").join("synth"),
            oido_core::cache::DEFAULT_MAX_BYTES,
        )
    })
}

/// Writes `clip` as 16-bit wav to `path`, creating parent directories.
///
/// # Errors
///
/// Fails if the directory or file cannot be written.
pub fn save(path: &Path, clip: &Clip) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    audio::write_wav16(path, clip).with_context(|| format!("writing {}", path.display()))
}

/// A fresh output path `<dir>/<prefix>-<unix nanos>.wav`.
pub fn timestamped_path(dir: &Path, prefix: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    dir.join(format!("{prefix}-{nanos}.wav"))
}
