//! Resident text-to-speech model built on llama.cpp's audio-generation API.
//!
//! [`Engine::load`] loads the backbone and its `mmproj` companion once; every
//! [`Engine::synthesize`] call afterwards reuses them, so the per-request cost is generation
//! only. Supported pipelines are whatever libmtmd implements (currently Qwen3-TTS and
//! Pocket TTS), detected from the `mmproj` file.
//!
//! An `Engine` is deliberately neither `Send` nor `Sync`: construct it on the thread that will
//! run it.

use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::Once;

use llama_sys as sys;

/// Context size used when [`EngineConfig::n_ctx`] is not changed.
///
/// One request needs its prompt (text tokens plus a few control rows) plus one position per
/// generated frame. 2048 covers a long chunk at the default frame cap. The model's training
/// context is far larger, and sizing the KV cache for it cost ~3.4 GB of RAM in measurements
/// against Qwen3-TTS 1.7B with identical output.
pub const DEFAULT_N_CTX: u32 = 2048;

/// Frame cap used when a request does not set one; matches llama-tts's default.
pub const DEFAULT_MAX_FRAMES: u32 = 512;

/// Prompt batch size. llama.cpp forces `n_batch == n_ubatch` when embeddings are enabled.
const N_BATCH: u32 = 512;

/// Positions reserved for the prompt when validating a request's frame cap against `n_ctx`.
const PROMPT_RESERVE: u32 = 256;

/// Errors from loading a model or synthesizing speech.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// The backbone gguf could not be loaded.
    #[error("failed to load model {}", .0.display())]
    ModelLoad(PathBuf),
    /// The `mmproj` gguf could not be loaded.
    #[error("failed to load mmproj {}", .0.display())]
    MmprojLoad(PathBuf),
    /// A llama.cpp context could not be created (usually out of memory).
    #[error("failed to create inference context")]
    ContextInit,
    /// The `mmproj` file has no audio-generation pipeline.
    #[error("{} does not support audio generation", .0.display())]
    NotAudioModel(PathBuf),
    /// The speaker reference clip could not be decoded.
    #[error("failed to load speaker reference {}", .0.display())]
    SpeakerLoad(PathBuf),
    /// A path or language string contains an interior NUL byte.
    #[error("argument contains a NUL byte: {0:?}")]
    Nul(String),
    /// The request cannot fit in the configured context.
    #[error("max_frames {max_frames} does not fit in n_ctx {n_ctx}")]
    FrameCapTooLarge {
        /// Requested frame cap.
        max_frames: u32,
        /// Configured context size.
        n_ctx: u32,
    },
    /// Generation failed inside llama.cpp.
    #[error("generation failed during {stage} (after {frames} frames)")]
    Generation {
        /// Pipeline stage that failed.
        stage: &'static str,
        /// Frames generated before the failure.
        frames: u32,
    },
    /// The pipeline rejected the request before generating.
    #[error("model rejected the input: unsupported language, or a speaker clip it cannot encode")]
    InvalidInput,
    /// The progress callback asked to stop.
    #[error("synthesis cancelled")]
    Cancelled,
}

/// Which audio-generation pipeline the loaded `mmproj` provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelKind {
    /// Qwen3-TTS (12 Hz codec, x-vector voice cloning, language selection).
    Qwen3Tts,
    /// Kyutai Pocket TTS (speaker reference required, language fixed by the weights).
    PocketTts,
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Qwen3Tts => "qwen3-tts",
            Self::PocketTts => "pocket-tts",
        })
    }
}

/// How to load a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    /// Backbone gguf.
    pub model: PathBuf,
    /// Matching `mmproj` gguf.
    pub mmproj: PathBuf,
    /// KV-cache size in positions; see [`DEFAULT_N_CTX`].
    pub n_ctx: u32,
    /// CPU threads for the backbone and the audio pipeline; `None` uses [`default_threads`].
    pub threads: Option<u32>,
    /// Backbone layers to offload to a GPU backend; `None` keeps llama.cpp's default.
    pub gpu_layers: Option<i32>,
}

impl EngineConfig {
    /// Config with default context size, threads, and offload.
    pub fn new(model: impl Into<PathBuf>, mmproj: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            mmproj: mmproj.into(),
            n_ctx: DEFAULT_N_CTX,
            threads: None,
            gpu_layers: None,
        }
    }
}

/// Sampling parameters for the backbone's codec tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// Top-k cutoff.
    pub top_k: i32,
    /// Nucleus cutoff.
    pub top_p: f32,
    /// Min-p cutoff.
    pub min_p: f32,
    /// Temperature.
    pub temp: f32,
    /// Repetition penalty (1.0 disables).
    pub repeat_penalty: f32,
}

impl Default for Sampling {
    /// llama-tts's defaults (common sampling defaults plus its 1.05 repetition penalty).
    fn default() -> Self {
        Self {
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            temp: 0.8,
            repeat_penalty: 1.05,
        }
    }
}

/// One synthesis request.
#[derive(Debug, Clone)]
pub struct Request<'a> {
    /// Text to speak.
    pub text: &'a str,
    /// Language code or name (`en`, `english`, `zh`, ...); `None` uses the model default.
    pub lang: Option<&'a str>,
    /// Reference clip (wav, mp3, flac) to clone; `None` uses the model's own voice.
    pub speaker: Option<&'a Path>,
    /// RNG seed; `None` draws a random one. A fixed seed makes output reproducible.
    pub seed: Option<u32>,
    /// Upper bound on generated frames.
    pub max_frames: u32,
    /// Sampling override; `None` uses the model's metadata defaults.
    pub sampling: Option<Sampling>,
}

impl<'a> Request<'a> {
    /// Request with the model's voice, default language, random seed, and default frame cap.
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            lang: None,
            speaker: None,
            seed: None,
            max_frames: DEFAULT_MAX_FRAMES,
            sampling: None,
        }
    }
}

/// Generated mono audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    /// Samples per second.
    pub sample_rate: u32,
    /// Mono PCM in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Codec frames generated.
    pub frames: u32,
}

impl Audio {
    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
}

/// A loaded model, ready to synthesize.
pub struct Engine {
    model: NonNull<sys::llama_model>,
    ctx: NonNull<sys::llama_context>,
    mtmd: NonNull<sys::mtmd_context>,
    gen_audio: NonNull<sys::mtmd_helper_gen_audio>,
    kind: ModelKind,
    sample_rate: u32,
    n_vocab: i32,
    n_ctx: u32,
    defaults: Sampling,
}

impl fmt::Debug for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("kind", &self.kind)
            .field("sample_rate", &self.sample_rate)
            .field("n_ctx", &self.n_ctx)
            .field("defaults", &self.defaults)
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Loads the backbone and `mmproj` and prepares a generation context.
    ///
    /// # Errors
    ///
    /// Fails if either file cannot be loaded, the context cannot be allocated, or the
    /// `mmproj` has no audio-generation pipeline.
    pub fn load(config: &EngineConfig) -> Result<Self, EngineError> {
        init_backend();
        let threads = config.threads.unwrap_or_else(default_threads);
        let threads = i32::try_from(threads).unwrap_or(i32::MAX);

        let model_path = path_cstring(&config.model)?;
        // SAFETY: returns a plain struct of defaults.
        let mut model_params = unsafe { sys::llama_model_default_params() };
        if let Some(layers) = config.gpu_layers {
            model_params.n_gpu_layers = layers;
        }
        // SAFETY: the path is a valid NUL-terminated string that outlives the call.
        let model = unsafe { sys::llama_model_load_from_file(model_path.as_ptr(), model_params) };
        let model =
            NonNull::new(model).ok_or_else(|| EngineError::ModelLoad(config.model.clone()))?;
        // From here on, `Partial` frees whatever has been created if a later step fails.
        let mut partial = Partial {
            model,
            ctx: None,
            mtmd: None,
        };

        // SAFETY: returns a plain struct of defaults.
        let mut ctx_params = unsafe { sys::llama_context_default_params() };
        ctx_params.n_ctx = config.n_ctx;
        ctx_params.n_batch = N_BATCH;
        ctx_params.n_ubatch = N_BATCH;
        ctx_params.n_seq_max = 1;
        ctx_params.n_threads = threads;
        ctx_params.n_threads_batch = threads;
        // The audio pipeline reads the backbone's hidden state after every step.
        ctx_params.embeddings = true;
        ctx_params.no_perf = true;
        // SAFETY: `model` is a live model.
        let ctx = unsafe { sys::llama_init_from_model(model.as_ptr(), ctx_params) };
        let ctx = NonNull::new(ctx).ok_or(EngineError::ContextInit)?;
        partial.ctx = Some(ctx);

        let mmproj_path = path_cstring(&config.mmproj)?;
        // SAFETY: returns a plain struct of defaults.
        let mut mtmd_params = unsafe { sys::mtmd_context_params_default() };
        mtmd_params.print_timings = false;
        // The code predictor and vocoder run here, once per frame; libmtmd's default is 4.
        mtmd_params.n_threads = threads;
        // SAFETY: valid path string and live model; the model outlives the mtmd context
        // because `Drop` frees the mtmd context first.
        let mtmd =
            unsafe { sys::mtmd_init_from_file(mmproj_path.as_ptr(), model.as_ptr(), mtmd_params) };
        let mtmd =
            NonNull::new(mtmd).ok_or_else(|| EngineError::MmprojLoad(config.mmproj.clone()))?;
        partial.mtmd = Some(mtmd);

        // SAFETY: `mtmd` is a live context.
        let info = unsafe { sys::mtmd_gen_audio_get_info(mtmd.as_ptr()) };
        let kind = match info.type_ {
            sys::MTMD_GEN_AUDIO_TYPE_QWEN3TTS => ModelKind::Qwen3Tts,
            sys::MTMD_GEN_AUDIO_TYPE_POCKETTTS => ModelKind::PocketTts,
            _ => return Err(EngineError::NotAudioModel(config.mmproj.clone())),
        };
        let sample_rate = u32::try_from(info.sample_rate)
            .map_err(|_| EngineError::NotAudioModel(config.mmproj.clone()))?;

        // SAFETY: both contexts are live and outlive the helper (`Drop` frees it first).
        let gen_audio = unsafe { sys::mtmd_helper_gen_audio_init(ctx.as_ptr(), mtmd.as_ptr()) };
        let gen_audio = NonNull::new(gen_audio).ok_or(EngineError::ContextInit)?;

        // SAFETY: `model` is live; the vocab is owned by the model.
        let n_vocab =
            unsafe { sys::llama_vocab_n_tokens(sys::llama_model_get_vocab(model.as_ptr())) };
        let defaults = metadata_sampling(model);
        std::mem::forget(partial);

        tracing::debug!(%kind, sample_rate, n_ctx = config.n_ctx, threads, ?defaults, "engine loaded");
        Ok(Self {
            model,
            ctx,
            mtmd,
            gen_audio,
            kind,
            sample_rate,
            n_vocab,
            n_ctx: config.n_ctx,
            defaults,
        })
    }

    /// The loaded pipeline.
    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// Output sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Sampling defaults read from the model's metadata.
    pub fn default_sampling(&self) -> Sampling {
        self.defaults
    }

    /// Synthesizes `req`, calling `progress` with the frame count after every frame.
    ///
    /// Returning [`ControlFlow::Break`] from `progress` stops generation with
    /// [`EngineError::Cancelled`].
    ///
    /// # Errors
    ///
    /// Fails if the speaker clip cannot be decoded, the frame cap does not fit the context,
    /// generation fails inside llama.cpp, or `progress` cancels.
    pub fn synthesize(
        &mut self,
        req: &Request<'_>,
        mut progress: impl FnMut(u32) -> ControlFlow<()>,
    ) -> Result<Audio, EngineError> {
        if req.max_frames + PROMPT_RESERVE > self.n_ctx {
            return Err(EngineError::FrameCapTooLarge {
                max_frames: req.max_frames,
                n_ctx: self.n_ctx,
            });
        }
        let sampling = req.sampling.unwrap_or(self.defaults);
        let seed = req.seed.unwrap_or(sys::LLAMA_DEFAULT_SEED);
        let lang = req
            .lang
            .map(|l| CString::new(l).map_err(|_| EngineError::Nul(l.to_owned())))
            .transpose()?;
        let speaker = req
            .speaker
            .map(|p| Bitmap::load(self.mtmd, p))
            .transpose()?;

        // The helper restarts positions at 0 on every input, so stale KV entries from the
        // previous request must go.
        // SAFETY: `ctx` is live; clearing memory has no other preconditions.
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(self.ctx.as_ptr()), true) };

        let inp = sys::mtmd_helper_gen_audio_inp {
            seq_id: 0,
            prompt: req.text.as_ptr().cast::<c_char>(),
            prompt_len: req.text.len(),
            speaker_ref: speaker
                .as_ref()
                .map_or(ptr::null_mut(), |b| b.bitmap.as_ptr()),
            lang: lang.as_ref().map_or(ptr::null(), |l| l.as_ptr()),
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            seed,
            out_type: sys::MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM,
        };
        // SAFETY: `inp`'s pointers (text, lang, bitmap) are valid for the call; the helper
        // copies the prompt and encodes the speaker before returning.
        let rc = unsafe { sys::mtmd_helper_gen_audio_set_input(self.gen_audio.as_ptr(), &inp) };
        drop(speaker);
        if rc != 0 {
            return Err(EngineError::InvalidInput);
        }

        loop {
            // SAFETY: input was set above.
            let remaining = unsafe {
                sys::mtmd_helper_gen_audio_step_prompt(self.gen_audio.as_ptr(), N_BATCH as i32)
            };
            match remaining {
                0 => break,
                r if r < 0 => {
                    return Err(EngineError::Generation {
                        stage: "prompt",
                        frames: 0,
                    });
                }
                _ => {}
            }
        }

        let sampler = Sampler::new(self.n_vocab, sampling, seed);
        let mut sampled = sampler.sample(self.ctx);
        // SAFETY: the prompt's last position was decoded with output enabled.
        let mut h_state: *const f32 =
            unsafe { sys::llama_get_embeddings_ith(self.ctx.as_ptr(), -1) };
        let mut frames = 0u32;
        let mut stop = false;
        while !stop && frames < req.max_frames {
            let mut h_next: *const f32 = ptr::null();
            // SAFETY: `h_state` points at a hidden state owned by llama.cpp or the helper,
            // valid until the next decode/step, which is this call.
            let rc = unsafe {
                sys::mtmd_helper_gen_audio_step_gen(
                    self.gen_audio.as_ptr(),
                    sampled,
                    h_state,
                    &mut h_next,
                    &mut stop,
                )
            };
            if rc != 0 {
                return Err(EngineError::Generation {
                    stage: "step_gen",
                    frames,
                });
            }
            if h_next.is_null() {
                break;
            }
            frames += 1;
            h_state = h_next;
            sampled = sampler.sample(self.ctx);
            if progress(frames).is_break() {
                return Err(EngineError::Cancelled);
            }
        }

        let mut rate = 0i32;
        let mut data: *const c_char = ptr::null();
        let mut len = 0usize;
        // SAFETY: generation finished; the output buffer stays valid until the next
        // get_output/reset, and is copied out before either.
        let rc = unsafe {
            sys::mtmd_helper_gen_audio_get_output(
                self.gen_audio.as_ptr(),
                &mut rate,
                &mut data,
                &mut len,
                ptr::null_mut(),
            )
        };
        if rc != 0 || (data.is_null() && len > 0) {
            return Err(EngineError::Generation {
                stage: "vocoder",
                frames,
            });
        }
        let samples = if len == 0 {
            Vec::new()
        } else {
            // SAFETY: PCM output is `len` bytes of native-endian f32 owned by the helper.
            // `read_unaligned` copes with a buffer that is not f32-aligned.
            (0..len / size_of::<f32>())
                .map(|i| unsafe { data.cast::<f32>().add(i).read_unaligned() })
                .collect()
        };
        Ok(Audio {
            sample_rate: u32::try_from(rate).unwrap_or(self.sample_rate),
            samples,
            frames,
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: each pointer is live and freed once, dependents before what they borrow.
        unsafe {
            sys::mtmd_helper_gen_audio_free(self.gen_audio.as_ptr());
            sys::mtmd_free(self.mtmd.as_ptr());
            sys::llama_free(self.ctx.as_ptr());
            sys::llama_model_free(self.model.as_ptr());
        }
    }
}

/// Frees partially constructed state when [`Engine::load`] fails midway.
struct Partial {
    model: NonNull<sys::llama_model>,
    ctx: Option<NonNull<sys::llama_context>>,
    mtmd: Option<NonNull<sys::mtmd_context>>,
}

impl Drop for Partial {
    fn drop(&mut self) {
        // SAFETY: each pointer is live and owned by this guard.
        unsafe {
            if let Some(m) = self.mtmd {
                sys::mtmd_free(m.as_ptr());
            }
            if let Some(c) = self.ctx {
                sys::llama_free(c.as_ptr());
            }
            sys::llama_model_free(self.model.as_ptr());
        }
    }
}

/// Decoded speaker reference.
struct Bitmap {
    bitmap: NonNull<sys::mtmd_bitmap>,
}

impl Bitmap {
    fn load(mtmd: NonNull<sys::mtmd_context>, path: &Path) -> Result<Self, EngineError> {
        let c_path = path_cstring(path)?;
        // SAFETY: live context, valid path; returns a plain struct of defaults.
        let wrapper = unsafe {
            sys::mtmd_helper_bitmap_init_from_file(
                mtmd.as_ptr(),
                c_path.as_ptr(),
                false,
                sys::mtmd_helper_init_opt_default(),
            )
        };
        if !wrapper.video_ctx.is_null() {
            // SAFETY: audio input never needs the video context; free the one we were handed.
            unsafe { sys::mtmd_helper_video_free(wrapper.video_ctx) };
        }
        NonNull::new(wrapper.bitmap)
            .map(|bitmap| Self { bitmap })
            .ok_or_else(|| EngineError::SpeakerLoad(path.to_owned()))
    }
}

impl Drop for Bitmap {
    fn drop(&mut self) {
        // SAFETY: owned bitmap, freed once.
        unsafe { sys::mtmd_bitmap_free(self.bitmap.as_ptr()) };
    }
}

/// Codec-token sampler chain, mirroring llama-tts's common sampler.
struct Sampler(NonNull<sys::llama_sampler>);

impl Sampler {
    fn new(n_vocab: i32, s: Sampling, seed: u32) -> Self {
        // SAFETY: every init returns a fresh sampler that the chain takes ownership of.
        unsafe {
            let chain = sys::llama_sampler_chain_init(sys::llama_sampler_chain_default_params());
            // Penalty window -1, as llama-tts sets it.
            sys::llama_sampler_chain_add(
                chain,
                sys::llama_sampler_init_penalties(n_vocab, -1, s.repeat_penalty, 0.0, 0.0),
            );
            sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_k(s.top_k));
            sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_p(s.top_p, 1));
            sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_min_p(s.min_p, 1));
            sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_temp(s.temp));
            sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_dist(seed));
            Self(NonNull::new(chain).expect("llama_sampler_chain_init returned null"))
        }
    }

    fn sample(&self, ctx: NonNull<sys::llama_context>) -> sys::llama_token {
        // SAFETY: live chain and context; the last decode produced logits at index -1.
        // `llama_sampler_sample` also accepts the token into the chain.
        unsafe { sys::llama_sampler_sample(self.0.as_ptr(), ctx.as_ptr(), -1) }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        // SAFETY: owned chain, freed once; it frees its members.
        unsafe { sys::llama_sampler_free(self.0.as_ptr()) };
    }
}

/// Physical core count (llama.cpp's own default), falling back to logical CPUs.
pub fn default_threads() -> u32 {
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
    let physical = std::fs::read_dir("/sys/devices/system/cpu")
        .ok()
        .map(|dir| {
            let mut cores = std::collections::BTreeSet::new();
            for entry in dir.flatten() {
                let name = entry.file_name();
                let is_cpu = name
                    .to_str()
                    .and_then(|n| n.strip_prefix("cpu"))
                    .is_some_and(|rest| {
                        !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
                    });
                if is_cpu
                    && let Ok(s) =
                        std::fs::read_to_string(entry.path().join("topology/thread_siblings_list"))
                {
                    cores.insert(s.trim().to_owned());
                }
            }
            cores.len()
        })
        .filter(|&n| n > 0)
        .unwrap_or(logical);
    u32::try_from(physical).unwrap_or(4)
}

fn init_backend() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: process-wide one-time init; the callback is a plain `extern "C" fn`.
        unsafe {
            sys::llama_backend_init();
            sys::llama_log_set(Some(log_to_tracing), ptr::null_mut());
            sys::mtmd_helper_log_set(Some(log_to_tracing), ptr::null_mut());
        }
    });
}

unsafe extern "C" fn log_to_tracing(
    level: sys::ggml_log_level,
    text: *const c_char,
    _user: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: llama.cpp passes a NUL-terminated string valid for this call.
    let msg = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let msg = msg.trim_end();
    if msg.is_empty() {
        return;
    }
    match level {
        sys::GGML_LOG_LEVEL_ERROR => tracing::error!(target: "llama", "{msg}"),
        sys::GGML_LOG_LEVEL_WARN => tracing::warn!(target: "llama", "{msg}"),
        sys::GGML_LOG_LEVEL_INFO => tracing::debug!(target: "llama", "{msg}"),
        _ => tracing::trace!(target: "llama", "{msg}"),
    }
}

/// Model metadata sampling defaults (`general.sampling.*`), over llama-tts's own defaults.
fn metadata_sampling(model: NonNull<sys::llama_model>) -> Sampling {
    let read = |key: &CStr| -> Option<String> {
        let mut buf = [0u8; 64];
        // SAFETY: live model; buffer length passed with the buffer.
        let n = unsafe {
            sys::llama_model_meta_val_str(
                model.as_ptr(),
                key.as_ptr(),
                buf.as_mut_ptr().cast(),
                buf.len(),
            )
        };
        let n = usize::try_from(n)
            .ok()
            .filter(|&n| n > 0 && n < buf.len())?;
        String::from_utf8(buf[..n].to_vec()).ok()
    };
    let mut s = Sampling::default();
    if let Some(v) = read(c"general.sampling.top_k").and_then(|v| v.parse().ok()) {
        s.top_k = v;
    }
    if let Some(v) = read(c"general.sampling.top_p").and_then(|v| v.parse().ok()) {
        s.top_p = v;
    }
    if let Some(v) = read(c"general.sampling.temp").and_then(|v| v.parse().ok()) {
        s.temp = v;
    }
    s
}

fn path_cstring(path: &Path) -> Result<CString, EngineError> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| EngineError::Nul(path.display().to_string()))
}
