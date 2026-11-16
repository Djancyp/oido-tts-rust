//! The engine thread for long-running servers.
//!
//! One thread owns the model and runs requests one at a time from a queue, so concurrent
//! callers share a single copy of the weights. The model loads on the first request and is
//! dropped after an idle period, so a server left running does not hold gigabytes of RAM.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use oido_core::jobs::Plan;
use oido_engine::EngineConfig;
use tokio::sync::oneshot;

use crate::synth::{self, RenderOptions, Rendered, Synth};

struct Task {
    plan: Plan,
    opts: RenderOptions,
    out: PathBuf,
    cancel: Arc<AtomicBool>,
    reply: oneshot::Sender<Result<Rendered>>,
}

/// Handle to the engine thread; dropping every handle stops the thread.
#[derive(Debug)]
pub struct Worker {
    tx: mpsc::Sender<Task>,
}

impl Worker {
    /// Starts the engine thread. `idle_unload` of `None` keeps the model loaded forever.
    ///
    /// # Panics
    ///
    /// Panics if the OS refuses to spawn a thread.
    pub fn spawn(config: EngineConfig, idle_unload: Option<Duration>) -> Self {
        let (tx, rx) = mpsc::channel::<Task>();
        thread::Builder::new()
            .name("oido-engine".into())
            .spawn(move || {
                let mut synth: Option<Synth> = None;
                loop {
                    let task = match idle_unload {
                        Some(idle) => match rx.recv_timeout(idle) {
                            Ok(t) => t,
                            Err(RecvTimeoutError::Timeout) => {
                                if synth.take().is_some() {
                                    tracing::info!("model unloaded after {}s idle", idle.as_secs());
                                }
                                continue;
                            }
                            Err(RecvTimeoutError::Disconnected) => break,
                        },
                        None => match rx.recv() {
                            Ok(t) => t,
                            Err(_) => break,
                        },
                    };
                    if task.cancel.load(Ordering::Relaxed) {
                        continue; // caller gave up while queued
                    }
                    let result = run_task(&mut synth, &config, &task);
                    let _ = task.reply.send(result);
                }
            })
            .expect("spawning engine thread");
        Self { tx }
    }

    /// Queues `plan`, waits for it, and writes the result to `out`.
    ///
    /// Generation stops at the next frame if `timeout` passes or this future is dropped.
    ///
    /// # Errors
    ///
    /// Fails if synthesis fails, times out, or the engine thread has stopped.
    pub async fn render(
        &self,
        plan: Plan,
        opts: RenderOptions,
        out: PathBuf,
        timeout: Duration,
    ) -> Result<Rendered> {
        let cancel = Arc::new(AtomicBool::new(false));
        let _cancel_on_drop = CancelOnDrop(Arc::clone(&cancel));
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Task {
                plan,
                opts,
                out,
                cancel,
                reply,
            })
            .map_err(|_| anyhow!("engine thread stopped"))?;
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => bail!("engine thread stopped"),
            Err(_) => bail!("timed out after {}s", timeout.as_secs()),
        }
    }
}

fn run_task(synth: &mut Option<Synth>, config: &EngineConfig, task: &Task) -> Result<Rendered> {
    let s = match synth {
        Some(s) => s,
        None => synth.insert(Synth::load(config, synth::default_cache())?),
    };
    let rendered = s.render(&task.plan, &task.opts, &task.cancel, |done, total| {
        tracing::debug!("chunk {done}/{total}");
    })?;
    synth::save(&task.out, &rendered.clip)?;
    Ok(rendered)
}

/// Sets the flag when dropped, so an abandoned request stops generating.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
