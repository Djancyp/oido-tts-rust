//! Text pipeline for oido, independent of the inference engine.
//!
//! - [`chunk`]: split text into model-sized chunks at sentence and clause boundaries.
//! - [`tags`]: inline `[pause:2s]` and `[style]` tags.
//! - [`script`]: `Speaker: line` podcast scripts.
//! - [`jobs`]: turn text or a script into an ordered list of synthesis jobs.
//! - [`audio`]: stitch clips with gaps or overlaps; wav I/O.
//! - [`cache`]: content-keyed on-disk cache of synthesized clips.

pub mod audio;
pub mod cache;
pub mod chunk;
pub mod jobs;
pub mod script;
pub mod tags;
