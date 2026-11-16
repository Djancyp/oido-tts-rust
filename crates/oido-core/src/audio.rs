//! Mono PCM clips: stitching and wav I/O.

use std::io;
use std::path::Path;

/// Mono PCM samples in `[-1, 1]` at a fixed rate.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Clip {
    /// Samples per second.
    pub sample_rate: u32,
    /// Mono samples.
    pub samples: Vec<f32>,
}

impl Clip {
    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
}

/// Errors from reading or writing wav files.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AudioError {
    /// The file could not be read, written, or parsed.
    #[error(transparent)]
    Wav(#[from] hound::Error),
    /// Clips with different sample rates cannot be joined.
    #[error("sample rate mismatch: {0} Hz vs {1} Hz")]
    RateMismatch(u32, u32),
}

/// Joins clips in order. `gaps_ms[i]` is the timing before `clips[i + 1]`: positive inserts
/// silence, negative overlaps the next clip into the previous clip's tail by mixing samples
/// (capped at the shorter of the two clips).
///
/// # Errors
///
/// Fails if the clips' sample rates differ.
///
/// # Panics
///
/// Panics if `gaps_ms.len() + 1 != clips.len()` for non-empty `clips`.
pub fn concat(clips: &[Clip], gaps_ms: &[i32]) -> Result<Clip, AudioError> {
    let Some(first) = clips.first() else {
        return Ok(Clip::default());
    };
    assert_eq!(
        gaps_ms.len() + 1,
        clips.len(),
        "need one gap per clip boundary"
    );
    let rate = first.sample_rate;
    let mut out = Vec::with_capacity(clips.iter().map(|c| c.samples.len()).sum());
    out.extend_from_slice(&first.samples);
    for (pair, &gap_ms) in clips.windows(2).zip(gaps_ms) {
        let (prev, next) = (&pair[0], &pair[1]);
        if next.sample_rate != rate {
            return Err(AudioError::RateMismatch(rate, next.sample_rate));
        }
        let gap = i64::from(rate) * i64::from(gap_ms) / 1000;
        if gap >= 0 {
            out.resize(
                out.len() + usize::try_from(gap).expect("non-negative gap fits usize"),
                0.0,
            );
            out.extend_from_slice(&next.samples);
            continue;
        }
        let overlap = usize::try_from(-gap)
            .unwrap_or(usize::MAX)
            .min(prev.samples.len())
            .min(next.samples.len());
        let base = out.len() - overlap;
        for (o, &n) in out[base..].iter_mut().zip(&next.samples[..overlap]) {
            *o = (*o + n).clamp(-1.0, 1.0);
        }
        out.extend_from_slice(&next.samples[overlap..]);
    }
    Ok(Clip {
        sample_rate: rate,
        samples: out,
    })
}

/// Writes 16-bit mono PCM, converting samples the way llama.cpp's `llama-tts` does
/// (clamp, scale by 32767, truncate) so outputs compare byte for byte.
///
/// # Errors
///
/// Fails if the file cannot be created or written.
pub fn write_wav16(path: &Path, clip: &Clip) -> Result<(), AudioError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: clip.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in &clip.samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// Writes 32-bit float mono PCM (lossless round trip through [`read_wav`]).
///
/// # Errors
///
/// Fails if the file cannot be created or written.
pub fn write_wav_f32(path: &Path, clip: &Clip) -> Result<(), AudioError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: clip.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in &clip.samples {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}

/// Reads a mono wav (16-bit int or 32-bit float).
///
/// # Errors
///
/// Fails on I/O errors, multi-channel files, or unsupported sample formats.
pub fn read_wav(path: &Path) -> Result<Clip, AudioError> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(AudioError::Wav(hound::Error::Unsupported));
    }
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            reader.into_samples::<f32>().collect::<Result<_, _>>()?
        }
        (hound::SampleFormat::Int, 16) => reader
            .into_samples::<i16>()
            .map(|s| s.map(|v| f32::from(v) / 32767.0))
            .collect::<Result<_, _>>()?,
        _ => return Err(AudioError::Wav(hound::Error::Unsupported)),
    };
    Ok(Clip {
        sample_rate: spec.sample_rate,
        samples,
    })
}

impl From<io::Error> for AudioError {
    fn from(e: io::Error) -> Self {
        Self::Wav(hound::Error::IoError(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(samples: &[f32]) -> Clip {
        Clip {
            sample_rate: 1000,
            samples: samples.to_vec(),
        }
    }

    #[test]
    fn gap_inserts_silence() {
        let out = concat(&[clip(&[0.5]), clip(&[0.25])], &[3]).unwrap();
        assert_eq!(out.samples, vec![0.5, 0.0, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn negative_gap_overlaps_and_clamps() {
        let out = concat(&[clip(&[0.1, 0.9, 0.9]), clip(&[0.5, 0.5, 0.3])], &[-2]).unwrap();
        assert_eq!(out.samples, vec![0.1, 1.0, 1.0, 0.3]);
    }

    #[test]
    fn overlap_capped_by_shorter_clip() {
        let out = concat(&[clip(&[0.1, 0.2]), clip(&[0.3])], &[-50]).unwrap();
        assert_eq!(out.samples.len(), 2);
    }

    #[test]
    fn rate_mismatch_is_an_error() {
        let other = Clip {
            sample_rate: 2000,
            samples: vec![0.0],
        };
        assert!(matches!(
            concat(&[clip(&[0.0]), other], &[0]),
            Err(AudioError::RateMismatch(1000, 2000))
        ));
    }

    #[test]
    fn f32_wav_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.wav");
        let c = Clip {
            sample_rate: 24_000,
            samples: vec![0.0, 0.123_456, -1.0, 1.0],
        };
        write_wav_f32(&path, &c).unwrap();
        assert_eq!(read_wav(&path).unwrap(), c);
    }
}
