//! On-disk cache of synthesized chunks, keyed on everything that affects the audio.
//!
//! Entries are lossless float wavs. A hit refreshes the entry's mtime; once the cache grows
//! past its byte budget the least recently used entries are deleted. Caching is an
//! optimization only: every failure reads as a miss and never fails synthesis.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::audio::{self, Clip};

/// Default byte budget.
pub const DEFAULT_MAX_BYTES: u64 = 500 * 1024 * 1024;

/// Content-derived identity of one synthesis call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey(String);

impl CacheKey {
    /// Starts a key.
    pub fn builder() -> CacheKeyBuilder {
        CacheKeyBuilder(blake3::Hasher::new())
    }

    /// Hex digest.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Accumulates named fields into a [`CacheKey`]. Field names and lengths are hashed, so
/// adjacent values cannot run together.
#[derive(Debug, Clone)]
pub struct CacheKeyBuilder(blake3::Hasher);

impl CacheKeyBuilder {
    /// Adds raw bytes.
    pub fn bytes(mut self, name: &str, value: &[u8]) -> Self {
        for part in [name.as_bytes(), value] {
            self.0.update(&(part.len() as u64).to_le_bytes());
            self.0.update(part);
        }
        self
    }

    /// Adds a string.
    pub fn str(self, name: &str, value: &str) -> Self {
        self.bytes(name, value.as_bytes())
    }

    /// Adds an optional string, distinguishing `None` from `Some("")`.
    pub fn opt_str(self, name: &str, value: Option<&str>) -> Self {
        match value {
            Some(v) => self.str(name, v),
            None => self.bytes(name, &[0xff]),
        }
    }

    /// Adds an integer.
    pub fn u64(self, name: &str, value: u64) -> Self {
        self.bytes(name, &value.to_le_bytes())
    }

    /// Adds a file's identity (path, size, mtime) without reading it; for large model files.
    ///
    /// # Errors
    ///
    /// Fails if the file's metadata cannot be read.
    pub fn file_identity(self, name: &str, path: &Path) -> io::Result<Self> {
        let meta = fs::metadata(path)?;
        let mtime = meta
            .modified()?
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        Ok(self
            .bytes(name, path.as_os_str().as_encoded_bytes())
            .u64("size", meta.len())
            .bytes("mtime", &mtime.to_le_bytes()))
    }

    /// Adds a file's full content; for reference clips, whose path may change while their
    /// content does not.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be read.
    pub fn file_content(mut self, name: &str, path: &Path) -> io::Result<Self> {
        let mut h = blake3::Hasher::new();
        h.update_reader(File::open(path)?)?;
        self.0.update(name.as_bytes());
        self.0.update(h.finalize().as_bytes());
        Ok(self)
    }

    /// Finishes the key.
    pub fn build(self) -> CacheKey {
        CacheKey(self.0.finalize().to_hex().to_string())
    }
}

/// A directory of cached clips.
#[derive(Debug, Clone)]
pub struct Cache {
    dir: PathBuf,
    max_bytes: u64,
}

impl Cache {
    /// Cache in `dir` (created on first write) holding at most `max_bytes`.
    pub fn new(dir: impl Into<PathBuf>, max_bytes: u64) -> Self {
        Self {
            dir: dir.into(),
            max_bytes,
        }
    }

    /// The cached clip for `key`, if any.
    pub fn get(&self, key: &CacheKey) -> Option<Clip> {
        let path = self.path(key);
        let clip = audio::read_wav(&path).ok()?;
        // Refresh recency for eviction; a failure only makes this entry evict sooner.
        if let Ok(f) = File::options().append(true).open(&path) {
            let _ = f.set_modified(SystemTime::now());
        }
        Some(clip)
    }

    /// Stores `clip` under `key`, then evicts old entries past the budget.
    ///
    /// # Errors
    ///
    /// Fails if the entry cannot be written; eviction errors are ignored.
    pub fn put(&self, key: &CacheKey, clip: &Clip) -> Result<(), audio::AudioError> {
        fs::create_dir_all(&self.dir)?;
        // Write then rename, so a concurrent reader never sees a partial file.
        let tmp = tempfile::Builder::new()
            .suffix(".tmp")
            .tempfile_in(&self.dir)?;
        audio::write_wav_f32(tmp.path(), clip)?;
        tmp.persist(self.path(key)).map_err(|e| e.error)?;
        self.evict();
        Ok(())
    }

    fn path(&self, key: &CacheKey) -> PathBuf {
        self.dir.join(format!("{}.wav", key.as_str()))
    }

    /// Deletes least recently used entries until the cache fits its budget.
    fn evict(&self) {
        let Ok(dir) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut entries: Vec<(SystemTime, u64, PathBuf)> = dir
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
            .filter_map(|e| {
                let meta = e.metadata().ok()?;
                Some((meta.modified().ok()?, meta.len(), e.path()))
            })
            .collect();
        let mut total: u64 = entries.iter().map(|(_, len, _)| len).sum();
        entries.sort();
        for (_, len, path) in entries {
            if total <= self.max_bytes {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                total -= len;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(n: usize) -> Clip {
        Clip {
            sample_rate: 24_000,
            samples: vec![0.25; n],
        }
    }

    #[test]
    fn key_fields_do_not_run_together() {
        let a = CacheKey::builder().str("a", "xy").str("b", "z").build();
        let b = CacheKey::builder().str("a", "x").str("b", "yz").build();
        assert_ne!(a, b);
        assert_ne!(
            CacheKey::builder().opt_str("l", None).build(),
            CacheKey::builder().opt_str("l", Some("")).build()
        );
    }

    #[test]
    fn put_then_get() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path().join("c"), DEFAULT_MAX_BYTES);
        let key = CacheKey::builder().str("text", "hi").build();
        assert!(cache.get(&key).is_none());
        cache.put(&key, &clip(10)).unwrap();
        assert_eq!(cache.get(&key), Some(clip(10)));
    }

    #[test]
    fn evicts_least_recently_used() {
        let dir = tempfile::tempdir().unwrap();
        // Each 1000-sample float entry is ~4 KB; the budget holds two.
        let cache = Cache::new(dir.path(), 9_000);
        let keys: Vec<CacheKey> = (0..3)
            .map(|i| CacheKey::builder().u64("i", i).build())
            .collect();
        let t0 = SystemTime::now() - std::time::Duration::from_secs(100);
        for (i, key) in keys.iter().take(2).enumerate() {
            cache.put(key, &clip(1000)).unwrap();
            let f = File::options().append(true).open(cache.path(key)).unwrap();
            f.set_modified(t0 + std::time::Duration::from_secs(i as u64))
                .unwrap();
        }
        assert!(
            cache.get(&keys[0]).is_some(),
            "touch key 0 so key 1 is the oldest"
        );
        cache.put(&keys[2], &clip(1000)).unwrap();
        assert!(cache.get(&keys[0]).is_some());
        assert!(cache.get(&keys[1]).is_none());
        assert!(cache.get(&keys[2]).is_some());
    }
}
