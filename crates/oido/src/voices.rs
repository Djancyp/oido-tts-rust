//! Named voice library: reference clips stored under the user data directory.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Audio formats the engine can decode as a reference clip.
const EXTENSIONS: &[&str] = &["wav", "mp3", "flac"];

/// Longest accepted voice name.
const MAX_NAME_LEN: usize = 64;

/// Directory of saved voices.
#[derive(Debug, Clone)]
pub struct Voices {
    dir: PathBuf,
}

impl Voices {
    /// The default library (`$XDG_DATA_HOME/oido/voices`).
    ///
    /// # Errors
    ///
    /// Fails if the platform has no data directory.
    pub fn open_default() -> Result<Self> {
        let dir = dirs::data_dir().context("no user data directory")?;
        Ok(Self {
            dir: dir.join("oido").join("voices"),
        })
    }

    /// Copies `clip` into the library as `name`, replacing any voice with that name.
    ///
    /// # Errors
    ///
    /// Fails on an invalid name, an unsupported format, or I/O errors.
    pub fn add(&self, name: &str, clip: &Path) -> Result<PathBuf> {
        validate_name(name)?;
        let ext = clip
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .filter(|e| EXTENSIONS.contains(&e.as_str()))
            .with_context(|| {
                format!("{} is not a {} file", clip.display(), EXTENSIONS.join("/"))
            })?;
        fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;
        self.remove(name).ok();
        let dest = self.dir.join(format!("{name}.{ext}"));
        fs::copy(clip, &dest).with_context(|| format!("copying {}", clip.display()))?;
        Ok(dest)
    }

    /// Deletes voice `name`.
    ///
    /// # Errors
    ///
    /// Fails if no such voice exists.
    pub fn remove(&self, name: &str) -> Result<()> {
        let path = self
            .find(name)
            .with_context(|| format!("no voice named {name:?}"))?;
        fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))
    }

    /// Saved voices as `(name, path)`, sorted by name.
    ///
    /// # Errors
    ///
    /// Fails if the library directory exists but cannot be read.
    pub fn list(&self) -> Result<Vec<(String, PathBuf)>> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e).with_context(|| format!("reading {}", self.dir.display())),
        };
        let mut voices: Vec<(String, PathBuf)> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| EXTENSIONS.contains(&e))
            })
            .filter_map(|p| Some((p.file_stem()?.to_str()?.to_owned(), p)))
            .collect();
        voices.sort();
        Ok(voices)
    }

    /// Resolves `spec` to a clip: an existing file path, else a saved voice name.
    ///
    /// # Errors
    ///
    /// Fails if `spec` is neither.
    pub fn resolve(&self, spec: &str) -> Result<PathBuf> {
        let path = Path::new(spec);
        if path.is_file() {
            return Ok(path.to_owned());
        }
        if validate_name(spec).is_ok()
            && let Some(p) = self.find(spec)
        {
            return Ok(p);
        }
        let names: Vec<String> = self.list()?.into_iter().map(|(n, _)| n).collect();
        bail!(
            "{spec:?} is not a file or a saved voice (saved: {})",
            if names.is_empty() {
                "none".to_owned()
            } else {
                names.join(", ")
            }
        )
    }

    fn find(&self, name: &str) -> Option<PathBuf> {
        EXTENSIONS
            .iter()
            .map(|ext| self.dir.join(format!("{name}.{ext}")))
            .find(|p| p.is_file())
    }
}

fn validate_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if !ok {
        bail!("voice names use 1-{MAX_NAME_LEN} letters, digits, '_' or '-'");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_resolve_remove() {
        let tmp = tempfile::tempdir().unwrap();
        let voices = Voices {
            dir: tmp.path().join("voices"),
        };
        let clip = tmp.path().join("ref.WAV");
        fs::write(&clip, b"RIFF").unwrap();

        let saved = voices.add("alice", &clip).unwrap();
        assert_eq!(saved.extension().unwrap(), "wav");
        assert_eq!(voices.resolve("alice").unwrap(), saved);
        assert_eq!(voices.resolve(clip.to_str().unwrap()).unwrap(), clip);
        assert_eq!(voices.list().unwrap(), vec![("alice".to_owned(), saved)]);

        voices.remove("alice").unwrap();
        assert!(voices.resolve("alice").is_err());
    }

    #[test]
    fn rejects_path_like_names() {
        let tmp = tempfile::tempdir().unwrap();
        let voices = Voices {
            dir: tmp.path().to_owned(),
        };
        assert!(voices.add("../evil", Path::new("x.wav")).is_err());
        assert!(voices.add("a b", Path::new("x.wav")).is_err());
    }
}
