//! Finding model files when no path was given.
//!
//! Searches, in order, `$XDG_DATA_HOME/oido/models/<name>/` (models converted locally, e.g.
//! Pocket TTS) and the Hugging Face hub cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
//! `~/.cache/huggingface/hub`) for a backbone gguf with its `mmproj-*.gguf` companion.

use std::fs;
use std::path::{Path, PathBuf};

/// Quantization preferred when a directory holds several backbones.
const PREFERRED_QUANT: &str = "Q4_K_M";

/// Model family, recognized from the backbone's file name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Family {
    /// Qwen3-TTS: multilingual, optional voice cloning.
    Qwen,
    /// Kyutai Pocket TTS: small and fast, one language per model, voice clip required.
    Pocket,
}

impl Family {
    /// Search order when no family is requested.
    const ALL: [Self; 2] = [Self::Qwen, Self::Pocket];

    /// Whether a backbone file name belongs to this family.
    pub fn matches(self, file_name: &str) -> bool {
        let name = file_name.to_lowercase();
        match self {
            Self::Qwen => name.contains("qwen3-tts"),
            Self::Pocket => name.contains("pocket-tts"),
        }
    }
}

/// A backbone gguf and its mmproj for `family` (or the first family found, Qwen first).
pub fn discover(family: Option<Family>) -> Option<(PathBuf, PathBuf)> {
    let candidates = candidate_dirs();
    let families: &[Family] = match &family {
        Some(f) => std::slice::from_ref(f),
        None => &Family::ALL,
    };
    families
        .iter()
        .find_map(|&f| candidates.iter().find_map(|dir| pair_in(dir, f)))
}

/// Directories that may hold a model pair, in search order.
fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs_out = Vec::new();
    if let Some(data) = dirs::data_dir() {
        dirs_out.extend(sorted_subdirs(&data.join("oido").join("models")));
    }
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| dirs::cache_dir().map(|c| c.join("huggingface").join("hub")));
    if let Some(hub) = hub {
        dirs_out.extend(hub_snapshots(&hub));
    }
    dirs_out
}

/// Snapshot directories of GGUF model repos in a hub cache.
fn hub_snapshots(hub: &Path) -> Vec<PathBuf> {
    sorted_subdirs(hub)
        .into_iter()
        .filter(|repo| {
            repo.file_name()
                .and_then(|n| n.to_str())
                .map(str::to_lowercase)
                .is_some_and(|n| n.starts_with("models--") && n.ends_with("-gguf"))
        })
        .flat_map(|repo| sorted_subdirs(&repo.join("snapshots")))
        .collect()
}

fn sorted_subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// The `family` backbone and its mmproj in one directory.
fn pair_in(dir: &Path, family: Family) -> Option<(PathBuf, PathBuf)> {
    let mut ggufs: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "gguf") && p.exists())
        .collect();
    ggufs.sort();
    let name = |p: &PathBuf| {
        p.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_owned()
    };
    let mmproj = ggufs
        .iter()
        .find(|p| name(p).starts_with("mmproj"))?
        .clone();
    let backbones: Vec<&PathBuf> = ggufs
        .iter()
        .filter(|p| !name(p).starts_with("mmproj") && family.matches(&name(p)))
        .collect();
    let model = backbones
        .iter()
        .find(|p| name(p).contains(PREFERRED_QUANT))
        .or_else(|| backbones.first())?;
    Some(((*model).clone(), mmproj))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path) {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, b"").unwrap();
    }

    #[test]
    fn finds_qwen_pair_preferring_q4_k_m() {
        let hub = tempfile::tempdir().unwrap();
        let snap = hub
            .path()
            .join("models--ggml-org--Qwen3-TTS-12Hz-1.7B-Base-GGUF/snapshots/abc");
        for f in [
            "Qwen3-TTS-Q8_0.gguf",
            "Qwen3-TTS-Q4_K_M.gguf",
            "mmproj-Qwen3-TTS-Q8_0.gguf",
            "README.md",
        ] {
            touch(&snap.join(f));
        }
        touch(
            &hub.path()
                .join("models--Qwen--Qwen3-Reranker-0.6B-GGUF/snapshots/x/model.gguf"),
        );
        let dirs = hub_snapshots(hub.path());
        let (model, mmproj) = dirs.iter().find_map(|d| pair_in(d, Family::Qwen)).unwrap();
        assert!(model.ends_with("Qwen3-TTS-Q4_K_M.gguf"), "{model:?}");
        assert!(mmproj.ends_with("mmproj-Qwen3-TTS-Q8_0.gguf"), "{mmproj:?}");
    }

    #[test]
    fn family_filters_backbones() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("pocket-tts-english-f16.gguf"));
        touch(&dir.path().join("mmproj-pocket-tts-english-f16.gguf"));
        assert!(pair_in(dir.path(), Family::Qwen).is_none());
        let (model, _) = pair_in(dir.path(), Family::Pocket).unwrap();
        assert!(model.ends_with("pocket-tts-english-f16.gguf"));
    }

    #[test]
    fn none_without_mmproj() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("Qwen3-TTS.gguf"));
        assert!(pair_in(dir.path(), Family::Qwen).is_none());
    }
}
