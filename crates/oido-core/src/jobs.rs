//! Turning text or a podcast script into an ordered list of synthesis jobs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::chunk::chunk_text;
use crate::script::Turn;
use crate::tags::split_segments;

/// Silence between consecutive jobs when no pause tag says otherwise.
pub const DEFAULT_GAP_MS: i32 = 200;

/// One model call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Job {
    /// Text to speak.
    pub text: String,
    /// Reference clip to clone; `None` uses the default voice.
    pub voice: Option<PathBuf>,
    /// Timing before this job from a pause tag; `None` uses [`DEFAULT_GAP_MS`].
    pub gap_before_ms: Option<i32>,
}

/// Jobs for one request, plus style tags found in the input that the engine cannot apply.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plan {
    /// Jobs in playback order.
    pub jobs: Vec<Job>,
    /// Distinct `[style]` tags that were stripped from the spoken text.
    pub ignored_styles: BTreeSet<String>,
}

impl Plan {
    /// Gaps between consecutive jobs, one per boundary.
    pub fn gaps_ms(&self) -> Vec<i32> {
        self.jobs
            .iter()
            .skip(1)
            .map(|j| j.gap_before_ms.unwrap_or(DEFAULT_GAP_MS))
            .collect()
    }
}

/// Plans single-voice text: split at inline tags, then chunked. Only the first chunk of a
/// segment carries that segment's pause.
pub fn compose(text: &str, voice: Option<PathBuf>) -> Plan {
    let mut plan = Plan::default();
    for seg in split_segments(text) {
        plan.ignored_styles.extend(seg.style);
        for (i, chunk) in chunk_text(&seg.text).into_iter().enumerate() {
            plan.jobs.push(Job {
                text: chunk,
                voice: voice.clone(),
                gap_before_ms: if i == 0 { seg.gap_before_ms } else { None },
            });
        }
    }
    plan
}

/// Plans a podcast: each turn chunked, with that speaker's voice from `voices` (default
/// voice if unmapped). Only a turn's first chunk carries the turn's pause.
pub fn podcast(turns: &[Turn], voices: &BTreeMap<String, PathBuf>) -> Plan {
    let mut plan = Plan::default();
    for turn in turns {
        plan.ignored_styles.extend(turn.style.clone());
        let voice = voices.get(&turn.speaker);
        for (i, chunk) in chunk_text(&turn.text).into_iter().enumerate() {
            plan.jobs.push(Job {
                text: chunk,
                voice: voice.cloned(),
                gap_before_ms: if i == 0 { turn.gap_before_ms } else { None },
            });
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script::parse_script;

    #[test]
    fn compose_carries_pause_to_first_chunk_only() {
        let long = "One two three four five six seven eight. Nine ten eleven twelve thirteen fourteen fifteen sixteen.";
        let plan = compose(&format!("Hello there friend. [pause:1s] {long}"), None);
        assert_eq!(plan.jobs.len(), 3);
        assert_eq!(plan.gaps_ms(), vec![1000, DEFAULT_GAP_MS]);
    }

    #[test]
    fn compose_reports_styles() {
        let plan = compose("[excited] Hi there. [whisper] psst", None);
        assert_eq!(
            plan.ignored_styles.into_iter().collect::<Vec<_>>(),
            ["excited", "whisper"]
        );
    }

    #[test]
    fn podcast_maps_voices_by_speaker() {
        let turns = parse_script("Host: Welcome.\n\nGuest: [pause:-300ms] Thanks!");
        let voices = BTreeMap::from([("Guest".to_owned(), PathBuf::from("/v/guest.wav"))]);
        let plan = podcast(&turns, &voices);
        assert_eq!(plan.jobs[0].voice, None);
        assert_eq!(
            plan.jobs[1].voice.as_deref(),
            Some(std::path::Path::new("/v/guest.wav"))
        );
        assert_eq!(plan.gaps_ms(), vec![-300]);
    }
}
