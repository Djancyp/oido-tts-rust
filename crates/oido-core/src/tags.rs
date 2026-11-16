//! Inline `[...]` tags in text: `[pause:2s]` timing tags and `[style]` tags.
//!
//! A style tag applies to everything after it, up to the next style tag. A pause tag sets
//! the timing before the text that follows it: positive values insert silence, negative
//! values overlap that text back into the previous audio.

use std::sync::LazyLock;

use regex::Regex;

/// Largest pause or overlap a tag may request; larger values are clamped.
pub const MAX_PAUSE_MS: i32 = 30_000;

static PAUSE_TAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:pause|silence):(-?[0-9]+(?:\.[0-9]+)?)(ms|s)$").expect("valid regex")
});

static INLINE_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[([^\[\]]+)\]").expect("valid regex"));

/// A stretch of text under one style, with its tag markers removed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Segment {
    /// Text to speak.
    pub text: String,
    /// Style tag in effect, if any.
    pub style: Option<String>,
    /// Timing before this segment from a preceding pause tag, in signed milliseconds.
    pub gap_before_ms: Option<i32>,
}

/// Splits `text` at every inline tag, stripping the tags from the spoken text.
pub fn split_segments(text: &str) -> Vec<Segment> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut style: Option<&str> = None;
    let mut pending_gap = None;
    let mut last = 0;
    let mut push = |piece: &str, style: Option<&str>, gap: &mut Option<i32>| {
        let piece = piece.trim();
        if !piece.is_empty() {
            segments.push(Segment {
                text: piece.to_owned(),
                style: style.map(str::to_owned),
                gap_before_ms: gap.take(),
            });
        }
    };
    for caps in INLINE_TAG.captures_iter(text) {
        let whole = caps.get(0).expect("group 0 always participates");
        push(&text[last..whole.start()], style, &mut pending_gap);
        let tag = caps.get(1).expect("group 1 always participates").as_str();
        match parse_pause_tag(tag) {
            Some(ms) => pending_gap = Some(ms),
            None => style = Some(tag),
        }
        last = whole.end();
    }
    push(&text[last..], style, &mut pending_gap);
    segments
}

/// Parses a tag body like `pause:2s` or `silence:-300ms` into signed, clamped milliseconds.
pub fn parse_pause_tag(tag: &str) -> Option<i32> {
    let caps = PAUSE_TAG.captures(tag.trim())?;
    let mut value: f64 = caps[1].parse().ok()?;
    if caps[2].eq_ignore_ascii_case("s") {
        value *= 1000.0;
    }
    // Truncate toward zero, then clamp; `as` saturates out-of-range floats.
    Some((value as i64).clamp(i64::from(-MAX_PAUSE_MS), i64::from(MAX_PAUSE_MS)) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_tags() {
        assert_eq!(parse_pause_tag("pause:2s"), Some(2000));
        assert_eq!(parse_pause_tag("SILENCE:-300ms"), Some(-300));
        assert_eq!(parse_pause_tag("pause:1.5s"), Some(1500));
        assert_eq!(parse_pause_tag("pause:999s"), Some(MAX_PAUSE_MS));
        assert_eq!(parse_pause_tag("whisper"), None);
    }
}
