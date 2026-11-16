//! Podcast scripts: one `Speaker: line` paragraph per turn.

use std::sync::LazyLock;

use regex::Regex;

use crate::tags::parse_pause_tag;

/// Longest speaker-name prefix (bytes before the colon) recognized on a turn's first line.
const MAX_SPEAKER_PREFIX: usize = 30;

static TRAILING_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[([^\[\]]+)\][\t\n\x0C\r ]*$").expect("valid regex"));
static LEADING_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[([^\[\]]+)\][\t\n\x0C\r ]*").expect("valid regex"));

/// One speaker's turn.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Turn {
    /// Speaker name as written in the script.
    pub speaker: String,
    /// Spoken text with tags removed.
    pub text: String,
    /// Style from a trailing `[style]` tag.
    pub style: Option<String>,
    /// Timing relative to the previous turn from a leading pause tag, in signed milliseconds.
    pub gap_before_ms: Option<i32>,
}

/// Parses a script into turns.
///
/// Paragraphs are separated by blank lines. A paragraph's first line must start with
/// `Name:` (letters, digits, spaces, `_`, `-`); following lines fold into the same turn.
/// A leading `[pause:..]` tag sets the turn's timing; a trailing `[...]` tag is its style.
/// Paragraphs without a speaker prefix are skipped.
pub fn parse_script(script: &str) -> Vec<Turn> {
    paragraphs(script)
        .into_iter()
        .filter_map(|para| {
            let (speaker, first) = split_speaker_line(para[0])?;
            let text_lines: Vec<&str> = std::iter::once(first)
                .chain(para[1..].iter().map(|l| l.trim()).filter(|l| !l.is_empty()))
                .collect();
            let text = text_lines.join(" ");
            let (text, gap_before_ms) = leading_pause(text.trim());
            let (text, style) = trailing_style(text);
            (!text.is_empty()).then(|| Turn {
                speaker: speaker.to_owned(),
                text: text.to_owned(),
                style: style.map(str::to_owned),
                gap_before_ms,
            })
        })
        .collect()
}

fn trailing_style(text: &str) -> (&str, Option<&str>) {
    match TRAILING_TAG.captures(text) {
        Some(caps) => {
            let whole = caps.get(0).expect("group 0 always participates");
            (
                text[..whole.start()].trim(),
                Some(caps.get(1).expect("group 1").as_str()),
            )
        }
        None => (text, None),
    }
}

fn leading_pause(text: &str) -> (&str, Option<i32>) {
    let Some(caps) = LEADING_TAG.captures(text) else {
        return (text, None);
    };
    match parse_pause_tag(&caps[1]) {
        Some(ms) => (text[caps.get(0).expect("group 0").end()..].trim(), Some(ms)),
        None => (text, None),
    }
}

/// Non-blank line groups separated by blank lines.
fn paragraphs(script: &str) -> Vec<Vec<&str>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for line in script.split('\n') {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(line);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Splits `Name: rest` into `("Name", "rest")`.
fn split_speaker_line(line: &str) -> Option<(&str, &str)> {
    let idx = line
        .find(':')
        .filter(|&i| i > 0 && i <= MAX_SPEAKER_PREFIX)?;
    let name = line[..idx].trim();
    let valid = !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b' ' | b'_' | b'-'));
    valid.then(|| (name, line[idx + 1..].trim()))
}
