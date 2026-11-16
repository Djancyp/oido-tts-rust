//! Parity with the Go implementation (oido-tts-v3): `fixtures/go_golden.json` holds the Go
//! `ChunkText`, `SplitInstructSegments`, and `ParseScript` outputs for every string literal
//! in the Go test suite plus extra edge cases.

use oido_core::chunk::chunk_text;
use oido_core::script::{Turn, parse_script};
use oido_core::tags::{Segment, split_segments};
use serde::Deserialize;

/// Go's sentinel for "no pause tag".
const GO_NO_GAP: i64 = -(1 << 30);

#[derive(Deserialize)]
struct Case {
    input: String,
    chunks: Option<Vec<String>>,
    segments: Option<Vec<GoSegment>>,
    turns: Option<Vec<GoTurn>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct GoSegment {
    text: String,
    instruct: String,
    gap_before_ms: i64,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct GoTurn {
    speaker: String,
    text: String,
    instruct: String,
    gap_before_ms: i64,
}

fn gap(ms: i64) -> Option<i32> {
    (ms != GO_NO_GAP).then(|| i32::try_from(ms).expect("pause fits i32"))
}

fn style(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}

fn cases() -> Vec<Case> {
    serde_json::from_str(include_str!("fixtures/go_golden.json")).expect("valid fixture")
}

#[test]
fn chunk_text_matches_go() {
    let mut failures = Vec::new();
    for c in cases() {
        let want = c.chunks.unwrap_or_default();
        let got = chunk_text(&c.input);
        if got != want {
            failures.push(format!(
                "input {:?}\n  want {want:?}\n  got  {got:?}",
                c.input
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn split_segments_matches_go() {
    let mut failures = Vec::new();
    for c in cases() {
        // Go quirk: tag-free input returns its single segment with gap 0 instead of NoGap.
        // The first job never has a gap before it, so both mean "no pause".
        let tag_free = !c.input.contains('[');
        let want: Vec<Segment> = c
            .segments
            .unwrap_or_default()
            .into_iter()
            .map(|s| Segment {
                text: s.text,
                style: style(s.instruct),
                gap_before_ms: if tag_free && s.gap_before_ms == 0 {
                    None
                } else {
                    gap(s.gap_before_ms)
                },
            })
            .collect();
        let got = split_segments(&c.input);
        if got != want {
            failures.push(format!(
                "input {:?}\n  want {want:?}\n  got  {got:?}",
                c.input
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn parse_script_matches_go() {
    let mut failures = Vec::new();
    for c in cases() {
        let want: Vec<Turn> = c
            .turns
            .unwrap_or_default()
            .into_iter()
            .map(|t| Turn {
                speaker: t.speaker,
                text: t.text,
                style: style(t.instruct),
                gap_before_ms: gap(t.gap_before_ms),
            })
            .collect();
        let got = parse_script(&c.input);
        if got != want {
            failures.push(format!(
                "input {:?}\n  want {want:?}\n  got  {got:?}",
                c.input
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
