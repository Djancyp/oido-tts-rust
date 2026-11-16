//! Splitting text into model-sized chunks.
//!
//! Qwen3-TTS generates a whole prompt as one autoregressive run; long prompts degrade and
//! start to sound rushed. Chunks of roughly [`MIN_CHUNK_WORDS`]..[`MAX_CHUNK_WORDS`] words,
//! cut at sentence or clause boundaries, keep each run in the range the model handles well.
//! Every chunk boundary becomes a short silence when the audio is stitched, so cuts with no
//! natural pause behind them are avoided where possible.

use std::sync::LazyLock;

use regex::Regex;

/// Chunks are flushed once they reach this many words.
pub const MIN_CHUNK_WORDS: usize = 8;
/// Pieces are not merged past this many words.
pub const MAX_CHUNK_WORDS: usize = 20;
/// A piece with no punctuation to split on is kept whole up to this many words, then cut on
/// raw word boundaries.
pub const HARD_MAX_CHUNK_WORDS: usize = 40;

/// Sentence-ending punctuation followed by ASCII whitespace or end of text.
static SENTENCE_SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:[.!?]+|[。！？]+)(?:[\t\n\x0C\r ]+|$)").expect("valid regex"));

/// Clause punctuation (comma, semicolon, colon) or a spaced dash.
static CLAUSE_SPLIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:[,;:，；：]+(?:[\t\n\x0C\r ]+|$))|(?:[\t\n\x0C\r ][-—–]+[\t\n\x0C\r ])")
        .expect("valid regex")
});

/// Sentence punctuation glued to the next sentence's capital (`week.Tell`), optionally with a
/// closing quote or bracket in between.
static MISSING_SPACE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"([.!?。！？]["'”’)\]]*)([\p{Lu}\p{Han}])"#).expect("valid regex")
});

static HAN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\p{Han}").expect("valid regex"));

/// Lower-cased abbreviations whose trailing period is not sentence-final (`Dr.Jones`).
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc", "inc", "ltd", "co", "no",
    "fig", "capt", "col", "gen", "lt", "sgt", "rev", "hon", "rep", "sen", "gov", "pres", "ave",
    "blvd", "corp", "dept", "univ", "assn", "bros",
];

/// Splits `text` into chunks for synthesis, preferring sentence, then clause boundaries.
///
/// Returns an empty vector for blank input and the trimmed input as one chunk for short text.
pub fn chunk_text(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let text = fix_missing_spaces(text);
    let sentences = split_on(&SENTENCE_SPLIT, &text);
    if sentences.len() <= 1 && word_count(&text) <= MAX_CHUNK_WORDS {
        return vec![text];
    }
    accumulate(sentences, split_long_sentence)
}

/// Number of word-equivalent tokens; each Han character counts as one word.
pub fn word_count(s: &str) -> usize {
    tokenize(s).len()
}

/// Inserts the space missing after sentence punctuation glued to a capital, except after a
/// known abbreviation or inside an acronym like `U.S.`.
fn fix_missing_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut last = 0;
    for caps in MISSING_SPACE.captures_iter(text) {
        let punct = caps.get(1).expect("group 1 always participates");
        let (word, word_start) = preceding_word(text, punct.start());
        let acronym_letter =
            word.len() == 1 && word_start > 0 && text.as_bytes()[word_start - 1] == b'.';
        if ABBREVIATIONS.contains(&word.to_lowercase().as_str()) || acronym_letter {
            continue;
        }
        out.push_str(&text[last..punct.end()]);
        out.push(' ');
        last = punct.end();
    }
    out.push_str(&text[last..]);
    out
}

/// The run of letters ending at byte index `end`, and the index it starts at.
fn preceding_word(text: &str, end: usize) -> (&str, usize) {
    let start = text[..end]
        .char_indices()
        .rev()
        .take_while(|&(_, c)| c.is_alphabetic())
        .last()
        .map_or(end, |(i, _)| i);
    (&text[start..end], start)
}

/// Groups pieces into chunks of about `MIN_CHUNK_WORDS..=MAX_CHUNK_WORDS` words; a piece
/// already over the maximum is broken up by `overflow` and emitted on its own.
fn accumulate(pieces: Vec<String>, overflow: fn(String) -> Vec<String>) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut cur_words = 0;
    let flush = |cur: &mut Vec<String>, cur_words: &mut usize, chunks: &mut Vec<String>| {
        if !cur.is_empty() {
            chunks.push(cur.join(" "));
            cur.clear();
            *cur_words = 0;
        }
    };

    for piece in pieces {
        let words = word_count(&piece);
        if words > MAX_CHUNK_WORDS {
            flush(&mut cur, &mut cur_words, &mut chunks);
            chunks.extend(overflow(piece));
            continue;
        }
        if cur_words > 0 && cur_words + words > MAX_CHUNK_WORDS {
            flush(&mut cur, &mut cur_words, &mut chunks);
        }
        cur.push(piece);
        cur_words += words;
        if cur_words >= MIN_CHUNK_WORDS {
            flush(&mut cur, &mut cur_words, &mut chunks);
        }
    }
    flush(&mut cur, &mut cur_words, &mut chunks);
    chunks
}

/// Splits an overlong sentence at clause punctuation, else keeps or word-splits it whole.
fn split_long_sentence(sentence: String) -> Vec<String> {
    let clauses = split_on(&CLAUSE_SPLIT, &sentence);
    if clauses.len() > 1 {
        accumulate(clauses, keep_whole_or_split_words)
    } else {
        keep_whole_or_split_words(sentence)
    }
}

fn keep_whole_or_split_words(s: String) -> Vec<String> {
    if word_count(&s) <= HARD_MAX_CHUNK_WORDS {
        return vec![s];
    }
    tokenize(&s)
        .chunks(MAX_CHUNK_WORDS)
        .map(join_tokens)
        .collect()
}

/// Splits after every match, trimming each piece; the match stays with the preceding piece.
fn split_on(re: &Regex, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        out.push(text[last..m.end()].trim().to_owned());
        last = m.end();
    }
    if last < text.len() {
        out.push(text[last..].trim().to_owned());
    }
    out
}

/// Whitespace-separated words, with every Han character split out as its own token, since
/// CJK prose has no spaces between words.
fn tokenize(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for field in s.split_whitespace() {
        let mut start = 0;
        for m in HAN.find_iter(field) {
            if m.start() > start {
                out.push(&field[start..m.start()]);
            }
            out.push(m.as_str());
            start = m.end();
        }
        if start < field.len() {
            out.push(&field[start..]);
        }
    }
    out
}

/// Re-joins tokens with spaces, except between two adjacent Han tokens.
fn join_tokens(tokens: &[&str]) -> String {
    let mut out = String::new();
    for (i, tok) in tokens.iter().enumerate() {
        if i > 0 && !(is_han_token(tokens[i - 1]) && is_han_token(tok)) {
            out.push(' ');
        }
        out.push_str(tok);
    }
    out
}

fn is_han_token(tok: &str) -> bool {
    HAN.find(tok).is_some_and(|m| m.start() == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn han_characters_count_individually() {
        assert_eq!(word_count("今天天气很好"), 6);
        assert_eq!(word_count("hello 世界 ok"), 4);
    }

    #[test]
    fn preserves_all_words() {
        let text = "One two three. Four five six seven eight nine ten eleven. Twelve thirteen fourteen \
                    fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two.";
        let joined = chunk_text(text).join(" ");
        assert_eq!(word_count(&joined), word_count(text), "chunks: {joined}");
    }
}
