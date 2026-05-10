//! Kokoro-js–style sentence splitting (port of [`kokoro.js/src/splitter.js`](../../kokoro.js/src/splitter.js)).
//!
//! Use [`split_sentences`] for the same synchronous behavior as `split(text)` in JavaScript.

use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

static URL_START: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?[,:]://").expect("url regex"));

/// Abbreviations that should not end a sentence after the period (kokoro-js `ABBREVIATIONS`).
static ABBREVIATIONS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "sgt", "col", "gen", "rep", "sen", "gov",
        "lt", "maj", "capt", "st", "mt", "etc", "co", "inc", "ltd", "dept", "vs", "p", "pg", "jan",
        "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "sun", "mon",
        "tu", "tue", "tues", "wed", "th", "thu", "thur", "thurs", "fri", "sat",
    ]
    .into_iter()
    .collect()
});

/// Closing delimiter → expected opening character on stack.
fn matching_map() -> &'static HashMap<char, char> {
    static M: LazyLock<HashMap<char, char>> = LazyLock::new(|| {
        [
            (')', '('),
            (']', '['),
            ('}', '{'),
            ('》', '《'),
            ('〉', '〈'),
            ('›', '‹'),
            ('»', '«'),
            ('〉', '〈'),
            ('」', '「'),
            ('』', '『'),
            ('〕', '〔'),
            ('】', '【'),
        ]
        .into_iter()
        .collect()
    });
    &M
}

fn opening_chars() -> &'static HashSet<char> {
    static O: LazyLock<HashSet<char>> =
        LazyLock::new(|| matching_map().values().copied().collect());
    &O
}

#[inline]
fn is_sentence_terminator(c: char, include_newlines: bool) -> bool {
    ".!?…。？！".contains(c) || (include_newlines && c == '\n')
}

#[inline]
fn is_trailing_char(c: char) -> bool {
    "\"')]}」』".contains(c)
}

fn get_token_from_buffer(buffer: &[char], start: usize) -> String {
    let mut end = start;
    while end < buffer.len() && !buffer[end].is_whitespace() {
        end += 1;
    }
    buffer[start..end].iter().collect()
}

fn is_abbreviation(token: &str) -> bool {
    let mut t = token.to_string();
    if t.ends_with("'s") || t.ends_with("'S") {
        t.truncate(t.len().saturating_sub(2));
    }
    while t.ends_with('.') {
        t.pop();
    }
    ABBREVIATIONS.contains(t.to_ascii_lowercase().as_str())
}

fn update_stack(c: char, stack: &mut Vec<char>, i: usize, buffer: &[char]) {
    if c == '"' || c == '\'' {
        if c == '\''
            && i > 0
            && i < buffer.len() - 1
            && buffer[i - 1].is_ascii_alphabetic()
            && buffer[i + 1].is_ascii_alphabetic()
        {
            return;
        }
        if stack.last().copied() == Some(c) {
            stack.pop();
        } else {
            stack.push(c);
        }
        return;
    }
    if opening_chars().contains(&c) {
        stack.push(c);
        return;
    }
    if let Some(&expected_opening) = matching_map().get(&c) {
        if stack.last().copied() == Some(expected_opening) {
            stack.pop();
        }
    }
}

fn scan_boundary(buffer: &[char], len: usize, idx: usize) -> (usize, usize) {
    let mut end = idx;
    while end + 1 < len && is_sentence_terminator(buffer[end + 1], false) {
        end += 1;
    }
    while end + 1 < len && is_trailing_char(buffer[end + 1]) {
        end += 1;
    }
    let mut next_non_space = end + 1;
    while next_non_space < len && buffer[next_non_space].is_whitespace() {
        next_non_space += 1;
    }
    (end, next_non_space)
}

/// Split `text` into sentences (kokoro-js `split(text)`).
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut stream = TextSplitterStream::new();
    stream.push_str(text);
    stream.finish_sync()
}

/// Incremental splitter matching [`TextSplitterStream`](https://github.com/hexgrad/kokoro-js/blob/main/src/splitter.js).
pub struct TextSplitterStream {
    buffer: Vec<char>,
    sentences: Vec<String>,
}

impl TextSplitterStream {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            sentences: Vec::new(),
        }
    }

    pub fn push_str(&mut self, txt: &str) {
        self.buffer.extend(txt.chars());
        self.process();
    }

    /// Flush trailing buffer like kokoro-js `flush()` (remainder becomes one sentence).
    pub fn flush_remainder(&mut self) {
        let s: String = self.buffer.iter().collect();
        let remainder = s.trim();
        if !remainder.is_empty() {
            self.sentences.push(remainder.to_string());
        }
        self.buffer.clear();
    }

    fn finish_sync(&mut self) -> Vec<String> {
        self.flush_remainder();
        std::mem::take(&mut self.sentences)
    }

    fn process(&mut self) {
        let mut sentence_start = 0usize;
        let mut i = 0usize;
        let mut stack: Vec<char> = Vec::new();

        while i < self.buffer.len() {
            let c = self.buffer[i];
            update_stack(c, &mut stack, i, &self.buffer);

            if stack.is_empty() && is_sentence_terminator(c, true) {
                let current_segment: String = self.buffer[sentence_start..i].iter().collect();
                static NUMBERED: LazyLock<Regex> =
                    LazyLock::new(|| Regex::new(r"(^|\n)\d+$").expect("numbered"));
                if NUMBERED.is_match(&current_segment) {
                    i += 1;
                    continue;
                }

                let len = self.buffer.len();
                let (boundary_end, next_non_space) = scan_boundary(&self.buffer, len, i);

                if i == next_non_space.saturating_sub(1) && c != '\n' {
                    i += 1;
                    continue;
                }

                if next_non_space == len {
                    break;
                }

                let mut token_start = i.saturating_sub(1);
                while token_start > 0 && !self.buffer[token_start].is_whitespace() {
                    token_start -= 1;
                }
                if self
                    .buffer
                    .get(token_start)
                    .map_or(false, |x| x.is_whitespace())
                {
                    token_start += 1;
                }
                token_start = token_start.max(sentence_start);
                let token = get_token_from_buffer(&self.buffer, token_start);
                if token.is_empty() {
                    i += 1;
                    continue;
                }

                if (URL_START.is_match(&token) || token.contains('@'))
                    && !token
                        .chars()
                        .last()
                        .map_or(false, |ch| is_sentence_terminator(ch, false))
                {
                    i = token_start + token.chars().count();
                    continue;
                }

                if is_abbreviation(&token) {
                    i += 1;
                    continue;
                }

                static INITIALS: LazyLock<Regex> =
                    LazyLock::new(|| Regex::new(r"^([A-Za-z]\.)+$").expect("initials"));
                if INITIALS.is_match(&token)
                    && next_non_space < len
                    && self
                        .buffer
                        .get(next_non_space)
                        .map_or(false, |ch| ch.is_ascii_uppercase())
                {
                    i += 1;
                    continue;
                }

                if c == '.'
                    && next_non_space < len
                    && self
                        .buffer
                        .get(next_non_space)
                        .map_or(false, |ch| ch.is_ascii_lowercase())
                {
                    i += 1;
                    continue;
                }

                let sentence: String = self.buffer[sentence_start..boundary_end + 1]
                    .iter()
                    .collect::<String>()
                    .trim()
                    .to_string();
                if sentence == "..." || sentence == "…" {
                    i += 1;
                    continue;
                }

                if !sentence.is_empty() {
                    self.sentences.push(sentence);
                }
                i = boundary_end + 1;
                sentence_start = i;
                continue;
            }
            i += 1;
        }

        if sentence_start > 0 && sentence_start <= self.buffer.len() {
            self.buffer = self.buffer[sentence_start..].to_vec();
        }
    }
}

impl Default for TextSplitterStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_two_sentences() {
        let s = split_sentences("This is a test. This is another test.");
        assert_eq!(
            s,
            vec![
                "This is a test.".to_string(),
                "This is another test.".to_string()
            ]
        );
    }

    #[test]
    fn dr_abbreviation() {
        let s = split_sentences("Dr. Smith is here. At 10 a.m. I saw him.");
        assert_eq!(
            s,
            vec![
                "Dr. Smith is here.".to_string(),
                "At 10 a.m. I saw him.".to_string()
            ]
        );
    }

    #[test]
    fn dollar_decimal() {
        let s = split_sentences("The price is $4.99. Do you want to buy it?");
        assert_eq!(
            s,
            vec![
                "The price is $4.99.".to_string(),
                "Do you want to buy it?".to_string()
            ]
        );
    }

    #[test]
    fn url_with_period() {
        let s = split_sentences("Visit https://example.com. It's a great site!");
        assert_eq!(
            s,
            vec![
                "Visit https://example.com.".to_string(),
                "It's a great site!".to_string()
            ]
        );
    }

    #[test]
    fn newlines() {
        let s = split_sentences("First sentence.\nSecond sentence.\nThird sentence.");
        assert_eq!(
            s,
            vec![
                "First sentence.".to_string(),
                "Second sentence.".to_string(),
                "Third sentence.".to_string(),
            ]
        );
    }
}
