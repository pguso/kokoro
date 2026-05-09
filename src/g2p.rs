/// Text-to-IPA conversion for English-only pipeline.
#[path = "g2p/backend.rs"]
mod backend;

use regex::{Error as RegexError, Regex};
use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

use crate::{letters_to_ipa, tokenizer::unknown_phonemes};
use backend::{BackendSource, HybridEnglishBackend, TokenContext};

#[derive(Debug)]
pub enum G2PError {
    Backend(String),
    Regex(RegexError),
    StrictInvalidPhoneme(String),
}

impl Display for G2PError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "G2PError: ")?;
        match self {
            Self::Backend(e) => Display::fmt(e, f),
            Self::Regex(e) => Display::fmt(e, f),
            Self::StrictInvalidPhoneme(e) => Display::fmt(e, f),
        }
    }
}

impl Error for G2PError {}

impl From<RegexError> for G2PError {
    fn from(value: RegexError) -> Self {
        Self::Regex(value)
    }
}

fn normalize_ipa_for_vocab(raw: &str, use_v11: bool) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut normalized = String::with_capacity(chars.len());
    for (idx, ch) in chars.iter().copied().enumerate() {
        let mapped = match ch {
            '\0' => continue,
            // Rhotic schwa alias seen in CMUdict path.
            'ɝ' => {
                if use_v11 {
                    'ɜ'
                } else {
                    'ɚ'
                }
            }
            // Unify ASCII quote-like marks.
            '`' | '´' => 'ˈ',
            _ => ch,
        };
        let prev = normalized.chars().last();
        let next = chars.get(idx + 1).copied();
        // Some eSpeak outputs duplicate consonants before reduced vowels (e.g. ððə).
        // Collapse only this narrow artifact pattern.
        if let (Some(p), Some(n)) = (prev, next)
            && p == mapped
            && "ðθszʃʒfvpbtdkgmnlrɹ".contains(mapped)
            && "əɐɚɜ".contains(n)
        {
            continue;
        }
        normalized.push(mapped);
    }
    normalized.trim_end_matches(['ˈ', 'ˌ']).trim().to_string()
}

fn is_acronym(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    chars.len() >= 2 && chars.len() <= 6 && chars.iter().all(|c| c.is_ascii_uppercase())
}

fn next_word(tokens: &[String], idx: usize) -> Option<String> {
    tokens
        .iter()
        .skip(idx + 1)
        .find(|t| t.chars().next().is_some_and(|c| c.is_ascii_alphabetic()))
        .cloned()
}

fn previous_word(tokens: &[String], idx: usize) -> Option<String> {
    if idx == 0 {
        return None;
    }
    tokens[..idx]
        .iter()
        .rev()
        .find(|t| t.chars().next().is_some_and(|c| c.is_ascii_alphabetic()))
        .cloned()
}

fn mixed_alnum_to_parts(token: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut current_kind: Option<bool> = None; // true = alpha, false = digit
    for ch in token.chars() {
        if !(ch.is_ascii_alphanumeric() || ch == '-') {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
                current_kind = None;
            }
            continue;
        }
        let kind = ch.is_ascii_alphabetic();
        if current_kind.is_some_and(|k| k != kind) && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
        current_kind = Some(kind);
        current.push(ch);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn num_to_words_en_u64(mut n: u64) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    fn under_1000(n: u64) -> String {
        const ONES: [&str; 20] = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ];
        const TENS: [&str; 10] = [
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        ];
        let mut out = String::new();
        let hundreds = n / 100;
        let rem = n % 100;
        if hundreds > 0 {
            out.push_str(ONES[hundreds as usize]);
            out.push_str(" hundred");
            if rem > 0 {
                out.push(' ');
            }
        }
        if rem >= 20 {
            out.push_str(TENS[(rem / 10) as usize]);
            if rem % 10 > 0 {
                out.push(' ');
                out.push_str(ONES[(rem % 10) as usize]);
            }
        } else if rem > 0 || out.is_empty() {
            out.push_str(ONES[rem as usize]);
        }
        out
    }

    if n < 20 {
        return ONES[n as usize].to_owned();
    }
    if n < 100 {
        let tens = TENS[(n / 10) as usize];
        if n % 10 == 0 {
            return tens.to_owned();
        }
        return format!("{} {}", tens, ONES[(n % 10) as usize]);
    }

    let scales = ["", "thousand", "million", "billion", "trillion"];
    let mut chunks: Vec<String> = Vec::new();
    let mut idx = 0;
    while n > 0 {
        let part = n % 1000;
        if part > 0 {
            let mut seg = under_1000(part);
            let scale = scales[idx];
            if !scale.is_empty() {
                seg.push(' ');
                seg.push_str(scale);
            }
            chunks.push(seg);
        }
        n /= 1000;
        idx += 1;
    }
    chunks.reverse();
    chunks.join(" ")
}

fn number_token_to_words_en(token: &str) -> Option<String> {
    if token.chars().all(|c| c.is_ascii_digit()) {
        if let Ok(n) = token.parse::<u64>() {
            return Some(num_to_words_en_u64(n));
        }
        // Fallback: spell per-digit.
        let words = token
            .chars()
            .filter_map(|c| match c {
                '0' => Some("zero"),
                '1' => Some("one"),
                '2' => Some("two"),
                '3' => Some("three"),
                '4' => Some("four"),
                '5' => Some("five"),
                '6' => Some("six"),
                '7' => Some("seven"),
                '8' => Some("eight"),
                '9' => Some("nine"),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");
        return Some(words);
    }
    None
}

fn append_lexical(result: &mut String, phoneme: &str) {
    if phoneme.is_empty() {
        return;
    }
    if !result.is_empty() && !result.ends_with([' ', '\n', '\t']) {
        result.push(' ');
    }
    result.push_str(phoneme);
    result.push(' ');
}

fn append_non_lexical(result: &mut String, token: &str) {
    for ch in token.chars() {
        if ch.is_whitespace() {
            if !result.ends_with(' ') {
                result.push(' ');
            }
            continue;
        }
        if [',', '.', '!', '?', ';', ':'].contains(&ch) && result.ends_with(' ') {
            result.pop();
        }
        result.push(ch);
        if [',', '.', '!', '?', ';', ':'].contains(&ch) {
            result.push(' ');
        }
    }
}

fn trace_token_g2p(token: &str, ipa: &str, source: BackendSource) {
    if std::env::var("KOKORO_G2P_TRACE").ok().as_deref() == Some("1") {
        let src = match source {
            BackendSource::Dictionary => "dict",
            BackendSource::Fallback => "fallback",
            BackendSource::Heuristic => "heuristic",
        };
        eprintln!(
            "kokoro g2p trace | source={} | token={:?} | ipa={}",
            src, token, ipa
        );
    }
}

pub fn g2p(text: &str, use_v11: bool) -> Result<String, G2PError> {
    let en_word_pattern = Regex::new(
        r"[A-Za-z]+(?:['-][A-Za-z]+)*|[A-Za-z]+\d+[A-Za-z\d-]*|\d+(?:\.\d+)?|[^A-Za-z\d]+",
    )?;
    let backend = HybridEnglishBackend;
    let mut result = String::new();
    let tokens = en_word_pattern
        .captures_iter(text)
        .map(|caps| caps[0].to_string())
        .collect::<Vec<_>>();
    for (idx, token) in tokens.iter().enumerate() {
        let c = token.chars().next().unwrap_or_default();
        if c.is_ascii_alphabetic() {
            let token_context = TokenContext {
                previous_word: previous_word(&tokens, idx),
                next_word: next_word(&tokens, idx),
            };
            if is_acronym(token) {
                let ipa = normalize_ipa_for_vocab(&letters_to_ipa(token), use_v11);
                trace_token_g2p(token, &ipa, BackendSource::Dictionary);
                append_lexical(&mut result, &ipa);
                continue;
            }
            let (ipa, source) = backend.resolve_word(token, &token_context)?;
            let ipa = normalize_ipa_for_vocab(&ipa, use_v11);
            trace_token_g2p(token, &ipa, source);
            append_lexical(&mut result, &ipa);
        } else if c.is_ascii_digit() {
            if let Some(words) = number_token_to_words_en(token) {
                for word in words.split_whitespace() {
                    let token_context = TokenContext {
                        previous_word: previous_word(&tokens, idx),
                        next_word: next_word(&tokens, idx),
                    };
                    let (ipa, source) = backend.resolve_word(word, &token_context)?;
                    let ipa = normalize_ipa_for_vocab(&ipa, use_v11);
                    trace_token_g2p(word, &ipa, source);
                    append_lexical(&mut result, &ipa);
                }
            } else {
                append_non_lexical(&mut result, token);
            }
        } else if token.chars().any(|ch| ch.is_ascii_alphabetic())
            && token.chars().any(|ch| ch.is_ascii_digit())
        {
            for part in mixed_alnum_to_parts(token) {
                if part.chars().all(|ch| ch.is_ascii_digit()) {
                    if let Some(words) = number_token_to_words_en(&part) {
                        for word in words.split_whitespace() {
                            let token_context = TokenContext {
                                previous_word: previous_word(&tokens, idx),
                                next_word: next_word(&tokens, idx),
                            };
                            let (ipa, source) = backend.resolve_word(word, &token_context)?;
                            let ipa = normalize_ipa_for_vocab(&ipa, use_v11);
                            trace_token_g2p(word, &ipa, source);
                            append_lexical(&mut result, &ipa);
                        }
                    }
                } else if part.chars().all(|ch| ch.is_ascii_alphabetic()) {
                    let token_context = TokenContext {
                        previous_word: previous_word(&tokens, idx),
                        next_word: next_word(&tokens, idx),
                    };
                    let (ipa, source) = if is_acronym(&part) {
                        (letters_to_ipa(&part), BackendSource::Dictionary)
                    } else {
                        backend.resolve_word(&part, &token_context)?
                    };
                    let ipa = normalize_ipa_for_vocab(&ipa, use_v11);
                    trace_token_g2p(&part, &ipa, source);
                    append_lexical(&mut result, &ipa);
                }
            }
        } else {
            append_non_lexical(&mut result, token);
        }
    }
    let compact_spaces = Regex::new(r"\s+")?;
    let compacted = compact_spaces.replace_all(result.trim(), " ").to_string();
    let strip_space_before_punct = Regex::new(r"\s+([,.;:!?])")?;
    let output = strip_space_before_punct
        .replace_all(compacted.as_str(), "$1")
        .trim()
        .to_string();

    if std::env::var("KOKORO_G2P_STRICT").ok().as_deref() == Some("1") {
        let unknown = unknown_phonemes(&output, use_v11);
        if !unknown.is_empty() {
            return Err(G2PError::StrictInvalidPhoneme(format!(
                "Unknown IPA symbols after normalization: {}",
                unknown
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_number_expansion() -> Result<(), super::G2PError> {
        use super::g2p;
        let output = g2p("I am 25.", false)?;
        assert!(!output.chars().any(|c| c.is_ascii_digit()));
        assert!(output.contains("twˈɛnti") || output.contains("twˈɛntaɪ"));
        Ok(())
    }

    #[test]
    fn test_g2p() -> Result<(), super::G2PError> {
        use super::g2p;

        let output = g2p("Hello, world!", false)?;
        assert!(output.contains("həlˈoʊ"));
        assert!(output.contains("wˈɚld"));

        Ok(())
    }

    #[test]
    fn test_g2p_keeps_english_numbers_non_cjk() -> Result<(), super::G2PError> {
        use super::g2p;

        let output = g2p("I'm 25 years old.", false)?;
        assert!(
            !output
                .chars()
                .any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
        );
        assert!(!output.chars().any(|c| c.is_ascii_digit()));
        assert!(!output.contains('\''));

        Ok(())
    }

    #[test]
    fn test_g2p_boundary_regression_sentence() -> Result<(), super::G2PError> {
        use super::g2p;

        let output = g2p(
            "Hello, world! I'm a 25 year old software engineer with a passion background?",
            false,
        )?;
        assert!(!output.contains("||"));
        assert!(!output.chars().any(|c| c.is_ascii_digit()));
        assert!(!output.contains("二十五"));
        assert!(!output.contains("ðð"));
        Ok(())
    }

    #[test]
    fn test_g2p_acronyms_and_mixed_tokens() -> Result<(), super::G2PError> {
        use super::g2p;

        let output = g2p("AI and GPT4 improve LLM tooling.", false)?;
        assert!(!output.is_empty());
        assert!(!output.chars().any(|c| c.is_ascii_digit()));
        assert!(output.contains("ˈA") || output.contains("ˈI"));
        Ok(())
    }

    #[test]
    fn test_g2p_long_quality_paragraph() -> Result<(), super::G2PError> {
        use super::g2p;

        let output = g2p(
            "I'm a 25 year old software engineer with a passion for python and everything related to AI. I also have a strong background in computer science and mathematics.",
            false,
        )?;
        assert!(!output.contains("||"));
        assert!(!output.chars().any(|c| c.is_ascii_digit()));
        assert!(!output.contains("fˈɔːwɒˌn"));
        assert!(!output.contains("sʃ"));
        Ok(())
    }

    #[test]
    fn test_g2p_homographs_context() -> Result<(), super::G2PError> {
        use super::g2p;

        let a = g2p("I read books every day.", false)?;
        let b = g2p("Yesterday I read that book.", false)?;
        assert_ne!(a, b, "Expected context-sensitive homograph handling.");
        Ok(())
    }
}
