/// Text-to-IPA conversion for English-only pipeline.
#[path = "g2p/backend.rs"]
mod backend;
#[path = "g2p/capability.rs"]
pub mod capability;
#[path = "g2p/espeak_cli.rs"]
mod espeak_cli;
mod lexicon;
#[path = "g2p/phonemize_js.rs"]
mod phonemize_js;

use regex::{Error as RegexError, Regex};
use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

use crate::{letters_to_ipa, tokenizer::unknown_phonemes};
use backend::{
    BackendSource, EnglishG2pBackend, HybridEnglishBackend, TokenContext, cmudict_has_entry,
};
use lexicon::lexicon_lookup;

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

/// Map Unicode apostrophes and paired quotes to ASCII so tokenization matches prose typography.
fn normalize_g2p_input(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\u{2018}' | '\u{2019}' | '\u{201B}' => out.push('\''),
            '\u{201C}' | '\u{201D}' => out.push('"'),
            _ => out.push(ch),
        }
    }
    out
}

/// French-style elision before a capitalized word (`L'Actrice`): pronounce the stem only so we do
/// not emit `L` as a separate letter and garble the apostrophe.
pub(super) fn strip_elided_l_article_word(token: &str) -> Option<&str> {
    let mut chars = token.chars();
    let first = chars.next()?;
    if !matches!(first, 'l' | 'L') {
        return None;
    }
    if chars.next() != Some('\'') {
        return None;
    }
    let prefix_len = first.len_utf8() + '\''.len_utf8();
    let rest = token.get(prefix_len..)?;
    let mut rest_chars = rest.chars();
    let r0 = rest_chars.next()?;
    if !r0.is_ascii_uppercase() {
        return None;
    }
    if rest.len() < 2 || !rest.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    Some(rest)
}

/// Full phoneme string plus any symbols not present in the Kokoro tokenizer vocabulary.
///
/// Use this to audit G2P output when comparing against another Kokoro implementation or
/// when tuning lexicon overrides. Prefer [`unknown_phonemes`] if you already have the string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct G2pAudit {
    pub phonemes: String,
    pub unknown_phoneme_chars: Vec<char>,
}

/// Run [`g2p`] and collect [`unknown_phonemes`] for the result (v10 vs v11 vocab).
pub fn g2p_audit(text: &str, use_v11: bool) -> Result<G2pAudit, G2PError> {
    let phonemes = g2p(text, use_v11)?;
    let unknown_phoneme_chars = unknown_phonemes(&phonemes, use_v11);
    Ok(G2pAudit {
        phonemes,
        unknown_phoneme_chars,
    })
}

/// Normalize raw eSpeak `TextToPhonemes` output to Kokoro tokenizer–safe IPA fragments.
///
/// Keeps the first variant before `||`, removes eSpeak pipe separators, ASCII stress
/// digits, and apostrophe markers that are not part of the model vocabulary.
pub fn sanitize_espeak_ipa(raw: &str) -> String {
    let primary = raw.split("||").next().unwrap_or(raw).trim();
    let mut cleaned = String::with_capacity(primary.len());
    for ch in primary.chars() {
        match ch {
            '|' | '\'' => {}
            '0'..='9' => {}
            _ => cleaned.push(ch),
        }
    }
    cleaned.trim().to_string()
}

impl From<RegexError> for G2PError {
    fn from(value: RegexError) -> Self {
        Self::Regex(value)
    }
}

fn normalize_ipa_for_vocab(raw: &str, use_v11: bool) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut normalized = String::with_capacity(chars.len());
    for (idx, ch) in chars.iter().copied().enumerate() {
        if matches!(ch, '\u{200c}' | '\u{200d}' | '\u{feff}') {
            continue;
        }
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
    if chars.len() < 2 || chars.len() > 6 || !chars.iter().all(|c| c.is_ascii_uppercase()) {
        return false;
    }
    // Short tokens where CMUdict homographs would mis-read initialisms (e.g. "ai" the word).
    if matches!(
        token,
        "AI" | "TV" | "PC" | "VR" | "US" | "UK" | "EU" | "UN" | "HD"
    ) {
        return true;
    }
    // Headlines use ALL CAPS for ordinary words ("TEXT"); prefer CMUdict when present so we
    // do not spell T–E–X–T. IBM/NASA use pronunciations when listed.
    if cmudict_has_entry(&token.to_ascii_lowercase()) {
        return false;
    }
    true
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

/// Split PascalCase / camelCase at a lowercase→uppercase boundary (`AlphaGo` → `Alpha` + `Go`).
fn split_camel_case_segments(token: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() < 3 {
        return None;
    }
    if !chars.iter().any(|c| c.is_ascii_lowercase())
        || !chars.iter().any(|c| c.is_ascii_uppercase())
    {
        return None;
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for i in 0..chars.len() {
        let ch = chars[i];
        if i > 0 && ch.is_ascii_uppercase() && chars[i - 1].is_ascii_lowercase() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
        cur.push(ch);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    (out.len() > 1).then_some(out)
}

/// Tokens before `idx` whose alphanumeric stem matches a simple past-narrative cue.
fn sentence_has_past_markers_before(tokens: &[String], idx: usize) -> bool {
    const MARKERS: &[&str] = &["yesterday", "ago", "last", "earlier", "previously"];
    for t in tokens.iter().take(idx) {
        let w = t
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if MARKERS.iter().any(|m| *m == w.as_str()) {
            return true;
        }
    }
    false
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
            BackendSource::Lexicon => "lexicon",
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

/// English grapheme-to-phoneme conversion.
///
/// By default uses Kokoro.js–aligned phonemization ([`phonemize_js`]): punctuation-split segments,
/// then **segment-level `espeak-ng`** when installed (`KOKORO_G2P_SEGMENT_ESPEAK` unset), then Misaki
/// + **default** crate feature `g2p-espeak` (bundled eSpeak OOV), else word-level Misaki + heuristics.
/// [`capability::g2p_espeak_capability`] logs once if neither bundled nor CLI eSpeak is available.
/// Set `KOKORO_G2P_REQUIRE_ESPEAK=1` to error instead of degrading. Disable subprocess eSpeak with
/// `KOKORO_ESPEAK_NG=0`. Slim build: `cargo build --no-default-features` (optionally `--features misaki-lean`).
/// Set `KOKORO_G2P_LEGACY=1` for the previous per-token CMUdict + Misaki pipeline.
///
/// Set `KOKORO_G2P_LANG=b` for British English (kokoro.js `language === "b"`); default is US (`a`).
///
/// Runtime flags are centralized in [`crate::G2pPipelineConfig`](crate::G2pPipelineConfig).
pub fn g2p(text: &str, use_v11: bool) -> Result<String, G2PError> {
    let cfg = crate::pipeline::G2pPipelineConfig::from_env();
    if cfg.legacy {
        return g2p_legacy(text, use_v11);
    }

    capability::g2p_espeak_capability();
    capability::enforce_require_espeak_if_configured()?;

    let trimmed_in = text.trim();
    if !trimmed_in.chars().any(char::is_whitespace)
        && let Some(ipa) = lexicon_lookup(trimmed_in)
        && !ipa.is_empty()
    {
        let out = normalize_ipa_for_vocab(&ipa, use_v11);
        return finish_g2p_output_checks(out, use_v11);
    }

    let normalized_text = normalize_g2p_input(text);
    let british = cfg.british;
    let raw = phonemize_js::phonemize_like_kokoro_js(&normalized_text, british, true, use_v11);
    let output = phonemize_js::finalize_g2p_output(&raw);
    finish_g2p_output_checks(output, use_v11)
}

fn finish_g2p_output_checks(output: String, use_v11: bool) -> Result<String, G2PError> {
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

/// Per-token CMUdict + Misaki pipeline used before Kokoro.js–parity phonemization.
///
/// Prefer [`g2p`] for output aligned with official kokoro.js; use this when you rely on
/// embedded CMUdict homograph rules (`read`, `finance`, …) or hyphen splitting behavior.
pub fn g2p_legacy(text: &str, use_v11: bool) -> Result<String, G2PError> {
    let normalized_text = normalize_g2p_input(text);
    let text = normalized_text.as_str();
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
                past_markers_before: sentence_has_past_markers_before(&tokens, idx),
            };
            if let Some(rest) = strip_elided_l_article_word(token) {
                let whole_hit = lexicon::lexicon_lookup(token).is_some()
                    || backend.lookup_word(token, &token_context).is_some();
                if !whole_hit {
                    let (ipa, source) = backend.resolve_word(rest, &token_context)?;
                    let ipa = normalize_ipa_for_vocab(&ipa, use_v11);
                    trace_token_g2p(rest, &ipa, source);
                    append_lexical(&mut result, &ipa);
                    continue;
                }
            }
            // PascalCase compounds ("AlphaGo"): no CMUdict headword; split like Alpha + Go instead
            // of eSpeak spelling each letter.
            if let Some(parts) = split_camel_case_segments(token) {
                let whole_hit = lexicon::lexicon_lookup(token).is_some()
                    || backend.lookup_word(token, &token_context).is_some();
                if whole_hit {
                    let (ipa, source) = backend.resolve_word(token, &token_context)?;
                    let ipa = normalize_ipa_for_vocab(&ipa, use_v11);
                    trace_token_g2p(token, &ipa, source);
                    append_lexical(&mut result, &ipa);
                    continue;
                }
                for part in parts {
                    if is_acronym(&part) {
                        let ipa = normalize_ipa_for_vocab(&letters_to_ipa(&part), use_v11);
                        trace_token_g2p(&part, &ipa, BackendSource::Dictionary);
                        append_lexical(&mut result, &ipa);
                        continue;
                    }
                    // "Py" is not in CMUdict; eSpeak spells letters. Use /paɪ/ like "Python" / "PyPI".
                    if part.eq_ignore_ascii_case("Py") {
                        let ipa = normalize_ipa_for_vocab("ˈpaɪ", use_v11);
                        trace_token_g2p(&part, &ipa, BackendSource::Dictionary);
                        append_lexical(&mut result, &ipa);
                        continue;
                    }
                    let (ipa, source) = backend.resolve_word(&part, &token_context)?;
                    let ipa = normalize_ipa_for_vocab(&ipa, use_v11);
                    trace_token_g2p(&part, &ipa, source);
                    append_lexical(&mut result, &ipa);
                }
                continue;
            }
            // Hyphenated tokens: prefer CMUdict/lexicon as one headword ("co-founder"); otherwise
            // phonemize segments ("Text-to-image" → text / to / image) instead of eSpeak on the
            // raw string (letter garbage).
            if token.contains('-') {
                let parts: Vec<&str> = token
                    .split('-')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                if parts.len() > 1 {
                    let whole_hit = lexicon::lexicon_lookup(token).is_some()
                        || backend.lookup_word(token, &token_context).is_some();
                    if whole_hit {
                        let (ipa, source) = backend.resolve_word(token, &token_context)?;
                        let ipa = normalize_ipa_for_vocab(&ipa, use_v11);
                        trace_token_g2p(token, &ipa, source);
                        append_lexical(&mut result, &ipa);
                        continue;
                    }
                    for part in parts {
                        let part = part.to_string();
                        if is_acronym(&part) {
                            let ipa = normalize_ipa_for_vocab(&letters_to_ipa(&part), use_v11);
                            trace_token_g2p(&part, &ipa, BackendSource::Dictionary);
                            append_lexical(&mut result, &ipa);
                            continue;
                        }
                        let (ipa, source) = backend.resolve_word(&part, &token_context)?;
                        let ipa = normalize_ipa_for_vocab(&ipa, use_v11);
                        trace_token_g2p(&part, &ipa, source);
                        append_lexical(&mut result, &ipa);
                    }
                    continue;
                }
            }
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
                        past_markers_before: sentence_has_past_markers_before(&tokens, idx),
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
                                past_markers_before: sentence_has_past_markers_before(&tokens, idx),
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
                        past_markers_before: sentence_has_past_markers_before(&tokens, idx),
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

    finish_g2p_output_checks(output, use_v11)
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
        assert!(
            output.contains("həlˈoʊ") || output.contains("həˈloʊ") || output.contains("həˈləʊ"),
            "expected US hello phones, got {output:?}"
        );
        assert!(
            output.contains("wˈɚld") || output.contains("wˈɜːld") || output.contains("wˈɝld"),
            "expected rhotic world, got {output:?}"
        );

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

    #[test]
    fn sanitize_espeak_ipa_strips_variants_and_stress_digits() {
        assert_eq!(super::sanitize_espeak_ipa("haʊ||haʊ"), "haʊ");
        assert_eq!(super::sanitize_espeak_ipa("twˈɛntaɪ2fˈɪv"), "twˈɛntaɪfˈɪv");
    }

    #[test]
    fn g2p_audit_wraps_phonemes_and_unknowns() -> Result<(), super::G2PError> {
        let a = super::g2p_audit("Hi.", false)?;
        assert_eq!(a.phonemes, super::g2p("Hi.", false)?);
        Ok(())
    }

    #[test]
    fn lexicon_override_applies_before_cmudict() -> Result<(), super::G2PError> {
        use std::collections::HashMap;

        use super::lexicon::set_test_lexicon;
        let mut m = HashMap::new();
        m.insert("xyzzy".into(), "zˈɪzi".into());
        set_test_lexicon(Some(m));
        let out = super::g2p("xyzzy", false)?;
        set_test_lexicon(None);
        assert!(
            out.contains("zˈɪzi"),
            "expected lexicon IPA in output, got {out:?}"
        );
        Ok(())
    }

    #[test]
    fn parse_lexicon_lines_tab_separated() {
        let m = super::lexicon::parse_g2p_lexicon_lines("Foo\tfˈu\n# ignored\nBar\tbˈɑ");
        assert_eq!(m.get("foo").map(String::as_str), Some("fˈu"));
        assert_eq!(m.get("bar").map(String::as_str), Some("bˈɑ"));
    }

    #[test]
    fn shout_case_text_uses_dictionary_not_spelling() -> Result<(), super::G2PError> {
        let p = super::g2p("TEXT.", false)?;
        assert!(
            !p.contains("tˈiˈi"),
            "expected word 'text', not letter spelling: {p:?}"
        );
        assert!(
            p.contains("tˈɛk") || p.contains("tɛk"),
            "expected EH vowel from CMUdict text: {p:?}"
        );
        Ok(())
    }

    #[test]
    fn transformative_unstressed_aa_is_schwa() -> Result<(), super::G2PError> {
        let p = super::g2p("transformative.", false)?;
        assert!(
            p.contains("mət"),
            "expected reduced vowel in '-mat-', got {p:?}"
        );
        Ok(())
    }

    #[test]
    fn hyphen_compound_splits_when_not_in_cmudict() -> Result<(), super::G2PError> {
        let p = super::g2p_legacy("Text-to-image.", false)?;
        assert!(
            p.contains("tˈɛkst") && p.contains("ˈɪmədʒ"),
            "expected per-segment phones, got {p:?}"
        );
        Ok(())
    }

    #[test]
    fn hyphen_compound_keeps_cmudict_whole_word() -> Result<(), super::G2PError> {
        let p = super::g2p_legacy("co-founder.", false)?;
        assert!(
            p.contains("fˈaʊnd"),
            "expected single-entry compound 'co-founder', got {p:?}"
        );
        Ok(())
    }

    #[test]
    fn finance_uses_noun_cmudict_variant() -> Result<(), super::G2PError> {
        let p = super::g2p_legacy("a finance professor.", false)?;
        assert!(
            p.contains("fˈaɪn"),
            "expected noun stress FI-nance (CMU finance(3)), got {p:?}"
        );
        assert!(
            !p.contains("fənˈæns"),
            "verb stress fi-NANCE should not be default for this phrase: {p:?}"
        );
        Ok(())
    }

    #[test]
    fn finance_after_to_keeps_verb_cmudict_variant() -> Result<(), super::G2PError> {
        let p = super::g2p_legacy("to finance the plan.", false)?;
        assert!(
            p.contains("fənˈæns") || p.contains("fɪnˈæns"),
            "expected verb fi-NANCE after 'to', got {p:?}"
        );
        Ok(())
    }

    #[test]
    fn camel_case_splits_alpha_go() -> Result<(), super::G2PError> {
        let p = super::g2p_legacy("AlphaGo.", false)?;
        assert!(
            !p.contains("ˈɛlpˈi"),
            "should not letter-spell PascalCase: {p:?}"
        );
        assert!(
            p.contains("ˈælfə") && p.contains("ɡˈoʊ"),
            "expected CMUdict alpha + go, got {p:?}"
        );
        Ok(())
    }

    #[test]
    fn camel_case_segment_boundaries() {
        assert_eq!(
            super::split_camel_case_segments("AlphaGo").unwrap(),
            vec!["Alpha".to_string(), "Go".to_string()]
        );
        assert!(super::split_camel_case_segments("alphago").is_none());
        assert!(super::split_camel_case_segments("XML").is_none());
    }

    #[test]
    fn financial_prefers_fi_nancial_variant() -> Result<(), super::G2PError> {
        let p = super::g2p("Financial.", false)?;
        assert!(
            !p.contains("fənˈæn"),
            "expected FI-nancial (CMU financial(3)), not fi-NAN-cial: {p:?}"
        );
        Ok(())
    }

    #[test]
    fn pytorch_py_is_pie_syllable() -> Result<(), super::G2PError> {
        let p = super::g2p_legacy("PyTorch.", false)?;
        assert!(
            p.contains("ˈpaɪ") && p.contains("tˈɔɹ"),
            "expected pie + torch, got {p:?}"
        );
        assert!(
            !p.contains("ˈiwˈI"),
            "should not spell Py letter-by-letter: {p:?}"
        );
        Ok(())
    }

    #[test]
    fn kokoro_js_normalize_text_dr_smith() {
        let n = super::phonemize_js::normalize_text("Dr. Smith works.");
        assert!(
            n.contains("Doctor"),
            "expected Doctor expansion like kokoro.js, got {n:?}"
        );
    }

    /// Loose alignment with [`kokoro.js/tests/phonemize.test.js`](../../kokoro.js/tests/phonemize.test.js)
    /// under Misaki G2P (strings differ from npm `phonemizer` / eSpeak).
    #[test]
    fn kokoro_js_golden_dr_smith_doctor_prefix() -> Result<(), super::G2PError> {
        let p = super::g2p("Dr. Smith works.", false)?;
        assert!(
            p.contains("kt") || p.contains("ˈd") || p.contains("dˈ"),
            "expected Doctor-style onset in {p:?}"
        );
        assert!(
            p.contains("smˈɪθ") || p.contains("mˈɪθ"),
            "expected Smith-like cluster in {p:?}"
        );
        Ok(())
    }

    #[test]
    fn kokoro_js_phonemize_hello_world_us() -> Result<(), super::G2PError> {
        let p = super::g2p("Hello World", false)?;
        assert!(
            p.contains("hə") && (p.contains("lˈo") || p.contains("ˈlo")),
            "expected Hello phrase phones (kokoro.js-style Misaki segment), got {p:?}"
        );
        assert!(
            p.contains("wˈ") || p.contains("ˈw"),
            "expected World in phrase, got {p:?}"
        );
        Ok(())
    }

    #[test]
    fn typographic_quotes_normalize_for_word_tokens() {
        assert_eq!(super::normalize_g2p_input("L\u{2019}Actrice"), "L'Actrice");
        assert_eq!(super::normalize_g2p_input("\u{201C}Hi\u{201D}"), "\"Hi\"");
    }

    #[test]
    fn strip_elided_l_article_keeps_lexicon_whole_word() {
        assert_eq!(
            super::strip_elided_l_article_word("L'Actrice"),
            Some("Actrice")
        );
        assert!(super::strip_elided_l_article_word("l'enfer").is_none());
        assert!(super::strip_elided_l_article_word("Let's").is_none());
    }

    #[test]
    fn l_actrice_and_curmer_use_word_phones_not_spelling() -> Result<(), super::G2PError> {
        let p = super::g2p(
            "titled \"L'Actrice,\" or \"The Actress,\" is taken from a book by Louis Curmer published in 1841.",
            false,
        )?;
        assert!(
            !p.contains("sˈi jˈuː ˈɑːɹ ˈɛm ˈiː ˈɑːɹ"),
            "should not letter-spell Curmer: {p:?}"
        );
        assert!(
            !p.contains("ˈɛl ˈA sˈiː tˈiː"),
            "should not letter-spell L'Actrice: {p:?}"
        );
        assert!(
            p.contains("kˈɜː") && p.contains("mˈɜː"),
            "Curmer should recover Cur-mer-style phones: {p:?}"
        );
        assert!(
            p.contains("ˈækt ɹˈaɪs") || p.contains("ˌɛl ˈækt"),
            "L'Actrice should use el + Act-rice heuristic: {p:?}"
        );
        Ok(())
    }

    #[test]
    fn guterson_lexicon_and_the_actress_use_di() -> Result<(), super::G2PError> {
        let p = super::g2p(
            "titled \"L'Actrice,\" or \"The Actress,\" is taken from a book by Louis Guterson published in 1841.",
            false,
        )?;
        assert!(
            p.contains("ɡ") && p.contains("ʌt") && p.contains("sən"),
            "Guterson should use embedded OOV IPA (not letter spelling): {p:?}"
        );
        assert!(
            !p.contains("ʤˈi jˈuː tˈiː"),
            "should not letter-spell Guterson: {p:?}"
        );
        assert!(
            p.contains("ði ˈækt"),
            "\"the\" before Actress should be /ði/: {p:?}"
        );
        Ok(())
    }
}
