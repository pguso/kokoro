use {
    cmudict_fast::{Cmudict, Rule},
    misaki_rs::{G2P as MisakiG2p, Language as MisakiLanguage},
    std::{str::FromStr, sync::LazyLock},
};

use super::{espeak_cli, lexicon::lexicon_lookup, sanitize_espeak_ipa};

/// Kokoro-aligned English G2P (misaki-rs): POS tagger + lexicon + rules; optional espeak OOV when
/// built with crate feature `g2p-espeak`.
static MISAKI_G2P_US: LazyLock<MisakiG2p> =
    LazyLock::new(|| MisakiG2p::new(MisakiLanguage::EnglishUS));

static MISAKI_G2P_GB: LazyLock<MisakiG2p> =
    LazyLock::new(|| MisakiG2p::new(MisakiLanguage::EnglishGB));

/// Shared Misaki instance for US English (matches kokoro.js `language === "a"` / `en-us`).
pub(crate) fn misaki_us() -> &'static MisakiG2p {
    &MISAKI_G2P_US
}

/// British English Misaki (kokoro.js `language === "b"` / `en`).
pub(crate) fn misaki_gb() -> &'static MisakiG2p {
    &MISAKI_G2P_GB
}

/// Sanitize Misaki/eSpeak IPA for a token or an entire phrase (whitespace splits token phones).
pub(crate) fn sanitize_espeak_ipa_phrase(raw: &str) -> String {
    let collapsed = raw.replace('❓', "");
    let mut parts = Vec::new();
    for w in collapsed.split_whitespace() {
        let s = sanitize_espeak_ipa(w);
        if !s.is_empty() {
            parts.push(s);
        }
    }
    parts.join(" ")
}

/// Phonemize a phrase or sentence like kokoro.js `phonemizer(text, lang)` — full Misaki `g2p` with context.
pub(crate) fn misaki_phonemize_segment(text: &str, british: bool) -> String {
    let g2p = if british { misaki_gb() } else { misaki_us() };
    match g2p.g2p(text.trim()) {
        Ok((ipa, _)) => {
            let s = sanitize_espeak_ipa_phrase(ipa.trim());
            if s.contains('❓') { String::new() } else { s }
        }
        Err(_) => String::new(),
    }
}

/// Letter-by-letter Misaki output for unknown proper names (no bundled eSpeak) often has about one
/// IPA stress group per letter; hyphen splits like `Cur-mer` / `Act-rice` recover a spoken word.
fn looks_like_letter_spelling(token: &str, ipa: &str) -> bool {
    let n = token.chars().filter(|c| c.is_ascii_alphabetic()).count();
    if n < 5 || n > 48 {
        return false;
    }
    if !token.chars().all(|c| c.is_ascii_alphabetic() || c == '\'') {
        return false;
    }
    let chunks = ipa.split_whitespace().filter(|s| !s.is_empty()).count();
    chunks >= n.saturating_sub(2) || (chunks >= 4 && chunks * 2 >= n)
}

/// Misaki/eSpeak may return no phones for some spellings; hyphenating can recover a plausible read
/// (`Cur-mer`). Prefer **fewest** IPA chunks, then **balanced** left/right lengths (so `Act-rice`
/// wins over `Ac-trice`), then shorter IPA.
fn hyphen_split_misaki_fallback(token: &str, british: bool) -> Option<String> {
    if !token.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let chars: Vec<char> = token.chars().collect();
    let n = chars.len();
    if n < 5 {
        return None;
    }
    let mut best_s: Option<String> = None;
    let mut best_chunks = usize::MAX;
    let mut best_neg_min_side = i32::MIN;
    let mut best_len = usize::MAX;

    for split in 2..=(n - 2) {
        let left_len = split;
        let right_len = n - split;
        let left: String = chars[..split].iter().collect();
        let right: String = chars[split..].iter().collect();
        let hyphenated = format!("{left}-{right}");
        let s = misaki_phonemize_segment(&hyphenated, british);
        if s.is_empty() {
            continue;
        }
        let chunks = s.split_whitespace().filter(|x| !x.is_empty()).count();
        let min_side = left_len.min(right_len) as i32;
        let better = best_s.as_ref().map_or(true, |_| {
            chunks < best_chunks
                || (chunks == best_chunks && min_side > best_neg_min_side)
                || (chunks == best_chunks && min_side == best_neg_min_side && s.len() < best_len)
        });
        if better {
            best_chunks = chunks;
            best_neg_min_side = min_side;
            best_len = s.len();
            best_s = Some(s);
        }
    }
    best_s
}

/// Per-word Misaki with hyphen retry for letter-spelled OOV; [`misaki_phonemize_segment`] for phrases.
pub(crate) fn misaki_phonemize_word_oov_retry(word: &str, british: bool) -> String {
    if let Some(ipa) = lexicon_lookup(word) {
        if !ipa.is_empty() {
            return sanitize_espeak_ipa_phrase(ipa.trim());
        }
    }

    if let Some(rest) = super::strip_elided_l_article_word(word) {
        let stem_ipa = misaki_phonemize_word_oov_retry(rest, british);
        let el_ipa = misaki_phonemize_segment("el", british);
        if stem_ipa.is_empty() {
            return el_ipa;
        }
        if el_ipa.is_empty() {
            return stem_ipa;
        }
        return format!("{el_ipa} {stem_ipa}");
    }

    let phonemes = misaki_phonemize_segment(word, british);
    if phonemes.is_empty() {
        return phonemes;
    }
    if looks_like_letter_spelling(word, &phonemes) {
        if let Some(raw) = espeak_cli::word_ipa(word, british) {
            let s = sanitize_espeak_ipa_phrase(raw.trim());
            if !s.is_empty() && !s.contains('❓') {
                let orig_c = phonemes.split_whitespace().count();
                let cli_c = s.split_whitespace().count();
                if cli_c < orig_c || !looks_like_letter_spelling(word, &s) {
                    return s;
                }
            }
        }
        if let Some(h) = hyphen_split_misaki_fallback(word, british) {
            let orig_c = phonemes.split_whitespace().count();
            let hy_c = h.split_whitespace().count();
            if hy_c < orig_c || h.len() + 12 < phonemes.len() {
                return h;
            }
        }
    }
    phonemes
}

static CMUDICT: LazyLock<Option<Cmudict>> =
    LazyLock::new(|| Cmudict::from_str(include_str!("../../dict/cmudict.dict")).ok());

/// True if `lower` is a headword in the embedded CMUdict (used to avoid spelling common
/// words letter-by-letter when they appear in ALL CAPS, e.g. "TEXT" → "text").
pub(crate) fn cmudict_has_entry(lower: &str) -> bool {
    CMUDICT
        .as_ref()
        .is_some_and(|dict| dict.get(lower).is_some())
}

#[derive(Clone, Copy, Debug)]
pub enum BackendSource {
    /// [`KOKORO_G2P_LEXICON`] file override.
    Lexicon,
    Dictionary,
    Fallback,
    Heuristic,
}

#[derive(Debug, Default, Clone)]
pub struct TokenContext {
    pub previous_word: Option<String>,
    pub next_word: Option<String>,
    /// When true, earlier tokens in this sentence often imply past tense (used for CMUdict homographs).
    pub past_markers_before: bool,
}

pub trait EnglishG2pBackend {
    fn lookup_word(&self, token: &str, context: &TokenContext) -> Option<String>;
    fn fallback_word(&self, token: &str, context: &TokenContext)
    -> Result<String, crate::G2PError>;
}

#[derive(Default)]
pub struct HybridEnglishBackend;

fn ipa_from_cmudict_rule(rule: &Rule) -> Option<String> {
    let pronunciation = rule.pronunciation();
    let ipa = pronunciation
        .iter()
        .map(|p| crate::arpa_to_ipa(&p.to_string()).unwrap_or_default())
        .collect::<String>();
    if ipa.is_empty() { None } else { Some(ipa) }
}

fn ipa_from_cmudict_rules(rules: &[Rule]) -> Option<String> {
    ipa_from_cmudict_rule(rules.first()?)
}

/// `Word` shape (single leading capital, rest lowercase). Used to detect headline-style tokens that
/// eSpeak often letter-spells when OOV.
fn is_title_case_ascii_word(token: &str) -> bool {
    let mut it = token.chars();
    let Some(first) = it.next() else {
        return false;
    };
    if !first.is_ascii_uppercase() || token.len() < 4 {
        return false;
    }
    it.all(|c| c.is_ascii_lowercase())
}

impl HybridEnglishBackend {
    pub fn resolve_word(
        &self,
        token: &str,
        context: &TokenContext,
    ) -> Result<(String, BackendSource), crate::G2PError> {
        if let Some(ipa) = lexicon_lookup(token) {
            if !ipa.is_empty() {
                return Ok((ipa, BackendSource::Lexicon));
            }
        }
        if let Some(ipa) = self.lookup_word(token, context) {
            return Ok((ipa, BackendSource::Dictionary));
        }
        let ipa = self.fallback_word(token, context)?;
        if ipa == crate::letters_to_ipa(token) {
            return Ok((ipa, BackendSource::Heuristic));
        }
        Ok((ipa, BackendSource::Fallback))
    }
}

impl EnglishG2pBackend for HybridEnglishBackend {
    fn lookup_word(&self, token: &str, context: &TokenContext) -> Option<String> {
        let dict = CMUDICT.as_ref()?;

        // CMUdict merges `word`, `word(2)`, … under one key; variants are `Rule`s in file order.
        if token.eq_ignore_ascii_case("read") {
            let rules = dict.get("read")?;
            let prefer_present = !context.past_markers_before;
            let idx = if prefer_present && rules.len() > 1 {
                1
            } else {
                0
            };
            let rule = rules.get(idx).or_else(|| rules.first())?;
            return ipa_from_cmudict_rule(rule);
        }

        if token.eq_ignore_ascii_case("finance") {
            let rules = dict.get("finance")?;
            let prev = context
                .previous_word
                .as_deref()
                .unwrap_or("")
                .to_ascii_lowercase();
            let verb_hint = matches!(
                prev.as_str(),
                "to" | "will"
                    | "would"
                    | "can"
                    | "could"
                    | "should"
                    | "must"
                    | "may"
                    | "might"
                    | "let's"
                    | "lets"
            );
            let idx = if verb_hint {
                0
            } else if rules.len() > 2 {
                2
            } else {
                0
            };
            let rule = rules.get(idx).or_else(|| rules.first())?;
            return ipa_from_cmudict_rule(rule);
        }

        if token.eq_ignore_ascii_case("financial") {
            let rules = dict.get("financial")?;
            // CMU order: (0) fi-NAN-cial, (2) FI-nancial — prefer FI-nancial for prose titles.
            let idx = if rules.len() > 2 { 2 } else { 0 };
            let rule = rules.get(idx).or_else(|| rules.first())?;
            return ipa_from_cmudict_rule(rule);
        }

        let token_candidates = [
            token,
            &token.to_ascii_lowercase(),
            &token.to_ascii_uppercase(),
        ];
        for candidate in token_candidates {
            if let Some(rules) = dict.get(candidate) {
                if let Some(ipa) = ipa_from_cmudict_rules(rules) {
                    return Some(ipa);
                }
            }
        }
        None
    }

    fn fallback_word(
        &self,
        token: &str,
        _context: &TokenContext,
    ) -> Result<String, crate::G2PError> {
        let _ = (
            _context.previous_word.as_deref(),
            _context.next_word.as_deref(),
        );
        if std::env::var("KOKORO_G2P_DISABLE_EXTERNAL").ok().as_deref() == Some("1") {
            return Ok(crate::letters_to_ipa(token));
        }

        let british = crate::pipeline::G2pPipelineConfig::from_env().british;
        let phonemes = misaki_phonemize_word_oov_retry(token, british);
        let letter_fallback = crate::letters_to_ipa(token);

        let try_lowercase_misaki = || -> Option<String> {
            let lower = token.to_ascii_lowercase();
            if lower == token || !lower.chars().all(|c| c.is_ascii_alphabetic()) {
                return None;
            }
            let p = misaki_phonemize_word_oov_retry(&lower, british);
            (!p.is_empty()).then_some(p)
        };

        if phonemes.is_empty() {
            if let Some(p) = try_lowercase_misaki() {
                return Ok(p);
            }
            if let Some(p) = hyphen_split_misaki_fallback(token, british) {
                return Ok(p);
            }
            if let Some(p) = hyphen_split_misaki_fallback(&token.to_ascii_lowercase(), british) {
                return Ok(p);
            }
            return Ok(letter_fallback);
        }

        // Capitalized unknowns (e.g. proper names) often come back as letter-by-letter phones;
        // retry with lowercase so eSpeak uses grapheme rules instead.
        if phonemes == letter_fallback {
            if let Some(p) = try_lowercase_misaki() {
                if p != letter_fallback {
                    return Ok(p);
                }
            }
        }

        // Misaki output may still resemble letter spelling for some capitalized OOV words; prefer a
        // lowercase read when it is much shorter.
        if is_title_case_ascii_word(token) {
            if let Some(p_lower) = try_lowercase_misaki() {
                if p_lower != phonemes && p_lower.len() >= 4 {
                    let spelling_like = phonemes.len() > p_lower.len().saturating_mul(2)
                        || phonemes.len() > p_lower.len().saturating_add(14);
                    if spelling_like {
                        return Ok(p_lower);
                    }
                }
            }
        }
        Ok(phonemes)
    }
}
