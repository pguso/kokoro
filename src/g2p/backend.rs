use {
    cmudict_fast::{Cmudict, Rule},
    log::warn,
    std::{str::FromStr, sync::LazyLock},
};

use super::{lexicon::lexicon_lookup, sanitize_espeak_ipa};

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

fn ipa_from_cmudict_rules(rules: &[Rule]) -> Option<String> {
    let pronunciation = rules.first()?.pronunciation();
    let ipa = pronunciation
        .iter()
        .map(|p| crate::arpa_to_ipa(&p.to_string()).unwrap_or_default())
        .collect::<String>();
    if ipa.is_empty() { None } else { Some(ipa) }
}

fn lookup_cmudict_key(dict: &Cmudict, key: &str) -> Option<String> {
    dict.get(key).and_then(ipa_from_cmudict_rules)
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

        if token.eq_ignore_ascii_case("read") {
            let prefer_present = !context.past_markers_before;
            let keys: &[&str] = if prefer_present {
                &["read(2)", "read"]
            } else {
                &["read", "read(2)"]
            };
            for key in keys {
                if let Some(ipa) = lookup_cmudict_key(dict, key) {
                    return Some(ipa);
                }
            }
            return None;
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

        static ESPEAK_AVAILABLE: LazyLock<bool> = LazyLock::new(|| {
            let available = voice_g2p::espeak::EspeakFallback::new().is_available();
            if !available {
                warn!(
                    "kokoro g2p: espeak-ng unavailable; using heuristic letter fallback for unknown words"
                );
            }
            available
        });
        if !*ESPEAK_AVAILABLE {
            return Ok(crate::letters_to_ipa(token));
        }

        let phonemes = voice_g2p::english_to_phonemes(token)
            .map_err(|e| crate::G2PError::Backend(e.to_string()))?;
        let phonemes = sanitize_espeak_ipa(phonemes.trim());
        if phonemes.is_empty() {
            return Ok(crate::letters_to_ipa(token));
        }
        Ok(phonemes)
    }
}
