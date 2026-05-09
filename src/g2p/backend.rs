use {
    cmudict_fast::Cmudict,
    log::warn,
    std::{str::FromStr, sync::LazyLock},
};

#[derive(Clone, Copy, Debug)]
pub enum BackendSource {
    Dictionary,
    Fallback,
    Heuristic,
}

#[derive(Debug, Default, Clone)]
pub struct TokenContext {
    pub previous_word: Option<String>,
    pub next_word: Option<String>,
}

pub trait EnglishG2pBackend {
    fn lookup_word(&self, token: &str, context: &TokenContext) -> Option<String>;
    fn fallback_word(&self, token: &str, context: &TokenContext)
    -> Result<String, crate::G2PError>;
}

#[derive(Default)]
pub struct HybridEnglishBackend;

impl HybridEnglishBackend {
    pub fn resolve_word(
        &self,
        token: &str,
        context: &TokenContext,
    ) -> Result<(String, BackendSource), crate::G2PError> {
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
    fn lookup_word(&self, token: &str, _context: &TokenContext) -> Option<String> {
        let _ = (
            _context.previous_word.as_deref(),
            _context.next_word.as_deref(),
        );
        static CMUDICT: LazyLock<Option<Cmudict>> =
            LazyLock::new(|| Cmudict::from_str(include_str!("../../dict/cmudict.dict")).ok());
        let dict = CMUDICT.as_ref()?;
        let token_candidates = [
            token,
            &token.to_ascii_lowercase(),
            &token.to_ascii_uppercase(),
        ];
        for candidate in token_candidates {
            if let Some(rules) = dict.get(candidate) {
                let pronunciation = rules.first()?.pronunciation();
                let ipa = pronunciation
                    .iter()
                    .map(|p| crate::arpa_to_ipa(&p.to_string()).unwrap_or_default())
                    .collect::<String>();
                if !ipa.is_empty() {
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
        let phonemes = phonemes.trim().to_string();
        if phonemes.is_empty() {
            return Ok(crate::letters_to_ipa(token));
        }
        Ok(phonemes)
    }
}
