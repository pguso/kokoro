//! Kokoro-style inference orchestration: env-backed G2P mode, KPipeline-style phoneme chunking.
//!
//! Python [`KPipeline`](https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py) keeps
//! phoneme strings under **510 characters** before calling the model. We mirror that budget here by
//! splitting IPA output at word boundaries (approximation of `en_tokenize` without Misaki tokens).

/// Maximum phoneme-string length (characters) per ONNX call, matching Python Kokoro pipelines.
pub const MAX_PHONEME_CHARS: usize = 510;

/// Resolved runtime flags for G2P (normally from environment variables).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G2pPipelineConfig {
    /// British English when `true` (kokoro.js `language === "b"` / `en`).
    pub british: bool,
    /// Use [`crate::g2p_legacy`](crate::g2p_legacy) instead of kokoro.js–style phonemization.
    pub legacy: bool,
}

impl Default for G2pPipelineConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl G2pPipelineConfig {
    /// `KOKORO_G2P_LANG=b` → British; `KOKORO_G2P_LEGACY=1` → legacy CMUdict + Misaki path.
    ///
    /// See also [`crate::g2p::capability`] and **`KOKORO_G2P_REQUIRE_ESPEAK=1`** (fail [`crate::g2p`] if no bundled or CLI eSpeak).
    pub fn from_env() -> Self {
        Self {
            british: std::env::var("KOKORO_G2P_LANG")
                .map(|s| s.eq_ignore_ascii_case("b"))
                .unwrap_or(false),
            legacy: std::env::var("KOKORO_G2P_LEGACY").ok().as_deref() == Some("1"),
        }
    }
}

/// Split an IPA phoneme line into segments no longer than `max_chars`, preferring whitespace
/// boundaries (greedy packing). Oversized single “words” are split into grapheme chunks.
pub fn chunk_phonemes(ps: &str, max_chars: usize) -> Vec<String> {
    let ps = ps.trim();
    if ps.is_empty() {
        return Vec::new();
    }
    if ps.chars().count() <= max_chars {
        return vec![ps.to_string()];
    }

    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_chars = 0usize;

    for word in ps.split_whitespace() {
        let wl = word.chars().count();
        if wl > max_chars {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
                cur_chars = 0;
            }
            push_word_chunks(word, max_chars, &mut out);
            continue;
        }

        let add = if cur.is_empty() { wl } else { wl + 1 };
        if cur_chars + add <= max_chars {
            if !cur.is_empty() {
                cur.push(' ');
                cur_chars += 1;
            }
            cur.push_str(word);
            cur_chars += wl;
        } else {
            out.push(std::mem::take(&mut cur));
            cur.push_str(word);
            cur_chars = wl;
        }
    }

    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn push_word_chunks(word: &str, max_chars: usize, out: &mut Vec<String>) {
    let chars: Vec<char> = word.chars().collect();
    for chunk in chars.chunks(max_chars) {
        out.push(chunk.iter().collect());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_empty() {
        assert!(chunk_phonemes("", MAX_PHONEME_CHARS).is_empty());
    }

    #[test]
    fn chunk_short_unchanged() {
        let s = "həlˈoʊ wˈɜːld";
        assert_eq!(chunk_phonemes(s, MAX_PHONEME_CHARS), vec![s.to_string()]);
    }

    #[test]
    fn chunk_splits_long_whitespace_runs() {
        let mut long = String::new();
        for i in 0..80 {
            long.push_str(&format!("w{i} "));
        }
        let long = long.trim();
        let chunks = chunk_phonemes(long, 40);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.chars().count() <= 40);
        }
    }
}
