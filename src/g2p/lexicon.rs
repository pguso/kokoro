//! Optional word → IPA overrides loaded from `KOKORO_G2P_LEXICON` (UTF-8 file).

use std::{collections::HashMap, fs, sync::LazyLock};

use log::warn;

#[cfg(test)]
thread_local! {
    static TEST_LEXICON: std::cell::RefCell<Option<HashMap<String, String>>> =
        std::cell::RefCell::new(None);
}

/// Parse `word<TAB>ipa` lines; `#` starts a comment; empty lines ignored.
/// Keys are stored lowercase for lookup.
pub(crate) fn parse_g2p_lexicon_lines(content: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((word, ipa)) = line.split_once('\t') else {
            continue;
        };
        let word = word.trim();
        if word.is_empty() {
            continue;
        }
        m.insert(word.to_ascii_lowercase(), ipa.trim().to_string());
    }
    m
}

fn load_lexicon_from_env() -> HashMap<String, String> {
    let Ok(path) = std::env::var("KOKORO_G2P_LEXICON") else {
        return HashMap::new();
    };
    match fs::read_to_string(&path) {
        Ok(content) => parse_g2p_lexicon_lines(&content),
        Err(e) => {
            warn!(
                "kokoro g2p: could not read KOKORO_G2P_LEXICON={path} ({e}); lexicon overrides disabled"
            );
            HashMap::new()
        }
    }
}

static LEXICON: LazyLock<HashMap<String, String>> = LazyLock::new(load_lexicon_from_env);

pub(crate) fn lexicon_lookup(token: &str) -> Option<String> {
    let lower = token.to_ascii_lowercase();
    #[cfg(test)]
    {
        let override_map = TEST_LEXICON.with(|r| r.borrow().clone());
        if let Some(map) = override_map {
            return map.get(&lower).cloned();
        }
    }
    LEXICON.get(&lower).cloned()
}

#[cfg(test)]
pub(crate) fn set_test_lexicon(map: Option<HashMap<String, String>>) {
    TEST_LEXICON.with(|r| *r.borrow_mut() = map);
}
