//! System [`espeak-ng`](https://github.com/espeak-ng/espeak-ng) subprocess — **kokoro-js** feeds each
//! non-punctuation section through eSpeak for IPA (see [`segment_ipa`]); [`word_ipa`] is a narrow
//! fallback when Misaki letter-spells a single token.
//!
//! Env: `KOKORO_ESPEAK_NG=0`/`off`/`false` disables both paths. `KOKORO_ESPEAK_NG_BIN` overrides the
//! binary name/path. `KOKORO_G2P_SEGMENT_ESPEAK=0` skips segment-level phonemization in
//! [`crate::g2p::phonemize_js::phonemize_like_kokoro_js`] (forces Misaki tiers only).

use std::io::Write;
use std::process::{Command, Stdio};

fn espeak_disabled() -> bool {
    matches!(
        std::env::var("KOKORO_ESPEAK_NG").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    )
}

fn voice_and_bin(british: bool) -> (String, String) {
    let bin = std::env::var("KOKORO_ESPEAK_NG_BIN").unwrap_or_else(|_| "espeak-ng".to_string());
    let voice = if british {
        "en-gb".to_string()
    } else {
        "en-us".to_string()
    };
    (bin, voice)
}

fn ipa_from_stdout(stdout: &str) -> Option<String> {
    let parts: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if parts.is_empty() {
        return None;
    }
    Some(parts.join(" "))
}

/// Whole-section IPA like kokoro-js `phonemizer(section)` (multi-word supported via stdin).
pub(crate) fn segment_ipa(text: &str, british: bool) -> Option<String> {
    if espeak_disabled() {
        return None;
    }
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 8192 || trimmed.contains('\0') {
        return None;
    }

    let (bin, voice) = voice_and_bin(british);

    let stdin_attempt = (|| {
        let mut child = Command::new(&bin)
            .args(["-q", "--ipa=3", "-v", &voice, "--stdin"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        {
            let stdin = child.stdin.as_mut()?;
            stdin.write_all(trimmed.as_bytes()).ok()?;
        }
        child.wait_with_output().ok()
    })();

    if let Some(output) = stdin_attempt {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(ipa) = ipa_from_stdout(&stdout) {
                return Some(ipa);
            }
        }
    }

    // Some installs lack `--stdin`; fall back to argv (short segments only).
    if trimmed.len() < 8000 {
        let output = Command::new(&bin)
            .args(["-q", "--ipa=3", "-v", &voice])
            .arg(trimmed)
            .output()
            .ok()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            return ipa_from_stdout(&stdout);
        }
    }

    None
}

pub(crate) fn word_ipa(word: &str, british: bool) -> Option<String> {
    if espeak_disabled() {
        return None;
    }
    if word.is_empty() || word.chars().count() > 512 {
        return None;
    }
    if !word
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '\'' | '-' | '‐'))
    {
        return None;
    }

    let (bin, voice) = voice_and_bin(british);

    let output = Command::new(&bin)
        .args(["-q", "--ipa=3", "-v", &voice])
        .arg(word)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    ipa_from_stdout(&stdout)
}
