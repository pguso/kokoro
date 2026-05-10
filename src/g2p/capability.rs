//! Runtime **eSpeak backend** detection for Kokoro.js–style OOV phonemization.
//!
//! - **Bundled:** crate feature [`g2p-espeak`](crate) links Misaki’s `espeak-rs` (no system install).
//! - **CLI:** system `espeak-ng` enables [`super::espeak_cli`] segment/word subprocesses.

use std::process::Command;
use std::sync::OnceLock;

/// Whether an eSpeak-class backend is available for arbitrary spellings (cf. kokoro-js phonemizer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G2pEspeakCapability {
    /// Bundled Misaki eSpeak and/or working `espeak-ng` CLI — OOV words get rule-based phones.
    KokoroJsParity,
    /// No bundled `g2p-espeak` **and** CLI unavailable/disabled — Misaki letter-spells unknown words.
    DegradedMisakiOnly,
}

fn cli_espeak_works() -> bool {
    if matches!(
        std::env::var("KOKORO_ESPEAK_NG").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    ) {
        return false;
    }
    let bin = std::env::var("KOKORO_ESPEAK_NG_BIN").unwrap_or_else(|_| "espeak-ng".to_string());
    Command::new(&bin)
        .args(["-q", "--ipa=3", "-v", "en-us", "kokoro"])
        .output()
        .is_ok_and(|o| o.status.success())
}

/// One-time probe; safe to call from any thread after logging is initialized.
pub fn g2p_espeak_capability() -> G2pEspeakCapability {
    static CELL: OnceLock<G2pEspeakCapability> = OnceLock::new();
    *CELL.get_or_init(|| {
        let bundled = cfg!(feature = "g2p-espeak");
        let cli = cli_espeak_works();
        let cap = if bundled || cli {
            G2pEspeakCapability::KokoroJsParity
        } else {
            G2pEspeakCapability::DegradedMisakiOnly
        };

        if matches!(cap, G2pEspeakCapability::DegradedMisakiOnly) {
            log::warn!(
                "kokoro-en G2P: degraded Misaki-only mode (no bundled g2p-espeak and no working espeak-ng CLI). \
                 Install espeak-ng or build with default features for kokoro-js-like OOV pronunciation."
            );
        }

        cap
    })
}

/// When `KOKORO_G2P_REQUIRE_ESPEAK=1`, fail fast if neither bundled nor CLI eSpeak is available.
pub(crate) fn enforce_require_espeak_if_configured() -> Result<(), super::G2PError> {
    if std::env::var("KOKORO_G2P_REQUIRE_ESPEAK").ok().as_deref() != Some("1") {
        return Ok(());
    }
    if matches!(
        g2p_espeak_capability(),
        G2pEspeakCapability::DegradedMisakiOnly
    ) {
        return Err(super::G2PError::Backend(
            "KOKORO_G2P_REQUIRE_ESPEAK=1 but no eSpeak backend (need default features with g2p-espeak and/or system espeak-ng)"
                .into(),
        ));
    }
    Ok(())
}
