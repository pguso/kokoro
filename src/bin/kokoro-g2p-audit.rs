//! Print G2P phonemes and any tokenizer-unknown symbols (for comparing to other Kokoro ports).
//!
//! ```text
//! KOKORO_G2P_V11=1 cargo run --bin kokoro-g2p-audit -- "Hello, world."
//! ```

use std::env;

use kokoro_en::g2p_audit;

fn main() {
    let use_v11 = env::var("KOKORO_G2P_V11").ok().as_deref() == Some("1");
    let text = env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello, world!".to_owned());

    match g2p_audit(&text, use_v11) {
        Ok(a) => {
            println!("phonemes: {}", a.phonemes);
            if a.unknown_phoneme_chars.is_empty() {
                println!("unknown tokenizer symbols: (none)");
            } else {
                println!(
                    "unknown tokenizer symbols: {}",
                    a.unknown_phoneme_chars
                        .iter()
                        .map(|c| format!("U+{:04X} {}", *c as u32, c))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}
