//! Kokoro.js [`phonemize`](https://github.com/hexgrad/kokoro/blob/main/kokoro-js/src/phonemize.js) parity:
//! normalize → punctuation split → **segment IPA** (same order as upstream priority):
//! 1. **system `espeak-ng`** on the whole section ([`super::espeak_cli::segment_ipa`]) unless
//!    `KOKORO_G2P_SEGMENT_ESPEAK=0` or `KOKORO_ESPEAK_NG` disables eSpeak.
//! 2. With crate feature `g2p-espeak`, phrase [`super::backend::misaki_phonemize_segment`] (Misaki +
//!    bundled `espeak-rs` OOV).
//! 3. Word-by-word Misaki + `L'` / hyphen / per-token `espeak-ng` ([`super::backend::misaki_phonemize_word_oov_retry`])
//!    when neither applies or returns empty.

use fancy_regex::Regex as FancyRegex;
use regex::{Captures, Regex as StdRegex};

#[cfg(feature = "g2p-espeak")]
use super::backend::misaki_phonemize_segment;
use super::backend::misaki_phonemize_word_oov_retry;
use super::espeak_cli;
use super::normalize_ipa_for_vocab;

fn phonemize_text_segment(text: &str, british: bool) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if std::env::var("KOKORO_G2P_SEGMENT_ESPEAK").ok().as_deref() != Some("0") {
        if let Some(raw) = espeak_cli::segment_ipa(trimmed, british) {
            let s = super::backend::sanitize_espeak_ipa_phrase(raw.trim());
            if !s.is_empty() && !s.contains('❓') {
                return s;
            }
        }
    }

    #[cfg(feature = "g2p-espeak")]
    {
        let s = misaki_phonemize_segment(trimmed, british);
        if !s.is_empty() && !s.contains('❓') {
            return s;
        }
    }

    trimmed
        .split_whitespace()
        .map(|w| misaki_phonemize_word_oov_retry(w, british))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split sections like kokoro.js `split(text, PUNCTUATION_PATTERN)`.
#[derive(Debug, Clone)]
pub struct Section {
    pub is_punctuation: bool,
    pub text: String,
}

fn escape_reg_exp(string: &str) -> String {
    let mut out = String::with_capacity(string.len() * 2);
    for ch in string.chars() {
        match ch {
            '.' | '*' | '+' | '?' | '^' | '$' | '{' | '}' | '(' | ')' | '|' | '[' | ']' | '\\'
            | '-' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

/// Same character class as kokoro.js `PUNCTUATION`.
const PUNCTUATION: &str = ";:,.!?¡¿—…\"«»“”(){}[]";

fn punctuation_pattern() -> StdRegex {
    let inner = escape_reg_exp(PUNCTUATION);
    let pat = format!(r"(\s*[{inner}]+\s*)+");
    StdRegex::new(&pat).expect("punctuation pattern")
}

/// Port of kokoro.js `split(text, regex)`.
pub fn split_keep_delimiters(text: &str, re: &StdRegex) -> Vec<Section> {
    let mut result = Vec::new();
    let mut prev = 0usize;
    for m in re.find_iter(text) {
        let match_start = m.start();
        if prev < match_start {
            result.push(Section {
                is_punctuation: false,
                text: text[prev..match_start].to_string(),
            });
        }
        let full = m.as_str();
        if !full.is_empty() {
            result.push(Section {
                is_punctuation: true,
                text: full.to_string(),
            });
        }
        prev = m.end();
    }
    if prev < text.len() {
        result.push(Section {
            is_punctuation: false,
            text: text[prev..].to_string(),
        });
    }
    result
}

fn split_num_cs(match_str: &str) -> String {
    if match_str.contains('.') {
        return match_str.to_string();
    }
    if match_str.contains(':') {
        let parts: Vec<&str> = match_str.split(':').collect();
        if parts.len() == 2 {
            let h: i32 = parts[0].parse().unwrap_or(0);
            let m: i32 = parts[1].parse().unwrap_or(0);
            if m == 0 {
                return format!("{h} o'clock");
            }
            if m < 10 {
                return format!("{h} oh {m}");
            }
            return format!("{h} {m}");
        }
    }
    let year: i32 = match_str[..std::cmp::min(4, match_str.len())]
        .parse()
        .unwrap_or(0);
    if year < 1100 || year % 1000 < 10 {
        return match_str.to_string();
    }
    if match_str.len() < 4 {
        return match_str.to_string();
    }
    let left = &match_str[..2];
    let right: i32 = match_str[2..4].parse().unwrap_or(0);
    let suffix = if match_str.ends_with('s') { "s" } else { "" };
    let y = year;
    if y % 1000 >= 100 && y % 1000 <= 999 {
        if right == 0 {
            return format!("{left} hundred{suffix}");
        }
        if right < 10 {
            return format!("{left} oh {right}{suffix}");
        }
    }
    format!("{left} {right}{suffix}")
}

fn flip_money_cs(m: &str) -> String {
    let bill = if m.starts_with('$') {
        "dollar"
    } else {
        "pound"
    };
    let rest = &m[1..];
    if rest.parse::<f64>().is_err() {
        return format!("{rest} {bill}s");
    }
    if !m.contains('.') {
        let suffix = if rest == "1" { "" } else { "s" };
        return format!("{rest} {bill}{suffix}");
    }
    let parts: Vec<&str> = m[1..].split('.').collect();
    if parts.len() < 2 {
        return m.to_string();
    }
    let b = parts[0];
    let c = parts[1];
    let d: i32 = format!("{:0<2}", c)
        .chars()
        .take(2)
        .collect::<String>()
        .parse()
        .unwrap_or(0);
    let coins = if m.starts_with('$') {
        if d == 1 { "cent" } else { "cents" }
    } else if d == 1 {
        "penny"
    } else {
        "pence"
    };
    let bill_plural = if b == "1" { "" } else { "s" };
    format!("{b} {bill}{bill_plural} and {d} {coins}")
}

fn point_num_cs(m: &str) -> String {
    let parts: Vec<&str> = m.split('.').collect();
    if parts.len() != 2 {
        return m.to_string();
    }
    let a = parts[0];
    let b = parts[1];
    let digits: Vec<String> = b.chars().map(|c| c.to_string()).collect();
    format!("{} point {}", a, digits.join(" "))
}

/// Port of kokoro.js `normalize_text`.
pub fn normalize_text(text: &str) -> String {
    let mut s = text.to_string();

    // 1. Quotes and brackets
    s = s.replace(['‘', '’', '\u{201b}'], "'");
    s = s.replace('«', "\u{201c}");
    s = s.replace('»', "\u{201d}");
    s = s.replace(['\u{201c}', '\u{201d}'], "\"");
    s = s.replace('(', "«");
    s = s.replace(')', "»");

    // 2. Uncommon punctuation
    s = s.replace('、', ", ");
    s = s.replace('。', ". ");
    s = s.replace('！', "! ");
    s = s.replace('，', ", ");
    s = s.replace('：', ": ");
    s = s.replace('；', "; ");
    s = s.replace('？', "? ");

    // 3. Whitespace (kokoro.js: odd whitespace → space; preserve \n)
    let mut t = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch == '\n' {
            t.push('\n');
        } else if ch.is_whitespace() {
            t.push(' ');
        } else {
            t.push(ch);
        }
    }
    s = t;
    // Collapse runs of spaces (kokoro.js uses one non-global replace; we collapse all for stable output)
    let re_spaces = StdRegex::new(r" +").unwrap();
    s = re_spaces.replace_all(&s, " ").to_string();
    let re_nl_spaces = FancyRegex::new(r"(?<=\n) +(?=\n)").unwrap();
    s = re_nl_spaces.replace_all(&s, "").to_string();

    // 4. Abbreviations (lookahead requires fancy-regex)
    let dr = FancyRegex::new(r"\bD[Rr]\.(?= [A-Z])").unwrap();
    s = dr.replace_all(&s, "Doctor").to_string();
    let mr = FancyRegex::new(r"\b(?:Mr\.|MR\.(?= [A-Z]))").unwrap();
    s = mr.replace_all(&s, "Mister").to_string();
    let ms = FancyRegex::new(r"\b(?:Ms\.|MS\.(?= [A-Z]))").unwrap();
    s = ms.replace_all(&s, "Miss").to_string();
    let mrs = FancyRegex::new(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))").unwrap();
    s = mrs.replace_all(&s, "Mrs").to_string();
    let etc = FancyRegex::new(r"(?i)\betc\.(?! [A-Z])").unwrap();
    s = etc.replace_all(&s, "etc").to_string();

    // 5. Casual words
    let yeah = StdRegex::new(r"(?i)\b(y)eah?\b").unwrap();
    s = yeah.replace_all(&s, "$1e'a").to_string();

    // 6. Numbers and currencies (order matters)
    let num_pat =
        FancyRegex::new(r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)").unwrap();
    s = replace_with_fn_fancy(&s, &num_pat, split_num_cs);

    let comma_digits = FancyRegex::new(r"(?<=\d),(?=\d)").unwrap();
    s = comma_digits.replace_all(&s, "").to_string();

    let money_pat = StdRegex::new(
        r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b",
    )
    .unwrap();
    s = replace_with_fn(&s, &money_pat, flip_money_cs);

    let dec_pat = StdRegex::new(r"\d*\.\d+").unwrap();
    s = replace_with_fn(&s, &dec_pat, point_num_cs);

    let dash_digits = FancyRegex::new(r"(?<=\d)-(?=\d)").unwrap();
    s = dash_digits.replace_all(&s, " to ").to_string();

    let digit_s = FancyRegex::new(r"(?<=\d)S\b").unwrap();
    s = digit_s.replace_all(&s, " S").to_string();

    // 6. Possessives
    let poss = FancyRegex::new(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b").unwrap();
    s = poss.replace_all(&s, "'S").to_string();
    let xs = FancyRegex::new(r"(?<=X')S\b").unwrap();
    s = xs.replace_all(&s, "s").to_string();

    // 7. Hyphenated letters
    let dotted = StdRegex::new(r"(?:[A-Za-z]\.){2,} [a-z]").unwrap();
    s = dotted
        .replace_all(&s, |caps: &Captures<'_>| caps[0].replace('.', "-"))
        .to_string();
    let caps_dot = FancyRegex::new(r"(?<=[A-Z])\.(?=[A-Z])").unwrap();
    s = caps_dot.replace_all(&s, "-").to_string();

    s.trim().to_string()
}

fn replace_with_fn<F>(text: &str, re: &StdRegex, f: F) -> String
where
    F: Fn(&str) -> String,
{
    let mut out = String::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        out.push_str(&text[last..m.start()]);
        out.push_str(&f(m.as_str()));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

fn replace_with_fn_fancy<F>(text: &str, re: &FancyRegex, f: F) -> String
where
    F: Fn(&str) -> String,
{
    let mut out = String::new();
    let mut last = 0;
    for m in re.find_iter(text).flatten() {
        out.push_str(&text[last..m.start()]);
        out.push_str(&f(m.as_str()));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

/// IPA post-processing chain from kokoro.js `phonemize` step 4–5.
pub fn post_process_phonemes(ps: &str, american: bool) -> String {
    let mut processed = ps.to_string();
    processed = processed.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ");
    processed = processed.replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ");
    processed = processed.replace('ʲ', "j");
    processed = processed.replace('r', "ɹ");
    processed = processed.replace('x', "k");
    processed = processed.replace('ɬ', "l");

    let hundred_gap = FancyRegex::new(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)").expect("hundred regex");
    processed = hundred_gap.replace_all(&processed, " ").to_string();

    let z_fix = FancyRegex::new(" z(?=[;:,.!?¡¿—…\"«»\u{201c}\u{201d} ]|$)").unwrap();
    processed = z_fix.replace_all(&processed, "z").to_string();

    // Weak /ðə/ → strong /ði/ before a vowel-initial syllable ("the actress" → /ði ˈæktɹəs/), closer to eSpeak/kokoro-js than Misaki's default ðə.
    let the_before_vowel =
        FancyRegex::new(r"ðə (?=(?:ˈ|ˌ)?[æɛɪɔʌʊəɜɝɑɐ])").expect("the-before-vowel");
    processed = the_before_vowel.replace_all(&processed, "ði ").to_string();

    if american {
        let ninety = FancyRegex::new(r"(?<=nˈaɪn)ti(?!ː)").expect("american ninety regex");
        processed = ninety.replace_all(&processed, "di").to_string();
    }

    processed.trim().to_string()
}

/// `british`: kokoro.js `language === "b"` (en); default US (`language === "a"`).
pub fn phonemize_like_kokoro_js(
    text: &str,
    british: bool,
    normalize: bool,
    use_v11: bool,
) -> String {
    let mut t = text.to_string();
    if normalize {
        t = normalize_text(&t);
    }

    let re = punctuation_pattern();
    let sections = split_keep_delimiters(&t, &re);

    let mut ps = String::new();
    for sec in sections {
        if sec.is_punctuation {
            ps.push_str(&sec.text);
        } else if !sec.text.is_empty() {
            let phones = phonemize_text_segment(&sec.text, british);
            ps.push_str(&phones);
        }
    }

    let processed = post_process_phonemes(&ps, !british);
    normalize_ipa_for_vocab(&processed, use_v11)
}

/// Full-document cleanup matching legacy `g2p` final passes (spaces before punctuation).
pub fn finalize_g2p_output(compacted: &str) -> String {
    let compact_spaces = StdRegex::new(r"\s+").unwrap();
    let compacted = compact_spaces
        .replace_all(compacted.trim(), " ")
        .to_string();
    let strip_space_before_punct = StdRegex::new(r"\s+([,.;:!?])").unwrap();
    strip_space_before_punct
        .replace_all(&compacted, "$1")
        .trim()
        .to_string()
}
