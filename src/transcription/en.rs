use regex::Regex;
use std::{collections::HashMap, sync::LazyLock};

static LETTERS_IPA_MAP: LazyLock<HashMap<char, &'static str>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert('a', "ɐ");
    map.insert('b', "bˈi");
    map.insert('c', "sˈi");
    map.insert('d', "dˈi");
    map.insert('e', "ˈi");
    map.insert('f', "ˈɛf");
    map.insert('g', "ʤˈi");
    map.insert('h', "ˈAʧ");
    map.insert('i', "ˈI");
    map.insert('j', "ʤˈA");
    map.insert('k', "kˈA");
    map.insert('l', "ˈɛl");
    map.insert('m', "ˈɛm");
    map.insert('n', "ˈɛn");
    map.insert('o', "ˈO");
    map.insert('p', "pˈi");
    map.insert('q', "kjˈu");
    map.insert('r', "ˈɑɹ");
    map.insert('s', "ˈɛs");
    map.insert('t', "tˈi");
    map.insert('u', "jˈu");
    map.insert('v', "vˈi");
    map.insert('w', "dˈʌbᵊlju");
    map.insert('x', "ˈɛks");
    map.insert('y', "wˈI");
    map.insert('z', "zˈi");
    map.insert('A', "ˈA");
    map.insert('B', "bˈi");
    map.insert('C', "sˈi");
    map.insert('D', "dˈi");
    map.insert('E', "ˈi");
    map.insert('F', "ˈɛf");
    map.insert('G', "ʤˈi");
    map.insert('H', "ˈAʧ");
    map.insert('I', "ˈI");
    map.insert('J', "ʤˈA");
    map.insert('K', "kˈA");
    map.insert('L', "ˈɛl");
    map.insert('M', "ˈɛm");
    map.insert('N', "ˈɛn");
    map.insert('O', "ˈO");
    map.insert('P', "pˈi");
    map.insert('Q', "kjˈu");
    map.insert('R', "ˈɑɹ");
    map.insert('S', "ˈɛs");
    map.insert('T', "tˈi");
    map.insert('U', "jˈu");
    map.insert('V', "vˈi");
    map.insert('W', "dˈʌbᵊlju");
    map.insert('X', "ˈɛks");
    map.insert('Y', "wˈI");
    map.insert('Z', "zˈi");
    map
});
static ARPA_IPA_MAP: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert("AA", "ɑ");
    map.insert("AE", "æ");
    map.insert("AH", "ə");
    map.insert("AO", "ɔ");
    map.insert("AW", "aʊ");
    map.insert("AY", "aɪ");
    map.insert("B", "b");
    map.insert("CH", "tʃ");
    map.insert("D", "d");
    map.insert("DH", "ð");
    map.insert("EH", "ɛ");
    map.insert("ER", "ɝ");
    map.insert("EY", "eɪ");
    map.insert("F", "f");
    map.insert("G", "ɡ");
    map.insert("HH", "h");
    map.insert("IH", "ɪ");
    map.insert("IY", "i");
    map.insert("JH", "dʒ");
    map.insert("K", "k");
    map.insert("L", "l");
    map.insert("M", "m");
    map.insert("N", "n");
    map.insert("NG", "ŋ");
    map.insert("OW", "oʊ");
    map.insert("OY", "ɔɪ");
    map.insert("P", "p");
    map.insert("R", "ɹ");
    map.insert("S", "s");
    map.insert("SH", "ʃ");
    map.insert("T", "t");
    map.insert("TH", "θ");
    map.insert("UH", "ʊ");
    map.insert("UW", "u");
    map.insert("V", "v");
    map.insert("W", "w");
    map.insert("Y", "j");
    map.insert("Z", "z");
    map.insert("ZH", "ʒ");
    map.insert("SIL", "");
    map
});

/// 支持2025新增符号（如：吸气音ʘ）
const SPECIAL_CASES: [(&str, &str); 3] = [("CLICK!", "ʘ"), ("TSK!", "ǀ"), ("TUT!", "ǁ")];

pub fn arpa_to_ipa(arpa: &str) -> Result<String, regex::Error> {
    let re = Regex::new(r"([A-Z!]+)(\d*)")?;

    let Some(caps) = re.captures(arpa) else {
        return Ok(Default::default());
    };

    let stress = caps.get(2).map_or("", |m| m.as_str());
    let arpa_phone = caps.get(1).map_or("", |m| m.as_str());

    // 处理特殊符号（2025新增）
    if let Some(sc) = SPECIAL_CASES.iter().find(|&&(s, _)| s == arpa_phone) {
        return Ok(sc.1.to_string());
    }

    // CMUdict `AH`: AH0 is reduced schwa; AH1/AH2/AH3 are stressed STRUT /ʌ/ (e.g. "but", "cut").
    // Mapping AH→ə unconditionally made stressed words sound like /bət/ instead of /bʌt/.
    //
    // CMUdict `AA`: stressed AA (LOT/PALM) → ɑ; unstressed AA0 → ə (e.g. "-mat-" in "transformative").
    // Mapping AA→ɑ for AA0 made unstressed syllables too dark compared to typical GenAm TTS.
    let phoneme: String = if arpa_phone == "AH" {
        match stress {
            "1" | "2" | "3" => "ʌ".to_owned(),
            _ => "ə".to_owned(),
        }
    } else if arpa_phone == "AA" {
        match stress {
            "1" | "2" | "3" => "ɑ".to_owned(),
            _ => "ə".to_owned(),
        }
    } else {
        ARPA_IPA_MAP
            .get(arpa_phone)
            .map_or_else(|| letters_to_ipa(arpa), |i| i.to_string())
    };

    let mut result = String::with_capacity(arpa.len() * 2);
    // 添加重音标记（支持三级重音）
    match stress {
        "1" => result.push('ˈ'),
        "2" => result.push('ˌ'),
        "3" => result.push('˧'), // 2025新增中级重音
        _ => {}
    }

    result.push_str(&phoneme);

    Ok(result)
}

pub fn letters_to_ipa(letters: &str) -> String {
    let mut res = String::with_capacity(letters.len());
    for i in letters.chars() {
        if let Some(p) = LETTERS_IPA_MAP.get(&i) {
            res.push_str(p);
        }
    }
    res
}

#[cfg(test)]
mod tests {
    #[test]
    fn ah_stressed_is_strut_not_schwa() {
        assert_eq!(super::arpa_to_ipa("AH1").unwrap(), "ˈʌ");
        assert_eq!(super::arpa_to_ipa("AH0").unwrap(), "ə");
    }

    #[test]
    fn aa_stressed_is_palm_unstressed_zero_is_schwa() {
        assert_eq!(super::arpa_to_ipa("AA1").unwrap(), "ˈɑ");
        assert_eq!(super::arpa_to_ipa("AA0").unwrap(), "ə");
    }

    #[test]
    fn but_word_cmudict_path_uses_strut() -> Result<(), crate::G2PError> {
        let p = crate::g2p("But.", false)?;
        assert!(
            p.contains("bˈʌt") || p.starts_with("bˈʌt"),
            "expected STRUT in 'But', got {p:?}"
        );
        Ok(())
    }
}
