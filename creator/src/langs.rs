pub struct Lang {

    pub code: &'static str,

    pub repo: &'static str,

    pub prefixes: &'static [&'static str],
}

pub const LANGS: &[Lang] = &[
    Lang { code: "bg", repo: FW2, prefixes: &["data/bul_Cyrl/train/"] },
    Lang { code: "cs", repo: FW2, prefixes: &["data/ces_Latn/train/"] },
    Lang { code: "da", repo: FW2, prefixes: &["data/dan_Latn/train/"] },
    Lang { code: "de", repo: FW2, prefixes: &["data/deu_Latn/train/"] },
    Lang { code: "el", repo: FW2, prefixes: &["data/ell_Grek/train/"] },
    Lang { code: "en", repo: "HuggingFaceFW/fineweb", prefixes: &["sample/", "data/"] },
    Lang { code: "es", repo: FW2, prefixes: &["data/spa_Latn/train/"] },
    Lang { code: "et", repo: FW2, prefixes: &["data/ekk_Latn/train/"] },
    Lang { code: "fi", repo: FW2, prefixes: &["data/fin_Latn/train/"] },
    Lang { code: "fr", repo: FW2, prefixes: &["data/fra_Latn/train/"] },
    Lang { code: "ga", repo: FW2, prefixes: &["data/gle_Latn/train/"] },
    Lang { code: "hr", repo: FW2, prefixes: &["data/hrv_Latn/train/"] },
    Lang { code: "hu", repo: FW2, prefixes: &["data/hun_Latn/train/"] },
    Lang { code: "it", repo: FW2, prefixes: &["data/ita_Latn/train/"] },
    Lang { code: "ja", repo: FW2, prefixes: &["data/jpn_Jpan/train/"] },
    Lang { code: "ko", repo: FW2, prefixes: &["data/kor_Hang/train/"] },
    Lang { code: "lt", repo: FW2, prefixes: &["data/lit_Latn/train/"] },
    Lang { code: "lv", repo: FW2, prefixes: &["data/lvs_Latn/train/"] },
    Lang { code: "mt", repo: FW2, prefixes: &["data/mlt_Latn/train/"] },
    Lang { code: "nl", repo: FW2, prefixes: &["data/nld_Latn/train/"] },
    Lang { code: "pl", repo: FW2, prefixes: &["data/pol_Latn/train/"] },
    Lang { code: "pt", repo: FW2, prefixes: &["data/por_Latn/train/"] },
    Lang { code: "ro", repo: FW2, prefixes: &["data/ron_Latn/train/"] },
    Lang { code: "ru", repo: FW2, prefixes: &["data/rus_Cyrl/train/"] },
    Lang { code: "sk", repo: FW2, prefixes: &["data/slk_Latn/train/"] },
    Lang { code: "sl", repo: FW2, prefixes: &["data/slv_Latn/train/"] },
    Lang { code: "sv", repo: FW2, prefixes: &["data/swe_Latn/train/"] },
    Lang { code: "uk", repo: FW2, prefixes: &["data/ukr_Cyrl/train/"] },
    Lang { code: "zh", repo: FW2, prefixes: &["data/cmn_Hani/train/"] },
];

const FW2: &str = "HuggingFaceFW/fineweb-2";


pub fn resolve(requested: &[String]) -> anyhow::Result<Vec<&'static Lang>> {
    if requested.is_empty() {
        return Ok(LANGS.iter().collect());
    }
    let mut out = Vec::with_capacity(requested.len());
    for code in requested {
        let lang = LANGS
            .iter()
            .find(|l| l.code == code)
            .ok_or_else(|| anyhow::anyhow!("unknown lang code: {code}"))?;
        out.push(lang);
    }
    Ok(out)
}
