use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::format::{
    self, SectionId, TopK, FLAG_LOW_RESOURCE, HEADER_LEN, MAGIC_EVLC, MAGIC_EVLD, SECTBL_OFF,
    SECTION_COUNT, VERSION,
};
use crate::hash::{fnv1a32, mix_u32};
use crate::normalize::{
    bin_len, class_of, fold_char, is_sentence_end, is_syllabic, CLASS_COUNT,
};


#[derive(Clone, Debug)]
pub struct Engine {
    pub docs: u32,
    pub chars_total: u64,
    pub tokens_total: u64,
    char_uni: BTreeMap<u32, u64>,
    char_bi: BTreeMap<u32, u64>,
    char_tri: BTreeMap<u32, u64>,
    tok_uni: BTreeMap<u32, u64>,

    tok_df: BTreeMap<u32, u64>,
    tok_bi: BTreeMap<u32, u64>,
    prefix: BTreeMap<u32, u64>,
    suffix: BTreeMap<u32, u64>,
    wordlen: [u64; 32],
    sentlen: [u64; 64],

    charpos: BTreeMap<(u32, u8), u64>,
    class_trans: [[u64; CLASS_COUNT]; CLASS_COUNT],

    doc_tokens: Vec<u32>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            docs: 0,
            chars_total: 0,
            tokens_total: 0,
            char_uni: BTreeMap::new(),
            char_bi: BTreeMap::new(),
            char_tri: BTreeMap::new(),
            tok_uni: BTreeMap::new(),
            tok_df: BTreeMap::new(),
            tok_bi: BTreeMap::new(),
            prefix: BTreeMap::new(),
            suffix: BTreeMap::new(),
            wordlen: [0; 32],
            sentlen: [0; 64],
            charpos: BTreeMap::new(),
            class_trans: [[0; CLASS_COUNT]; CLASS_COUNT],
            doc_tokens: Vec::new(),
        }
    }


    pub fn ingest(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.docs += 1;
        self.doc_tokens.clear();


        const MAX_CHARS: usize = 65_536;
        let mut chars: Vec<u32> = Vec::new();
        // Single pass: fold chars and accumulate class transitions inline,
        // avoiding a second loop + char::from_u32 per codepoint.
        let mut prev_cls: Option<usize> = None;
        for c in text.chars() {
            if chars.len() >= MAX_CHARS {
                break;
            }
            let cc = class_of(c) as usize;
            if let Some(p) = prev_cls {
                self.class_trans[p.min(CLASS_COUNT - 1)][cc.min(CLASS_COUNT - 1)] += 1;
            }
            prev_cls = Some(cc);
            fold_char(c, |f| {
                if chars.len() < MAX_CHARS {
                    chars.push(f as u32);
                }
            });
        }
        self.chars_total += chars.len() as u64;

        for &cp in &chars {
            *self.char_uni.entry(cp).or_insert(0) += 1;
        }
        let mut win = [0u8; 12];
        for (i, &cp) in chars.iter().enumerate() {
            if i >= 1 {
                win[0..4].copy_from_slice(&chars[i - 1].to_le_bytes());
                win[4..8].copy_from_slice(&cp.to_le_bytes());
                *self.char_bi.entry(fnv1a32(&win[0..8])).or_insert(0) += 1;
            }
            if i >= 2 {
                win[0..4].copy_from_slice(&chars[i - 2].to_le_bytes());
                win[4..8].copy_from_slice(&chars[i - 1].to_le_bytes());
                win[8..12].copy_from_slice(&cp.to_le_bytes());
                *self.char_tri.entry(fnv1a32(&win)).or_insert(0) += 1;
            }
        }


        let mut h = 0x811c_9dc5u32;
        let mut tlen = 0usize;
        let mut aff: [u32; 8] = [0; 8];
        let mut aff_n = 0usize;
        let mut prev_tok: Option<u32> = None;
        let mut sent_len = 0u32;

        let flush = |me: &mut Engine,
                         h: &mut u32,
                         tlen: &mut usize,
                         aff: &[u32; 8],
                         aff_n: usize,
                         prev_tok: &mut Option<u32>,
                         sent_len: &mut u32| {
            if *tlen == 0 {
                return;
            }
            let th = *h;
            *me.tok_uni.entry(th).or_insert(0) += 1;
            me.doc_tokens.push(th);
            me.tokens_total += 1;
            if let Some(p) = *prev_tok {
                *me.tok_bi.entry(mix_u32(p, th)).or_insert(0) += 1;
            }
            *prev_tok = Some(th);
            me.wordlen[bin_len(*tlen, 32)] += 1;
            *sent_len += 1;
            let m = aff_n.min(8);
            if m >= 2 {
                let pn = m.min(4);
                let mut pb = [0u8; 16];
                for k in 0..pn {
                    pb[k * 4..k * 4 + 4].copy_from_slice(&aff[k].to_le_bytes());
                }
                *me.prefix.entry(fnv1a32(&pb[0..pn * 4])).or_insert(0) += 1;
                let sn = m.min(4);
                let mut sb = [0u8; 16];
                for k in 0..sn {
                    sb[k * 4..k * 4 + 4].copy_from_slice(&aff[m - sn + k].to_le_bytes());
                }
                *me.suffix.entry(fnv1a32(&sb[0..sn * 4])).or_insert(0) += 1;
                *me.charpos.entry((aff[0], 0)).or_insert(0) += 1;
                if m > 1 {
                    *me.charpos.entry((aff[m - 1], 2)).or_insert(0) += 1;
                }

                if m > 2 {
                    for k in 1..m - 1 {
                        *me.charpos.entry((aff[k], 1)).or_insert(0) += 1;
                    }
                }
            }
            *h = 0x811c_9dc5;
            *tlen = 0;
        };

        for c in text.chars() {
            if is_sentence_end(c) {
                flush(
                    self, &mut h, &mut tlen, &aff, aff_n, &mut prev_tok, &mut sent_len,
                );
                aff_n = 0;
                prev_tok = None;
                if sent_len > 0 {
                    self.sentlen[bin_len(sent_len as usize, 64)] += 1;
                    sent_len = 0;
                }
                continue;
            }
            let mut folded = [0u32; 3];
            let mut nf = 0usize;
            fold_char(c, |f| {
                if nf < 3 {
                    folded[nf] = f as u32;
                    nf += 1;
                }
            });
            for k in 0..nf {
                let fc = char::from_u32(folded[k]).unwrap_or('\u{FFFD}');
                if is_syllabic(fc) {
                    flush(
                        self, &mut h, &mut tlen, &aff, aff_n, &mut prev_tok, &mut sent_len,
                    );
                    aff_n = 0;
                    let th = fnv1a32(&folded[k].to_le_bytes());
                    *self.tok_uni.entry(th).or_insert(0) += 1;
                    self.doc_tokens.push(th);
                    self.tokens_total += 1;
                    if let Some(p) = prev_tok {
                        *self.tok_bi.entry(mix_u32(p, th)).or_insert(0) += 1;
                    }
                    prev_tok = Some(th);
                    self.wordlen[bin_len(1, 32)] += 1;
                    sent_len += 1;
                } else if fc.is_alphabetic() || fc.is_numeric() {
                    for &byte in &folded[k].to_le_bytes() {
                        h ^= byte as u32;
                        h = h.wrapping_mul(0x0100_0193);
                    }
                    tlen += 1;
                    if aff_n < 8 {
                        aff[aff_n] = folded[k];
                        aff_n += 1;
                    }
                } else {
                    flush(
                        self, &mut h, &mut tlen, &aff, aff_n, &mut prev_tok, &mut sent_len,
                    );
                    aff_n = 0;
                }
            }
        }
        flush(
            self, &mut h, &mut tlen, &aff, aff_n, &mut prev_tok, &mut sent_len,
        );
        if sent_len > 0 {
            self.sentlen[bin_len(sent_len as usize, 64)] += 1;
        }


        self.doc_tokens.sort_unstable();
        self.doc_tokens.dedup();
        let dt = core::mem::take(&mut self.doc_tokens);
        for th in dt {
            *self.tok_df.entry(th).or_insert(0) += 1;
        }
    }


    pub fn merge(&mut self, other: Engine) {
        self.docs += other.docs;
        self.chars_total += other.chars_total;
        self.tokens_total += other.tokens_total;
        merge_map(&mut self.char_uni, other.char_uni);
        merge_map(&mut self.char_bi, other.char_bi);
        merge_map(&mut self.char_tri, other.char_tri);
        merge_map(&mut self.tok_uni, other.tok_uni);
        merge_map(&mut self.tok_df, other.tok_df);
        merge_map(&mut self.tok_bi, other.tok_bi);
        merge_map(&mut self.prefix, other.prefix);
        merge_map(&mut self.suffix, other.suffix);
        merge_map(&mut self.charpos, other.charpos);
        for i in 0..32 {
            self.wordlen[i] += other.wordlen[i];
        }
        for i in 0..64 {
            self.sentlen[i] += other.sentlen[i];
        }
        for a in 0..CLASS_COUNT {
            for b in 0..CLASS_COUNT {
                self.class_trans[a][b] += other.class_trans[a][b];
            }
        }
    }

    pub fn token_df(&self) -> &BTreeMap<u32, u64> {
        &self.tok_df
    }
}

fn merge_map<K: Ord>(dst: &mut BTreeMap<K, u64>, src: BTreeMap<K, u64>) {
    for (k, v) in src {
        *dst.entry(k).or_insert(0) += v;
    }
}


fn top_k(map: &BTreeMap<u32, u64>, k: usize) -> Vec<(u32, u64)> {
    let mut v: Vec<(u32, u64)> = map.iter().map(|(&a, &b)| (a, b)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(k);
    v
}


pub fn finish(
    eng: &Engine,
    lang_code: &str,
    low_resource: bool,
    idf: &BTreeMap<u32, f32>,
) -> Vec<u8> {
    let total_uni: u64 = eng.char_uni.values().sum::<u64>().max(1);
    let total_bi: u64 = eng.char_bi.values().sum::<u64>().max(1);
    let total_tri: u64 = eng.char_tri.values().sum::<u64>().max(1);
    let total_tok: u64 = eng.tok_uni.values().sum::<u64>().max(1);
    let total_tokbi: u64 = eng.tok_bi.values().sum::<u64>().max(1);
    let total_pre: u64 = eng.prefix.values().sum::<u64>().max(1);
    let total_suf: u64 = eng.suffix.values().sum::<u64>().max(1);


    let oov_q = |total: u64| format::quantize_logprob(0.5 / (total as f32 + 0.5));


    let mut secs: [Vec<u8>; SECTION_COUNT] = Default::default();


    fn hq(top: Vec<(u32, u64)>, total: u64) -> Vec<u8> {
        let mut recs: Vec<(u32, i16)> = top
            .into_iter()
            .map(|(k, c)| {
                (
                    k,
                    format::quantize_logprob((c as f32 + 0.5) / (total as f32 + 0.5)),
                )
            })
            .collect();
        recs.sort_by_key(|r| r.0);
        let mut out = Vec::with_capacity(recs.len() * 6);
        for (k, q) in recs {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&q.to_le_bytes());
        }
        out
    }

    secs[SectionId::CharUnigram.index()] = {
        let top = top_k(&eng.char_uni, TopK::CHAR_UNIGRAM);
        let mut recs: Vec<(u32, i16)> = top
            .into_iter()
            .map(|(k, c)| {
                (
                    k,
                    format::quantize_logprob((c as f32 + 0.5) / (total_uni as f32 + 0.5)),
                )
            })
            .collect();
        recs.sort_by_key(|r| r.0);
        let mut out = Vec::with_capacity(recs.len() * 6);
        for (k, q) in recs {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&q.to_le_bytes());
        }
        out
    };
    let oov_tri = oov_q(total_tri);
    secs[SectionId::CharBigram.index()] =
        hq(top_k(&eng.char_bi, TopK::CHAR_BIGRAM), total_bi);
    secs[SectionId::CharTrigram.index()] =
        hq(top_k(&eng.char_tri, TopK::CHAR_TRIGRAM), total_tri);
    secs[SectionId::TokenUnigram.index()] =
        hq(top_k(&eng.tok_uni, TopK::TOKEN_UNIGRAM), total_tok);
    secs[SectionId::TokenBigram.index()] =
        hq(top_k(&eng.tok_bi, TopK::TOKEN_BIGRAM), total_tokbi);
    secs[SectionId::WordLen.index()] = {
        let tot: u64 = eng.wordlen.iter().sum::<u64>().max(1);
        let mut out = Vec::with_capacity(64);
        for &c in &eng.wordlen {
            out.extend_from_slice(&format::quantize_prob(c as f32 / tot as f32).to_le_bytes());
        }
        out
    };
    secs[SectionId::SentLen.index()] = {
        let tot: u64 = eng.sentlen.iter().sum::<u64>().max(1);
        let mut out = Vec::with_capacity(128);
        for &c in &eng.sentlen {
            out.extend_from_slice(&format::quantize_prob(c as f32 / tot as f32).to_le_bytes());
        }
        out
    };
    secs[SectionId::Prefix.index()] = hq(top_k(&eng.prefix, TopK::PREFIX), total_pre);
    secs[SectionId::Suffix.index()] = hq(top_k(&eng.suffix, TopK::SUFFIX), total_suf);
    secs[SectionId::TokenIdf.index()] = {
        let top = top_k(&eng.tok_uni, TopK::TOKEN_IDF);
        let mut recs: Vec<(u32, u16)> = top
            .into_iter()
            .map(|(k, _)| (k, format::quantize_idf(idf.get(&k).copied().unwrap_or(0.0))))
            .collect();
        recs.sort_by_key(|r| r.0);
        let mut out = Vec::with_capacity(recs.len() * 6);
        for (k, q) in recs {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&q.to_le_bytes());
        }
        out
    };


    let uni_keys: Vec<u32> = {
        let mut v: Vec<(u32, u64)> = eng
            .char_uni
            .iter()
            .map(|(&a, &b)| (a, b))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(TopK::CHARPOS_CHARS);
        let mut ks: Vec<u32> = v.into_iter().map(|(k, _)| k).collect();
        ks.sort_unstable();
        ks
    };
    secs[SectionId::CharPos.index()] = {
        let mut out = Vec::with_capacity(uni_keys.len() * 3 * 2);
        for &cp in &uni_keys {
            let ci: u64 = eng.char_uni.get(&cp).copied().unwrap_or(1).max(1);
            for slot in 0u8..3 {
                let c = eng.charpos.get(&(cp, slot)).copied().unwrap_or(0);
                let p = (c as f32 + 0.5) / (ci as f32 + 1.5);
                out.extend_from_slice(&format::quantize_logprob(p).to_le_bytes());
            }
        }
        out
    };
    secs[SectionId::ClassTrans.index()] = {
        let mut out = Vec::with_capacity(CLASS_COUNT * CLASS_COUNT * 2);
        for a in 0..CLASS_COUNT {
            let row: u64 = eng.class_trans[a].iter().sum::<u64>().max(1);
            for b in 0..CLASS_COUNT {
                let p = (eng.class_trans[a][b] as f32 + 0.5) / (row as f32 + 0.5);
                out.extend_from_slice(&format::quantize_prob(p).to_le_bytes());
            }
        }
        out
    };


    let mut blob = alloc::vec![0u8; HEADER_LEN];
    blob[0..4].copy_from_slice(&MAGIC_EVLD);
    blob[4..6].copy_from_slice(&VERSION.to_le_bytes());
    let mut lang = [0u8; 8];
    let lb = lang_code.as_bytes();
    let nl = lb.len().min(8);
    lang[..nl].copy_from_slice(&lb[..nl]);
    blob[6..14].copy_from_slice(&lang);
    let flags: u16 = if low_resource { FLAG_LOW_RESOURCE } else { 0 };
    blob[14..16].copy_from_slice(&flags.to_le_bytes());
    blob[16..20].copy_from_slice(&eng.docs.to_le_bytes());
    blob[20..28].copy_from_slice(&eng.chars_total.to_le_bytes());
    blob[28..36].copy_from_slice(&eng.tokens_total.to_le_bytes());
    let avgdl = if eng.docs > 0 {
        eng.tokens_total as f32 / eng.docs as f32
    } else {
        0.0
    };
    blob[36..40].copy_from_slice(&avgdl.to_le_bytes());
    blob[40..44].copy_from_slice(&format::DEFAULT_BM25_K1.to_le_bytes());
    blob[44..48].copy_from_slice(&format::DEFAULT_BM25_B.to_le_bytes());
    blob[48..50].copy_from_slice(&oov_tri.to_le_bytes());

    let mut off = HEADER_LEN as u32;
    for (i, sec) in secs.iter().enumerate() {
        let o = SECTBL_OFF + i * 12;

        let count = match i {
            x if x == SectionId::WordLen.index() => (sec.len() / 2) as u32,
            x if x == SectionId::SentLen.index() => (sec.len() / 2) as u32,
            x if x == SectionId::CharPos.index() => (sec.len() / 6) as u32,
            x if x == SectionId::ClassTrans.index() => (sec.len() / 2) as u32,
            _ => (sec.len() / 6) as u32,
        };
        blob[o..o + 4].copy_from_slice(&off.to_le_bytes());
        blob[o + 4..o + 8].copy_from_slice(&(sec.len() as u32).to_le_bytes());
        blob[o + 8..o + 12].copy_from_slice(&count.to_le_bytes());
        off += sec.len() as u32 + 4;
    }
    let hcrc = format::crc32(&blob[0..252]);
    blob[252..256].copy_from_slice(&hcrc.to_le_bytes());

    let mut out = blob;
    for sec in &secs {
        out.extend_from_slice(sec);
        out.extend_from_slice(&format::crc32(sec).to_le_bytes());
    }
    out
}


pub fn encode_combined(items: &[(alloc::string::String, Vec<u8>)]) -> Vec<u8> {
    let count = items.len().min(u16::MAX as usize) as u16;
    let mut head = alloc::vec![0u8; HEADER_LEN];
    head[0..4].copy_from_slice(&MAGIC_EVLC);
    head[4..6].copy_from_slice(&VERSION.to_le_bytes());
    head[6..8].copy_from_slice(&count.to_le_bytes());
    let hcrc = format::crc32(&head[0..252]);
    head[252..256].copy_from_slice(&hcrc.to_le_bytes());

    let mut index = Vec::with_capacity(items.len() * 16);
    let mut off = (HEADER_LEN + items.len() * 16) as u32;
    for (code, blob) in items {
        let mut lang = [0u8; 8];
        let lb = code.as_bytes();
        let nl = lb.len().min(8);
        lang[..nl].copy_from_slice(&lb[..nl]);
        index.extend_from_slice(&lang);
        index.extend_from_slice(&off.to_le_bytes());
        index.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        off += blob.len() as u32;
    }
    let mut out = head;
    out.extend_from_slice(&index);
    for (_, blob) in items {
        out.extend_from_slice(blob);
    }
    out
}
