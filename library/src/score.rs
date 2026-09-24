use alloc::vec::Vec;

use crate::format::SectionId;
use crate::hash::{fnv1a32, mix_u32};
use crate::model::LangDb;
use crate::normalize::{
    bin_len, class_of, fold_char, is_sentence_end, is_syllabic, CLASS_COUNT,
};


#[derive(Clone, Debug)]
pub struct Score {

    pub lang_index: usize,

    pub lang: [u8; 8],

    pub score: f32,

    pub confidence: f32,
}

impl Score {
    pub fn lang_code(&self) -> &str {
        let len = self.lang.iter().position(|&b| b == 0).unwrap_or(8);
        core::str::from_utf8(&self.lang[..len]).unwrap_or("???")
    }
}


struct TokAcc {
    h: u32,
    len: usize,
    aff: [u32; 8],
    aff_n: usize,
    prev: Option<u32>,
    uni: f64,
    n: u32,
    bi: f64,
    bi_n: u32,
    words: u32,
    word_hist: [u32; 32],
    sent_len: u32,
    sent_hist: [u32; 64],
    sents: u32,
    pos: f64,
    pos_n: u32,
}

impl TokAcc {
    fn new() -> Self {
        Self {
            h: 0x811c_9dc5,
            len: 0,
            aff: [0; 8],
            aff_n: 0,
            prev: None,
            uni: 0.0,
            n: 0,
            bi: 0.0,
            bi_n: 0,
            words: 0,
            word_hist: [0; 32],
            sent_len: 0,
            sent_hist: [0; 64],
            sents: 0,
            pos: 0.0,
            pos_n: 0,
        }
    }

    fn push_folded(&mut self, cp: u32) {
        for &byte in &cp.to_le_bytes() {
            self.h ^= byte as u32;
            self.h = self.h.wrapping_mul(0x0100_0193);
        }
        self.len += 1;
        if self.aff_n < 8 {
            self.aff[self.aff_n] = cp;
            self.aff_n += 1;
        }
    }

    fn score_token(&mut self, db: &LangDb<'_>, th: u32, tok_len: usize) {
        let lp = db.ngram_q(SectionId::TokenUnigram, th) as f64;
        let idf = db.token_idf(th) as f64;

        self.uni += lp * (1.0 + 0.25 * idf);
        self.n += 1;
        if let Some(p) = self.prev {
            self.bi += db.ngram_q(SectionId::TokenBigram, mix_u32(p, th)) as f64;
            self.bi_n += 1;
        }
        self.prev = Some(th);
        self.word_hist[bin_len(tok_len, 32)] += 1;
        self.words += 1;
        self.sent_len += 1;
    }

    fn affixes(&mut self, db: &LangDb<'_>) {
        let m = self.aff_n.min(8);
        if m >= 2 {
            let pn = m.min(4);
            let mut pb = [0u8; 16];
            for k in 0..pn {
                pb[k * 4..k * 4 + 4].copy_from_slice(&self.aff[k].to_le_bytes());
            }
            self.uni += 0.2 * db.ngram_q(SectionId::Prefix, fnv1a32(&pb[0..pn * 4])) as f64;
            let sn = m.min(4);
            let mut sb = [0u8; 16];
            for k in 0..sn {
                sb[k * 4..k * 4 + 4].copy_from_slice(&self.aff[m - sn + k].to_le_bytes());
            }
            self.uni += 0.2 * db.ngram_q(SectionId::Suffix, fnv1a32(&sb[0..sn * 4])) as f64;
            let ui = db.uni_index(self.aff[0]);
            self.pos += db.charpos_q(ui, 0) as f64;
            self.pos_n += 1;
            let ul = db.uni_index(self.aff[m - 1]);
            self.pos += db.charpos_q(ul, 2) as f64;
            self.pos_n += 1;
        }
    }

    fn flush(&mut self, db: &LangDb<'_>) {
        if self.len == 0 {
            return;
        }
        let th = self.h;
        let len = self.len;
        self.score_token(db, th, len);
        self.affixes(db);
        self.h = 0x811c_9dc5;
        self.len = 0;
        self.aff_n = 0;
    }


    fn push_syllabic(&mut self, db: &LangDb<'_>, cp: u32) {
        self.flush(db);
        let th = fnv1a32(&cp.to_le_bytes());
        self.score_token(db, th, 1);
    }

    fn end_sentence(&mut self) {
        self.prev = None;
        if self.sent_len > 0 {
            self.sent_hist[bin_len(self.sent_len as usize, 64)] += 1;
            self.sents += 1;
            self.sent_len = 0;
        }
    }
}


pub fn score_one(db: &LangDb<'_>, text: &str) -> f32 {
    const MAX_CHARS: usize = 32_768;
    let mut chars: Vec<u32> = Vec::new();
    chars.reserve(text.len().min(4096));
    for c in text.chars() {
        if chars.len() >= MAX_CHARS {
            break;
        }
        fold_char(c, |f| {
            if chars.len() < MAX_CHARS {
                chars.push(f as u32);
            }
        });
    }
    if chars.is_empty() {
        return f32::NEG_INFINITY;
    }
    let n = chars.len() as f32;


    let mut tri = 0f64;
    let mut tri_n = 0u32;
    let mut bi = 0f64;
    let mut bi_n = 0u32;
    let mut uni = 0f64;
    let mut win = [0u8; 12];
    for (i, &cp) in chars.iter().enumerate() {
        uni += db.char_unigram_q(cp) as f64;
        if i >= 1 {
            win[0..4].copy_from_slice(&chars[i - 1].to_le_bytes());
            win[4..8].copy_from_slice(&cp.to_le_bytes());
            bi += db.ngram_q(SectionId::CharBigram, fnv1a32(&win[0..8])) as f64;
            bi_n += 1;
        }
        if i >= 2 {
            win[0..4].copy_from_slice(&chars[i - 2].to_le_bytes());
            win[4..8].copy_from_slice(&chars[i - 1].to_le_bytes());
            win[8..12].copy_from_slice(&cp.to_le_bytes());
            tri += db.ngram_q(SectionId::CharTrigram, fnv1a32(&win)) as f64;
            tri_n += 1;
        }
    }


    let mut t = TokAcc::new();
    for c in text.chars() {
        if is_sentence_end(c) {
            t.flush(db);
            t.end_sentence();
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
                t.push_syllabic(db, folded[k]);
            } else if fc.is_alphabetic() || fc.is_numeric() {
                t.push_folded(folded[k]);
            } else {
                t.flush(db);
            }
        }
    }
    t.flush(db);
    t.end_sentence();


    let mut cls = 0f64;
    let mut cls_n = 0u32;
    let mut prev_cls: Option<usize> = None;
    for &cp in &chars {
        let cc = class_of(char::from_u32(cp).unwrap_or('\u{FFFD}')) as usize;
        if let Some(p) = prev_cls {
            cls += db.class_trans_q(p.min(CLASS_COUNT - 1), cc.min(CLASS_COUNT - 1)) as f64;
            cls_n += 1;
        }
        prev_cls = Some(cc);
    }


    let mut wl_dot = 0f32;
    if t.words > 0 {
        let m = db.wordlen();
        for (i, &c) in t.word_hist.iter().enumerate() {
            if i < m.len() {
                wl_dot += (c as f32 / t.words as f32) * m[i];
            }
        }
    }
    let mut sl_dot = 0f32;
    if t.sents > 0 {
        let m = db.sentlen();
        for (i, &c) in t.sent_hist.iter().enumerate() {
            if i < m.len() {
                sl_dot += (c as f32 / t.sents as f32) * m[i];
            }
        }
    }
    let dist = (libm::logf(wl_dot.max(1e-6)) + libm::logf(sl_dot.max(1e-6))) as f64;


    fn avg(s: f64, cnt: u32) -> f64 {
        if cnt == 0 {
            0.0
        } else {
            s / (cnt as f64 * 4096.0)
        }
    }
    let short = n < 150.0;
    let (w_tri, w_bi, w_tok) = if short {
        (0.55, 0.30, 0.05)
    } else {
        (0.45, 0.25, 0.15)
    };
    (avg(tri, tri_n) * w_tri
        + avg(bi, bi_n) * w_bi
        + avg(uni, chars.len() as u32) * 0.05
        + avg(t.uni, t.n.max(1)) * w_tok
        + avg(t.bi, t.bi_n) * 0.05
        + avg(t.pos, t.pos_n) * 0.03
        + avg(cls, cls_n) * 0.02
        + dist * 0.05) as f32
}


pub fn classify<'a>(dbs: &[LangDb<'a>], text: &str) -> Vec<Score> {
    let mut out: Vec<Score> = Vec::with_capacity(dbs.len());
    for (i, db) in dbs.iter().enumerate() {
        out.push(Score {
            lang_index: i,
            lang: db.header().lang,
            score: score_one(db, text),
            confidence: 0.0,
        });
    }
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let t = 0.15f32;
    let max = out.first().map(|s| s.score).unwrap_or(0.0);
    if max.is_finite() {
        let mut sum = 0.0f32;
        for s in out.iter_mut() {
            let e = libm::expf(((s.score - max) / t).clamp(-20.0, 0.0));
            s.confidence = e;
            sum += e;
        }
        if sum > 0.0 {
            for s in out.iter_mut() {
                s.confidence /= sum;
            }
        }
    }
    out
}


pub fn classify_top<'a>(dbs: &[LangDb<'a>], text: &str) -> Option<Score> {
    classify(dbs, text).into_iter().next()
}
