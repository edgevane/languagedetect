use alloc::vec::Vec;
use core::{fmt, str};

use crate::format::{
    self, SectionId, HEADER_LEN, MAGIC_EVLC, MAGIC_EVLD, SECTBL_OFF, SECTION_COUNT, VERSION,
};
use crate::normalize::CLASS_COUNT as NCLASS;


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    TooShort,
    BadMagic,
    BadVersion,
    BadSectionTable,
    BadCrc,
    BadLangCode,
    TruncatedSection,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::TooShort => "buffer shorter than header",
            Self::BadMagic => "bad magic, not an .evld/.evlc file",
            Self::BadVersion => "unsupported version",
            Self::BadSectionTable => "section offsets out of bounds",
            Self::BadCrc => "CRC32 mismatch (corrupt file)",
            Self::BadLangCode => "lang code is not valid UTF-8",
            Self::TruncatedSection => "section payload truncated",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

fn le_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn le_i16(b: &[u8]) -> i16 {
    i16::from_le_bytes([b[0], b[1]])
}
fn le_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn le_u64(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}


#[derive(Clone, Debug)]
pub struct Header {
    pub lang: [u8; 8],
    pub flags: u16,
    pub docs: u32,
    pub chars_total: u64,
    pub tokens_total: u64,
    pub avg_doc_len: f32,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub oov_logprob_q: i16,

    pub sections: [(u32, u32, u32); SECTION_COUNT],
}

impl Header {
    pub fn lang_code(&self) -> &str {
        let len = self.lang.iter().position(|&b| b == 0).unwrap_or(8);
        str::from_utf8(&self.lang[..len]).unwrap_or("???")
    }
}

fn parse_header(bytes: &[u8], magic: &[u8; 4]) -> Result<Header, ParseError> {
    if bytes.len() < HEADER_LEN {
        return Err(ParseError::TooShort);
    }
    if &bytes[0..4] != magic {
        return Err(ParseError::BadMagic);
    }
    if le_u16(&bytes[4..6]) != VERSION {
        return Err(ParseError::BadVersion);
    }
    if format::crc32(&bytes[0..252]) != le_u32(&bytes[252..256]) {
        return Err(ParseError::BadCrc);
    }
    let mut lang = [0u8; 8];
    lang.copy_from_slice(&bytes[6..14]);
    if str::from_utf8(&lang).is_err() {
        return Err(ParseError::BadLangCode);
    }
    let mut sections = [(0u32, 0u32, 0u32); SECTION_COUNT];
    for i in 0..SECTION_COUNT {
        let o = SECTBL_OFF + i * 12;
        sections[i] = (
            le_u32(&bytes[o..o + 4]),
            le_u32(&bytes[o + 4..o + 8]),
            le_u32(&bytes[o + 8..o + 12]),
        );
        let (off, len, _) = sections[i];
        let end = (off as usize).saturating_add(len as usize);
        if off as usize != 0 && (off as usize) < HEADER_LEN || end > bytes.len() {
            if (off as usize) < HEADER_LEN && len != 0 {
                return Err(ParseError::BadSectionTable);
            }
            if end > bytes.len() {
                return Err(ParseError::BadSectionTable);
            }
        }
    }
    Ok(Header {
        lang,
        flags: le_u16(&bytes[14..16]),
        docs: le_u32(&bytes[16..20]),
        chars_total: le_u64(&bytes[20..28]),
        tokens_total: le_u64(&bytes[28..36]),
        avg_doc_len: le_f32(&bytes[36..40]),
        bm25_k1: le_f32(&bytes[40..44]),
        bm25_b: le_f32(&bytes[44..48]),
        oov_logprob_q: le_i16(&bytes[48..50]),
        sections,
    })
}


#[derive(Clone, Debug)]
pub struct LangDb<'a> {
    bytes: &'a [u8],
    hdr: Header,
}

impl<'a> LangDb<'a> {

    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ParseError> {
        let hdr = parse_header(bytes, &MAGIC_EVLD)?;
        let db = Self { bytes, hdr };

        for i in 0..SECTION_COUNT {
            db.section_payload(i)?;
        }
        Ok(db)
    }

    pub fn header(&self) -> &Header {
        &self.hdr
    }

    pub fn lang_code(&self) -> &str {
        self.hdr.lang_code()
    }

    fn section_payload(&self, idx: usize) -> Result<&'a [u8], ParseError> {
        let (off, len, _) = self.hdr.sections[idx];
        if len == 0 {
            return Ok(&[]);
        }
        let off = off as usize;
        let len = len as usize;
        let body = self
            .bytes
            .get(off..off + len)
            .ok_or(ParseError::TruncatedSection)?;
        let crc = self
            .bytes
            .get(off + len..off + len + 4)
            .ok_or(ParseError::TruncatedSection)?;
        if format::crc32(body) != le_u32(crc) {
            return Err(ParseError::BadCrc);
        }
        Ok(body)
    }

    fn section_raw(&self, id: SectionId) -> &'a [u8] {

        let (off, len, _) = self.hdr.sections[id.index()];
        if len == 0 {
            return &[];
        }
        &self.bytes[off as usize..off as usize + len as usize]
    }


    fn lookup_rec(section: &[u8], key: u32) -> Option<i32> {
        let n = section.len() / 6;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let o = mid * 6;
            let k = u32::from_le_bytes([
                section[o],
                section[o + 1],
                section[o + 2],
                section[o + 3],
            ]);
            if k < key {
                lo = mid + 1;
            } else if k > key {
                hi = mid;
            } else {
                return Some(i16::from_le_bytes([section[o + 4], section[o + 5]]) as i32);
            }
        }
        None
    }


    pub fn ngram_q(&self, id: SectionId, hash: u32) -> i32 {
        debug_assert!(matches!(
            id,
            SectionId::CharBigram
                | SectionId::CharTrigram
                | SectionId::TokenUnigram
                | SectionId::TokenBigram
                | SectionId::Prefix
                | SectionId::Suffix
        ));
        Self::lookup_rec(self.section_raw(id), hash)
            .unwrap_or(self.hdr.oov_logprob_q as i32)
    }


    pub fn char_unigram_q(&self, codepoint: u32) -> i32 {
        Self::lookup_rec(self.section_raw(SectionId::CharUnigram), codepoint)
            .unwrap_or(self.hdr.oov_logprob_q as i32)
    }


    pub fn token_idf(&self, hash: u32) -> f32 {
        let sec = self.section_raw(SectionId::TokenIdf);
        let n = sec.len() / 6;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let o = mid * 6;
            let k = u32::from_le_bytes([sec[o], sec[o + 1], sec[o + 2], sec[o + 3]]);
            if k < hash {
                lo = mid + 1;
            } else if k > hash {
                hi = mid;
            } else {
                let q = u16::from_le_bytes([sec[o + 4], sec[o + 5]]);
                return format::dequantize_idf(q);
            }
        }
        0.0
    }


    pub fn wordlen(&self) -> Vec<f32> {
        let raw = self.section_raw(SectionId::WordLen);
        raw.chunks_exact(2)
            .map(|c| (u16::from_le_bytes([c[0], c[1]]) as f32) / 65535.0)
            .collect()
    }


    pub fn sentlen(&self) -> Vec<f32> {
        let raw = self.section_raw(SectionId::SentLen);
        raw.chunks_exact(2)
            .map(|c| (u16::from_le_bytes([c[0], c[1]]) as f32) / 65535.0)
            .collect()
    }


    pub fn charpos_q(&self, uni_index: Option<usize>, slot: usize) -> i32 {
        let raw = self.section_raw(SectionId::CharPos);
        match uni_index {
            Some(i) => {
                let o = (i * 3 + slot.min(2)) * 2;
                if let Some(c) = raw.get(o..o + 2) {
                    return i16::from_le_bytes([c[0], c[1]]) as i32;
                }
                self.hdr.oov_logprob_q as i32
            }
            None => self.hdr.oov_logprob_q as i32,
        }
    }


    pub fn uni_index(&self, codepoint: u32) -> Option<usize> {
        let sec = self.section_raw(SectionId::CharUnigram);
        let n = sec.len() / 6;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let o = mid * 6;
            let k = u32::from_le_bytes([sec[o], sec[o + 1], sec[o + 2], sec[o + 3]]);
            if k < codepoint {
                lo = mid + 1;
            } else if k > codepoint {
                hi = mid;
            } else {
                return Some(mid);
            }
        }
        None
    }


    pub fn class_trans_q(&self, a: usize, b: usize) -> i32 {
        debug_assert!(a < NCLASS && b < NCLASS);
        let raw = self.section_raw(SectionId::ClassTrans);
        let o = (a * NCLASS + b) * 2;
        raw.get(o..o + 2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as i32)
            .unwrap_or(0)
    }
}


#[derive(Clone, Debug)]
pub struct CombinedDb<'a> {
    bytes: &'a [u8],
    count: usize,
}

impl<'a> CombinedDb<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ParseError> {
        if bytes.len() < HEADER_LEN {
            return Err(ParseError::TooShort);
        }
        if &bytes[0..4] != MAGIC_EVLC {
            return Err(ParseError::BadMagic);
        }
        if le_u16(&bytes[4..6]) != VERSION {
            return Err(ParseError::BadVersion);
        }
        if format::crc32(&bytes[0..252]) != le_u32(&bytes[252..256]) {
            return Err(ParseError::BadCrc);
        }
        let count = le_u16(&bytes[6..8]) as usize;
        let need = HEADER_LEN + count * 16;
        if bytes.len() < need {
            return Err(ParseError::TooShort);
        }

        let db = Self { bytes, count };
        for i in 0..count {
            db.get(i)?;
        }
        Ok(db)
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, i: usize) -> Result<LangDb<'a>, ParseError> {
        if i >= self.count {
            return Err(ParseError::TruncatedSection);
        }
        let o = HEADER_LEN + i * 16;
        let off = le_u32(&self.bytes[o + 8..o + 12]) as usize;
        let len = le_u32(&self.bytes[o + 12..o + 16]) as usize;
        let blob = self.bytes.get(off..off + len).ok_or(ParseError::BadSectionTable)?;
        LangDb::from_bytes(blob)
    }


    pub fn find(&self, code: &str) -> Option<LangDb<'a>> {
        (0..self.count).find_map(|i| {
            let db = self.get(i).ok()?;
            (db.lang_code() == code).then_some(db)
        })
    }
}
