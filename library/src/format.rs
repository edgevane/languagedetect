pub const MAGIC_EVLD: [u8; 4] = *b"EVLD";

pub const MAGIC_EVLC: [u8; 4] = *b"EVLC";

pub const VERSION: u16 = 1;

pub const HEADER_LEN: usize = 256;

pub const SECTBL_OFF: usize = 64;

pub const SECTION_COUNT: usize = 12;

pub const FLAG_LOW_RESOURCE: u16 = 0x0001;


pub const DEFAULT_BM25_K1: f32 = 1.2;
pub const DEFAULT_BM25_B: f32 = 0.75;


#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionId {
    CharUnigram = 0,
    CharBigram = 1,
    CharTrigram = 2,
    TokenUnigram = 3,
    TokenBigram = 4,
    WordLen = 5,
    SentLen = 6,
    Prefix = 7,
    Suffix = 8,
    CharPos = 9,
    ClassTrans = 10,
    TokenIdf = 11,
}

impl SectionId {
    pub const fn index(self) -> usize {
        self as usize
    }
}


pub struct TopK;
impl TopK {
    pub const CHAR_UNIGRAM: usize = 512;
    pub const CHAR_BIGRAM: usize = 6000;
    pub const CHAR_TRIGRAM: usize = 12000;
    pub const TOKEN_UNIGRAM: usize = 6000;
    pub const TOKEN_BIGRAM: usize = 3000;
    pub const PREFIX: usize = 1024;
    pub const SUFFIX: usize = 1024;
    pub const TOKEN_IDF: usize = 6000;
    pub const WORDLEN_BINS: usize = 32;
    pub const SENTLEN_BINS: usize = 64;
    pub const CHARPOS_CHARS: usize = 256;
    pub const CLASS_COUNT: usize = 12;
}


pub fn quantize_logprob(p: f32) -> i16 {
    let q = libm::roundf(libm::logf(p.max(f32::MIN_POSITIVE)) * 4096.0) as i32;
    q.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}


pub fn dequantize_logprob(q: i16) -> f32 {
    (q as f32) / 4096.0
}


pub fn quantize_idf(idf: f32) -> u16 {
    libm::roundf(idf.clamp(0.0, 63.0) * 1024.0) as u16
}


pub fn dequantize_idf(q: u16) -> f32 {
    (q as f32) / 1024.0
}


pub fn quantize_prob(p: f32) -> u16 {
    libm::roundf(p.clamp(0.0, 1.0) * 65535.0) as u16
}


pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_roundtrip() {
        let q = quantize_logprob(0.001);
        let back = dequantize_logprob(q);
        assert!((back - 0.001f32.ln()).abs() < 0.001);
        assert_eq!(dequantize_idf(quantize_idf(4.5)), 4.5);
    }

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
