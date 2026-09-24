#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Lower = 0,
    Upper = 1,
    Digit = 2,
    Space = 3,
    Punct = 4,
    LatinExt = 5,
    Greek = 6,
    Cyrillic = 7,
    Cjk = 8,
    Kana = 9,
    Hangul = 10,
    Other = 11,
}

pub const CLASS_COUNT: usize = 12;

pub const fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0xF900..=0xFAFF)
}

pub const fn is_kana(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x309F | 0x30A0..=0x30FF)
}

pub const fn is_hangul(c: char) -> bool {
    matches!(c as u32, 0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F)
}


pub const fn is_syllabic(c: char) -> bool {
    is_cjk(c) || is_kana(c) || is_hangul(c)
}

pub const fn is_latin_ext(c: char) -> bool {
    matches!(c as u32,
        0x00C0..=0x00FF | 0x0100..=0x017F | 0x0180..=0x024F | 0x1E00..=0x1EFF)
}

pub const fn is_greek(c: char) -> bool {
    matches!(c as u32, 0x0370..=0x03FF | 0x1F00..=0x1FFF)
}

pub const fn is_cyrillic(c: char) -> bool {
    matches!(c as u32, 0x0400..=0x04FF | 0x0500..=0x052F)
}


pub fn class_of(c: char) -> Class {
    if c.is_whitespace() {
        return Class::Space;
    }
    if c.is_ascii_lowercase() || (c.is_alphabetic() && c.is_lowercase()) {
        if (c as u32) < 128 {
            return Class::Lower;
        }
        if is_latin_ext(c) {
            return Class::LatinExt;
        }
        if is_greek(c) {
            return Class::Greek;
        }
        if is_cyrillic(c) {
            return Class::Cyrillic;
        }
        if is_kana(c) {
            return Class::Kana;
        }
        if is_hangul(c) {
            return Class::Hangul;
        }
        if is_cjk(c) {
            return Class::Cjk;
        }
        return Class::Other;
    }
    if c.is_ascii_uppercase() || (c.is_alphabetic() && c.is_uppercase()) {
        return Class::Upper;
    }
    if c.is_numeric() {
        return Class::Digit;
    }
    if is_greek(c) {
        return Class::Greek;
    }
    if is_cyrillic(c) {
        return Class::Cyrillic;
    }
    if is_syllabic(c) {
        if is_kana(c) {
            return Class::Kana;
        }
        if is_hangul(c) {
            return Class::Hangul;
        }
        return Class::Cjk;
    }
    if c.is_alphabetic() {
        return Class::Other;
    }
    Class::Punct
}


pub fn fold_char(c: char, mut emit: impl FnMut(char)) {
    if c.is_ascii() {
        emit(c.to_ascii_lowercase());
        return;
    }
    for f in c.to_lowercase() {
        emit(f);
    }
}


pub const fn is_sentence_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？' | '\n')
}


pub const fn bin_len(n: usize, bins: usize) -> usize {
    if n >= bins { bins - 1 } else { n }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_cover_scripts() {
        assert_eq!(class_of('a'), Class::Lower);
        assert_eq!(class_of('Ą'), Class::Upper);
        assert_eq!(class_of('ą'), Class::LatinExt);
        assert_eq!(class_of('α'), Class::Greek);
        assert_eq!(class_of('ж'), Class::Cyrillic);
        assert_eq!(class_of('中'), Class::Cjk);
        assert_eq!(class_of('あ'), Class::Kana);
        assert_eq!(class_of('한'), Class::Hangul);
        assert_eq!(class_of('5'), Class::Digit);
        assert_eq!(class_of(' '), Class::Space);
    }

    #[test]
    fn fold_ascii_and_unicode() {
        let mut out = [0u32; 4];
        let mut n = 0;
        fold_char('A', |c| {
            out[n] = c as u32;
            n += 1;
        });
        assert_eq!(out[0], 'a' as u32);
    }
}
