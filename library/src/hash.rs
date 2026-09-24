pub fn fnv1a32(data: &[u8]) -> u32 {
    const OFFSET: u32 = 0x811c_9dc5;
    const PRIME: u32 = 0x0100_0193;
    let mut h = OFFSET;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(PRIME);
    }
    h
}


pub fn fnv1a32_str(s: &str) -> u32 {
    fnv1a32(s.as_bytes())
}


pub fn mix_u32(a: u32, b: u32) -> u32 {
    const PRIME: u32 = 0x0100_0193;
    let mut h = a.wrapping_mul(PRIME) ^ 0x9e37_79b9;
    h = h.wrapping_mul(PRIME) ^ b;
    h ^= h >> 15;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_vectors() {

        assert_eq!(fnv1a32(b""), 0x811c_9dc5);
        assert_eq!(fnv1a32(b"a"), 0xe40c_292c);
        assert_eq!(fnv1a32(b"foobar"), 0xbf9c_f968);
    }

    #[test]
    fn mix_is_order_sensitive() {
        assert_ne!(mix_u32(1, 2), mix_u32(2, 1));
    }
}
