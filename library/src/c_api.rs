use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::model::LangDb;
use crate::score;
use crate::{MAGIC_EVLD, VERSION};


pub struct EvldHandle {
    bytes: Vec<u8>,
}


#[repr(C)]
pub struct EvldScore {

    pub lang_index: u32,

    pub lang: [u8; 8],

    pub confidence: f32,
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn evld_version() -> u32 {
    VERSION as u32
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn evld_load(bytes: *const u8, len: usize) -> *mut EvldHandle {
    if bytes.is_null() || len < 256 {
        return core::ptr::null_mut();
    }
    let slice = unsafe { core::slice::from_raw_parts(bytes, len) };
    if slice.len() < 4 || &slice[0..4] != MAGIC_EVLD {
        return core::ptr::null_mut();
    }
    let owned = slice.to_vec();
    if LangDb::from_bytes(&owned).is_err() {
        return core::ptr::null_mut();
    }
    Box::into_raw(Box::new(EvldHandle { bytes: owned }))
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn evld_free(handle: *mut EvldHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn evld_classify(
    handles: *const *mut EvldHandle,
    n: usize,
    text: *const u8,
    text_len: usize,
    out: *mut EvldScore,
    max_out: usize,
) -> i32 {
    if handles.is_null() || text.is_null() || out.is_null() || max_out == 0 {
        return -1;
    }
    let text_slice = unsafe { core::slice::from_raw_parts(text, text_len) };
    let text_str = match core::str::from_utf8(text_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let mut dbs = Vec::with_capacity(n.min(256));
    for i in 0..n.min(256) {
        let h = unsafe { *handles.add(i) };
        if h.is_null() {
            return -1;
        }
        let bytes = unsafe { &(*h).bytes };
        match LangDb::from_bytes(bytes) {
            Ok(db) => dbs.push(db),
            Err(_) => return -1,
        }
    }
    let ranked = score::classify(&dbs, text_str);
    let k = ranked.len().min(max_out);
    for (i, s) in ranked.iter().take(k).enumerate() {
        unsafe {
            *out.add(i) = EvldScore {
                lang_index: s.lang_index as u32,
                lang: s.lang,
                confidence: s.confidence,
            };
        }
    }
    k as i32
}
