#![cfg_attr(not(any(test, feature = "std")), no_std)]

extern crate alloc;

#[cfg(not(any(test, feature = "std")))]
mod platform {
    use dlmalloc::GlobalDlmalloc;

    #[global_allocator]
    static ALLOC: GlobalDlmalloc = GlobalDlmalloc;

    #[panic_handler]
    fn panic_handler(_: &core::panic::PanicInfo) -> ! {
        unsafe { libc_abort() }
    }

    unsafe extern "C" {
        fn abort() -> !;
    }

    unsafe fn libc_abort() -> ! {
        unsafe { abort() }
    }
}

pub mod builder;
pub mod c_api;
pub mod format;
pub mod hash;
pub mod model;
pub mod normalize;
pub mod score;

pub use format::{SectionId, TopK, FLAG_LOW_RESOURCE, MAGIC_EVLC, MAGIC_EVLD, VERSION};
pub use model::{CombinedDb, LangDb, ParseError};
pub use score::{Score, classify, classify_top};

#[cfg(test)]
mod e2e;
