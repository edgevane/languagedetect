# languagedetect

Small, dependency-light language detection engine in Rust. `no_std`-compatible
runtime (`library/`), plus a training tool (`creator/`) that builds compact
models.

- Crate: `edgevane_langdetect` (lib name `edgevanelang`)
- Formats: `EVLD` (single language DB), `EVLC` (combined multi-language DB)
- Signals: char uni/bi/trigrams, token uni/bigrams + TF-IDF, prefixes/suffixes,
  char position, class transitions, word/sentence length distributions
- Unicode folding in `normalize.rs`, FNV-1a hashing in `hash.rs`
- C API (`evld_load` / `evld_classify` / `evld_free`) for embedding

## Layout

- `library/` — runtime + trainer engine (`Engine::ingest`, `finish`,
  `encode_combined`, `LangDb`, `CombinedDb`, `classify`, `classify_top`, C API)
- `creator/` — CLI that trains `.evld` per language and packs `.evlc`

## Use (Rust)

```rust
use edgevanelang::{CombinedDb, classify};

// combined.chunks / combined.dbs depending on model.rs API
let scores = classify(&dbs, "Wczoraj padał deszcz, więc zostałem w domu.");
let top = &scores[0];
println!("{} {:.2}", top.lang_code(), top.confidence);
```

Train a single language:

```rust
use edgevanelang::builder::{Engine, finish, encode_combined};
use std::collections::BTreeMap;

let mut eng = Engine::new();
eng.ingest("Wczoraj padał deszcz, więc zostałem w domu.");
eng.ingest("Polskie znaki: ą ć ę ł ń ó ś ź ż.");
let idf = BTreeMap::new(); // real IDF comes from multi-language DF pass
let evld = finish(&eng, "pl", false, &idf);
```

## Use (C)

```c
void *h = evld_load(bytes, len);
EvldScore out[8];
int n = evld_classify(&h, 1, (uint8_t*)text, text_len, out, 8);
evld_free(h);
```

## Build / test

```sh
cargo test -p edgevane_langdetect
cargo build -p edgevane_langdetect --release
# no_std check
cargo build -p edgevane_langdetect --release --no-default-features
```

## Model format

Header (256 B): magic `EVLD`/`EVLC`, version, lang code, flags
(`FLAG_LOW_RESOURCE`), doc/char/token counts, BM25 params, section table with
per-section CRC32. Sections: char n-grams, token n-grams + IDF, affixes,
`CharPos`, `ClassTrans`, length histograms. See `library/src/format.rs`.

## License

MIT — see `library/LICENSE-MIT`.
