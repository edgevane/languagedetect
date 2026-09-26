use edgevanelang::model::LangDb;
use edgevanelang::score::classify;
use rayon::prelude::*;


pub fn report(
    dbs: &[LangDb],
    engines: &[(&crate::langs::Lang, Vec<Box<str>>)],
    _want: usize,
) {
    println!("eval: top-1 accuracy (held-out strided sample)");
    let mut tot_ok = 0usize;
    let mut tot_n = 0usize;
    for (i, (lang, docs)) in engines.iter().enumerate() {
        if docs.is_empty() {
            println!("  {:<3} no eval docs", lang.code);
            continue;
        }
        // Score docs in parallel; classify is read-only over dbs.
        let ok: usize = docs
            .par_iter()
            .map(|d| {
                let ranked = classify(dbs, d);
                (ranked.first().map(|s| s.lang_index) == Some(i)) as usize
            })
            .sum();
        tot_ok += ok;
        tot_n += docs.len();
        println!(
            "  {:<3} {}/{} = {:.1}%",
            lang.code,
            ok,
            docs.len(),
            100.0 * ok as f32 / docs.len() as f32
        );
    }
    if tot_n > 0 {
        println!(
            "eval: TOTAL {}/{} = {:.1}%",
            tot_ok,
            tot_n,
            100.0 * tot_ok as f32 / tot_n as f32
        );
    }
}
