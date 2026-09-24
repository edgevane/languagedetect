use edgevanelang::model::LangDb;
use edgevanelang::score::classify;


pub fn report(
    dbs: &[LangDb],
    engines: &[(&crate::langs::Lang, edgevanelang::builder::Engine, Vec<Box<str>>)],
    _want: usize,
) {
    println!("eval: top-1 accuracy (held-out strided sample)");
    let mut tot_ok = 0usize;
    let mut tot_n = 0usize;
    for (i, (lang, _, docs)) in engines.iter().enumerate() {
        if docs.is_empty() {
            println!("  {:<3} no eval docs", lang.code);
            continue;
        }
        let mut ok = 0usize;
        for d in docs {
            let ranked = classify(dbs, d);
            if ranked.first().map(|s| s.lang_index) == Some(i) {
                ok += 1;
            }
        }
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
