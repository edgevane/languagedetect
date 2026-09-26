use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use edgevanelang::model::{CombinedDb, LangDb};
use edgevanelang::score::classify;

struct Case {
    text: String,
    want: String,
}

fn pick_field(o: &serde_json::Map<String, serde_json::Value>) -> Option<Case> {
    let get = |keys: &[&str]| -> Option<&serde_json::Value> {
        keys.iter().find_map(|k| o.get(*k))
    };
    let text = get(&["text", "sentence", "content"])?.as_str()?.to_string();
    let want = get(&["lang", "label", "code", "expected", "language"])?
        .as_str()?
        .to_string();
    Some(Case { text, want })
}

fn load_cases(path: &str) -> Result<Vec<Case>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let trimmed = raw.trim_start();
    if trimmed.starts_with('[') {
        let v: serde_json::Value =
            serde_json::from_str(&raw).context("parse test.json as JSON array")?;
        let arr = v.as_array().context("top-level JSON must be an array")?;
        arr.iter()
            .filter_map(|e| e.as_object().and_then(pick_field))
            .collect::<Vec<_>>()
            .pipe_ok()
    } else {
        let mut out = Vec::new();
        for (i, line) in raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value =
                serde_json::from_str(line).with_context(|| format!("parse JSONL line {}", i + 1))?;
            if let Some(o) = v.as_object().and_then(pick_field) {
                out.push(o);
            }
        }
        Ok(out)
    }
}

trait Pipe: Sized {
    fn pipe_ok(self) -> Result<Self> {
        Ok(self)
    }
}
impl<T> Pipe for Vec<T> {}

fn resolve_model(cli_model: Option<&str>, out_dir: &str) -> Result<PathBuf> {
    if let Some(m) = cli_model {
        return Ok(PathBuf::from(m));
    }
    let combined = Path::new(out_dir).join("combined.evld");
    if combined.exists() {
        return Ok(combined);
    }
    // Fall back to first .evld in out_dir.
    let mut first: Option<PathBuf> = None;
    if let Ok(rd) = fs::read_dir(out_dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "evld") {
                first = Some(p);
                break;
            }
        }
    }
    first.with_context(|| {
        format!("no model: pass --model <file.evld>, or expected {combined:?}")
    })
}

pub fn run(benchmark: &str, cli_model: Option<&str>, out_dir: &str) -> Result<()> {
    let cases = load_cases(benchmark)?;
    anyhow::ensure!(!cases.is_empty(), "no benchmark cases in {benchmark}");
    let model_path = resolve_model(cli_model, out_dir)?;
    println!(
        "benchmark: {} cases, model {}",
        cases.len(),
        model_path.display()
    );

    let bytes = fs::read(&model_path)
        .with_context(|| format!("read {}", model_path.display()))?;
    // Both views borrow from `bytes`; keep it alive for the whole run.
    let cdb = CombinedDb::from_bytes(&bytes).ok();
    let single = if cdb.is_none() {
        Some(LangDb::from_bytes(&bytes).context("parse model as LangDb/CombinedDb")?)
    } else {
        None
    };
    let dbs: Vec<LangDb> = if let Some(c) = &cdb {
        (0..c.len()).map(|i| c.get(i).unwrap()).collect()
    } else {
        vec![single.unwrap().clone()]
    };

    let codes: Vec<String> = dbs.iter().map(|d| d.lang_code().to_string()).collect();
    println!("benchmark: langs [{}]", codes.join(", "));

    let t0 = Instant::now();
    let mut ok = 0usize;
    let mut per: std::collections::BTreeMap<String, (usize, usize)> =
        std::collections::BTreeMap::new();
    let mut conf_sum = 0f64;
    for c in &cases {
        let ranked = classify(&dbs, &c.text);
        let got = ranked.first().map(|s| s.lang_code().to_string()).unwrap_or_default();
        let conf = ranked.first().map(|s| s.confidence as f64).unwrap_or(0.0);
        conf_sum += conf;
        let e = per.entry(c.want.clone()).or_insert((0, 0));
        e.1 += 1;
        if got == c.want {
            ok += 1;
            e.0 += 1;
        } else {
            println!("  MISS want={} got={} conf={conf:.2} :: {}", c.want, got, snippet(&c.text));
        }
    }
    let dt = t0.elapsed();
    let n = cases.len() as f64;
    println!(
        "benchmark: {}/{} = {:.1}%  avg_conf={:.2}  {:.0} docs/s  {:.1} ms total",
        ok,
        cases.len(),
        100.0 * ok as f64 / n,
        conf_sum / n,
        n / dt.as_secs_f64(),
        dt.as_secs_f64() * 1000.0
    );
    for (lang, (o, t)) in &per {
        println!("  {lang:<6} {o}/{t} = {:.1}%", 100.0 * *o as f64 / *t as f64);
    }
    Ok(())
}

fn snippet(s: &str) -> String {
    const N: usize = 80;
    let mut out: String = s.chars().take(N).collect();
    if s.chars().count() > N {
        out.push('…');
    }
    out.replace('\n', " ")
}
