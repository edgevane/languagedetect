use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::thread;

use anyhow::{Context, Result};
use edgevanelang::builder::{encode_combined, finish, Engine};
use edgevanelang::model::LangDb;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parquet::file::reader::ChunkReader;
use rayon::prelude::*;

use crate::cli::Cli;
use crate::langs::{self, Lang};
use crate::read_parquet::decode_text_auto;
use crate::source::{self, HttpChunkReader};

fn docs_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg:>4} [{bar:30}] {pos}/{len} docs {eta_precise}")
        .unwrap()
        .progress_chars("=>-")
}

fn bytes_style() -> ProgressStyle {
    ProgressStyle::with_template("  dl [{bar:30}] {bytes}/{total_bytes} {msg}")
        .unwrap()
        .progress_chars("=>-")
}


pub fn run(cli: &Cli) -> Result<()> {
    let wanted = langs::resolve(&cli.langs)?;
    fs::create_dir_all(&cli.out_dir).context("create out_dir")?;
    let cache_dir: Option<PathBuf> = if cli.no_cache {
        None
    } else {
        Some(match &cli.cache_dir {
            Some(d) => PathBuf::from(d),
            None => PathBuf::from(&cli.out_dir).join(".cache"),
        })
    };

    // Snapshot pre-existing cache files so cleanup below only removes
    // files fetched during this run (never the user's own files).
    let cache_before: std::collections::HashSet<PathBuf> = match &cache_dir {
        Some(dir) => fs::read_dir(dir)
            .map(|rd| rd.flatten().map(|e| e.path()).collect())
            .unwrap_or_default(),
        None => Default::default(),
    };
    // Plan inputs once: HF listing happens a single time even though
    // the two passes below read the data twice.
    let mut plans: Vec<(&Lang, Vec<JobSource>)> = Vec::with_capacity(wanted.len());
    for lang in wanted {
        plans.push((lang, plan_sources(lang, cli)?));
    }
    let n_langs = plans.len();

    let mp = MultiProgress::new();
    let total = mp.add(ProgressBar::new((n_langs * 2) as u64));
    total.set_style(docs_style());
    total.set_message("all");
    println!(
        "creator: {} langs, cap {} docs/lang -> {}",
        n_langs,
        cli.max_docs,
        cli.out_dir
    );


    // Pass 1: train each lang, keep only its token_df key set (small),
    // drop the heavy Engine immediately. Peak RAM = one Engine.
    let mut df_keys: Vec<Vec<u32>> = Vec::with_capacity(plans.len());
    for (lang, sources) in &plans {
        let docs_bar = mp.add(ProgressBar::new(cli.max_docs));
        docs_bar.set_style(docs_style());
        docs_bar.set_message(lang.code.to_string());
        let (engine, _eval) = build_one(lang, cli, sources, cache_dir.as_deref(), &mp, &docs_bar)?;
        docs_bar.finish_and_clear();
        println!(
            "  {:<3} docs={:<8} tokens={:<12} (pass 1/2)",
            lang.code,
            engine.docs,
            engine.tokens_total,
        );
        df_keys.push(engine.token_df().keys().copied().collect());
        // `engine` + `_eval` drop here.
        total.inc(1);
    }

    let idf = global_idf_keys(&df_keys);
    println!("creator: global IDF over {} langs", plans.len());
    // Key sets no longer needed.
    drop(df_keys);

    // Pass 2: re-train each lang, finish + save immediately, drop Engine.
    // Re-training the same docs in the same order yields the same state,
    // so quality is unchanged vs the old single-pass build.
    // With the download cache on, pass 2 reads from local disk.
    let mut eval_docs: Vec<(&Lang, Vec<Box<str>>)> = Vec::with_capacity(plans.len());
    let mut codes: Vec<String> = Vec::with_capacity(plans.len());
    for (lang, sources) in &plans {
        let docs_bar = mp.add(ProgressBar::new(cli.max_docs));
        docs_bar.set_style(docs_style());
        docs_bar.set_message(lang.code.to_string());
        let (engine, eval) = build_one(lang, cli, sources, cache_dir.as_deref(), &mp, &docs_bar)?;
        docs_bar.finish_and_clear();
        let low = engine.docs < cli.max_docs as u32 / 2;
        println!(
            "  {:<3} docs={:<8} tokens={:<12} {} (pass 2/2)",
            lang.code,
            engine.docs,
            engine.tokens_total,
            if low { "(low_resource)" } else { "" }
        );
        let blob = finish(&engine, lang.code, low, &idf);
        // Engine no longer needed once the blob is built.
        drop(engine);
        let path = PathBuf::from(&cli.out_dir).join(format!("{}.evld", lang.code));
        fs::write(&path, &blob).with_context(|| format!("write {}", path.display()))?;
        println!("  wrote {} ({} KB)", path.display(), blob.len() / 1024);
        codes.push(lang.code.to_string());
        eval_docs.push((*lang, eval));
        total.inc(1);
    }
    total.finish_and_clear();
    drop(idf);

    // Pass 2 is done: cached parquets are no longer needed.
    // Remove only files fetched during this run, unless --leave-files.
    if let Some(dir) = cache_dir.as_deref() {
        if cli.leave_files {
            println!("creator: leaving cached files in {}", dir.display());
        } else if let Ok(rd) = fs::read_dir(dir) {
            let mut removed = 0u64;
            for e in rd.flatten() {
                let p = e.path();
                if cache_before.contains(&p) {
                    continue;
                }
                if p.is_file() && fs::remove_file(&p).is_ok() {
                    removed += 1;
                }
            }
            if removed > 0 {
                println!("creator: removed {removed} cached parquet files");
            }
        }
    }

    let mut items: Vec<(String, Vec<u8>)> = Vec::with_capacity(codes.len());
    for code in &codes {
        let path = PathBuf::from(&cli.out_dir).join(format!("{code}.evld"));
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        items.push((code.clone(), bytes));
    }
    let mut dbs: Vec<LangDb> = Vec::with_capacity(items.len());
    for (code, blob) in &items {
        let db = LangDb::from_bytes(blob).context("re-parse written blob")?;
        assert_eq!(db.lang_code(), code);
        dbs.push(db);
    }

    if !cli.no_combined {
        let bundle = encode_combined(&items);
        let path = PathBuf::from(&cli.out_dir).join("combined.evld");
        fs::write(&path, &bundle).with_context(|| format!("write {}", path.display()))?;
        println!(
            "  wrote {} ({} MB)",
            path.display(),
            bundle.len() / (1024 * 1024)
        );
    }
    crate::eval::report(&dbs, &eval_docs, cli.eval_docs);
    Ok(())
}


/// Where a language's training docs come from. Planned once up front
/// so HF listing + file discovery happen a single time for both passes.
enum JobSource {
    Lines(PathBuf),
    LocalParquet(PathBuf),
    Http { repo: String, path: String },
}

fn plan_sources(lang: &Lang, cli: &Cli) -> Result<Vec<JobSource>> {
    if let Some(dir) = &cli.from_text {
        let txt = PathBuf::from(dir).join(format!("{}.txt", lang.code));
        let pq = PathBuf::from(dir).join(format!("{}.parquet", lang.code));
        if pq.exists() {
            return Ok(vec![JobSource::LocalParquet(pq)]);
        }
        return Ok(vec![JobSource::Lines(txt)]);
    }
    let repo = source::open(lang.repo, cli.hf_token.as_deref())?;
    let files = source::list_train_files(&repo, lang.prefixes)?;
    println!("  {:<3} {} files in {}", lang.code, files.len(), lang.repo);
    Ok(files
        .into_iter()
        .map(|path| JobSource::Http {
            repo: lang.repo.to_string(),
            path,
        })
        .collect())
}

fn build_one(
    _lang: &Lang,
    cli: &Cli,
    sources: &[JobSource],
    cache_dir: Option<&std::path::Path>,
    mp: &MultiProgress,
    docs_bar: &ProgressBar,
) -> Result<(Engine, Vec<Box<str>>)> {
    let jobs: Vec<Job> = sources
        .iter()
        .map(|s| match s {
            JobSource::Lines(p) => Job::Lines(p.clone()),
            JobSource::LocalParquet(p) => Job::LocalParquet(p.clone()),
            JobSource::Http { repo, path } => {
                if let Some(dir) = cache_dir {
                    Job::CachedParquet {
                        repo: repo.clone(),
                        path: path.clone(),
                        dir: dir.to_path_buf(),
                    }
                } else {
                    Job::Http {
                        repo: repo.clone(),
                        path: path.clone(),
                    }
                }
            }
        })
        .collect();

    let mut engine = Engine::new();
    let mut eval: Vec<Box<str>> = Vec::new();
    let mut taken = 0u64;
    let stride = (cli.max_docs / cli.eval_docs.max(1) as u64).max(1);
    let mut since_eval = 0u64;
    let done = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::sync_channel::<Result<Vec<Box<str>>>>(8);

    thread::scope(|s| -> Result<()> {

        let done_p = done.clone();
        let text_col = cli.text_col.clone();
        let token = cli.hf_token.clone();
        let mp_p = mp.clone();
        s.spawn(move || {
            for job in &jobs {
                if done_p.load(Ordering::Relaxed) {
                    break;
                }
                if let Err(e) = produce_job(job, &text_col, token.as_deref(), &mp_p, &done_p, &tx)
                {
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        });

        for msg in rx {
            let batch = msg?;
            let mut train: Vec<&Box<str>> = Vec::with_capacity(batch.len());
            for doc in &batch {
                if taken >= cli.max_docs {
                    break;
                }
                since_eval += 1;
                if eval.len() < cli.eval_docs && since_eval >= stride {
                    since_eval = 0;
                    eval.push(doc.clone());
                    continue;
                }
                train.push(doc);
                taken += 1;
            }
            if !train.is_empty() {
                let partial: Engine = train
                    .par_iter()
                    .fold(Engine::new, |mut e, doc| {
                        e.ingest(doc);
                        e
                    })
                    .reduce(Engine::new, |mut a, b| {
                        a.merge(b);
                        a
                    });
                engine.merge(partial);
                docs_bar.inc(train.len() as u64);
            }
            if taken >= cli.max_docs {
                done.store(true, Ordering::Relaxed);
                break;
            }
        }
        Ok(())
    })?;
    Ok((engine, eval))
}


enum Job {
    Lines(PathBuf),
    LocalParquet(PathBuf),
    CachedParquet {
        repo: String,
        path: String,
        dir: PathBuf,
    },
    Http { repo: String, path: String },
}


fn produce_job(
    job: &Job,
    text_col: &str,
    token: Option<&str>,
    mp: &MultiProgress,
    done: &Arc<AtomicBool>,
    tx: &mpsc::SyncSender<Result<Vec<Box<str>>>>,
) -> Result<()> {
    match job {
        Job::Lines(path) => {
            let f = fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
            let mut buf: Vec<Box<str>> = Vec::with_capacity(16384);
            for line in std::io::BufReader::new(f).lines() {
                if done.load(Ordering::Relaxed) {
                    break;
                }
                let line = line.context("read line")?;
                if line.trim().is_empty() {
                    continue;
                }
                buf.push(line.into());
                if buf.len() >= 16384 {
                    if tx.send(Ok(std::mem::take(&mut buf))).is_err() {
                        break;
                    }
                    buf = Vec::with_capacity(4096);
                }
            }
            if !buf.is_empty() {
                let _ = tx.send(Ok(buf));
            }
            Ok(())
        }
        Job::LocalParquet(path) => {
            let f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
            decode_channel_auto(f, text_col, done, tx)
        }
        Job::CachedParquet { repo, path, dir } => {
            // First pass downloads, second pass hits the local file.
            let dl_bar = mp.add(ProgressBar::new(0));
            dl_bar.set_style(bytes_style());
            dl_bar.set_message(short_name(path));
            let local = source::fetch_cached(repo, path, token, dir, dl_bar)?;
            let f = fs::File::open(&local).with_context(|| format!("open {}", local.display()))?;
            decode_channel_auto(f, text_col, done, tx)
        }
        Job::Http { repo, path } => {
            let net_bar = mp.add(ProgressBar::new(0));
            net_bar.set_style(bytes_style());
            net_bar.set_message(short_name(path));
            let src = HttpChunkReader::open(repo, path, token, net_bar.clone())?;
            let r = decode_channel_auto(src, text_col, done, tx);
            let pulled = net_bar.position() as f64 / 1e6;
            let total = net_bar.length().unwrap_or(0) as f64 / 1e6;
            net_bar.finish_and_clear();
            println!("  pulled {pulled:.1} MB / {total:.0} MB ({path})");
            r
        }
    }
}


fn decode_channel_auto<C: ChunkReader + 'static>(
    src: C,
    text_col: &str,
    done: &Arc<AtomicBool>,
    tx: &mpsc::SyncSender<Result<Vec<Box<str>>>>,
) -> Result<()> {
    decode_text_auto(src, text_col, |batch| {
        if done.load(Ordering::Relaxed) {
            return false;
        }
        tx.send(Ok(batch.to_vec())).is_ok()
    })?;
    Ok(())
}

fn short_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}


fn global_idf_keys(df_keys: &[Vec<u32>]) -> BTreeMap<u32, f32> {
    let n = df_keys.len() as f32;
    let mut df: HashMap<u32, u32> = HashMap::new();
    for keys in df_keys {
        for &tok in keys {
            *df.entry(tok).or_insert(0) += 1;
        }
    }
    df.into_iter()
        .map(|(tok, d)| {
            let idf = ((n - d as f32 + 0.5) / (d as f32 + 0.5) + 1.0).ln();
            (tok, idf)
        })
        .collect()
}
