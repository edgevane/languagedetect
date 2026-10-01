use clap::Parser;


#[derive(Parser, Debug)]
#[command(name = "creator", version)]
pub struct Cli {

    #[arg(long, value_delimiter = ',')]
    pub langs: Vec<String>,


    #[arg(long, default_value = "./evld")]
    pub out_dir: String,


    #[arg(long, default_value_t = 512_000)]
    pub max_docs: u64,


    #[arg(long, default_value = "text")]
    pub text_col: String,


    #[arg(long)]
    pub from_text: Option<String>,


    #[arg(long, default_value_t = false)]
    pub no_combined: bool,


    #[arg(long, default_value_t = 300)]
    pub eval_docs: usize,


    #[arg(long)]
    pub workers: Option<usize>,


    #[arg(long)]
    pub hf_token: Option<String>,

    /// Path to test.json / .jsonl for benchmark mode.
    /// Each entry needs text + label, e.g.
    /// {"text": "...", "lang": "pl"} (also accepts `label`/`code`/`expected`).
    /// Supports a JSON array or JSON-lines file.
    #[arg(long)]
    pub benchmark: Option<String>,

    /// Model file for benchmark (.evld single or combined).
    /// Defaults to `<out_dir>/combined.evld`.
    #[arg(long)]
    pub model: Option<String>,

    /// Cache dir for downloaded parquet files. Defaults to `<out_dir>/.cache`.
    /// Pass 2 re-reads the same files, so the cache turns the second
    /// pass into local-speed I/O instead of a second download.
    #[arg(long)]
    pub cache_dir: Option<String>,

    /// Disable the download cache (pass 2 re-downloads everything).
    #[arg(long, default_value_t = false)]
    pub no_cache: bool,

    /// Keep downloaded parquet files in the cache dir after the build.
    /// By default cached files fetched during this run are deleted
    /// once pass 2 is done.
    #[arg(long, visible_alias = "leave_files", default_value_t = false)]
    pub leave_files: bool,
}
