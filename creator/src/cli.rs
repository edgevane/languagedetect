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
}
