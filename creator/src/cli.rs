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
}
