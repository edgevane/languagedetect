mod cli;
mod eval;
mod langs;
mod pipeline;
mod read_parquet;
mod source;

use anyhow::Result;
use clap::Parser;

use cli::Cli;

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(w) = cli.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(w)
            .build_global()
            .map_err(|e| anyhow::anyhow!("rayon pool: {e}"))?;
    }
    pipeline::run(&cli)
}
