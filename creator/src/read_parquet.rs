use anyhow::{Context, Result};
use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use parquet::file::reader::ChunkReader;

const BATCH: usize = 16384;


/// Single-open variant: resolves `text_col` and streams in one pass,
/// avoiding a second footer/metadata fetch (matters for HTTP).
pub(crate) fn decode_text_auto<C: ChunkReader + 'static>(
    src: C,
    text_col: &str,
    mut on_batch: impl FnMut(&[Box<str>]) -> bool,
) -> Result<u64> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(src).context("parquet open")?;
    let mut leaves: Vec<String> = Vec::new();
    collect_leaves(
        builder.parquet_schema().root_schema(),
        String::new(),
        &mut leaves,
    );
    let leaf_idx = leaves
        .iter()
        .position(|n| n == text_col || n.ends_with(&format!(".{text_col}")))
        .with_context(|| format!("column '{text_col}' not found; leaves: {leaves:?}"))?;
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [leaf_idx]);
    let batches = builder
        .with_projection(mask)
        .with_batch_size(BATCH)
        .build()
        .context("parquet build")?;
    let mut decoded = 0u64;
    let mut buf: Vec<Box<str>> = Vec::with_capacity(BATCH);
    for batch in batches {
        let batch: RecordBatch = batch.context("record batch")?;
        push_text(batch.column(0), &mut buf);
        if buf.len() >= BATCH {
            decoded += buf.len() as u64;
            if !on_batch(&buf) {
                return Ok(decoded);
            }
            buf.clear();
        }
    }
    if !buf.is_empty() {
        decoded += buf.len() as u64;
        on_batch(&buf);
    }
    Ok(decoded)
}

fn push_text(col: &dyn arrow_array::Array, buf: &mut Vec<Box<str>>) {
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        for i in 0..a.len() {
            if !a.is_null(i) {
                let s = a.value(i);
                if !s.is_empty() {
                    buf.push(s.into());
                }
            }
        }
    } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        for i in 0..a.len() {
            if !a.is_null(i) {
                let s = a.value(i);
                if !s.is_empty() {
                    buf.push(s.into());
                }
            }
        }
    }
}

fn collect_leaves(t: &parquet::schema::types::Type, prefix: String, out: &mut Vec<String>) {
    use parquet::schema::types::Type;
    match t {
        Type::PrimitiveType { basic_info, .. } => {
            let n = if prefix.is_empty() {
                basic_info.name().to_string()
            } else {
                format!("{prefix}.{}", basic_info.name())
            };
            out.push(n);
        }
        Type::GroupType { basic_info, fields, .. } => {
            let p = if prefix.is_empty() && basic_info.name() == "schema" {
                String::new()
            } else if prefix.is_empty() {
                basic_info.name().to_string()
            } else {
                format!("{prefix}.{}", basic_info.name())
            };
            for f in fields {
                collect_leaves(f, p.clone(), out);
            }
        }
    }
}
