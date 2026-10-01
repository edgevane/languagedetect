use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, OnceLock,
};
use std::time::Duration;

use anyhow::{Context, Result};
use bytes::Bytes;
use hf_hub::{HFClient, HFRepositorySync, RepoTypeDataset};
use indicatif::ProgressBar;
use parquet::errors::{ParquetError, Result as PqResult};
use parquet::file::reader::{ChunkReader, Length};

pub mod listing {
    use super::*;
    use hf_hub::repository::RepoTreeEntry;

    pub type DatasetRepo = HFRepositorySync<RepoTypeDataset>;


    pub fn open(repo_id: &str, token: Option<&str>) -> Result<DatasetRepo> {
        let (owner, name) = repo_id
            .split_once('/')
            .context("repo id must be 'owner/name'")?;
        let mut builder = HFClient::builder();
        if let Some(t) = token {
            builder = builder.token(t.to_string());
        }
        let client = builder.build_sync().context("hf client")?;
        Ok(client.dataset(owner.to_string(), name.to_string()))
    }


    pub fn list_train_files(repo: &DatasetRepo, prefixes: &[&str]) -> Result<Vec<String>> {
        for prefix in prefixes {
            let dir = prefix.trim_end_matches('/');
            let entries: Vec<RepoTreeEntry> = repo
                .list_tree()
                .path_in_repo(dir.to_string())
                .recursive(true)
                .send()
                .context("list repo tree (network?)")?;
            let mut files: Vec<String> = entries
                .into_iter()
                .filter_map(|e| match e {
                    RepoTreeEntry::File { path, .. }
                        if path.starts_with(prefix) && path.ends_with(".parquet") =>
                    {
                        Some(path)
                    }
                    _ => None,
                })
                .collect();
            if !files.is_empty() {
                files.sort();
                return Ok(files);
            }
        }
        anyhow::bail!("no .parquet files under any of {prefixes:?}")
    }
}

pub use listing::{list_train_files, open};


const BLOCK: u64 = 4 * 1024 * 1024;
const MAX_BLOCKS: usize = 16;

struct Inner {
    client: reqwest::blocking::Client,
    url: String,
    token: Option<String>,
    total: u64,
    cache: Mutex<BlockCache>,
    fetched: AtomicU64,
    bar: ProgressBar,
    /// Optional dir for write-through block cache. Blocks fetched from the
    /// network are stored as files, so a later reader (pass 2) gets them
    /// from disk. Streaming behaviour is unchanged: blocks are still
    /// fetched on demand, never as one full download.
    persist: Option<PathBuf>,
}

#[derive(Default)]
struct BlockCache {
    map: BTreeMap<u64, Bytes>,
    order: VecDeque<u64>,
}


#[derive(Clone)]
pub struct HttpChunkReader {
    inner: Arc<Inner>,
}

pub(crate) fn shared_client() -> Result<reqwest::blocking::Client> {
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    if let Some(c) = CLIENT.get() {
        return Ok(c.clone());
    }
    let c = reqwest::blocking::Client::builder()
        .user_agent("edgevanelang-creator/0.1")
        .timeout(Duration::from_secs(300))
        .build()
        .context("http client")?;
    let _ = CLIENT.set(c.clone());
    Ok(c)
}

pub(crate) fn dataset_url(repo_id: &str, path: &str) -> String {
    format!("https://huggingface.co/datasets/{repo_id}/resolve/main/{path}")
}

impl HttpChunkReader {

    pub fn open(repo_id: &str, path: &str, token: Option<&str>, bar: ProgressBar) -> Result<Self> {
        Self::open_inner(repo_id, path, token, bar, None)
    }

    /// Streaming reader with a write-through block cache in `cache_dir`.
    /// Pass 1 streams from the network and fills the cache on the fly;
    /// pass 2 opens the same location and reads blocks from disk.
    pub fn open_cached(
        repo_id: &str,
        path: &str,
        token: Option<&str>,
        bar: ProgressBar,
        cache_dir: &std::path::Path,
    ) -> Result<Self> {
        let safe_repo: String = repo_id.replace('/', "__");
        let safe_path: String = path.replace('/', "__");
        let dir = cache_dir.join(format!("{safe_repo}__{safe_path}.blocks"));
        std::fs::create_dir_all(&dir).context("create cache dir")?;
        Self::open_inner(repo_id, path, token, bar, Some(dir))
    }

    fn open_inner(
        repo_id: &str,
        path: &str,
        token: Option<&str>,
        bar: ProgressBar,
        persist: Option<PathBuf>,
    ) -> Result<Self> {
        let client = shared_client()?;
        let url = dataset_url(repo_id, path);
        let mut head = client.head(&url);
        if let Some(t) = token {
            head = head.bearer_auth(t);
        }
        let resp = head.send().with_context(|| format!("HEAD {path}"))?;
        anyhow::ensure!(resp.status().is_success(), "HEAD {path}: {}", resp.status());
        let total = resp.content_length().unwrap_or(0);
        bar.set_length(total);
        Ok(Self {
            inner: Arc::new(Inner {
                client,
                url,
                token: token.map(str::to_string),
                total,
                cache: Mutex::new(BlockCache::default()),
                fetched: AtomicU64::new(0),
                bar,
                persist,
            }),
        })
    }

    fn block_path(&self, idx: u64) -> Option<PathBuf> {
        self.inner
            .persist
            .as_ref()
            .map(|d| d.join(format!("blk_{idx:08x}")))
    }

    fn block(&self, idx: u64) -> Result<Bytes> {

        {
            let cache = self.inner.cache.lock().unwrap();
            if let Some(b) = cache.map.get(&idx) {
                return Ok(b.clone());
            }
        }
        // Write-through cache: pass 2 reads this file instead of the network.
        // Atomic write (tmp + rename) so an interrupted run never leaves
        // a partial block behind.
        if let Some(p) = self.block_path(idx) {
            if let Ok(data) = std::fs::read(&p) {
                let b = Bytes::from(data);
                let mut cache = self.inner.cache.lock().unwrap();
                if cache.map.len() >= MAX_BLOCKS {
                    if let Some(old) = cache.order.pop_front() {
                        cache.map.remove(&old);
                    }
                }
                cache.order.push_back(idx);
                cache.map.insert(idx, b.clone());
                return Ok(b);
            }
        }
        let start = idx * BLOCK;
        let end = (start + BLOCK).min(self.inner.total).saturating_sub(1);
        let mut req = self
            .inner
            .client
            .get(&self.inner.url)
            .header("Range", format!("bytes={start}-{end}"));
        if let Some(t) = &self.inner.token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().with_context(|| format!("range {start}-{end}"))?;
        anyhow::ensure!(
            resp.status() == 206,
            "range request not honored (HTTP {}); server must support Range",
            resp.status()
        );
        let bytes = resp.bytes().context("range body")?;
        if let Some(p) = self.block_path(idx) {
            let tmp = p.with_extension("part");
            if std::fs::write(&tmp, &bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &p);
            }
        }
        let mut cache = self.inner.cache.lock().unwrap();
        if cache.map.len() >= MAX_BLOCKS {
            if let Some(old) = cache.order.pop_front() {
                cache.map.remove(&old);
            }
        }
        cache.order.push_back(idx);
        cache.map.insert(idx, bytes.clone());
        self.inner.fetched.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        self.inner.bar.inc(bytes.len() as u64);
        Ok(bytes)
    }


    fn read_at(&self, start: u64, out: &mut [u8]) -> Result<()> {
        let total = self.inner.total;
        if total > 0 {
            anyhow::ensure!(start + out.len() as u64 <= total, "read past EOF");
        }
        let mut off = 0;
        while off < out.len() {
            let idx = (start + off as u64) / BLOCK;
            let b = self.block(idx)?;
            let inner_off = ((start + off as u64) % BLOCK) as usize;
            let n = (b.len() - inner_off).min(out.len() - off);
            out[off..off + n].copy_from_slice(&b[inner_off..inner_off + n]);
            off += n;
        }
        Ok(())
    }
}

impl Length for HttpChunkReader {
    fn len(&self) -> u64 {
        self.inner.total
    }
}


pub struct HttpRead {
    src: HttpChunkReader,
    pos: u64,
}

impl io::Read for HttpRead {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let total = self.src.inner.total;
        if total > 0 && self.pos >= total {
            return Ok(0);
        }
        let mut n = buf.len() as u64;
        if total > 0 {
            n = n.min(total - self.pos);
        }

        let block_end = (self.pos / BLOCK + 1) * BLOCK;
        n = n.min(block_end - self.pos);
        let n = n as usize;
        self.src
            .read_at(self.pos, &mut buf[..n])
            .map_err(io::Error::other)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl ChunkReader for HttpChunkReader {
    type T = HttpRead;

    fn get_read(&self, start: u64) -> PqResult<HttpRead> {
        Ok(HttpRead {
            src: self.clone(),
            pos: start,
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> PqResult<Bytes> {
        let mut buf = vec![0u8; length];
        self.read_at(start, &mut buf).map_err(|e| {
            ParquetError::General(format!("http range read failed: {e:#}"))
        })?;
        Ok(Bytes::from(buf))
    }
}
