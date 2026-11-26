use std::time::Instant;

use crate::dataset::{random_points, random_queries};
use crate::flat::search_flat;
use crate::hnsw::HnswIndex;
use crate::ivf::{search_lists, train_ivf_flat};
use crate::nsw::NswIndex;
use crate::recall::recall_at_k;

/// Fixed measurement contract for comparing indexes.
#[derive(Clone, Debug)]
pub struct BenchConfig {
    pub n: usize,
    pub dim: usize,
    pub n_queries: usize,
    pub k: usize,
    pub data_seed: u64,
    pub query_seed: u64,
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_level: usize,
    pub graph_seed: u64,
    pub nlist: usize,
    pub probes: usize,
    pub ivf_iters: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            n: 1000,
            dim: 32,
            n_queries: 50,
            k: 10,
            data_seed: 42,
            query_seed: 99,
            max_connections: 24,
            ef_construction: 100,
            ef_search: 100,
            max_level: 4,
            graph_seed: 7,
            nlist: 32,
            probes: 8,
            ivf_iters: 10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IndexReport {
    pub name: &'static str,
    pub build_ms: f64,
    pub avg_search_ms: f64,
    pub avg_recall: f32,
}

#[derive(Clone, Debug)]
pub struct BenchReport {
    pub config: BenchConfig,
    pub indexes: Vec<IndexReport>,
}

fn mean_f32(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

fn mean_f64(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn time_ms(f: impl FnOnce()) -> f64 {
    let t0 = Instant::now();
    f();
    t0.elapsed().as_secs_f64() * 1000.0
}

/// Build each index once, then average search latency and recall over many queries.
pub fn run_benchmark(cfg: &BenchConfig) -> BenchReport {
    let points = random_points(cfg.n, cfg.dim, cfg.data_seed);
    let queries = random_queries(cfg.n_queries, cfg.dim, cfg.query_seed);

    // Ground truth for every query (exact).
    let exact: Vec<Vec<(f32, usize)>> = queries
        .iter()
        .map(|q| search_flat(&points, &q.embeddings, cfg.k))
        .collect();

    let mut indexes = Vec::new();

    // --- Flat (search only; build is free) ---
    {
        let mut search_ms = Vec::with_capacity(cfg.n_queries);
        for q in &queries {
            let t0 = Instant::now();
            let _ = search_flat(&points, &q.embeddings, cfg.k);
            search_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        indexes.push(IndexReport {
            name: "flat",
            build_ms: 0.0,
            avg_search_ms: mean_f64(&search_ms),
            avg_recall: 1.0,
        });
    }

    // --- IVFFlat ---
    {
        let mut ivf = None;
        let build_ms = time_ms(|| {
            ivf = Some(train_ivf_flat(
                &points,
                cfg.nlist,
                cfg.ivf_iters,
                cfg.data_seed,
            ));
        });
        let ivf = ivf.expect("ivf trained");

        let mut search_ms = Vec::with_capacity(cfg.n_queries);
        let mut recalls = Vec::with_capacity(cfg.n_queries);
        for (qi, q) in queries.iter().enumerate() {
            let t0 = Instant::now();
            let hits = search_lists(&ivf, &points, &q.embeddings, cfg.k, cfg.probes);
            search_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            recalls.push(recall_at_k(&exact[qi], &hits, cfg.k));
        }
        indexes.push(IndexReport {
            name: "ivf",
            build_ms,
            avg_search_ms: mean_f64(&search_ms),
            avg_recall: mean_f32(&recalls),
        });
    }

    // --- NSW ---
    {
        let mut nsw = None;
        let build_ms = time_ms(|| {
            nsw = Some(NswIndex::build(
                &points,
                cfg.max_connections,
                cfg.ef_construction,
            ));
        });
        let nsw = nsw.expect("nsw built");

        let mut search_ms = Vec::with_capacity(cfg.n_queries);
        let mut recalls = Vec::with_capacity(cfg.n_queries);
        for (qi, q) in queries.iter().enumerate() {
            let t0 = Instant::now();
            let hits = nsw.search(&points, q, cfg.k, cfg.ef_search);
            search_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            recalls.push(recall_at_k(&exact[qi], &hits, cfg.k));
        }
        indexes.push(IndexReport {
            name: "nsw",
            build_ms,
            avg_search_ms: mean_f64(&search_ms),
            avg_recall: mean_f32(&recalls),
        });
    }

    // --- HNSW ---
    {
        let mut hnsw = None;
        let build_ms = time_ms(|| {
            hnsw = Some(HnswIndex::build(
                &points,
                cfg.max_connections,
                cfg.ef_construction,
                cfg.max_level,
                cfg.graph_seed,
            ));
        });
        let hnsw = hnsw.expect("hnsw built");

        let mut search_ms = Vec::with_capacity(cfg.n_queries);
        let mut recalls = Vec::with_capacity(cfg.n_queries);
        for (qi, q) in queries.iter().enumerate() {
            let t0 = Instant::now();
            let hits = hnsw.search(&points, q, cfg.k, cfg.ef_search);
            search_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            recalls.push(recall_at_k(&exact[qi], &hits, cfg.k));
        }
        indexes.push(IndexReport {
            name: "hnsw",
            build_ms,
            avg_search_ms: mean_f64(&search_ms),
            avg_recall: mean_f32(&recalls),
        });
    }

    BenchReport {
        config: cfg.clone(),
        indexes,
    }
}

pub fn print_report(report: &BenchReport) {
    let c = &report.config;
    println!("=== vector_db benchmark ===");
    println!(
        "n={}  dim={}  queries={}  k={}  metric=euclidean",
        c.n, c.dim, c.n_queries, c.k
    );
    println!(
        "ivf: nlist={} probes={} iters={} | graph: M={} ef_c={} ef_s={} max_level={}",
        c.nlist,
        c.probes,
        c.ivf_iters,
        c.max_connections,
        c.ef_construction,
        c.ef_search,
        c.max_level
    );
    println!();
    println!(
        "{:<8} {:>12} {:>14} {:>12}",
        "index", "build_ms", "avg_search_ms", "avg_recall"
    );
    println!("{}", "-".repeat(50));
    for row in &report.indexes {
        println!(
            "{:<8} {:>12.2} {:>14.3} {:>12.3}",
            row.name, row.build_ms, row.avg_search_ms, row.avg_recall
        );
    }
}
