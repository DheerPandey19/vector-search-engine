# vector_db

An in-memory vector search engine in Rust, built from scratch 

Exact search is the ground truth. Three approximate indexes trade a bit of recall for less work per query. A small benchmark harness compares them under one fixed contract: same data, same queries, Euclidean distance, same `k`.

## Indexes

### 1. Flat (exact)

Score the query against **every** stored vector. Sort by distance. Keep top-`k`.

- **Correctness:** always exact (recall = 1)
- **Cost:** `O(n · dim)` per query
- **Role:** baseline for recall and latency

Use Flat whenever you need the true nearest neighbors, or as the yardstick for ANN quality.

### 2. IVFFlat (Inverted File + Flat refine)

**Build:** run k-means to learn `nlist` centroids, then put each point into the list of its nearest centroid.

**Search:** find the `probes` nearest centroids, gather only those lists, then exact-score that candidate set and keep top-`k`.

| Knob | Meaning |
| --- | --- |
| `nlist` | number of clusters / inverted lists |
| `probes` | how many lists to open at query time |

- **Idea:** don’t scan the whole collection — only a few partitions
- **Tradeoff:** more `probes` → higher recall, slower search
- **Typical profile:** fast build and search; recall depends heavily on `probes`

### 3. NSW (Navigable Small World)

A **single-layer proximity graph**. Each point keeps up to `M` bidirectional neighbor edges.

**Build:** insert points one by one. For each new point, beam-search the graph of earlier points (`ef_construction`), link to the nearest selected neighbors, prune both ends to degree `M`.

**Search:** start at the entry point (last inserted), expand a beam of width `ef_search`, follow edges toward the query, return top-`k`.

| Knob | Meaning |
| --- | --- |
| `M` (`max_connections`) | max neighbors per node |
| `ef_construction` | search width while inserting |
| `ef_search` | search width at query time |

- **Idea:** walk a sparse graph instead of scanning all points
- **Tradeoff:** small `M` / `ef` → fast but can miss neighbors; large values → high recall but may exceed Flat latency
- **Limit:** one layer + a fixed entry point can struggle as `n` grows unless knobs are tuned

### 4. HNSW (Hierarchical NSW)

NSW plus a **stack of layers**. Layer 0 holds every point. Higher layers are sparser “highways.”

**Build:** sample a level for each point (most stay low; few reach the top). Greedy-route above the new point’s level; on shared layers, beam-search + link + prune like NSW.

**Search:** greedy descent from the top entry through layers `top … 1`, then beam search on layer 0 with `ef_search`.

| Knob | Meaning |
| --- | --- |
| same as NSW | `M`, `ef_construction`, `ef_search` |
| `max_level` | highest layer index allowed |

- **Idea:** long-range routing upstairs, careful local search downstairs
- **Tradeoff:** slower / heavier build than NSW; usually better recall routing on large collections
- **On small/medium synthetic data:** latency vs Flat depends on knobs — too wide and HNSW can be slower than exact

## Shared contract

- Vectors stored as `f32`; identity is the point’s index in the dataset
- Metric: Euclidean distance (lower is better)
- ANN quality: **recall@k** = fraction of Flat’s top-`k` indexes also returned by the approximate index
- Same `k`, data, and queries for every index in a benchmark run

## Layout

```text
src/
  metric.rs    Point + Euclidean distance
  flat.rs      exact top-k
  ivf.rs       IVFFlat train + search
  nsw.rs       NSW graph, build, search
  hnsw.rs      layers, greedy descent, build, search
  dataset.rs   seeded synthetic points / queries
  recall.rs    recall@k
  bench.rs     multi-query build/search/recall harness
  main.rs      runs the benchmark
```

## Run

```powershell
cd vector_db
cargo run --release
cargo test
```

Always prefer `--release` for timing. Debug builds make graph indexes look unfairly slow.

Tune the run in `main.rs` via `BenchConfig` (`n`, `dim`, `n_queries`, `k`, `nlist`, `probes`, `max_connections`, `ef_*`, …).

## Example results (illustrative)

On synthetic data (`n=10_000`, `dim=32`, 50 queries, `k=10`), with middle graph knobs roughly `M=24`, `ef=100`:

| index | role in that run |
| --- | --- |
| Flat | truth; ~1 ms / query |
| IVFFlat | fastest search; recall ~0.7 with `probes=8` |
| NSW | under Flat latency; recall ~0.98 |
| HNSW | recall ~0.98; search near Flat at this size |

Knobs move the needle a lot: tight graphs can drop NSW recall badly; very wide `ef_search` can make graphs slower than Flat with perfect recall. The harness exists so you can see that dial.

## What’s next

- **IVF-PQ** — product quantization for compressed candidate scoring + exact rerank
- Larger / real corpora (e.g. SIFT) under the same measurement contract
- Optional DataFusion / SQL path later

## References

- Course book: https://skyzh.github.io/vector-db-from-scratch/
- Classic HNSW paper: Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*
