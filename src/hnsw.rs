use crate::metric::{euclidean_distance_slice, Point};
use crate::nsw::{connect_on_layer, prune_neighbours, search_nsw, NswGraph};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Greedy walk on one layer: always move to a strictly closer neighbor.
/// Returns the local minimum reached (one handoff index for the next layer).
pub fn greedy_search(
    graph: &NswGraph,
    points: &[Point],
    query: &Point,
    entry: usize,
) -> usize {
    let mut current = entry;
    let mut current_dist =
        euclidean_distance_slice(&query.embeddings, &points[current].embeddings);

    loop {
        let mut best = current;
        let mut best_dist = current_dist;

        for &neighbor in graph.neighbors(current) {
            if neighbor >= points.len() {
                continue;
            }
            let d =
                euclidean_distance_slice(&query.embeddings, &points[neighbor].embeddings);
            if d < best_dist || (d == best_dist && neighbor < best) {
                best = neighbor;
                best_dist = d;
            }
        }

        if best == current {
            return current;
        }
        current = best;
        current_dist = best_dist;
    }
}

/// One HNSW index: several NSW graphs stacked.
///
/// `layers[0]` = full graph (everyone).
/// `layers[ℓ]` = only points whose level is >= ℓ (others keep an empty neighbor list).
/// `levels[i]` = highest layer that point `i` appears in.
pub struct HnswIndex {
    pub layers: Vec<NswGraph>,
    pub levels: Vec<usize>,
    pub entry: usize,
    pub max_connections: usize,
    pub ef_construction: usize,
}

impl HnswIndex {
    pub fn build(
        points: &[Point],
        max_connections: usize,
        ef_construction: usize,
        max_level: usize,
        seed: u64,
    ) -> Self {
        build_hnsw(
            points,
            max_connections,
            ef_construction,
            max_level,
            seed,
        )
    }

    /// Greedy descent through upper layers, then beam search on layer 0.
    pub fn search(
        &self,
        points: &[Point],
        query: &Point,
        k: usize,
        ef_search: usize,
    ) -> Vec<(f32, usize)> {
        search_hnsw(self, points, query, k, ef_search)
    }
}

/// Sample a level in `0..=max_level`. Most points stay low; few reach the top.
pub fn sample_level(rng: &mut StdRng, max_level: usize) -> usize {
    let mut level = 0;
    while level < max_level && rng.gen_bool(0.5) {
        level += 1;
    }
    level
}

fn ensure_layers(layers: &mut Vec<NswGraph>, up_to: usize, n: usize) {
    while layers.len() <= up_to {
        layers.push(NswGraph::new(n));
    }
}

/// Build a hierarchical NSW.
pub fn build_hnsw(
    points: &[Point],
    max_connections: usize,
    ef_construction: usize,
    max_level: usize,
    seed: u64,
) -> HnswIndex {
    let n = points.len();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut layers: Vec<NswGraph> = Vec::new();
    let mut levels = vec![0; n];

    if n == 0 {
        return HnswIndex {
            layers,
            levels,
            entry: 0,
            max_connections,
            ef_construction,
        };
    }

    let level0 = sample_level(&mut rng, max_level);
    ensure_layers(&mut layers, level0, n);
    levels[0] = level0;
    let mut entry = 0;

    for r in 1..n {
        let target = sample_level(&mut rng, max_level);
        ensure_layers(&mut layers, target, n);
        levels[r] = target;

        let mut ep = entry;
        let top = layers.len() - 1;

        // Layers above target: greedy handoff only.
        let mut lc = top;
        while lc > target {
            ep = greedy_search(&layers[lc], &points[..r], &points[r], ep);
            lc -= 1;
        }

        // Shared layers target..=0: beam search + connect.
        lc = target.min(top);
        loop {
            let found = search_nsw(
                &layers[lc],
                &points[..r],
                &points[r],
                ep,
                ef_construction,
            );
            let cand: Vec<usize> = found.iter().map(|(_, i)| *i).collect();
            let selected = prune_neighbours(points, r, &cand, max_connections);
            connect_on_layer(&mut layers[lc], points, r, &selected, max_connections);

            if let Some((_, nearest)) = found.first() {
                ep = *nearest;
            }

            if lc == 0 {
                break;
            }
            lc -= 1;
        }

        if target == layers.len() - 1 && target > levels[entry] {
            entry = r;
        }
    }

    HnswIndex {
        layers,
        levels,
        entry,
        max_connections,
        ef_construction,
    }
}

/// Query: greedy on layers top..=1, then `search_nsw` on layer 0.
pub fn search_hnsw(
    index: &HnswIndex,
    points: &[Point],
    query: &Point,
    k: usize,
    ef_search: usize,
) -> Vec<(f32, usize)> {
    if points.is_empty() || index.layers.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut ep = index.entry;
    let top = index.layers.len() - 1;

    let mut lc = top;
    while lc >= 1 {
        ep = greedy_search(&index.layers[lc], points, query, ep);
        lc -= 1;
    }

    let ef = ef_search.max(k).max(1);
    let mut hits = search_nsw(&index.layers[0], points, query, ep, ef);
    hits.truncate(k.min(hits.len()));
    hits
}
