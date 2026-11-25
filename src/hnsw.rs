use crate::metric::{euclidean_distance_slice, Point};
use crate::nsw::NswGraph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Greedy walk on one layer: always move to a strictly closer neighbor.
/// Returns the local minimum reached (one handoff index for the next layer).

pub fn greedy_search(
    graph: &NswGraph,
    points: &[Point],
    query: & Point,
    entry: usize,
) ->usize {
    let mut current = entry;
    let mut current_dist =
             euclidean_distance_slice(&query.embeddings,&points[current].embeddings)


    loop {
        let mut best = current;
        let mut best_dist = current_dist;

        for &neighbor in graph.neighbors(current){
            // Skip neighbors outside this points slice (during build we pass 0..r).
            if neighbor >= points.len(){
                 continue;
            }

            let d = euclidean_distance_slice(&query.embeddings,&points[neighbor].embeddings);
            if d < best_dist || (d == best_dist && neighbor < best){
                best = neighbor;
                best_dist = d;
            }
        }

        if best == current{
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

/// Sample a level in `0..max_level` (inclusive upper bound is max_level - 1
/// if we treat max_level as "number of layers", or 0..=max_level inclusive —
/// we'll use: return value in 0..=max_level, capped).
///
/// Geometric idea: keep "promoting" while a coin flip succeeds.
/// Most points stay at 0; few reach the top.
pub fn sample_level(rng: &mut StdRng, max_level: usize) -> usize {
    let mut level = 0;
    // ~50% chance to go up each time; stop at max_level.
    while level < max_level && rng.gen::<bool>() {
        level += 1;
    }
    level
}