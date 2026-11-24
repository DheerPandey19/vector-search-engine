use crate::metric::{euclidean_distance_slice, Point};
use crate::nsw::NswGraph;

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