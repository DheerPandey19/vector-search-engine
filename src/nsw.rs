/// Edges of a Navigable Small World graph.
///
/// `neighbors[i]` is the list of point indexes that point `i` links to.
/// The vectors stay in the caller's `points` slice. This struct stores edges only.
use crate::metric::{euclidean_distance_slice, Point};

pub struct NswGraph {
    neighbors: Vec<Vec<usize>>,
}

impl NswGraph {
    /// `n` is how many points will be in the graph.
    /// Every point starts with an empty neighbor list.
    pub fn new(n: usize) -> Self {
        Self {
            neighbors: vec![Vec::new(); n],
        }
    }

    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// Borrow the neighbor indexes of one point.
    pub fn neighbors(&self, index: usize) -> &[usize] {
        &self.neighbors[index]
    }

    /// Link `a` and `b` both ways.
    pub fn add_bidirectional(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.add_one_way(a, b);
        self.add_one_way(b, a);
    }

    fn add_one_way(&mut self, from: usize, to: usize) {
        let list = &mut self.neighbors[from];
        if !list.contains(&to) {
            list.push(to);
        }
    }
}

/// Best-first walk on the graph.
///
/// - `entry`: start node (point index)
/// - `ef`: search width — how many good points to keep while exploring
/// - returns up to `ef` hits, nearest first: `(distance, point_index)`
pub fn search_nsw(
    graph: &NswGraph,
    points: &[Point],
    query: &Point,
    entry: usize,
    ef: usize,
) -> Vec<(f32, usize)> {
    if points.is_empty() {
        return Vec::new();
    }

    let ef = ef.max(1).min(points.len());

    let entry_distance =
        euclidean_distance_slice(&query.embeddings, &points[entry].embeddings);

    // C: still to expand — we always expand the nearest pending point.
    let mut candidates: Vec<(f32, usize)> = vec![(entry_distance, entry)];
    // W: best results so far — trimmed to `ef`.
    let mut results: Vec<(f32, usize)> = vec![(entry_distance, entry)];
    let mut visited = vec![false; points.len()];
    visited[entry] = true;

    while !candidates.is_empty() {
        // Nearest first in C.
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (cand_dist, cand_idx) = candidates.remove(0);

        // Nearest first in W so the last item is the worst when W is full.
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let worst_dist = results.last().unwrap().0;

        // Stop only when the next candidate is *strictly* worse than W's worst.
        if results.len() >= ef && cand_dist > worst_dist {
            break;
        }

        for &neighbor in graph.neighbors(cand_idx) {
            if visited[neighbor] {
                continue;
            }
            visited[neighbor] = true;

            let d = euclidean_distance_slice(&query.embeddings, &points[neighbor].embeddings);

            // Keep this neighbor if W still has room, or it beats W's worst.
            results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let worst = results.last().unwrap().0;
            if results.len() < ef || d < worst {
                candidates.push((d, neighbor));
                results.push((d, neighbor));
                results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                results.truncate(ef);
            }
        }
    }

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    results
}
