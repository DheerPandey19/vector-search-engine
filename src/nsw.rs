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
    pub fn new(n: usize) -> Self {
        Self {
            neighbors: vec![Vec::new(); n],
        }
    }

    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }

    /// Borrow the neighbor indexes of one point.
    pub fn neighbors(&self, index: usize) -> &[usize] {
        &self.neighbors[index]
    }

    /// Replace the neighbor list for one point (used after prune).
    pub fn set_neighbors(&mut self, index: usize, list: Vec<usize>) {
        self.neighbors[index] = list;
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

/// Built NSW index: graph + entry point + construction knobs.
pub struct NswIndex {
    pub graph: NswGraph,
    pub entry: usize,
    pub max_connections: usize,
    pub ef_construction: usize,
}

impl NswIndex {
    pub fn build(
        points: &[Point],
        max_connections: usize,
        ef_construction: usize,
    ) -> Self {
        let (graph, entry) = build_nsw(points, max_connections, ef_construction);
        Self {
            graph,
            entry,
            max_connections,
            ef_construction,
        }
    }

    /// Search with width `ef_search.max(k)`, then return at most `k` hits.
    pub fn search(
        &self,
        points: &[Point],
        query: &Point,
        k: usize,
        ef_search: usize,
    ) -> Vec<(f32, usize)> {
        let ef = ef_search.max(k).max(1);
        let mut hits = search_nsw(&self.graph, points, query, self.entry, ef);
        hits.truncate(k.min(hits.len()));
        hits
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

    let mut candidates: Vec<(f32, usize)> = vec![(entry_distance, entry)];
    let mut results: Vec<(f32, usize)> = vec![(entry_distance, entry)];
    let mut visited = vec![false; points.len()];
    visited[entry] = true;

    while !candidates.is_empty() {
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (cand_dist, cand_idx) = candidates.remove(0);

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let worst_dist = results.last().unwrap().0;

        if results.len() >= ef && cand_dist > worst_dist {
            break;
        }

        for &neighbor in graph.neighbors(cand_idx) {
            if neighbor >= points.len() || visited[neighbor] {
                continue;
            }
            visited[neighbor] = true;

            let d = euclidean_distance_slice(&query.embeddings, &points[neighbor].embeddings);

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

pub fn prune_neighbours(
    points: &[Point],
    owner: usize,
    candidates: &[usize],
    max_connections: usize,
) -> Vec<usize> {
    if max_connections == 0 {
        return Vec::new();
    }

    let mut unique: Vec<usize> = Vec::new();
    for &idx in candidates {
        if idx == owner {
            continue;
        }
        if !unique.contains(&idx) {
            unique.push(idx);
        }
    }

    unique.sort_by(|&a, &b| {
        let da = euclidean_distance_slice(&points[owner].embeddings, &points[a].embeddings);
        let db = euclidean_distance_slice(&points[owner].embeddings, &points[b].embeddings);
        da.partial_cmp(&db).unwrap().then(a.cmp(&b))
    });

    unique.truncate(max_connections.min(unique.len()));
    unique
}

/// Link `r` to `selected`, then prune both ends so the graph stays reciprocal
/// and degree-bounded.
pub fn connect_on_layer(
    graph: &mut NswGraph,
    points: &[Point],
    r: usize,
    selected: &[usize],
    max_connections: usize,
) {
    for &m in selected {
        graph.add_bidirectional(r, m);
    }

    let pruned_r = prune_neighbours(points, r, graph.neighbors(r), max_connections);
    graph.set_neighbors(r, pruned_r);

    let linked: Vec<usize> = graph.neighbors(r).to_vec();
    for &m in &linked {
        let pruned_m = prune_neighbours(points, m, graph.neighbors(m), max_connections);
        graph.set_neighbors(m, pruned_m);
        if !graph.neighbors(m).contains(&r) {
            let filtered: Vec<usize> = graph
                .neighbors(r)
                .iter()
                .copied()
                .filter(|&x| x != m)
                .collect();
            graph.set_neighbors(r, filtered);
        }
    }
}

/// Insert points 0..n-1 one at a time. Returns (graph, entry_point).
pub fn build_nsw(
    points: &[Point],
    max_connections: usize,
    ef_construction: usize,
) -> (NswGraph, usize) {
    let mut graph = NswGraph::new(points.len());
    if points.is_empty() {
        return (graph, 0);
    }

    let mut entry = 0;

    for r in 1..points.len() {
        let found = search_nsw(
            &graph,
            &points[..r],
            &points[r],
            entry,
            ef_construction,
        );
        let cand: Vec<usize> = found.into_iter().map(|(_, i)| i).collect();
        let selected = prune_neighbours(points, r, &cand, max_connections);
        connect_on_layer(&mut graph, points, r, &selected, max_connections);
        entry = r;
    }

    (graph, entry)
}
