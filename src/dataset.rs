use crate::metric::Point;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Build `n` points in `dim` dimensions, values in [-1, 1), seeded for repeatability.
pub fn random_points(n: usize, dim: usize, seed: u64) -> Vec<Point> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(n);

    for i in 0..n {
        let mut embeddings = Vec::with_capacity(dim);
        for _ in 0..dim {
            embeddings.push(rng.gen_range(-1.0f32..1.0));
        }
        points.push(Point {
            id: format!("p{i}"),
            embeddings,
        });
    }
    points
}

/// One random query vector with the same dimension.
pub fn random_query(dim: usize, seed: u64) -> Point {
    let mut queries = random_queries(1, dim, seed);
    queries.pop().expect("requested one query")
}

/// `q` random query vectors, seeded for repeatability.
pub fn random_queries(q: usize, dim: usize, seed: u64) -> Vec<Point> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut queries = Vec::with_capacity(q);

    for i in 0..q {
        let mut embeddings = Vec::with_capacity(dim);
        for _ in 0..dim {
            embeddings.push(rng.gen_range(-1.0f32..1.0));
        }
        queries.push(Point {
            id: format!("q{i}"),
            embeddings,
        });
    }
    queries
}
