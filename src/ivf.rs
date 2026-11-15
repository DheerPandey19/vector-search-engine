use crate::metric::{euclidean_distance_slice, Point};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub struct IvfFlatIndex {
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<Vec<usize>>,
}

pub fn seed_centroids(points: &[Point], nlist: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut indices: Vec<usize> = (0..points.len()).collect();
    indices.shuffle(&mut rng);

    let mut centroids = Vec::new();
    for i in 0..nlist {
        centroids.push(points[indices[i]].embeddings.clone());
    }
    centroids
}

pub fn assign_to_list(points: &[Point], centroids: &[Vec<f32>]) -> Vec<Vec<usize>> {
    let nlist = centroids.len();
    let mut lists: Vec<Vec<usize>> = vec![Vec::new(); nlist];

    for (point_i, point) in points.iter().enumerate() {
        let mut best_i = 0;
        let mut best_distance =
            euclidean_distance_slice(&point.embeddings, &centroids[0]);

        for c_i in 1..nlist {
            let d = euclidean_distance_slice(&point.embeddings, &centroids[c_i]);
            if d < best_distance {
                best_i = c_i;
                best_distance = d;
            }
        }
        lists[best_i].push(point_i);
    }

    lists
}

pub fn update_centroids(points: &[Point], lists: &[Vec<usize>]) -> Vec<Vec<f32>> {
    let dim = points[0].embeddings.len();
    let mut new_centroids: Vec<Vec<f32>> = Vec::new();

    for list in lists {
        if list.is_empty() {
            new_centroids.push(vec![0.0; dim]);
            continue;
        }

        let mut sums = vec![0.0_f32; dim];
        for &point_i in list {
            let emb = &points[point_i].embeddings;
            for d in 0..dim {
                sums[d] += emb[d];
            }
        }

        let n = list.len() as f32;
        for d in 0..dim {
            sums[d] /= n;
        }
        new_centroids.push(sums);
    }

    new_centroids
}
