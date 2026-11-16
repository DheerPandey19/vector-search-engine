use crate::metric::{euclidean_distance_slice, Point};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub struct IvfFlatIndex {
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<Vec<usize>>,
}
pub fn train_ivf_flat(
    points: &[Point],
    nlist: usize,
    iters: usize,
    seed: u64,
) -> IvfFlatIndex {
    let mut centroids = seed_centroids(points, nlist, seed);

    for _ in 0..iters {
        let lists = assign_to_list(points, &centroids);
        centroids = update_centroids(points, &lists);
    }

    let lists = assign_to_list(points, &centroids);
    IvfFlatIndex { centroids, lists }
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

pub fn search_lists(
    index: &IvfFlatIndex,
    points: &[Point],
    query: &[f32],
    k: usize,
    probes:usize,
) -> Vec<(f32,usize)> {

    //score the distance to each centriod vs query
    let mut centroid_scores : Vec<(f32, usize)> = Vec::new();
    for(i,centroid) in index.centroids.iter().enumerate() {
        let d = euclidean_distance_slice(query, centroid);
        centroid_scores.push((d,i));
    }

    centroid_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    //gather the candidates from nearest pc(probe count) lists

    let probe_count = probes.min(centroid_scores.len());
    
    let mut candidates : Vec<usize> = Vec::new();
    for i in 0..probe_count{
        let list_i = centroid_scores[i].1;
        for &point_i in &index.lists[list_i] {
            candidates.push(point_i);
        }
    }

       // 3) exact distances for candidates, keep top-k
       let mut scored: Vec<(f32, usize)> = Vec::new();
       for point_i in candidates {
           let d = euclidean_distance_slice(query, &points[point_i].embeddings);
           scored.push((d, point_i));
       }
       scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
       scored.truncate(k.min(scored.len()));
       scored





}

