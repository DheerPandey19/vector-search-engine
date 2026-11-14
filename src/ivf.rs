use crate::metric::Point;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;


pub struct IvfFlatIndex {
    pub centroids : Vec<Vec<f32>>,
    pub lists : Vec<Vec<usize>>,
}

pub fn seed_centroids(point : &[Point], nlist : usize , seed : u64)-> Vec<Vec<f32>>{
    let mut rng = StdRng::seed_from_u64(seed);

    let mut indices : Vec<usize> = (0..point.len()).collect();
    indices.shuffle(&mut rng);

    let mut centroids = Vec:: new();
    for i in 0..nlist{
        centroids.push(point[indices[i]].embeddings.clone());
    }
    centroids
}
