use crate::metric::{euclidean_distance_slice, Point};

pub fn search_flat(points: &[Point],query: &[f32], k:usize)-> Vec<(f32, usize)>{
    let mut scored: Vec<(f32,usize)> = Vec::new();

    for(i,point) in points.iter().enumerate() {
        let score = euclidean_distance_slice(query,&point.embeddings);

        scored.push((score,i));

    }

    scored.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
    scored.truncate(k.min(scored.len()));

    return scored;
}