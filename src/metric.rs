#[derive(Clone)]
pub struct Point {
    pub embeddings: Vec<f32>,
    pub id: String,
}

pub fn euclidean_distance_slice(a: &[f32], b: &[f32]) -> f32 {
    let mut distance = 0.0;
    for i in 0..a.len() {
        distance += (a[i] - b[i]).powi(2);
    }
    distance.sqrt()
}

pub fn euclidean_distance(a: &Point, b: &Point) -> f32 {
    euclidean_distance_slice(&a.embeddings, &b.embeddings)
}