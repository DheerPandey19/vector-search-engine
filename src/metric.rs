#[derive(Clone)]
pub struct Point {
    pub embeddings: Vec<f32>,
    pub id: String,
}

pub fn euclidean_distance(a: &Point, b: &Point) -> f32 {
    let mut distance = 0.0;
    for i in 0..a.embeddings.len() {
        distance += (a.embeddings[i] - b.embeddings[i]).powi(2);
    }
    distance.sqrt()
}