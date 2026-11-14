use crate::metric::Point;

pub struct IvfFlatIndex {
    pub centroids : Vec<Vec<f32>>,
    pub lists : Vec<Vec<usize>>,
}