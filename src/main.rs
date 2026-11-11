struct Point {
    embeddings: Vec<f32>,
    id: String,
}

fn  euclidean_distance(a: &Point, b: &Point) -> f32 {
    let mut distance=0.0;
    for i in 0..a.embeddings.len() {
        distance +=(a.embeddings[i] - b.embeddings[i]).powi(2);
        
    }
    distance=distance.sqrt();
    return distance;
}
fn main() {

    let  k = 2;
    let query = Point {
        id : String::from("query"),
        embeddings: vec![1.0,0.0],
    };

    let p0 = Point {
        id : String::from("p0"),
        embeddings: vec![0.9, 0.1],
    };

    let p1 = Point {
        id : String::from("p1"),
        embeddings: vec![0.1, 0.9],
    };

    let p2 = Point {
        id: String::from("2"),
        embeddings: vec![0.0, 1.0],
    };


    let points = vec![p0, p1,p2];
    

    let mut scored : Vec<(f32,String)> = Vec::new();
    for point in &points{
        let distance = euclidean_distance(&query,point);
        scored.push((distance, point.id.clone()));
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for i in 0..k {
        println!("id {} -> distance {}", scored[i].1, scored[i].0);
    }


}
