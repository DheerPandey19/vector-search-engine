use vector_db::metric::{Point, euclidean_distance};

fn main() {
    let k = 2;
    let query = Point {
        id: String::from("query"),
        embeddings: vec![1.0, 0.0],
    };

    let points = vec![
        Point {
            id: String::from("p0"),
            embeddings: vec![0.9, 0.1],
        },
        Point {
            id: String::from("p1"),
            embeddings: vec![0.1, 0.9],
        },
        Point {
            id: String::from("2"),
            embeddings: vec![0.0, 1.0],
        },
    ];

    // --- exact baseline (scan all points) ---
    println!("exact top-k:");
    let mut scored: Vec<(f32, String)> = Vec::new();
    for point in &points {
        let distance = euclidean_distance(&query, point);
        scored.push((distance, point.id.clone()));
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for i in 0..k {
        println!("id {} -> distance {}", scored[i].1, scored[i].0);
    }

    // --- toy IVFFlat: centroids + inverted lists ---
    let centroids = vec![
        Point {
            id: String::from("c0"),
            embeddings: vec![0.0, 1.0],
        },
        Point {
            id: String::from("c1"),
            embeddings: vec![1.0, 0.0],
        },
    ];

    let mut lists: Vec<Vec<Point>> = Vec::new();
    for _ in 0..centroids.len() {
        lists.push(Vec::new());
    }

    // BUILD: assign each point to nearest centroid
    for point in &points {
        let mut best_i = 0;
        let mut best_distance = euclidean_distance(point, &centroids[0]);

        for i in 1..centroids.len() {
            let d = euclidean_distance(point, &centroids[i]);
            if d < best_distance {
                best_distance = d;
                best_i = i;
            }
        }
        println!("{} -> list {}", point.id, best_i);
        lists[best_i].push(point.clone());
    }

    // QUERY: probe nearest centroid's list, find best point in it
    let mut best_centroid = 0;
    let mut best_centroid_distance = euclidean_distance(&query, &centroids[0]);

    for i in 1..centroids.len() {
        let d = euclidean_distance(&query, &centroids[i]);
        if d < best_centroid_distance {
            best_centroid_distance = d;
            best_centroid = i;
        }
    }
    println!("query probes list {}", best_centroid);

    let candidates = &lists[best_centroid];
    let mut scored : Vec<(f32, String)> = Vec::new();
    for point in candidates{
        let distance = euclidean_distance(point, &query);
        scored.push((distance,point.id.clone()));
    }
    scored.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap());

    let take = k.min(scored.len());
    for i in 0..take {
        println!("id {} -> distance {}", scored[i].1, scored[i].0);
    }

}
