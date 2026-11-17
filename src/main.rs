use vector_db::ivf::{assign_to_list, seed_centroids, update_centroids, train_ivf_flat,search_lists};
use vector_db::metric::{euclidean_distance, euclidean_distance_slice, Point};
use vector_db::flat::search_flat;


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

    let nlist = 2;
    let seed = 42;
    let mut centroids = seed_centroids(&points, nlist, seed);
    println!("seeded centroids: {:?}", centroids);

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


    let mut lists: Vec<Vec<Point>> = Vec::new();
    for _ in 0..centroids.len() {
        lists.push(Vec::new());
    }

    // BUILD: assign each point to nearest centroid
    for point in &points {
        let mut best_i = 0;
        let mut best_distance =
            euclidean_distance_slice(&point.embeddings, &centroids[0]);

        for i in 1..centroids.len() {
            let d = euclidean_distance_slice(&point.embeddings, &centroids[i]);
            if d < best_distance {
                best_distance = d;
                best_i = i;
            }
        }
        println!("{} -> list {}", point.id, best_i);
        lists[best_i].push(point.clone());
    }
    // UPDATE: move each centroid to the mean of its list
    for list_i in 0..nlist{
        if lists[list_i].is_empty(){
            continue;
        }

        let dim = centroids[list_i].len();
        let mut new_centroid = vec![0.0;dim];

        for points in &lists[list_i]{
            for d in 0..dim{
                new_centroid[d] += points.embeddings[d];
            }
        }

        let count = lists[list_i].len() as f32;
        for d in 0..dim{
            new_centroid[d]/=count;
        }

        centroids[list_i] = new_centroid;
    }

    // QUERY: probe nearest centroid
    let mut best_centroid = 0;
    let mut best_centroid_distance =
        euclidean_distance_slice(&query.embeddings, &centroids[0]);

    for i in 1..centroids.len() {
        let d = euclidean_distance_slice(&query.embeddings, &centroids[i]);
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

    let lists = assign_to_list(&points, &centroids); // your function name
    let centroids = update_centroids(&points, &lists);
    println!("lists: {:?}", lists);
    println!("updated centroids: {:?}", centroids);

    // let index = train_ivf_flat(&points, nlist, 5, seed);
    // println!("trained centroids: {:?}", index.centroids);
    // println!("trained lists: {:?}", index.lists);

    let index = train_ivf_flat(&points, nlist, 5, seed);
    let results = search_lists(&index, &points, &query.embeddings, k, 2);
    println!("ivf search: {:?}", results);
    
    let flat = search_flat(&points, &query.embeddings, k);
    println!("flat search: {:?}", flat);
}
