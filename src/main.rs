use vector_db::flat::search_flat;
use vector_db::ivf::{search_lists, train_ivf_flat};
use vector_db::metric::Point;

fn main() {
    let k = 2;
    let nlist = 2;
    let probes = 2;
    let seed = 42;
    let iters = 5;

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

    let index = train_ivf_flat(&points, nlist, iters, seed);
    println!("trained centroids: {:?}", index.centroids);
    println!("trained lists: {:?}", index.lists);

    let flat = search_flat(&points, &query.embeddings, k);
    let ivf = search_lists(&index, &points, &query.embeddings, k, probes);

    println!("flat search: {:?}", flat);
    println!("ivf search (probes={}): {:?}", probes, ivf);
}