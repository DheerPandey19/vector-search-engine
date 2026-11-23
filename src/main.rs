use vector_db::flat::search_flat;
use vector_db::metric::Point;
use vector_db::nsw::{build_nsw, search_nsw};

fn main() {
    let points = vec![
        Point {
            id: String::from("A"),
            embeddings: vec![0.0, 0.0],
        },
        Point {
            id: String::from("B"),
            embeddings: vec![1.0, 0.0],
        },
        Point {
            id: String::from("C"),
            embeddings: vec![2.0, 0.0],
        },
        Point {
            id: String::from("D"),
            embeddings: vec![1.0, 2.0],
        },
        Point {
            id: String::from("E"),
            embeddings: vec![3.0, 1.0],
        },
    ];

    let query = Point {
        id: String::from("query"),
        embeddings: vec![0.2, 0.1],
    };

    // M=2 is too tight on this insert order: B prunes D away, then E only
    // links to D, and the walk from E can never reach A. Use M=3 so the
    // long-range edge survives.
    let max_connections = 3;
    let ef_construction = 4;
    let ef_search = 2;
    let k = 2;

    let (graph, entry) = build_nsw(&points, max_connections, ef_construction);
    let nsw = search_nsw(&graph, &points, &query, entry, ef_search);
    let flat = search_flat(&points, &query.embeddings, k);

    println!("entry: {}", entry);
    for i in 0..points.len() {
        println!("neighbors of {}: {:?}", points[i].id, graph.neighbors(i));
    }
    println!("nsw: {:?}", &nsw[..k.min(nsw.len())]);
    println!("flat: {:?}", flat);
}
