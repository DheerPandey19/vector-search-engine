use vector_db::flat::search_flat;
use vector_db::metric::Point;
use vector_db::nsw::{search_nsw, NswGraph};

fn main() {
    // A=(0,0) B=(1,0) C=(2,0) D=(1,2) E=(3,1)
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

    // Hand-built edges (same as the walk we traced):
    // A—B—C—E
    //    |
    //    D
    let mut graph = NswGraph::new(points.len());
    graph.add_bidirectional(0, 1); // A—B
    graph.add_bidirectional(1, 2); // B—C
    graph.add_bidirectional(1, 3); // B—D
    graph.add_bidirectional(2, 4); // C—E

    let entry = 4; // start at E (last inserted)
    let ef = 2;
    let k = 2;

    let nsw = search_nsw(&graph, &points, &query, entry, ef);
    let flat = search_flat(&points, &query.embeddings, k);

    println!("neighbors of B: {:?}", graph.neighbors(1));
    println!("nsw (entry=E, ef={}): {:?}", ef, &nsw[..k.min(nsw.len())]);
    println!("flat: {:?}", flat);
}