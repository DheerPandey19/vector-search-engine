use vector_db::flat::search_flat;
use vector_db::hnsw::HnswIndex;
use vector_db::metric::Point;
use vector_db::nsw::NswIndex;

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

    let max_connections = 3;
    let ef_construction = 4;
    let ef_search = 4;
    let max_level = 2;
    let seed = 42;
    let k = 2;

    let nsw = NswIndex::build(&points, max_connections, ef_construction);
    let hnsw = HnswIndex::build(
        &points,
        max_connections,
        ef_construction,
        max_level,
        seed,
    );

    let flat = search_flat(&points, &query.embeddings, k);
    let nsw_hits = nsw.search(&points, &query, k, ef_search);
    let hnsw_hits = hnsw.search(&points, &query, k, ef_search);

    println!("nsw entry: {}", nsw.entry);
    println!("hnsw entry: {} (levels: {:?})", hnsw.entry, hnsw.levels);
    println!("flat: {:?}", flat);
    println!("nsw:  {:?}", nsw_hits);
    println!("hnsw: {:?}", hnsw_hits);
}
