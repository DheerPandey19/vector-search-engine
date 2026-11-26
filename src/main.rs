use vector_db::bench::{print_report, run_benchmark, BenchConfig};

fn main() {
    // Bump n / n_queries for a heavier evaluation (`cargo run --release`).
    let cfg = BenchConfig {
        n: 10000,
        dim: 32,
        n_queries: 50,
        k: 10,
        nlist: 32,
        probes: 8,
        ..BenchConfig::default()
    };

    let report = run_benchmark(&cfg);
    print_report(&report);
}
