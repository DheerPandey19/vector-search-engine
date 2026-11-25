/// Fraction of exact top-k indexes also returned by the approximate result.
///
/// Both sides are `(distance, point_index)`. Only indexes matter for overlap.
pub fn recall_at_k(exact: &[(f32, usize)], approx: &[(f32, usize)], k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }

    let exact_k = k.min(exact.len());
    let approx_k = k.min(approx.len());
    if exact_k == 0 {
        return 0.0;
    }

    let mut hits = 0usize;
    for (_, idx) in exact.iter().take(exact_k) {
        if approx.iter().take(approx_k).any(|(_, j)| j == idx) {
            hits += 1;
        }
    }

    hits as f32 / exact_k as f32
}

#[cfg(test)]
mod tests {
    use super::recall_at_k;

    #[test]
    fn perfect_overlap_is_one() {
        let exact = vec![(0.1, 0), (0.2, 1), (0.3, 2)];
        let approx = vec![(0.1, 0), (0.25, 1), (0.4, 2)];
        assert!((recall_at_k(&exact, &approx, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn half_overlap_is_half() {
        let exact = vec![(0.1, 0), (0.2, 1)];
        let approx = vec![(0.1, 0), (0.3, 9)];
        assert!((recall_at_k(&exact, &approx, 2) - 0.5).abs() < 1e-6);
    }
}
