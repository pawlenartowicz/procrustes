//! Linear assignment via Jonker-Volgenant.
//!
//! Ported from [Antti/lapjv-rust](https://github.com/Antti/lapjv-rust) v0.3.0
//! (commit `cce816a08c014f484456223754b167d4aef0e01d`), MIT-licensed by Andrii
//! Dmytrenko (2018-2021). Reworked to f64-only over flat row-major buffers,
//! removing ndarray / generic-Float / cancellation / log dependencies for
//! in-tree use. See `LICENSE-THIRDPARTY` at crate root for the upstream notice.
//!
//! Original algorithm: Jonker & Volgenant, "A Shortest Augmenting Path
//! Algorithm for Dense and Sparse Linear Assignment Problems", *Computing*
//! 38, 325-340 (1987).

/// Solve `argmax_p Σ_k |dot[p[k]·k + k]|` over permutations p of `0..k`.
///
/// `dot` is a row-major K×K buffer with `dot[i*k + j] = ⟨a[:, i], reference[:, j]⟩`.
/// Returns `assigned`: length-K vector where `assigned[k]` is the source row of
/// `dot` mapped onto column `k` (i.e. the column of `a` matched to column `k`
/// of `reference`).
///
/// Internally negates and absolute-values the cost to convert
/// `signed_permutation`'s max-|dot| problem into JV's min-cost problem.
///
/// # Invariants (caller-enforced)
/// - `dot.len() == k * k`
/// - `k > 0`
/// - all entries finite (caller validates inputs upstream)
///
/// Errors are not returned: violations are programmer errors; debug builds
/// trip a `debug_assert!`. The JV core itself never fails on a valid square
/// dense cost matrix.
#[allow(clippy::many_single_char_names)]
pub(crate) fn solve_max_abs(dot: &[f64], k: usize) -> Vec<usize> {
    debug_assert_eq!(dot.len(), k * k);
    debug_assert!(k > 0);

    // Build lap_cost[i*k + j] = -|dot[i*k + j]|
    let lap_cost: Vec<f64> = dot.iter().map(|x| -x.abs()).collect();

    // JV state vectors
    let mut free_rows: Vec<usize> = Vec::with_capacity(k);
    let mut v: Vec<f64> = Vec::with_capacity(k);
    let mut in_col: Vec<usize> = Vec::with_capacity(k);
    let mut in_row: Vec<usize> = vec![0; k];

    ccrrt(
        &lap_cost,
        k,
        &mut free_rows,
        &mut v,
        &mut in_col,
        &mut in_row,
    );

    let mut i = 0;
    while !free_rows.is_empty() && i < 2 {
        carr(
            &lap_cost,
            k,
            &mut free_rows,
            &mut v,
            &mut in_col,
            &mut in_row,
        );
        i += 1;
    }

    if !free_rows.is_empty() {
        ca(
            &lap_cost,
            k,
            &mut free_rows,
            &mut v,
            &mut in_col,
            &mut in_row,
        );
    }

    // After JV: in_col[j] = row assigned to column j (col→row, y-array).
    // The caller wants assigned[j] = source row for column j, which is in_col.
    in_col
}

// Column-reduction and reduction transfer for a dense cost matrix.
// Ported from ccrrt_dense in lapjv-rust.
#[allow(clippy::many_single_char_names)]
fn ccrrt(
    cost: &[f64],
    k: usize,
    free_rows: &mut Vec<usize>,
    v: &mut Vec<f64>,
    in_col: &mut Vec<usize>,
    in_row: &mut [usize],
) {
    let mut unique = vec![true; k];
    let mut in_row_not_set = vec![true; k];

    // For each column j, find the row with minimum cost → initial row-per-column assignment.
    // The original lapjv-rust iterates lanes(Axis(0)) = columns of the 2-D cost matrix.
    // in_col[j] = row with minimum cost in column j  (col → row, y-array in JV)
    // v[j]      = minimum cost in column j           (column dual variable)
    for j in 0..k {
        let mut min_row = 0;
        let mut min_val = cost[j]; // cost[row=0, col=j]
        for i in 1..k {
            let c = cost[i * k + j];
            if c < min_val {
                min_val = c;
                min_row = i;
            }
        }
        in_col.push(min_row);
        v.push(min_val);
    }

    for j in (0..k).rev() {
        let i = in_col[j];
        if in_row_not_set[i] {
            in_row[i] = j;
            in_row_not_set[i] = false;
        } else {
            unique[i] = false;
            in_col[j] = usize::MAX;
        }
    }

    for i in 0..k {
        if in_row_not_set[i] {
            free_rows.push(i);
        } else if unique[i] {
            let j = in_row[i];
            let mut min = f64::INFINITY;
            for j2 in 0..k {
                if j2 == j {
                    continue;
                }
                let c = reduced_cost(cost, k, v, i, j2);
                if c < min {
                    min = c;
                }
            }
            v[j] -= min;
        }
    }
}

// Augmenting row reduction for a dense cost matrix.
// Ported from carr_dense in lapjv-rust.
#[allow(clippy::many_single_char_names)]
fn carr(
    cost: &[f64],
    k: usize,
    free_rows: &mut Vec<usize>,
    v: &mut [f64],
    in_col: &mut [usize],
    in_row: &mut [usize],
) {
    let mut current = 0;
    let mut new_free_rows = 0;
    let mut rr_cnt = 0;
    let num_free_rows = free_rows.len();

    while current < num_free_rows {
        rr_cnt += 1;
        let free_i = free_rows[current];
        current += 1;

        let (v1, v2, mut j1, j2) = find_umins(cost, k, v, free_i);

        let mut i0 = in_col[j1];
        let v1_new = v[j1] - (v2 - v1);
        let v1_lowers = v1_new < v[j1];

        if rr_cnt < current * k {
            if v1_lowers {
                v[j1] = v1_new;
            } else if i0 != usize::MAX && j2.is_some() {
                j1 = j2.unwrap();
                i0 = in_col[j1];
            }
            if i0 != usize::MAX {
                if v1_lowers {
                    current -= 1;
                    free_rows[current] = i0;
                } else {
                    free_rows[new_free_rows] = i0;
                    new_free_rows += 1;
                }
            }
        } else if i0 != usize::MAX {
            free_rows[new_free_rows] = i0;
            new_free_rows += 1;
        }
        in_row[free_i] = j1;
        in_col[j1] = free_i;
    }
    free_rows.truncate(new_free_rows);
}

// Full augmenting-path phase.
// Ported from ca_dense in lapjv-rust.
#[allow(clippy::many_single_char_names)]
fn ca(
    cost: &[f64],
    k: usize,
    free_rows: &mut Vec<usize>,
    v: &mut [f64],
    in_col: &mut [usize],
    in_row: &mut [usize],
) {
    let mut pred = vec![0usize; k];

    let rows = std::mem::take(free_rows);
    for freerow in rows {
        let mut i = usize::MAX;
        let mut cnt = 0;
        let mut j = find_path_dense(cost, k, v, in_col, freerow, &mut pred);
        debug_assert!(j < k);
        while i != freerow {
            i = pred[j];
            in_col[j] = i;
            std::mem::swap(&mut j, &mut in_row[i]);
            cnt += 1;
            if cnt > k {
                // Square dense matrix — this path cannot exceed k steps.
                unreachable!("ca: augmenting path exceeded k steps on a valid square matrix");
            }
        }
    }
}

// Single iteration of modified Dijkstra shortest path algorithm (find_path_dense).
// Ported from find_path_dense in lapjv-rust.
#[allow(clippy::many_single_char_names)]
fn find_path_dense(
    cost: &[f64],
    k: usize,
    v: &mut [f64],
    in_col: &[usize],
    start_i: usize,
    pred: &mut [usize],
) -> usize {
    let mut collist: Vec<usize> = (0..k).collect();
    let mut d: Vec<f64> = (0..k)
        .map(|i| reduced_cost(cost, k, v, start_i, i))
        .collect();
    for p in pred.iter_mut().take(k) {
        *p = start_i;
    }

    let mut lo = 0;
    let mut hi = 0;
    let mut n_ready = 0;

    let mut final_j = None;
    while final_j.is_none() {
        if lo == hi {
            n_ready = lo;
            hi = find_dense(k, lo, &d, &mut collist);
            for &j in collist.iter().take(hi).skip(lo) {
                if in_col[j] == usize::MAX {
                    final_j = Some(j);
                }
            }
        }

        if final_j.is_none() {
            final_j = scan_dense(
                cost,
                k,
                v,
                in_col,
                &mut lo,
                &mut hi,
                &mut d,
                &mut collist,
                pred,
            );
        }
    }

    let mind = d[collist[lo]];
    for &j in collist.iter().take(n_ready) {
        v[j] += d[j] - mind;
    }
    final_j.unwrap()
}

// Scan all columns in TODO starting from arbitrary column in SCAN.
// Ported from scan_dense in lapjv-rust.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
fn scan_dense(
    cost: &[f64],
    k: usize,
    v: &[f64],
    in_col: &[usize],
    plo: &mut usize,
    phi: &mut usize,
    d: &mut [f64],
    collist: &mut [usize],
    pred: &mut [usize],
) -> Option<usize> {
    let mut lo = *plo;
    let mut hi = *phi;
    while lo != hi {
        let j = collist[lo];
        lo += 1;
        let i = in_col[j];
        let mind = d[j];
        let h = reduced_cost(cost, k, v, i, j) - mind;
        // Iterate over TODO columns (hi..collist.len()), but hi may grow as we
        // move columns into the SCAN set. Use a while loop so the bound is
        // re-evaluated each iteration (unlike a for-range which fixes the end).
        let mut idx = hi;
        while idx < collist.len() {
            let j2 = collist[idx];
            let cred_ij = reduced_cost(cost, k, v, i, j2) - h;
            if cred_ij < d[j2] {
                d[j2] = cred_ij;
                pred[j2] = i;
                if (cred_ij - mind).abs() < f64::EPSILON {
                    if in_col[j2] == usize::MAX {
                        return Some(j2);
                    }
                    collist[idx] = collist[hi];
                    collist[hi] = j2;
                    hi += 1;
                }
            }
            idx += 1;
        }
    }
    *plo = lo;
    *phi = hi;
    None
}

// Find range of columns at minimum distance in collist[lo..].
// Ported from find_dense in lapjv-rust.
fn find_dense(k: usize, lo: usize, d: &[f64], collist: &mut [usize]) -> usize {
    let mut hi = lo + 1;
    let mut mind = d[collist[lo]];
    // Use a while loop so `hi` can grow dynamically as we extend the frontier.
    let mut idx = hi;
    while idx < k {
        let j = collist[idx];
        let h = d[j];
        if h <= mind {
            if h < mind {
                hi = lo;
                mind = h;
            }
            collist[idx] = collist[hi];
            collist[hi] = j;
            hi += 1;
        }
        idx += 1;
    }
    hi
}

// Minimum and second minimum reduced cost for row `row_i` over all columns.
// Ported from find_umins_plain in lapjv-rust.
#[allow(clippy::many_single_char_names)]
fn find_umins(cost: &[f64], k: usize, v: &[f64], row_i: usize) -> (f64, f64, usize, Option<usize>) {
    let row = &cost[row_i * k..(row_i + 1) * k];
    let mut umin = row[0] - v[0];
    let mut usubmin = f64::INFINITY;
    let mut j1 = 0;
    let mut j2 = None;
    for j in 1..k {
        let h = row[j] - v[j];
        if h < usubmin {
            if h >= umin {
                usubmin = h;
                j2 = Some(j);
            } else {
                usubmin = umin;
                umin = h;
                j2 = Some(j1);
                j1 = j;
            }
        }
    }
    (umin, usubmin, j1, j2)
}

fn reduced_cost(cost: &[f64], k: usize, v: &[f64], i: usize, j: usize) -> f64 {
    cost[i * k + j] - v[j]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: check that a slice is a valid permutation of 0..k.
    fn is_valid_permutation(assigned: &[usize], k: usize) -> bool {
        if assigned.len() != k {
            return false;
        }
        let mut seen = vec![false; k];
        for &a in assigned {
            if a >= k || seen[a] {
                return false;
            }
            seen[a] = true;
        }
        true
    }

    #[test]
    fn all_zero_dot_returns_valid_permutation() {
        let k = 4;
        let dot = vec![0.0_f64; k * k];
        let assigned = solve_max_abs(&dot, k);
        assert!(
            is_valid_permutation(&assigned, k),
            "expected valid permutation, got {assigned:?}",
        );
    }

    #[test]
    fn diagonal_optimal_returns_identity_permutation() {
        let k = 5;
        let mut dot = vec![0.0_f64; k * k];
        for i in 0..k {
            dot[i * k + i] = 1.0;
        }
        let assigned = solve_max_abs(&dot, k);
        let expected: Vec<usize> = (0..k).collect();
        assert_eq!(
            assigned, expected,
            "diagonal dot: expected identity permutation, got {assigned:?}",
        );
    }

    #[test]
    fn reverse_permutation_optimal() {
        let k = 5;
        let mut dot = vec![0.0_f64; k * k];
        for i in 0..k {
            dot[i * k + (k - 1 - i)] = 1.0;
        }
        let assigned = solve_max_abs(&dot, k);
        let expected: Vec<usize> = (0..k).rev().collect();
        assert_eq!(
            assigned, expected,
            "reverse dot: expected reverse permutation, got {assigned:?}",
        );
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn cyclic_shift_optimal_breaks_self_inverse() {
        // Cost matrix where the unique optimum picks a non-self-inverse permutation:
        // assigned should be [1, 2, 3, 0] (its inverse is [3, 0, 1, 2] — different).
        //
        // We need solve_max_abs to maximize Σ |dot[assigned[c]*4 + c]|.
        // Set dot[i*4 + j] so the unique maximum sums when row 1 → col 0,
        // row 2 → col 1, row 3 → col 2, row 0 → col 3.
        //
        // Place a large value (10) at each of (assigned[c], c) = (1,0), (2,1),
        // (3,2), (0,3). Place small distractors (1) elsewhere so the optimum
        // is unique by a wide margin.
        let mut dot = vec![1.0_f64; 16];
        dot[1 * 4 + 0] = 10.0; // assigned[0] = 1
        dot[2 * 4 + 1] = 10.0; // assigned[1] = 2
        dot[3 * 4 + 2] = 10.0; // assigned[2] = 3
        dot[0 * 4 + 3] = 10.0; // assigned[3] = 0

        let assigned = super::solve_max_abs(&dot, 4);

        assert_eq!(
            assigned,
            vec![1, 2, 3, 0],
            "cyclic-shift optimum must be assigned[c] = (c + 1) mod 4; \
             a self-inverse return value would mask row/col transposition bugs in ccrrt"
        );
    }
}
