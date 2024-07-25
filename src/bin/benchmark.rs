use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::current_num_threads;
use sparse::{JacobiParams, SparseBlockMat};
use std::time::Instant;

fn benchmark<F: Fn()>(f: F, duration: f64) {
    let start = Instant::now();
    let mut iter = 0;
    while start.elapsed().as_secs_f64() < duration {
        f();
        iter += 1;
    }
    println!(
        "{:.1e} s / iter ({} iterations)",
        start.elapsed().as_secs_f64() / iter as f64,
        iter
    );
}

fn get_laplacian_2d(ni: usize, nj: usize) -> SparseBlockMat {
    let dx = 1.0 / (ni as f64 + 1.0);
    let dy = 1.0 / (nj as f64 + 1.0);

    let idx = |i, j| i + ni * j;

    let mut edgs = Vec::new();
    for i in 0..ni {
        for j in 0..nj {
            if i < ni - 1 {
                edgs.push([idx(i, j), idx(i + 1, j)]);
            }
            if j < nj - 1 {
                edgs.push([idx(i, j), idx(i, j + 1)]);
            }
        }
    }
    let mut mat = SparseBlockMat::from_edges(ni * nj, edgs.iter().copied(), true);
    for i in 0..ni {
        for j in 0..nj {
            mat.set(idx(i, j), idx(i, j), -2.0 * (1.0 / dx / dx + 1.0 / dy / dy));
            if i > 0 {
                mat.set(idx(i, j), idx(i - 1, j), 1.0 / dx / dx);
            }
            if i < ni - 1 {
                mat.set(idx(i, j), idx(i + 1, j), 1.0 / dx / dx);
            }
            if j > 0 {
                mat.set(idx(i, j), idx(i, j - 1), 1.0 / dy / dy);
            }
            if j < nj - 1 {
                mat.set(idx(i, j), idx(i, j + 1), 1.0 / dy / dy);
            }
        }
    }
    mat
}

#[derive(Debug)]
enum BenchmarkType {
    MultSeq,
    MultPar,
    MultParChunks,
    JacobiPar,
}

fn run(bench_type: BenchmarkType, n: usize, duration: f64) {
    println!(
        "Running {bench_type:?} on {} threads",
        current_num_threads()
    );
    let mat = get_laplacian_2d(n, n);

    let mut rng = StdRng::seed_from_u64(1234);
    let x = (0..mat.n())
        .map(|_| rng.gen::<f64>() - 0.5)
        .collect::<Vec<_>>();

    match bench_type {
        BenchmarkType::MultSeq => benchmark(
            || {
                let _y = mat.seq_mult(&x);
            },
            duration,
        ),
        BenchmarkType::MultPar => benchmark(
            || {
                let _y = mat.mult(&x);
            },
            duration,
        ),
        BenchmarkType::MultParChunks => benchmark(
            || {
                let chunk_size = mat.n() / current_num_threads();
                let _y = mat.mult_chunks(&x, chunk_size);
            },
            duration,
        ),
        BenchmarkType::JacobiPar => benchmark(
            || {
                let mut sol = vec![0.0; mat.n()];
                mat.jacobi(
                    &x,
                    &mut sol,
                    JacobiParams {
                        max_iter: 10,
                        ..Default::default()
                    },
                );
            },
            duration,
        ),
    }
}

fn main() {
    let n = 1000;
    let duration = 10.0;

    if current_num_threads() == 1 {
        run(BenchmarkType::MultSeq, n, duration);
    }
    run(BenchmarkType::MultPar, n, duration);
    run(BenchmarkType::MultParChunks, n, duration);
    run(BenchmarkType::JacobiPar, n, duration);
}
