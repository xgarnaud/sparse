use crate::{MatVec, SparseMat};
use nalgebra::{SMatrix, SVector};

impl<const N: usize> MatVec for SMatrix<f64, N, N> {
    type Mat = SMatrix<f64, N, N>;
    type Vect = SVector<f64, N>;

    fn mat_zero() -> Self::Mat {
        Self::Mat::zeros()
    }

    fn vect_zero() -> Self::Vect {
        Self::Vect::zeros()
    }

    fn inverse(mat: &Self::Mat) -> Self::Mat {
        mat.try_inverse().unwrap()
    }

    fn mult(mat: &Self::Mat, vec: &Self::Vect) -> Self::Vect {
        let mut out = Self::Vect::zeros();
        mat.mul_to(&vec, &mut out);
        out
    }

    fn norm2(vec: &Self::Vect) -> f64 {
        vec.norm_squared()
    }
}

pub type SparseBlockMat<const N: usize> = SparseMat<SMatrix<f64, N, N>>;

#[cfg(test)]
mod tests {
    use crate::{nalgebra::SparseBlockMat, IterativeParams, SparseMat};
    use nalgebra::{SMatrix, SVector};

    fn get_laplacian_2d(ni: usize, nj: usize) -> SparseBlockMat<2> {
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
        let k = SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 2.0);

        let mut mat = SparseMat::from_edges(ni * nj, edgs.iter().copied(), true).unwrap();
        for i in 0..ni {
            for j in 0..nj {
                mat.set(
                    idx(i, j),
                    idx(i, j),
                    -2.0 * (1.0 / dx / dx + 1.0 / dy / dy) * k,
                );
                if i > 0 {
                    mat.set(idx(i, j), idx(i - 1, j), 1.0 / dx / dx * k);
                }
                if i < ni - 1 {
                    mat.set(idx(i, j), idx(i + 1, j), 1.0 / dx / dx * k);
                }
                if j > 0 {
                    mat.set(idx(i, j), idx(i, j - 1), 1.0 / dy / dy * k);
                }
                if j < nj - 1 {
                    mat.set(idx(i, j), idx(i, j + 1), 1.0 / dy / dy * k);
                }
            }
        }
        mat
    }

    #[test]
    fn test_1() {
        let mat = get_laplacian_2d(5, 5);

        let row = mat.row(0);
        assert_eq!(row.len(), 3);
        for i in 0..mat.n() {
            if i == 0 || i == 1 || i == 5 {
                assert!(row.get(i).is_some())
            } else {
                assert!(row.get(i).is_none())
            }
        }
        let row = mat.row(6);
        assert_eq!(row.len(), 5);
        for i in 0..mat.n() {
            if i == 1 || i == 5 || i == 6 || i == 7 || i == 11 {
                assert!(row.get(i).is_some())
            } else {
                assert!(row.get(i).is_none())
            }
        }
    }

    #[test]
    fn test_2() {
        let mat = get_laplacian_2d(5, 5);

        for i in 0..mat.n() {
            let row = mat.row(i);
            for j in 0..mat.n() {
                if let Some(&v) = row.get(j) {
                    if i == j {
                        assert_eq!(v[0], -144.0);
                        assert_eq!(v[1], 0.0);
                        assert_eq!(v[2], -0.0);
                        assert_eq!(v[3], -288.0);
                    } else {
                        assert_eq!(v[0], 36.0);
                        assert_eq!(v[1], 0.0);
                        assert_eq!(v[2], 0.0);
                        assert_eq!(v[3], 72.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_residual() {
        let mat = get_laplacian_2d(5, 5);
        let b = vec![SVector::<f64, 2>::new(3.0, 4.0); mat.n()];
        let rhs = mat.mult(&b);
        let res = mat.residual(&rhs, &b);
        assert!(res < 1e-12);
    }

    #[test]
    fn test_jacobi() {
        let mat = get_laplacian_2d(16, 16);
        let rhs = vec![SVector::<f64, 2>::new(3.0, 4.0); mat.n()];
        let mut b = vec![SVector::<f64, 2>::zeros(); mat.n()];

        let params = IterativeParams {
            max_iter: 2000,
            rel_tol: 1e-4,
            abs_tol: 1.0,
        };
        let (niter, residual) = mat.jacobi(&rhs, &mut b, params);
        assert!(niter < params.max_iter);
        assert!(residual < params.rel_tol * SparseBlockMat::<2>::l2_norm(&rhs));

        let residual = mat.residual(&rhs, &b);
        assert!(residual < params.rel_tol * SparseBlockMat::<2>::l2_norm(&rhs));
    }

    #[test]
    fn test_seq_sgs() {
        let mat = get_laplacian_2d(16, 16);
        let rhs = vec![SVector::<f64, 2>::new(3.0, 4.0); mat.n()];

        let params = IterativeParams {
            max_iter: 200,
            rel_tol: 2.5e-4,
            abs_tol: 1.0,
        };

        let mut b = vec![SVector::<f64, 2>::zeros(); mat.n()];
        let (niter, residual_jac) = mat.jacobi(&rhs, &mut b, params);
        assert_eq!(niter, params.max_iter);

        let mut b = vec![SVector::<f64, 2>::zeros(); mat.n()];
        let (niter, residual_sgs) = mat.seq_sgs(&rhs, &mut b, params);
        assert!(niter < params.max_iter);
        assert!(residual_sgs < residual_jac * 0.01);
    }

    #[test]
    fn test_sgs() {
        let mat = get_laplacian_2d(16, 16);
        let rhs = vec![SVector::<f64, 2>::new(3.0, 4.0); mat.n()];

        let params = IterativeParams {
            max_iter: 200,
            rel_tol: 2.5e-4,
            abs_tol: 1.0,
        };

        let mut b = vec![SVector::<f64, 2>::zeros(); mat.n()];
        let (niter_seq_sgs, residual_seq_sgs) = mat.seq_sgs(&rhs, &mut b, params);
        assert!(niter_seq_sgs < params.max_iter);

        let mut b = vec![SVector::<f64, 2>::zeros(); mat.n()];
        let (niter_sgs_1, residual_sgs_1) = mat.sgs(&rhs, &mut b, params, 16 * 16);
        assert!(niter_sgs_1 < params.max_iter);
        assert_eq!(niter_sgs_1, niter_seq_sgs);
        assert_eq!(residual_sgs_1, residual_seq_sgs);

        let mut b = vec![SVector::<f64, 2>::zeros(); mat.n()];
        let (niter_sgs_2, residual_sgs_2) = mat.sgs(&rhs, &mut b, params, 16 * 16 / 5);
        assert!(niter_sgs_2 < params.max_iter);
        assert!(niter_sgs_2 > niter_seq_sgs);
        assert!(residual_sgs_2 > residual_seq_sgs);
        assert!(residual_sgs_2 < residual_seq_sgs * 2.0);
    }
}
