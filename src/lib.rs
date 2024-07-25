#![feature(raw_slice_split)]
#![feature(test)]
extern crate test;

use iterators::{ParallelRowsIterator, ParallelRowsMutIterator, RowsIterator, RowsMutIterator};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::ParallelSliceMut;
use std::{fmt::Display, ops::Range};
mod iterators;

pub struct Row<'a>(&'a [(usize, f64)]);

impl<'a> Row<'a> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &(usize, f64)> {
        self.0.iter()
    }
    pub fn get(&self, j: usize) -> Option<&f64> {
        let idx = self.0.binary_search_by(|x| x.0.cmp(&j));
        if let Ok(idx) = idx {
            Some(&self.0[idx].1)
        } else {
            None
        }
    }
    pub fn mult(&self, b: &[f64]) -> f64 {
        self.0.iter().fold(0.0, |x, (j, v)| x + v * b[*j])
    }
}

impl<'a> Display for Row<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cols: ")?;
        for (j, _) in self.0.iter() {
            write!(f, "{} ", j)?;
        }
        write!(f, ", vals: ")?;
        for (_, v) in self.0.iter() {
            write!(f, "{} ", v)?;
        }
        Ok(())
    }
}

pub struct RowMut<'a>(&'a mut [(usize, f64)]);

impl<'a> RowMut<'a> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut (usize, f64)> {
        self.0.iter_mut()
    }
    pub fn get(&mut self, j: usize) -> Option<&mut f64> {
        let idx = self.0.binary_search_by(|x| x.0.cmp(&j));
        if let Ok(idx) = idx {
            Some(&mut self.0[idx].1)
        } else {
            None
        }
    }
    pub fn get_mut(&mut self, j: usize) -> Option<&mut f64> {
        let idx = self.0.binary_search_by(|x| x.0.cmp(&j));
        if let Ok(idx) = idx {
            Some(&mut self.0[idx].1)
        } else {
            None
        }
    }
    pub fn sort(&mut self) {
        self.0.sort_by(|xi, xj| xi.0.cmp(&xj.0));
    }
    pub fn zero(&mut self) {
        self.iter_mut().for_each(|x| x.1 = 0.0);
    }
}

impl<'a> Display for RowMut<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cols: ")?;
        for (j, _) in self.0.iter() {
            write!(f, "{} ", j)?;
        }
        write!(f, ", vals: ")?;
        for (_, v) in self.0.iter() {
            write!(f, "{} ", v)?;
        }
        Ok(())
    }
}

pub struct SparseBlockMat {
    ptr: Vec<usize>,
    data: Vec<(usize, f64)>,
}

#[derive(Clone, Copy)]
pub struct JacobiParams {
    pub max_iter: usize,
    pub rel_tol: f64,
    pub abs_tol: f64,
}

impl Default for JacobiParams {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rel_tol: 1e-6,
            abs_tol: 1e-12,
        }
    }
}

impl SparseBlockMat {
    pub fn from_edges<I: Iterator<Item = [usize; 2]> + Clone>(
        n_verts: usize,
        edges: I,
        with_diagonal: bool,
    ) -> Self {
        let mut ptr = if with_diagonal {
            vec![1; n_verts + 1]
        } else {
            vec![0; n_verts + 1]
        };

        ptr[0] = 0;
        for [i0, i1] in edges.clone() {
            ptr[i0 + 1] += 1;
            ptr[i1 + 1] += 1;
        }

        for i in 0..n_verts {
            ptr[i + 1] += ptr[i];
        }
        let nnz = ptr[n_verts];
        let mut data = vec![(usize::MAX, 0.0); nnz];

        // diagonal part
        if with_diagonal {
            ptr.iter().take(n_verts).enumerate().for_each(|(i, &idx)| {
                data[idx].0 = i;
            });
        }

        for [i0, i1] in edges {
            #[allow(clippy::needless_range_loop)]
            for idx in ptr[i0]..ptr[i0 + 1] {
                if data[idx].0 == usize::MAX {
                    data[idx].0 = i1;
                    break;
                }
            }
            #[allow(clippy::needless_range_loop)]
            for idx in ptr[i1]..ptr[i1 + 1] {
                if data[idx].0 == usize::MAX {
                    data[idx].0 = i0;
                    break;
                }
            }
        }

        let mut res = Self { ptr, data };

        for i in 0..n_verts {
            res.row_mut(i).sort();
        }

        res
    }
    pub fn n(&self) -> usize {
        self.ptr.len() - 1
    }
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
    fn range(&self, i: usize) -> Range<usize> {
        self.ptr[i]..self.ptr[i + 1]
    }
    pub fn row(&self, i: usize) -> Row<'_> {
        let range = self.range(i);
        Row(&self.data[range])
    }
    pub fn rows(&self) -> impl IndexedParallelIterator<Item = Row<'_>> {
        ParallelRowsIterator::new(self)
    }
    pub fn row_chunks(
        &self,
        chunk_size: usize,
    ) -> impl IndexedParallelIterator<Item = RowsIterator<'_>> {
        let mut n_chunks = self.n() / chunk_size;
        if n_chunks * chunk_size < self.n() {
            n_chunks += 1;
        }
        (0..n_chunks).into_par_iter().map(move |i_chunk| {
            let start = i_chunk * chunk_size;
            let end = (start + chunk_size).min(self.n());
            RowsIterator::new_range(self, start, end)
        })
    }
    pub fn seq_rows(&self) -> impl ExactSizeIterator<Item = Row<'_>> {
        RowsIterator::new(self)
    }
    pub fn row_mut(&mut self, i: usize) -> RowMut<'_> {
        let range = self.range(i);
        RowMut(&mut self.data[range])
    }
    pub fn rows_mut(&mut self) -> impl IndexedParallelIterator<Item = RowMut<'_>> {
        ParallelRowsMutIterator::new(self)
    }
    pub fn seq_rows_mut(&mut self) -> impl ExactSizeIterator<Item = RowMut<'_>> {
        RowsMutIterator::new(self)
    }
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        *(self.row_mut(i).get(j).unwrap()) = v;
    }
    pub fn diag(&self) -> Vec<f64> {
        self.rows()
            .enumerate()
            .map(|(i, row)| *row.get(i).unwrap())
            .collect::<Vec<_>>()
    }
    pub fn mult(&self, b: &[f64]) -> Vec<f64> {
        let mut res = vec![0.0; self.n()];
        self.rows()
            .zip(res.par_iter_mut())
            .for_each(|(row, y)| *y = row.mult(b));
        res
    }
    pub fn seq_mult(&self, b: &[f64]) -> Vec<f64> {
        let mut res = vec![0.0; self.n()];
        self.seq_rows()
            .zip(res.iter_mut())
            .for_each(|(row, y)| *y = row.mult(b));
        res
    }
    pub fn mult_chunks(&self, b: &[f64], chunk_size: usize) -> Vec<f64> {
        let mut res = vec![0.0; self.n()];
        self.row_chunks(chunk_size)
            .zip(res.par_chunks_mut(chunk_size))
            .for_each(|(rows, res)| {
                rows.zip(res.iter_mut())
                    .for_each(|(row, y)| *y = row.mult(b))
            });
        res
    }
    pub fn residual(&self, rhs: &[f64], b: &[f64]) -> f64 {
        self.rows()
            .zip(rhs.par_iter())
            .map(|(row, rhs)| {
                let tmp = row.mult(b) - rhs;
                tmp * tmp
            })
            .sum::<f64>()
            .sqrt()
    }
    pub fn l2_norm(b: &[f64]) -> f64 {
        b.par_iter().map(|x| x * x).sum::<f64>().sqrt()
    }
    pub fn jacobi(&self, rhs: &[f64], b: &mut [f64], params: JacobiParams) -> (usize, f64) {
        let nrm = Self::l2_norm(rhs);
        assert!(nrm > f64::EPSILON);
        let tol = params.abs_tol.min(params.rel_tol * nrm);
        let mut res = 0.0;

        let mut diag = self.diag();
        diag.par_iter_mut().for_each(|x| *x = 1.0 / *x);

        let mut tmp = vec![0.0; self.n()];
        for iter in 0..params.max_iter {
            self.rows()
                .enumerate()
                .zip(tmp.par_iter_mut())
                .for_each(|((i_row, row), x)| {
                    *x = rhs[i_row];
                    *x -= row
                        .iter()
                        .filter(|&(j, _)| *j != i_row)
                        .fold(0.0, |x, (j, v)| x + v * b[*j]);
                });
            b.par_iter_mut()
                .zip(tmp.par_iter())
                .zip(diag.par_iter())
                .for_each(|((b, tmp), d)| *b = tmp * d);

            res = self.residual(rhs, b);
            if res < tol {
                return (iter + 1, res);
            }
        }
        (params.max_iter, res)
    }
}

#[cfg(test)]
mod tests {
    use crate::{JacobiParams, SparseBlockMat};
    use rayon::iter::ParallelIterator;

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
                        assert_eq!(v, -144.0);
                    } else {
                        assert_eq!(v, 36.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_mult() {
        let n = 100;
        let edges = (0..(n - 1)).map(|i| [i, i + 1]);
        let mut mat = SparseBlockMat::from_edges(n, edges, false);
        assert_eq!(mat.n(), n);
        assert_eq!(mat.nnz(), 2 * (n - 2) + 2);

        mat.rows_mut().for_each(|mut row| {
            row.iter_mut().for_each(|(_, v)| *v = 0.5);
        });
        mat.set(0, 1, 1.0);
        mat.set(n - 1, n - 2, 1.0);

        let d = 1.0 / (n as f64 - 1.0);
        let x = |i: usize| d * i as f64;

        let f = (0..n).map(|i| x(i).sin()).collect::<Vec<_>>();

        let g = mat.mult(&f);

        assert_eq!(g[0], f[1]);
        assert_eq!(g[n - 1], f[n - 2]);
        for (i, &v) in g.iter().enumerate().skip(1).take(n - 2) {
            let tmp = 0.5 * (x(i - 1).sin() + x(i + 1).sin());
            assert_eq!(tmp, v);
        }
    }

    #[test]
    fn test_chunks() {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mat = get_laplacian_2d(5, 5);
        assert_eq!(mat.row_chunks(4).map(|rows| rows.len()).sum::<usize>(), 25);

        let mut rng = StdRng::seed_from_u64(1234);

        let x = (0..mat.n())
            .map(|_| rng.gen::<f64>() - 0.5)
            .collect::<Vec<_>>();

        let y = mat.mult(&x);
        let y_chunks = mat.mult_chunks(&x, 4);

        let err = y
            .iter()
            .zip(y_chunks.iter())
            .map(|(y, y_chunks)| (y - y_chunks).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-12 * SparseBlockMat::l2_norm(&y));
    }

    #[test]
    fn test_residual() {
        let mat = get_laplacian_2d(5, 5);
        let b = vec![1.0; mat.n()];
        let rhs = mat.mult(&b);
        let res = mat.residual(&rhs, &b);
        println!("res = {res}");
        assert!(res < 1e-12);
    }

    #[test]
    fn test_jacobi() {
        let mat = get_laplacian_2d(16, 16);
        let rhs = vec![1.0; mat.n()];
        let mut b = vec![0.0; mat.n()];

        let params = JacobiParams {
            max_iter: 2000,
            rel_tol: 1e-4,
            abs_tol: 1.0,
        };
        let (niter, residual) = mat.jacobi(&rhs, &mut b, params);
        assert!(niter < params.max_iter);
        assert!(residual < params.rel_tol * SparseBlockMat::l2_norm(&rhs));

        let residual = mat.residual(&rhs, &b);
        assert!(residual < params.rel_tol * SparseBlockMat::l2_norm(&rhs));
    }
}
