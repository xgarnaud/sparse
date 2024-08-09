#![feature(raw_slice_split)]
use core::fmt;
use iterators::{ParallelRowsIterator, ParallelRowsMutIterator, RowsIterator, RowsMutIterator};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::ParallelSliceMut;
use std::iter::repeat;
use std::ops::{Add, AddAssign, Sub, SubAssign};
use std::{fmt::Display, ops::Range};

pub mod iterators;
pub mod matrix_market;
#[cfg(feature = "nalgebra")]
pub mod nalgebra;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
pub struct Error(String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.0)
    }
}
impl std::error::Error for Error {}
impl Error {
    #[must_use]
    pub fn from(msg: &str) -> Box<Self> {
        Box::new(Self(msg.into()))
    }
}

pub trait MatVec {
    type Mat: Send
        + Sync
        + Display
        + Clone
        + Copy
        + AddAssign
        + SubAssign
        + Add<Self::Mat, Output = Self::Mat>
        + Sub<Self::Mat, Output = Self::Mat>;
    type Vect: Send
        + Sync
        + Display
        + Clone
        + Copy
        + AddAssign
        + SubAssign
        + Add<Self::Vect, Output = Self::Vect>
        + Sub<Self::Vect, Output = Self::Vect>;
    fn mat_zero() -> Self::Mat;
    fn vect_zero() -> Self::Vect;
    fn inverse(mat: &Self::Mat) -> Self::Mat;
    fn mult(mat: &Self::Mat, vec: &Self::Vect) -> Self::Vect;
    fn norm2(vec: &Self::Vect) -> f64;
}

impl MatVec for f64 {
    type Mat = f64;
    type Vect = f64;
    fn mat_zero() -> Self::Mat {
        0.0
    }
    fn vect_zero() -> Self::Vect {
        0.0
    }
    fn inverse(mat: &Self::Mat) -> Self::Mat {
        1.0 / mat
    }
    fn mult(mat: &Self::Mat, vec: &Self::Vect) -> Self::Vect {
        mat * vec
    }
    fn norm2(vec: &Self::Vect) -> f64 {
        vec * vec
    }
}

pub struct Row<'a, T: MatVec>(&'a [(usize, T::Mat)]);

impl<'a, T: MatVec> Row<'a, T> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &(usize, T::Mat)> {
        self.0.iter()
    }
    pub fn get(&self, j: usize) -> Option<&T::Mat> {
        let idx = self.0.binary_search_by(|x| x.0.cmp(&j));
        if let Ok(idx) = idx {
            Some(&self.0[idx].1)
        } else {
            None
        }
    }
    pub fn mult(&self, b: &[T::Vect]) -> T::Vect {
        self.0
            .iter()
            .fold(T::vect_zero(), |x, (j, v)| x + T::mult(v, &b[*j]))
    }
}

impl<'a, T: MatVec> Display for Row<'a, T> {
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

pub struct RowMut<'a, T: MatVec>(&'a mut [(usize, T::Mat)]);

impl<'a, T: MatVec> RowMut<'a, T> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut (usize, T::Mat)> {
        self.0.iter_mut()
    }
    pub fn get(&mut self, j: usize) -> Option<&mut T::Mat> {
        let idx = self.0.binary_search_by(|x| x.0.cmp(&j));
        if let Ok(idx) = idx {
            Some(&mut self.0[idx].1)
        } else {
            None
        }
    }
    pub fn get_mut(&mut self, j: usize) -> Option<&mut T::Mat> {
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
        self.iter_mut().for_each(|x| x.1 = T::mat_zero());
    }
}

impl<'a, T: MatVec> Display for RowMut<'a, T> {
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

#[derive(Clone, Copy, Debug)]
pub struct IterativeParams {
    pub max_iter: usize,
    pub rel_tol: f64,
    pub abs_tol: f64,
}

impl Default for IterativeParams {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rel_tol: 1e-6,
            abs_tol: 1e-12,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum IterativeType {
    Jacobi,
    Sgs,
    SgsSeq,
}

pub struct SparseMat<T: MatVec> {
    ptr: Vec<usize>,
    data: Vec<(usize, T::Mat)>,
}

impl<T: MatVec> SparseMat<T> {
    pub fn from_ij<I1: Iterator<Item = [usize; 2]> + Clone, I2: Iterator<Item = T::Mat> + Clone>(
        n: usize,
        ij: I1,
        vals: I2,
    ) -> Result<Self> {
        let mut ptr = vec![0; n + 1];

        ptr[0] = 0;
        for [i, _] in ij.clone() {
            ptr[i + 1] += 1;
        }

        for i in 0..n {
            ptr[i + 1] += ptr[i];
        }
        let nnz = ptr[n];
        let mut data = vec![(usize::MAX, T::mat_zero()); nnz];

        for ([i, j], val) in ij.zip(vals) {
            #[allow(clippy::needless_range_loop)]
            for idx in ptr[i]..ptr[i + 1] {
                if data[idx].0 == j {
                    return Err(Error::from("Entry ({i},{j}) already present"));
                } else if data[idx].0 == usize::MAX {
                    data[idx] = (j, val);
                    break;
                }
            }
        }

        let mut res = Self { ptr, data };

        for i in 0..n {
            res.row_mut(i).sort();
        }

        Ok(res)
    }

    pub fn from_edges<I: Iterator<Item = [usize; 2]> + Clone>(
        n: usize,
        edges: I,
        with_diagonal: bool,
    ) -> Result<Self> {
        if with_diagonal {
            Self::from_ij(
                n,
                edges
                    .clone()
                    .chain(edges.map(|[i, j]| [j, i]))
                    .chain((0..n).map(|i| [i, i])),
                repeat(T::mat_zero()),
            )
        } else {
            Self::from_ij(
                n,
                edges.clone().chain(edges.map(|[i, j]| [j, i])),
                repeat(T::mat_zero()),
            )
        }
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
    pub fn row(&self, i: usize) -> Row<'_, T> {
        let range = self.range(i);
        Row(&self.data[range])
    }
    pub fn rows(&self) -> impl IndexedParallelIterator<Item = Row<'_, T>> {
        ParallelRowsIterator::new(self)
    }
    pub fn row_chunks(
        &self,
        chunk_size: usize,
    ) -> impl IndexedParallelIterator<Item = ((usize, usize), RowsIterator<'_, T>)> {
        let mut n_chunks = self.n() / chunk_size;
        if n_chunks * chunk_size < self.n() {
            n_chunks += 1;
        }
        (0..n_chunks).into_par_iter().map(move |i_chunk| {
            let start = i_chunk * chunk_size;
            let end = (start + chunk_size).min(self.n());
            ((start, end), RowsIterator::new_range(self, start, end))
        })
    }
    pub fn seq_rows(
        &self,
    ) -> impl ExactSizeIterator<Item = Row<'_, T>> + DoubleEndedIterator<Item = Row<'_, T>> {
        RowsIterator::new(self)
    }
    pub fn row_mut(&mut self, i: usize) -> RowMut<'_, T> {
        let range = self.range(i);
        RowMut(&mut self.data[range])
    }
    pub fn rows_mut(&mut self) -> impl IndexedParallelIterator<Item = RowMut<'_, T>> {
        ParallelRowsMutIterator::new(self)
    }
    pub fn seq_rows_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = RowMut<'_, T>> + DoubleEndedIterator<Item = RowMut<'_, T>>
    {
        RowsMutIterator::new(self)
    }
    pub fn set(&mut self, i: usize, j: usize, v: T::Mat) {
        *(self.row_mut(i).get(j).unwrap()) = v;
    }
    pub fn diag(&self) -> Vec<T::Mat> {
        self.rows()
            .enumerate()
            .map(|(i, row)| *row.get(i).unwrap())
            .collect::<Vec<_>>()
    }
    pub fn mult(&self, b: &[T::Vect]) -> Vec<T::Vect> {
        let mut res = vec![T::vect_zero(); self.n()];
        self.rows()
            .zip(res.par_iter_mut())
            .for_each(|(row, y)| *y = row.mult(b));
        res
    }
    pub fn seq_mult(&self, b: &[T::Vect]) -> Vec<T::Vect> {
        let mut res = vec![T::vect_zero(); self.n()];
        self.seq_rows()
            .zip(res.iter_mut())
            .for_each(|(row, y)| *y = row.mult(b));
        res
    }
    pub fn mult_chunks(&self, b: &[T::Vect], chunk_size: usize) -> Vec<T::Vect> {
        let mut res = vec![T::vect_zero(); self.n()];
        self.row_chunks(chunk_size)
            .zip(res.par_chunks_mut(chunk_size))
            .for_each(|((_, rows), res)| {
                rows.zip(res.iter_mut())
                    .for_each(|(row, y)| *y = row.mult(b))
            });
        res
    }
    pub fn residual(&self, rhs: &[T::Vect], b: &[T::Vect]) -> f64 {
        self.rows()
            .zip(rhs.par_iter())
            .map(|(row, rhs)| {
                let tmp = row.mult(b) - *rhs;
                T::norm2(&tmp)
            })
            .sum::<f64>()
            .sqrt()
    }
    pub fn l2_norm(b: &[T::Vect]) -> f64 {
        b.par_iter().map(|x| T::norm2(x)).sum::<f64>().sqrt()
    }
    pub fn jacobi(
        &self,
        rhs: &[T::Vect],
        b: &mut [T::Vect],
        params: IterativeParams,
    ) -> (usize, f64) {
        let nrm = Self::l2_norm(rhs);
        assert!(nrm > f64::EPSILON);
        let tol = params.abs_tol.min(params.rel_tol * nrm);
        let mut res = 0.0;

        let mut diag = self.diag();
        diag.par_iter_mut().for_each(|x| *x = T::inverse(x));

        let mut tmp = vec![T::vect_zero(); self.n()];
        for iter in 0..params.max_iter {
            self.rows()
                .enumerate()
                .zip(tmp.par_iter_mut())
                .for_each(|((i_row, row), x)| {
                    *x = rhs[i_row];
                    *x -= row
                        .iter()
                        .filter(|&(j, _)| *j != i_row)
                        .fold(T::vect_zero(), |x, (j, v)| x + T::mult(v, &b[*j]));
                });
            b.par_iter_mut()
                .zip(tmp.par_iter())
                .zip(diag.par_iter())
                .for_each(|((b, tmp), d)| *b = T::mult(d, tmp));

            res = self.residual(rhs, b);
            if res < tol {
                return (iter + 1, res);
            }
        }
        (params.max_iter, res)
    }
    pub fn sgs(
        &self,
        rhs: &[T::Vect],
        b: &mut [T::Vect],
        params: IterativeParams,
        chunk_size: usize,
    ) -> (usize, f64) {
        let nrm = Self::l2_norm(rhs);
        assert!(nrm > f64::EPSILON);
        let tol = params.abs_tol.min(params.rel_tol * nrm);
        let mut res = 0.0;

        let mut diag = self.diag();
        diag.par_iter_mut().for_each(|x| *x = T::inverse(x));

        for iter in 0..params.max_iter {
            let b_copy = b.to_vec();
            self.row_chunks(chunk_size)
                .zip(b.par_chunks_mut(chunk_size))
                .for_each(|(((start, end), rows), b)| {
                    rows.enumerate().for_each(|(i_row, row)| {
                        let i_row = i_row + start;
                        let mut tmp = rhs[i_row];
                        tmp -= row
                            .iter()
                            .filter(|&(j, _)| *j != i_row && *j >= start && *j < end)
                            .fold(T::vect_zero(), |x, (j, v)| x + T::mult(v, &b[*j - start]));
                        tmp -= row
                            .iter()
                            .filter(|&(j, _)| *j < start || *j >= end)
                            .fold(T::vect_zero(), |x, (j, v)| x + T::mult(v, &b_copy[*j]));
                        b[i_row - start] = T::mult(&diag[i_row], &tmp);
                    });
                });
            self.row_chunks(chunk_size)
                .zip(b.par_chunks_mut(chunk_size))
                .for_each(|(((start, end), rows), b)| {
                    rows.enumerate().rev().for_each(|(i_row, row)| {
                        let i_row = i_row + start;
                        let mut tmp = rhs[i_row];
                        tmp -= row
                            .iter()
                            .filter(|&(j, _)| *j != i_row && *j >= start && *j < end)
                            .fold(T::vect_zero(), |x, (j, v)| x + T::mult(v, &b[*j - start]));
                        tmp -= row
                            .iter()
                            .filter(|&(j, _)| *j < start || *j >= end)
                            .fold(T::vect_zero(), |x, (j, v)| x + T::mult(v, &b_copy[*j]));
                        b[i_row - start] = T::mult(&diag[i_row], &tmp);
                    });
                });

            res = self.residual(rhs, b);
            if res < tol {
                return (iter + 1, res);
            }
        }
        (params.max_iter, res)
    }
    pub fn seq_sgs(
        &self,
        rhs: &[T::Vect],
        b: &mut [T::Vect],
        params: IterativeParams,
    ) -> (usize, f64) {
        let nrm = Self::l2_norm(rhs);
        assert!(nrm > f64::EPSILON);
        let tol = params.abs_tol.min(params.rel_tol * nrm);
        let mut res = 0.0;

        let mut diag = self.diag();
        diag.par_iter_mut().for_each(|x| *x = T::inverse(x));

        for iter in 0..params.max_iter {
            self.seq_rows().enumerate().for_each(|(i_row, row)| {
                let mut tmp = rhs[i_row];
                tmp -= row
                    .iter()
                    .filter(|&(j, _)| *j != i_row)
                    .fold(T::vect_zero(), |x, (j, v)| x + T::mult(v, &b[*j]));
                b[i_row] = T::mult(&diag[i_row], &tmp);
            });
            self.seq_rows().enumerate().rev().for_each(|(i_row, row)| {
                let mut tmp = rhs[i_row];
                tmp -= row
                    .iter()
                    .filter(|&(j, _)| *j != i_row)
                    .fold(T::vect_zero(), |x, (j, v)| x + T::mult(v, &b[*j]));
                b[i_row] = T::mult(&diag[i_row], &tmp);
            });
            res = self.residual(rhs, b);
            if res < tol {
                return (iter + 1, res);
            }
        }
        (params.max_iter, res)
    }
    pub fn solve_iterative(
        &self,
        rhs: &[T::Vect],
        b: &mut [T::Vect],
        params: IterativeParams,
        solver_type: IterativeType,
    ) -> (usize, f64) {
        match solver_type {
            IterativeType::Jacobi => self.jacobi(rhs, b, params),
            IterativeType::Sgs => {
                let chunk_size = self.n() / rayon::current_num_threads();
                self.sgs(rhs, b, params, chunk_size)
            }
            IterativeType::SgsSeq => self.seq_sgs(rhs, b, params),
        }
    }
}

pub type SparseMatF64 = SparseMat<f64>;

#[cfg(test)]
mod tests {
    use crate::{
        matrix_market::{MTXReader, MTXWriter},
        IterativeParams, Result, SparseMat, SparseMatF64,
    };
    use rayon::iter::ParallelIterator;
    use tempfile::NamedTempFile;

    fn get_laplacian_2d(ni: usize, nj: usize) -> SparseMatF64 {
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
        let mut mat = SparseMat::from_edges(ni * nj, edgs.iter().copied(), true).unwrap();
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
        let mut mat = SparseMatF64::from_edges(n, edges, false).unwrap();
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
        assert_eq!(
            mat.row_chunks(4).map(|(_, rows)| rows.len()).sum::<usize>(),
            25
        );

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
        assert!(err < 1e-12 * SparseMatF64::l2_norm(&y));
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

        let params = IterativeParams {
            max_iter: 2000,
            rel_tol: 1e-4,
            abs_tol: 1.0,
        };
        let (niter, residual) = mat.jacobi(&rhs, &mut b, params);
        assert!(niter < params.max_iter);
        assert!(residual < params.rel_tol * SparseMatF64::l2_norm(&rhs));

        let residual = mat.residual(&rhs, &b);
        assert!(residual < params.rel_tol * SparseMatF64::l2_norm(&rhs));
    }

    #[test]
    fn test_seq_sgs() {
        let mat = get_laplacian_2d(16, 16);
        let rhs = vec![1.0; mat.n()];

        let params = IterativeParams {
            max_iter: 200,
            rel_tol: 2.5e-4,
            abs_tol: 1.0,
        };

        let mut b = vec![0.0; mat.n()];
        let (niter, residual_jac) = mat.jacobi(&rhs, &mut b, params);
        assert_eq!(niter, params.max_iter);

        let mut b = vec![0.0; mat.n()];
        let (niter, residual_sgs) = mat.seq_sgs(&rhs, &mut b, params);
        assert!(niter < params.max_iter);
        assert!(residual_sgs < residual_jac * 0.01);
    }

    #[test]
    fn test_sgs() {
        let mat = get_laplacian_2d(16, 16);
        let rhs = vec![1.0; mat.n()];

        let params = IterativeParams {
            max_iter: 200,
            rel_tol: 2.5e-4,
            abs_tol: 1.0,
        };

        let mut b = vec![0.0; mat.n()];
        let (niter_seq_sgs, residual_seq_sgs) = mat.seq_sgs(&rhs, &mut b, params);
        assert!(niter_seq_sgs < params.max_iter);

        let mut b = vec![0.0; mat.n()];
        let (niter_sgs_1, residual_sgs_1) = mat.sgs(&rhs, &mut b, params, 16 * 16);
        assert!(niter_sgs_1 < params.max_iter);
        assert_eq!(niter_sgs_1, niter_seq_sgs);
        assert_eq!(residual_sgs_1, residual_seq_sgs);

        let mut b = vec![0.0; mat.n()];
        let (niter_sgs_2, residual_sgs_2) = mat.sgs(&rhs, &mut b, params, 16 * 16 / 5);
        assert!(niter_sgs_2 < params.max_iter);
        assert!(niter_sgs_2 > niter_seq_sgs);
        assert!(residual_sgs_2 > residual_seq_sgs);
        assert!(residual_sgs_2 < residual_seq_sgs * 2.0);
    }

    #[test]
    fn test_io() -> Result<()> {
        let mat = get_laplacian_2d(5, 5);

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".mtx";
        let writer = MTXWriter::new(&fname)?;
        writer.write_matrix(&mat)?;

        let mut reader = MTXReader::new(&fname)?;
        let mat1 = reader.read_matrix()?;

        assert_eq!(mat.n(), mat1.n());
        assert_eq!(mat.nnz(), mat1.nnz());

        for (row, row1) in mat.seq_rows().zip(mat1.seq_rows()) {
            assert_eq!(row.len(), row1.len());
            for ((j, v), (j1, v1)) in row.iter().zip(row1.iter()) {
                assert_eq!(j, j1);
                assert_eq!(v, v1);
            }
        }

        Ok(())
    }
}
