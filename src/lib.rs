#![feature(raw_slice_split)]
use std::{fmt::Display, ops::Range};

use iterators::{ParallelRowsIterator, ParallelRowsMutIterator, RowsIterator, RowsMutIterator};
use rayon::iter::IndexedParallelIterator;
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

impl SparseBlockMat {
    pub fn new<I: Iterator<Item = [usize; 2]> + Clone>(n_verts: usize, edges: I) -> Self {
        let mut ptr = vec![1; n_verts + 1];
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
        ptr.iter().take(n_verts).enumerate().for_each(|(i, &idx)| {
            data[idx].0 = i;
        });

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
}

#[cfg(test)]
mod tests {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    use crate::SparseBlockMat;

    fn get_laplacian_2d(ni: usize, nj: usize) -> SparseBlockMat {
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
        SparseBlockMat::new(ni * nj, edgs.iter().copied())
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
        let mut mat = get_laplacian_2d(5, 5);

        mat.rows_mut().enumerate().for_each(|(i, mut row)| {
            let val = row.get(i).unwrap();
            *val = 1.0;
        });

        for i in 0..mat.n() {
            let row = mat.row(i);
            for j in 0..mat.n() {
                if let Some(&v) = row.get(j) {
                    if i == j {
                        assert_eq!(v, 1.0);
                    } else {
                        assert_eq!(v, 0.0);
                    }
                }
            }
        }
    }
}
