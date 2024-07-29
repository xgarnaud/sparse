use crate::{MatVec, Row, RowMut, SparseBlockMat};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

pub struct RowsIterator<'a, T: MatVec> {
    ptr: &'a [usize],
    data: &'a [(usize, T::Mat)],
}

impl<'a, T: MatVec> RowsIterator<'a, T> {
    pub fn new(mat: &'a SparseBlockMat<T>) -> Self {
        Self {
            ptr: &mat.ptr[..mat.n()],
            data: &mat.data,
        }
    }

    pub fn new_range(mat: &'a SparseBlockMat<T>, start: usize, end: usize) -> Self {
        Self {
            ptr: &mat.ptr[start..end],
            data: &mat.data[mat.ptr[start]..mat.ptr[end]],
        }
    }

    fn length(&self) -> usize {
        self.ptr.len()
    }

    fn empty(&self) -> bool {
        self.ptr.is_empty()
    }

    fn n(&self, i: usize) -> usize {
        let start = self.ptr[i] - self.ptr[0];
        let end = if i + 1 < self.ptr.len() {
            self.ptr[i + 1] - self.ptr[0]
        } else {
            self.data.len()
        };
        end - start
    }
}

impl<'a, T: MatVec> Iterator for RowsIterator<'a, T> {
    type Item = Row<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.empty() {
            return None;
        }
        let (head, tail) = self.data.split_at(self.n(0));
        self.ptr = &self.ptr[1..];
        self.data = tail;

        Some(Row(head))
    }
}

impl<'a, T: MatVec> DoubleEndedIterator for RowsIterator<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.empty() {
            return None;
        }
        let m = self.length() - 1;
        let nnz = self.data.len();
        let (head, tail) = self.data.split_at(nnz - self.n(m));
        self.ptr = &self.ptr[..m];
        self.data = head;

        Some(Row(tail))
    }
}

impl<'a, T: MatVec> ExactSizeIterator for RowsIterator<'a, T> {
    fn len(&self) -> usize {
        self.length()
    }
}

pub struct ParallelRowsIterator<'a, T: MatVec> {
    ptr: &'a [usize],
    data: &'a [(usize, T::Mat)],
}

impl<'a, T: MatVec> ParallelRowsIterator<'a, T> {
    pub fn new(mat: &'a SparseBlockMat<T>) -> Self {
        Self {
            ptr: &mat.ptr[..mat.n()],
            data: &mat.data,
        }
    }
}

impl<'a, T: MatVec> ParallelIterator for ParallelRowsIterator<'a, T> {
    type Item = Row<'a, T>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.ptr.len())
    }
}

impl<'a, T: MatVec> IndexedParallelIterator for ParallelRowsIterator<'a, T> {
    fn len(&self) -> usize {
        self.ptr.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        callback.callback(ParallelRowProducer {
            ptr: self.ptr,
            data: self.data,
        })
    }
}

pub struct ParallelRowProducer<'a, T: MatVec> {
    ptr: &'a [usize],
    data: &'a [(usize, T::Mat)],
}

impl<'a, T: MatVec> rayon::iter::plumbing::Producer for ParallelRowProducer<'a, T> {
    type Item = Row<'a, T>;
    type IntoIter = RowsIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        RowsIterator {
            ptr: self.ptr,
            data: self.data,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let n = self.ptr[index] - self.ptr[0];
        let (ptr0, ptr1) = self.ptr.split_at(index);
        let (data0, data1) = self.data.split_at(n);

        (
            Self {
                ptr: ptr0,
                data: data0,
            },
            Self {
                ptr: ptr1,
                data: data1,
            },
        )
    }
}

pub struct RowsMutIterator<'a, T: MatVec> {
    ptr: &'a [usize],
    data: *mut [(usize, T::Mat)],
}

impl<'a, T: MatVec> RowsMutIterator<'a, T> {
    pub fn new(mat: &'a mut SparseBlockMat<T>) -> Self {
        Self {
            ptr: &mat.ptr[..mat.n()],
            data: mat.data.as_mut_slice(),
        }
    }

    fn length(&self) -> usize {
        self.ptr.len()
    }

    fn empty(&self) -> bool {
        self.ptr.is_empty()
    }

    fn n(&self, i: usize) -> usize {
        let start = self.ptr[i] - self.ptr[0];
        let end = if i + 1 < self.ptr.len() {
            self.ptr[i + 1] - self.ptr[0]
        } else {
            self.data.len()
        };
        end - start
    }
}

impl<'a, T: MatVec> Iterator for RowsMutIterator<'a, T>
where
    <T as MatVec>::Mat: 'a,
{
    type Item = RowMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.empty() {
            return None;
        }
        let (head, tail) = unsafe { self.data.split_at_mut(self.n(0)) };
        self.ptr = &self.ptr[1..];
        self.data = tail;

        Some(RowMut(unsafe { &mut *head }))
    }
}

impl<'a, T: MatVec> DoubleEndedIterator for RowsMutIterator<'a, T>
where
    <T as MatVec>::Mat: 'a,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.empty() {
            return None;
        }
        let m = self.length() - 1;
        let nnz = self.data.len();
        let (head, tail) = unsafe { self.data.split_at_mut(nnz - self.n(m)) };
        self.ptr = &self.ptr[1..];
        self.data = head;

        Some(RowMut(unsafe { &mut *tail }))
    }
}

impl<'a, T: MatVec> ExactSizeIterator for RowsMutIterator<'a, T>
where
    <T as MatVec>::Mat: 'a,
{
    fn len(&self) -> usize {
        self.length()
    }
}

pub struct ParallelRowsMutIterator<'a, T: MatVec>
where
    <T as MatVec>::Mat: 'a,
{
    ptr: &'a [usize],
    data: &'a mut [(usize, T::Mat)],
}

impl<'a, T: MatVec> ParallelRowsMutIterator<'a, T>
where
    <T as MatVec>::Mat: 'a,
{
    pub fn new(mat: &'a mut SparseBlockMat<T>) -> Self {
        Self {
            ptr: &mat.ptr[..mat.n()],
            data: mat.data.as_mut_slice(),
        }
    }
}

impl<'a, T: MatVec> ParallelIterator for ParallelRowsMutIterator<'a, T>
where
    <T as MatVec>::Mat: 'a,
{
    type Item = RowMut<'a, T>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.ptr.len())
    }
}

impl<'a, T: MatVec> IndexedParallelIterator for ParallelRowsMutIterator<'a, T>
where
    <T as MatVec>::Mat: 'a,
{
    fn len(&self) -> usize {
        self.ptr.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        callback.callback(ParallelRowMutProducer {
            ptr: self.ptr,
            data: self.data,
        })
    }
}

pub struct ParallelRowMutProducer<'a, T: MatVec>
where
    <T as MatVec>::Mat: 'a,
{
    ptr: &'a [usize],
    data: &'a mut [(usize, T::Mat)],
}

impl<'a, T: MatVec> rayon::iter::plumbing::Producer for ParallelRowMutProducer<'a, T>
where
    <T as MatVec>::Mat: 'a,
{
    type Item = RowMut<'a, T>;
    type IntoIter = RowsMutIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        RowsMutIterator {
            ptr: self.ptr,
            data: self.data,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let n = self.ptr[index] - self.ptr[0];
        let (ptr0, ptr1) = self.ptr.split_at(index);
        let (data0, data1) = self.data.split_at_mut(n);

        (
            Self {
                ptr: ptr0,
                data: data0,
            },
            Self {
                ptr: ptr1,
                data: data1,
            },
        )
    }
}

// pub struct SparseBlockMatRowIterator<'a>(&'a SparseBlockMat);

// pub struct RowsMutIterator<'a> {
//     ptr: &'a [usize],
//     data: *mut [(usize, f64)],
// }

// impl<'a> RowsMutIterator<'a> {
//     pub fn new(mat: &'a mut SparseBlockMat) -> Self {
//         Self {
//             ptr: &mat.ptr,
//             data: mat.data.as_mut_slice(),
//         }
//     }
// }

// impl<'a> Iterator for RowsMutIterator<'a> {
//     type Item = RowMut<'a>;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.ptr.len() < 2 {
//             return None;
//         }
//         let n = self.ptr[1] - self.ptr[0];
//         let (head, tail) = unsafe { self.data.split_at_mut(n) };
//         self.ptr = &self.ptr[1..];
//         self.data = tail;

//         Some(RowMut(unsafe { &mut *head }))
//     }
// }

// impl<'a> ExactSizeIterator for RowsMutIterator<'a> {
//     fn len(&self) -> usize {
//         self.ptr.len() - 1
//     }
// }
