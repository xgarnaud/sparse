use crate::{Row, RowMut, SparseBlockMat};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

pub struct RowsIterator<'a> {
    ptr: &'a [usize],
    data: &'a [(usize, f64)],
}

impl<'a> RowsIterator<'a> {
    pub fn new(mat: &'a SparseBlockMat) -> Self {
        Self {
            ptr: &mat.ptr[..mat.n()],
            data: &mat.data,
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

impl<'a> Iterator for RowsIterator<'a> {
    type Item = Row<'a>;

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

impl<'a> DoubleEndedIterator for RowsIterator<'a> {
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

impl<'a> ExactSizeIterator for RowsIterator<'a> {
    fn len(&self) -> usize {
        self.length()
    }
}

pub struct ParallelRowsIterator<'a> {
    ptr: &'a [usize],
    data: &'a [(usize, f64)],
}

impl<'a> ParallelRowsIterator<'a> {
    pub fn new(mat: &'a SparseBlockMat) -> Self {
        Self {
            ptr: &mat.ptr[..mat.n()],
            data: &mat.data,
        }
    }
}

impl<'a> ParallelIterator for ParallelRowsIterator<'a> {
    type Item = Row<'a>;

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

impl<'a> IndexedParallelIterator for ParallelRowsIterator<'a> {
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

pub struct ParallelRowProducer<'a> {
    ptr: &'a [usize],
    data: &'a [(usize, f64)],
}

impl<'a> rayon::iter::plumbing::Producer for ParallelRowProducer<'a> {
    type Item = Row<'a>;
    type IntoIter = RowsIterator<'a>;

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

pub struct RowsMutIterator<'a> {
    ptr: &'a [usize],
    data: *mut [(usize, f64)],
}

impl<'a> RowsMutIterator<'a> {
    pub fn new(mat: &'a mut SparseBlockMat) -> Self {
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

impl<'a> Iterator for RowsMutIterator<'a> {
    type Item = RowMut<'a>;

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

impl<'a> DoubleEndedIterator for RowsMutIterator<'a> {
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

impl<'a> ExactSizeIterator for RowsMutIterator<'a> {
    fn len(&self) -> usize {
        self.length()
    }
}

pub struct ParallelRowsMutIterator<'a> {
    ptr: &'a [usize],
    data: &'a mut [(usize, f64)],
}

impl<'a> ParallelRowsMutIterator<'a> {
    pub fn new(mat: &'a mut SparseBlockMat) -> Self {
        Self {
            ptr: &mat.ptr[..mat.n()],
            data: mat.data.as_mut_slice(),
        }
    }
}

impl<'a> ParallelIterator for ParallelRowsMutIterator<'a> {
    type Item = RowMut<'a>;

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

impl<'a> IndexedParallelIterator for ParallelRowsMutIterator<'a> {
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

pub struct ParallelRowMutProducer<'a> {
    ptr: &'a [usize],
    data: &'a mut [(usize, f64)],
}

impl<'a> rayon::iter::plumbing::Producer for ParallelRowMutProducer<'a> {
    type Item = RowMut<'a>;
    type IntoIter = RowsMutIterator<'a>;

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
