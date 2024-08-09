use crate::{Error, Result, SparseMatF64};
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

#[derive(Debug)]
pub enum MTXDataType {
    RealSymmetricMatrix,
    RealGeneralMatrix,
    Array,
}

pub struct MTXReader {
    reader: BufReader<File>,
    dtype: MTXDataType,
}

impl MTXReader {
    pub fn new(fname: &str) -> Result<Self> {
        if !fname.ends_with(".mtx") {
            return Err(Error::from(&format!("Invalid file extension for {fname}")));
        }
        let file = File::open(fname)?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let header = line.trim();
        if !header.starts_with("%%MatrixMarket") {
            return Err(Error::from(&format!("Invalid header : {header}")));
        }
        let kwds = header.split(' ').skip(1).collect::<Vec<_>>();
        if kwds[0] != "matrix" {
            unimplemented!("type not implemented: {header}");
        }

        let dtype = if kwds[1] == "coordinate" && kwds[2] == "real" && kwds[3] == "symmetric" {
            MTXDataType::RealSymmetricMatrix
        } else if kwds[1] == "coordinate" && kwds[2] == "real" && kwds[3] == "general" {
            MTXDataType::RealGeneralMatrix
        } else if kwds[1] == "array" && kwds[2] == "real" && kwds[3] == "general" {
            MTXDataType::Array
        } else {
            unimplemented!("type not implemented: {header}");
        };
        Ok(Self { reader, dtype })
    }

    pub fn read_array(&mut self) -> Result<(Vec<usize>, Vec<f64>)> {
        match self.dtype {
            MTXDataType::Array => {
                let mut line = String::new();
                self.reader.read_line(&mut line)?;
                let trimmed_line = line.trim();
                let shape = trimmed_line
                    .split(' ')
                    .map(|x| x.parse().unwrap())
                    .collect::<Vec<usize>>();
                let n = shape.iter().product();
                let mut res = Vec::with_capacity(n);
                for _ in 0..n {
                    line.clear();
                    self.reader.read_line(&mut line)?;
                    let trimmed_line = line.trim();
                    res.push(trimmed_line.parse().unwrap());
                }
                Ok((shape, res))
            }
            _ => Err(Error::from(&format!(
                "Cannot read an array: dtype = {:?}",
                self.dtype
            ))),
        }
    }
    pub fn read_matrix(&mut self) -> Result<SparseMatF64> {
        match self.dtype {
            MTXDataType::RealGeneralMatrix | MTXDataType::RealSymmetricMatrix => {
                let mut line = String::new();
                self.reader.read_line(&mut line)?;
                let trimmed_line = line.trim();
                let shape = trimmed_line
                    .split(' ')
                    .map(|x| x.parse().unwrap())
                    .collect::<Vec<usize>>();
                assert_eq!(shape.len(), 3);
                assert_eq!(shape[0], shape[1]);
                let n = shape[0];
                let nnz = shape[2];

                let mut ij = Vec::with_capacity(nnz);
                let mut vals = Vec::with_capacity(nnz);
                for _ in 0..nnz {
                    line.clear();
                    self.reader.read_line(&mut line)?;
                    let trimmed_line = line.trim();
                    let split = trimmed_line.split(' ');
                    let split = split.filter(|x| !x.is_empty()).collect::<Vec<_>>();
                    let i = split[0].parse().unwrap();
                    let j = split[1].parse().unwrap();
                    ij.push([i, j]);
                    let v = split[2].parse().unwrap();
                    vals.push(v);
                }

                Ok(SparseMatF64::from_ij(
                    n,
                    ij.iter().copied(),
                    vals.iter().copied(),
                )?)
            }
            _ => Err(Error::from(&format!(
                "Cannot read a matrix: dtype = {:?}",
                self.dtype
            ))),
        }
    }
}

pub struct MTXWriter {
    writer: BufWriter<File>,
}

impl MTXWriter {
    pub fn new(fname: &str) -> Result<Self> {
        if !fname.ends_with(".mtx") {
            return Err(Error::from(&format!("Invalid file extension for {fname}")));
        }

        let file = File::create(fname)?;

        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn write_array(mut self, shape: &[usize], data: &[f64]) -> Result<()> {
        if shape.iter().product::<usize>() != data.len() {
            return Err(Error::from("Invalid shape"));
        }

        writeln!(self.writer, "%%MatrixMarket matrix array real general")?;
        let mut line = format!("{} ", shape[0]);
        for s in shape.iter().skip(1) {
            line += &format!("{} ", s);
        }
        writeln!(self.writer, "{}", line)?;

        for v in data.iter() {
            writeln!(self.writer, "{v}")?;
        }
        Ok(())
    }

    pub fn write_matrix(mut self, mat: &SparseMatF64) -> Result<()> {
        writeln!(self.writer, "%%MatrixMarket matrix coordinate real general")?;
        writeln!(self.writer, "{} {} {}", mat.n(), mat.n(), mat.nnz())?;

        for (i, row) in mat.seq_rows().enumerate() {
            for (j, v) in row.iter() {
                writeln!(self.writer, "{i} {j} {v}")?;
            }
        }
        Ok(())
    }
}
