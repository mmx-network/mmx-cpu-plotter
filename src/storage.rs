//! Temporary streams shared by the RAM and disk plotting paths.
use crate::io::Policy;
use std::{
    cell::RefCell,
    io::{self, Read, Write},
    path::Path,
    rc::Rc,
};
use tempfile::{TempDir, TempPath};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    Ram,
    Disk,
}

enum Data {
    Ram(RefCell<Vec<u8>>),
    Disk(TempPath),
}
#[derive(Clone)]
pub(crate) struct TempData(Rc<Data>, Policy);

pub(crate) struct Storage {
    directory: Option<TempDir>,
    policy: Policy,
}
impl Storage {
    pub fn new(mode: Mode, directory: &Path, direct_io: bool) -> io::Result<Self> {
        Ok(Self {
            policy: Policy::new(direct_io),
            directory: match mode {
                Mode::Ram => None,
                Mode::Disk => Some(
                    tempfile::Builder::new()
                        .prefix(".mmx-plot-")
                        .tempdir_in(directory)?,
                ),
            },
        })
    }
    pub fn is_ram(&self) -> bool {
        self.directory.is_none()
    }
    pub fn create(&self) -> io::Result<(TempData, Box<dyn Write>)> {
        self.create_sized(0)
    }
    pub fn create_sized(&self, capacity: u64) -> io::Result<(TempData, Box<dyn Write>)> {
        let data = TempData(
            Rc::new(if let Some(dir) = &self.directory {
                Data::Disk(tempfile::NamedTempFile::new_in(dir.path())?.into_temp_path())
            } else {
                let capacity = usize::try_from(capacity).map_err(io::Error::other)?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(capacity)
                    .map_err(io::Error::other)?;
                Data::Ram(RefCell::new(bytes))
            }),
            self.policy.clone(),
        );
        let writer: Box<dyn Write> = match data.0.as_ref() {
            Data::Ram(_) => Box::new(MemoryStream {
                data: data.clone(),
                position: 0,
            }),
            Data::Disk(path) => self.policy.writer(path, true)?,
        };
        Ok((data, writer))
    }
}
impl TempData {
    pub fn reader(&self) -> io::Result<Box<dyn Read>> {
        Ok(match self.0.as_ref() {
            Data::Ram(_) => Box::new(MemoryStream {
                data: self.clone(),
                position: 0,
            }),
            Data::Disk(path) => self.1.reader(path)?,
        })
    }
}
struct MemoryStream {
    data: TempData,
    position: usize,
}
impl Read for MemoryStream {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let Data::Ram(bytes) = self.data.0.as_ref() else {
            unreachable!()
        };
        let bytes = bytes.borrow();
        let n = output.len().min(bytes.len() - self.position);
        output[..n].copy_from_slice(&bytes[self.position..self.position + n]);
        self.position += n;
        Ok(n)
    }
}
impl Write for MemoryStream {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        let Data::Ram(bytes) = self.data.0.as_ref() else {
            unreachable!()
        };
        let mut bytes = bytes.borrow_mut();
        let end = self
            .position
            .checked_add(input.len())
            .ok_or_else(|| io::Error::other("temporary stream too large"))?;
        let len = bytes.len().max(end);
        bytes.resize(len, 0);
        bytes[self.position..end].copy_from_slice(input);
        self.position = end;
        Ok(input.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
