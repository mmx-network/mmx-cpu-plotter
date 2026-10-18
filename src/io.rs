//! Sequential, aligned file I/O. Partial writes are padded on disk and truncated
//! to the logical length, so the plot and temporary formats need no padding rule.
use std::{
    cell::Cell,
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    rc::Rc,
};

const ALIGN: usize = 4096;
const BUFFER: usize = 1024 * 1024;

#[derive(Clone)]
pub(crate) struct Policy {
    enabled: bool,
    warned: Rc<Cell<bool>>,
}
impl Policy {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            warned: Rc::new(Cell::new(false)),
        }
    }
    fn fallback(&self, path: &Path) {
        if self.enabled && !self.warned.replace(true) {
            eprintln!(
                "Direct I/O unavailable for {}; using buffered I/O",
                path.display()
            );
        }
    }
    pub fn reader(&self, path: &Path) -> io::Result<Box<dyn Read>> {
        let file = FileIo::open(path, false, false, self.clone())?;
        let remaining = file.file.metadata()?.len();
        Ok(Box::new(AlignedReader {
            file,
            buffer: Aligned::new(),
            used: 0,
            filled: 0,
            eof: false,
            remaining,
        }))
    }
    pub fn writer(&self, path: &Path, truncate: bool) -> io::Result<Box<dyn Write>> {
        Ok(Box::new(AlignedWriter {
            file: FileIo::open(path, true, truncate, self.clone())?,
            buffer: Aligned::new(),
            used: 0,
            committed: 0,
            truncate,
            seek_needed: false,
        }))
    }
}
struct Aligned {
    bytes: Vec<u8>,
    start: usize,
}
impl Aligned {
    fn new() -> Self {
        let bytes = vec![0; BUFFER + ALIGN - 1];
        let start = (ALIGN - bytes.as_ptr() as usize % ALIGN) % ALIGN;
        Self { bytes, start }
    }
    fn data(&mut self) -> &mut [u8] {
        &mut self.bytes[self.start..self.start + BUFFER]
    }
}
struct FileIo {
    file: File,
    path: PathBuf,
    write: bool,
    direct: bool,
    policy: Policy,
}
fn unsupported(error: &io::Error) -> bool {
    #[cfg(target_os = "linux")]
    {
        matches!(
            error.raw_os_error(),
            Some(libc::EINVAL | libc::EOPNOTSUPP | libc::ENOSYS)
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = error;
        false
    }
}
impl FileIo {
    fn open(path: &Path, write: bool, truncate: bool, policy: Policy) -> io::Result<Self> {
        let mut options = OpenOptions::new();
        options
            .read(!write)
            .write(write)
            .truncate(write && truncate);
        #[cfg(target_os = "linux")]
        let direct = policy.enabled;
        #[cfg(not(target_os = "linux"))]
        let direct = false;
        #[cfg(target_os = "linux")]
        if direct {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_DIRECT);
        }
        match options.open(path) {
            Ok(file) => {
                if policy.enabled && !direct {
                    policy.fallback(path);
                }
                Ok(Self {
                    file,
                    path: path.to_owned(),
                    write,
                    direct,
                    policy,
                })
            }
            Err(error) if direct && unsupported(&error) => {
                policy.fallback(path);
                let file = OpenOptions::new()
                    .read(!write)
                    .write(write)
                    .truncate(write && truncate)
                    .open(path)?;
                Ok(Self {
                    file,
                    path: path.to_owned(),
                    write,
                    direct: false,
                    policy,
                })
            }
            Err(error) => Err(error),
        }
    }
    fn reopen_buffered(&mut self) -> io::Result<()> {
        let position = self.file.stream_position()?;
        self.policy.fallback(&self.path);
        self.file = OpenOptions::new()
            .read(!self.write)
            .write(self.write)
            .open(&self.path)?;
        self.file.seek(SeekFrom::Start(position))?;
        self.direct = false;
        Ok(())
    }
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        loop {
            match self.file.read(bytes) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if self.direct && unsupported(&error) => self.reopen_buffered()?,
                result => return result,
            }
        }
    }
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        let mut offset = 0;
        while offset < bytes.len() {
            match self.file.write(&bytes[offset..]) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if self.direct && unsupported(&error) => self.reopen_buffered()?,
                Err(error) => return Err(error),
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => offset += n,
            }
        }
        Ok(())
    }
}
struct AlignedReader {
    file: FileIo,
    buffer: Aligned,
    used: usize,
    filled: usize,
    eof: bool,
    remaining: u64,
}
impl Read for AlignedReader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        if self.used == self.filled {
            if self.eof {
                return Ok(0);
            }
            self.filled = self.file.read(self.buffer.data())?;
            self.used = 0;
            self.remaining = self.remaining.saturating_sub(self.filled as u64);
            self.eof = self.filled == 0 || self.remaining == 0;
        }
        let n = bytes.len().min(self.filled - self.used);
        bytes[..n].copy_from_slice(&self.buffer.data()[self.used..self.used + n]);
        self.used += n;
        Ok(n)
    }
}
struct AlignedWriter {
    file: FileIo,
    buffer: Aligned,
    used: usize,
    committed: u64,
    truncate: bool,
    seek_needed: bool,
}
impl AlignedWriter {
    fn flush_buffer(&mut self) -> io::Result<()> {
        if self.used == 0 {
            return Ok(());
        }
        let padded = self.used.div_ceil(ALIGN) * ALIGN;
        self.buffer.data()[self.used..padded].fill(0);
        if self.seek_needed {
            self.file.file.seek(SeekFrom::Start(self.committed))?;
        }
        self.file.write_all(&self.buffer.data()[..padded])?;
        if self.truncate && self.used < BUFFER {
            self.file.file.set_len(self.committed + self.used as u64)?;
        }
        if self.used == BUFFER {
            self.committed += BUFFER as u64;
            self.used = 0;
            self.seek_needed = false;
        } else {
            self.seek_needed = true;
        }
        Ok(())
    }
}
impl Write for AlignedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = bytes.len().min(BUFFER - self.used);
        self.buffer.data()[self.used..self.used + n].copy_from_slice(&bytes[..n]);
        self.used += n;
        if self.used == BUFFER {
            self.flush_buffer()?;
        }
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()?;
        self.file.file.flush()
    }
}

pub(crate) struct PlotWriter {
    inner: Box<dyn Write>,
    position: u64,
}
impl PlotWriter {
    pub fn new(inner: Box<dyn Write>) -> Self {
        Self { inner, position: 0 }
    }
    pub fn position(&self) -> u64 {
        self.position
    }
}
impl Write for PlotWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(bytes)?;
        self.position += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_io_alignment_tail_truncation_flush_and_prefix_overwrite() {
        let mut buffer = Aligned::new();
        assert_eq!(buffer.data().as_ptr() as usize % ALIGN, 0);
        for enabled in [false, true] {
            let file = tempfile::NamedTempFile::new().unwrap();
            let policy = Policy::new(enabled);
            let mut writer = policy.writer(file.path(), true).unwrap();
            let bytes: Vec<u8> = (0..BUFFER * 2 + 17).map(|i| (i % 251) as u8).collect();
            writer.write_all(&bytes[..123]).unwrap();
            writer.flush().unwrap();
            assert_eq!(std::fs::metadata(file.path()).unwrap().len(), 123);
            writer.write_all(&bytes[123..]).unwrap();
            writer.flush().unwrap();
            drop(writer);
            assert_eq!(std::fs::read(file.path()).unwrap(), bytes);
            let mut actual = Vec::new();
            policy
                .reader(file.path())
                .unwrap()
                .read_to_end(&mut actual)
                .unwrap();
            assert_eq!(actual, bytes);
            let mut writer = policy.writer(file.path(), false).unwrap();
            writer.write_all(&[7; ALIGN]).unwrap();
            writer.flush().unwrap();
            drop(writer);
            actual[..ALIGN].fill(7);
            assert_eq!(std::fs::read(file.path()).unwrap(), actual);
        }
    }
}
