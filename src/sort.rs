//! Fixed-record external merge sort. Disk mode never loads a full table.
use crate::storage::{Storage, TempData};
use rayon::prelude::*;
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    io::{self, Read, Write},
};

pub(crate) const RUN_BYTES: usize = 64 * 1024 * 1024;
const MERGE_FAN_IN: usize = 32;
pub(crate) const ENTRY_BYTES: usize = 76;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Entry {
    pub y: u32,
    pub meta: [u32; 14],
    pub left: u64,
    pub right: u64,
}
impl Entry {
    pub fn read(reader: &mut dyn Read) -> io::Result<Option<Self>> {
        let mut bytes = [0; ENTRY_BYTES];
        if reader.read(&mut bytes[..1])? == 0 {
            return Ok(None);
        }
        reader.read_exact(&mut bytes[1..])?;
        let mut words = bytes[..60]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        Ok(Some(Self {
            y: words.next().unwrap(),
            meta: std::array::from_fn(|_| words.next().unwrap()),
            left: u64::from_le_bytes(bytes[60..68].try_into().unwrap()),
            right: u64::from_le_bytes(bytes[68..76].try_into().unwrap()),
        }))
    }
    pub fn write(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut bytes = [0; ENTRY_BYTES];
        bytes[..4].copy_from_slice(&self.y.to_le_bytes());
        for (dst, word) in bytes[4..60].chunks_exact_mut(4).zip(self.meta) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        bytes[60..68].copy_from_slice(&self.left.to_le_bytes());
        bytes[68..].copy_from_slice(&self.right.to_le_bytes());
        writer.write_all(&bytes)
    }
    fn same_hash(&self, other: &Self) -> bool {
        self.y == other.y && self.meta == other.meta
    }
}

pub(crate) struct Table {
    pub data: TempData,
    pub count: u64,
}
pub(crate) struct SortedTable {
    pub count: u64,
    source: SortedSource,
}
enum SortedSource {
    Ram(Vec<Entry>),
    Stream(TempData),
}
pub(crate) enum EntryReader<'a> {
    Ram(std::slice::Iter<'a, Entry>),
    Stream(Box<dyn Read>),
}
impl EntryReader<'_> {
    pub fn next(&mut self) -> io::Result<Option<Entry>> {
        match self {
            Self::Ram(iter) => Ok(iter.next().copied()),
            Self::Stream(reader) => Entry::read(reader),
        }
    }
}
impl SortedTable {
    pub fn as_slice(&self) -> Option<&[Entry]> {
        match &self.source {
            SortedSource::Ram(entries) => Some(entries),
            SortedSource::Stream(_) => None,
        }
    }
    pub fn from_ram(mut entries: Vec<Entry>) -> Self {
        entries.par_sort_unstable();
        entries.dedup_by(|a, b| a.same_hash(b));
        Self {
            count: entries.len() as u64,
            source: SortedSource::Ram(entries),
        }
    }
    pub fn reader(&self) -> io::Result<EntryReader<'_>> {
        Ok(match &self.source {
            SortedSource::Ram(entries) => EntryReader::Ram(entries.iter()),
            SortedSource::Stream(data) => EntryReader::Stream(data.reader()?),
        })
    }
}
pub(crate) struct Sorter<'a> {
    storage: &'a Storage,
    entries: Vec<Entry>,
    runs: Vec<Table>,
    limit: usize,
}
impl<'a> Sorter<'a> {
    pub fn new(storage: &'a Storage) -> Self {
        Self::with_limit(
            storage,
            if storage.is_ram() {
                usize::MAX
            } else {
                RUN_BYTES / size_of::<Entry>()
            },
        )
    }
    pub(crate) fn with_limit(storage: &'a Storage, limit: usize) -> Self {
        Self {
            storage,
            entries: Vec::new(),
            runs: Vec::new(),
            limit,
        }
    }
    pub fn push(&mut self, entry: Entry) -> io::Result<()> {
        // Avoid doubling a multi-GiB allocation in RAM mode, and never exceed
        // the fixed run budget in disk mode.
        if self.entries.len() == self.entries.capacity() {
            let extra = if self.storage.is_ram() {
                (self.entries.len() / 8).max(8192)
            } else {
                self.limit - self.entries.len()
            };
            self.entries
                .try_reserve_exact(extra)
                .map_err(io::Error::other)?;
        }
        self.entries.push(entry);
        if self.entries.len() >= self.limit {
            self.flush_run()?;
        }
        Ok(())
    }
    pub fn reserve_expected(&mut self, count: u64) -> io::Result<()> {
        if self.storage.is_ram() {
            let count = usize::try_from(count).map_err(io::Error::other)?;
            self.entries
                .try_reserve_exact(count)
                .map_err(io::Error::other)?;
        }
        Ok(())
    }
    fn flush_run(&mut self) -> io::Result<()> {
        if self.entries.is_empty() {
            return Ok(());
        }
        self.entries.par_sort_unstable();
        // Identical metadata has identical successors. Keep one valid backtrace.
        self.entries.dedup_by(|a, b| a.same_hash(b));
        let count = self.entries.len() as u64;
        let (data, mut writer) = self.storage.create_sized(count * ENTRY_BYTES as u64)?;
        for entry in self.entries.drain(..) {
            entry.write(&mut writer)?;
        }
        writer.flush()?;
        self.runs.push(Table { data, count });
        Ok(())
    }
    pub fn finish(mut self) -> io::Result<SortedTable> {
        if self.storage.is_ram() && self.runs.is_empty() {
            // Keep the sorted allocation itself: no serialization, second table
            // allocation, or copy through byte streams on the full-RAM path.
            return Ok(SortedTable::from_ram(self.entries));
        }
        self.flush_run()?;
        drop(self.entries);
        if self.runs.is_empty() {
            let (data, mut writer) = self.storage.create()?;
            writer.flush()?;
            return Ok(SortedTable {
                source: SortedSource::Stream(data),
                count: 0,
            });
        }
        while self.runs.len() > 1 {
            let mut next = Vec::new();
            let mut runs = self.runs.into_iter();
            loop {
                let group: Vec<_> = runs.by_ref().take(MERGE_FAN_IN).collect();
                if group.is_empty() {
                    break;
                }
                next.push(merge(self.storage, group)?);
            }
            self.runs = next;
        }
        let table = self.runs.pop().unwrap();
        Ok(SortedTable {
            count: table.count,
            source: SortedSource::Stream(table.data),
        })
    }
}
fn merge(storage: &Storage, runs: Vec<Table>) -> io::Result<Table> {
    let mut readers = runs
        .iter()
        .map(|run| run.data.reader())
        .collect::<io::Result<Vec<_>>>()?;
    let (data, mut writer) = storage.create()?;
    let mut heap = BinaryHeap::new();
    for (i, reader) in readers.iter_mut().enumerate() {
        if let Some(entry) = Entry::read(reader)? {
            heap.push(Reverse((entry, i)));
        }
    }
    let mut previous: Option<Entry> = None;
    let mut count = 0;
    while let Some(Reverse((entry, run))) = heap.pop() {
        if previous.as_ref().is_none_or(|prev| !prev.same_hash(&entry)) {
            entry.write(&mut writer)?;
            previous = Some(entry);
            count += 1;
        }
        if let Some(entry) = Entry::read(&mut readers[run])? {
            heap.push(Reverse((entry, run)));
        }
    }
    writer.flush()?;
    Ok(Table { data, count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Mode;
    #[test]
    fn merge_across_many_runs_deduplicates_and_preserves_order() {
        for mode in [Mode::Ram, Mode::Disk] {
            let dir = tempfile::tempdir().unwrap();
            let store = Storage::new(mode, dir.path(), true).unwrap();
            let mut sorter = Sorter::with_limit(&store, 3);
            for i in (0..250).rev() {
                sorter
                    .push(Entry {
                        y: i % 37,
                        meta: [i % 37; 14],
                        left: i as u64,
                        right: 0,
                    })
                    .unwrap();
            }
            let table = sorter.finish().unwrap();
            assert_eq!(table.count, 37);
            let mut reader = table.reader().unwrap();
            for y in 0..37 {
                assert_eq!(reader.next().unwrap().unwrap().y, y);
            }
            assert!(reader.next().unwrap().is_none());
        }
    }
}
