//! MMX's little-endian park bitstream and base-three delta code.
use anyhow::{ensure, Result};

pub(crate) fn symbol_bits(symbol: u64) -> Result<u32> {
    ensure!(
        symbol < 48,
        "park delta {symbol} exceeds the MMX format limit (47)"
    );
    Ok(2 * (symbol as u32 / 3) + 2)
}
pub(crate) fn write_bits(bytes: &mut [u8], offset: &mut usize, mut value: u64, bits: usize) {
    let end = *offset + bits;
    while *offset < end {
        let shift = *offset % 8;
        let n = (8 - shift).min(end - *offset);
        bytes[*offset / 8] |= ((value & ((1 << n) - 1)) as u8) << shift;
        value >>= n;
        *offset += n;
    }
}
pub(crate) fn write_symbol(bytes: &mut [u8], offset: &mut usize, symbol: u64) -> Result<()> {
    let bits = symbol_bits(symbol)?;
    let index = symbol / 3;
    let prefix = (1u64 << (2 * index)) - 1;
    write_bits(
        bytes,
        offset,
        prefix | ((symbol % 3) << (2 * index)),
        bits as usize,
    );
    Ok(())
}
pub(crate) fn line_point(a: u64, b: u64) -> u64 {
    let x = a.max(b);
    x * (x - 1) / 2 + a.min(b)
}

/// Live-entry bitmap with constant-time rank in a 512-bit block.
pub(crate) struct LiveSet {
    words: Vec<u64>,
    prefix: Vec<u64>,
    len: u64,
}
impl LiveSet {
    pub fn new(len: u64) -> Result<Self> {
        let words: usize = len.div_ceil(64).try_into()?;
        let mut data = Vec::new();
        data.try_reserve_exact(words)?;
        data.resize(words, 0);
        Ok(Self {
            words: data,
            prefix: Vec::new(),
            len,
        })
    }
    pub fn into_words(self) -> Vec<u64> {
        self.words
    }
    pub fn from_words(len: u64, words: Vec<u64>) -> Self {
        debug_assert_eq!(words.len() as u64, len.div_ceil(64));
        Self {
            words,
            prefix: Vec::new(),
            len,
        }
    }
    #[cfg(test)]
    pub fn insert(&mut self, index: u64) -> Result<()> {
        ensure!(
            index < self.len,
            "temporary pointer {index} outside table of {} entries",
            self.len
        );
        self.words[index as usize / 64] |= 1 << (index % 64);
        Ok(())
    }
    pub fn contains(&self, index: u64) -> bool {
        index < self.len && self.words[index as usize / 64] & (1 << (index % 64)) != 0
    }
    pub fn index(&self, index: u64) -> u64 {
        let word = index as usize / 64;
        self.prefix[word / 8]
            + self.words[word / 8 * 8..word]
                .iter()
                .map(|w| w.count_ones() as u64)
                .sum::<u64>()
            + (self.words[word] & ((1u64 << (index % 64)) - 1)).count_ones() as u64
    }
    pub fn build_index(&mut self) -> u64 {
        self.prefix.clear();
        let mut count = 0;
        for block in self.words.chunks(8) {
            self.prefix.push(count);
            count += block.iter().map(|w| w.count_ones() as u64).sum::<u64>();
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_symbols_and_line_points() {
        let mut bytes = [0; 8];
        let mut offset = 0;
        for sym in [0, 1, 2, 3, 4, 5] {
            write_symbol(&mut bytes, &mut offset, sym).unwrap();
        }
        assert_eq!(offset, 18);
        assert_eq!(&bytes[..3], &[0xe4, 0xdc, 2]);
        assert!(symbol_bits(48).is_err());
        assert_eq!(line_point(0, 1), 0);
        assert_eq!(
            line_point(u32::MAX as u64, u32::MAX as u64 - 1),
            (1u64 << 63) - (1u64 << 31) - 1
        );
    }
    #[test]
    fn rank_across_block_boundaries() {
        let mut live = LiveSet::new(2050).unwrap();
        for i in [0, 63, 64, 511, 512, 1024, 2049] {
            live.insert(i).unwrap();
        }
        assert!(live.insert(2050).is_err());
        assert_eq!(live.build_index(), 7);
        for (rank, i) in [0, 63, 64, 511, 512, 1024, 2049].into_iter().enumerate() {
            assert!(live.contains(i));
            assert_eq!(live.index(i), rank as u64);
        }
    }
}
