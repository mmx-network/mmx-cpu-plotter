//! Native CPU proof recomputation. Consensus proofs contain 256 uncompressed X
//! values; compressed plotting/proving, remote compute and GPU paths are separate.
use crate::Hash;
use rayon::prelude::*;
use sha2::{Digest, Sha256, Sha512};
use std::collections::BTreeSet;

pub const N_META: usize = 14;
pub const META_BYTES: usize = 48;
pub const PROOF_SIZE: usize = 256;
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(&'static str),
    #[error("zero matches at table {0}")]
    ZeroMatches(u8),
}
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub y: u32,
    pub meta: [u8; META_BYTES],
    pub proof: Vec<u32>,
}
const INIT: [u32; 16] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
];
/// The native 4 KiB memory hash, with explicit little-endian words and wrapping
/// arithmetic so debug/release and CPU architecture have identical behavior.
pub fn memory_hash(key: &[u8; 64]) -> [u8; 128] {
    let mut state = [0u32; 32];
    for (i, bytes) in key.as_chunks::<4>().0.iter().enumerate() {
        state[i] = u32::from_le_bytes(*bytes);
    }
    state[16..].copy_from_slice(&INIT);
    let (mut b, mut c) = (0u32, 0u32);
    let mut memory = [0u32; 1024];
    for row in memory.as_chunks_mut::<32>().0.iter_mut() {
        for _ in 0..4 {
            for k in 0..16 {
                let (mut a, mut d) = (state[k], state[16 + k]);
                a = a.wrapping_add(b);
                d = (d ^ a).rotate_left(16);
                c = c.wrapping_add(d);
                b = (b ^ c).rotate_left(12);
                a = a.wrapping_add(b);
                d = (d ^ a).rotate_left(8);
                c = c.wrapping_add(d);
                b = (b ^ c).rotate_left(7);
                state[k] = a;
                state[16 + k] = d;
            }
        }
        row.copy_from_slice(&state);
    }
    state.copy_from_slice(&memory[992..]);
    for iter in 0..256 {
        let sum = state
            .iter()
            .enumerate()
            .fold(0u32, |s, (i, v)| s.wrapping_add(v.rotate_left(i as u32)));
        let dir = sum.wrapping_add(sum << 11).wrapping_add(sum << 22);
        let bits = (dir >> 22) % 32;
        let offset = (dir >> 27) as usize;
        for (i, v) in state.iter_mut().enumerate() {
            *v = v.wrapping_add(memory[offset * 32 + (iter % 32 + i) % 32].rotate_left(bits) ^ sum);
        }
        for (i, v) in state.iter().enumerate() {
            memory[offset * 32 + i] ^= v;
        }
    }
    let mut hash = [0; 128];
    for (dst, v) in hash.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    hash
}
fn mask(ksize: u8) -> Result<u32> {
    if !(8..=32).contains(&ksize) {
        return Err(Error::Invalid("invalid ksize"));
    }
    Ok(((1u64 << ksize) - 1) as u32)
}
fn hash_meta(bytes: &[u8], kmask: u32) -> (u32, [u32; N_META]) {
    let hash = Sha512::digest(bytes);
    let mut y = 0;
    let mut meta = [0; N_META];
    for (i, v) in hash.as_chunks::<4>().0.iter().take(N_META).enumerate() {
        let word = u32::from_le_bytes(*v);
        y ^= word;
        meta[i] = word & kmask;
    }
    (y & kmask, meta)
}
pub fn compute_f1(x: u32, id: &Hash, ksize: u8) -> Result<(u32, [u32; N_META])> {
    let kmask = mask(ksize)?;
    let mut message = [0; 36];
    message[..4].copy_from_slice(&x.to_le_bytes());
    message[4..].copy_from_slice(id);
    let key: [u8; 64] = Sha512::digest(message).into();
    let mut bytes = [0; 192];
    bytes[..64].copy_from_slice(&key);
    bytes[64..].copy_from_slice(&memory_hash(&key));
    Ok(hash_meta(&bytes, kmask))
}
/// Native compute(..., xbits=0), including deduplicated X input, metadata sorting,
/// all eight match tables, final full-metadata deduplication and proof backtrace.
pub fn compute(xs: &[u32], id: &Hash, ksize: u8) -> Result<Vec<Entry>> {
    let kmask = mask(ksize)?;
    let xs: Vec<_> = xs
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut meta = Vec::with_capacity(xs.len());
    let mut entries = Vec::with_capacity(xs.len());
    // F1 inputs are independent. Indexed collection preserves native X order
    // before deterministic metadata sorting and proof backtracing.
    let initial: Vec<_> = xs.par_iter().map(|x| compute_f1(*x, id, ksize)).collect();
    for (i, value) in initial.into_iter().enumerate() {
        let (y, m) = value?;
        entries.push((y, i));
        meta.push(m);
    }
    let mut links: [Vec<(usize, usize)>; 10] = std::array::from_fn(|_| Vec::new());
    for (table, table_links) in links.iter_mut().enumerate().skip(2) {
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| meta[a.1].cmp(&meta[b.1])));
        let mut next_meta = Vec::new();
        let mut matches = Vec::new();
        for (i, &(yl, pl)) in entries.iter().enumerate() {
            for &(yr, pr) in &entries[i + 1..] {
                if yr == yl.wrapping_add(1) {
                    let mut message = [0; N_META * 8];
                    for (dst, word) in message
                        .as_chunks_mut::<4>()
                        .0
                        .iter_mut()
                        .zip(meta[pl].iter().chain(meta[pr].iter()))
                    {
                        dst.copy_from_slice(&word.to_le_bytes());
                    }
                    let (y, m) = hash_meta(&message, kmask);
                    matches.push((y, next_meta.len()));
                    next_meta.push(m);
                    table_links.push((pl, pr));
                } else if yr > yl {
                    break;
                }
            }
        }
        if matches.is_empty() {
            return Err(Error::ZeroMatches(table as u8));
        }
        entries = matches;
        meta = next_meta;
    }
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| meta[a.1].cmp(&meta[b.1])));
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (y, index) in entries {
        if !seen.insert(meta[index]) {
            continue;
        }
        let mut indices = vec![index];
        for table in (2..=9).rev() {
            indices = indices
                .into_iter()
                .flat_map(|i| [links[table][i].0, links[table][i].1])
                .collect();
        }
        let mut bytes = [0; META_BYTES];
        for (dst, v) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(meta[index]) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        out.push(Entry {
            y,
            meta: bytes,
            proof: indices.into_iter().map(|i| xs[i]).collect(),
        });
    }
    Ok(out)
}
fn prefixed_hash(prefix: &[u8], challenge: &Hash, meta: &[u8; META_BYTES]) -> Hash {
    let mut hash = Sha256::new();
    hash.update(prefix);
    hash.update(challenge);
    hash.update(meta);
    hash.finalize().into()
}
pub fn calc_quality(challenge: &Hash, meta: &[u8; META_BYTES]) -> Hash {
    prefixed_hash(b"proof_quality", challenge, meta)
}
pub fn leading_zero_bits(hash: &Hash, bits: u32) -> bool {
    if bits > 256 {
        return false;
    }
    (0..bits).all(|i| hash[31 - (i / 8) as usize] & (1u8 << (7 - i % 8)) == 0)
}
pub fn check_post_filter(challenge: &Hash, meta: &[u8; META_BYTES], bits: u32) -> bool {
    leading_zero_bits(&prefixed_hash(b"post_filter", challenge, meta), bits)
}
pub fn verify(
    xs: &[u32],
    challenge: &Hash,
    id: &Hash,
    plot_filter: u32,
    post_filter: u32,
    ksize: u8,
    hard_fork: bool,
) -> Result<Hash> {
    if xs.len() != PROOF_SIZE {
        return Err(Error::Invalid("invalid proof size"));
    }
    if plot_filter > 30 || post_filter > 256 {
        return Err(Error::Invalid("invalid filter size"));
    }
    let entries = compute(xs, id, ksize)?;
    if entries.is_empty() {
        return Err(Error::Invalid("invalid proof"));
    }
    if entries.len() > 1 {
        return Err(Error::Invalid("more than one proof found"));
    }
    let entry = &entries[0];
    let y0 = u32::from_le_bytes(challenge[..4].try_into().unwrap()) & mask(ksize)?;
    if entry.y < y0 || entry.y as u64 >= y0 as u64 + (1u64 << plot_filter) {
        return Err(Error::Invalid("invalid Y value"));
    }
    if entry.proof != xs {
        return Err(Error::Invalid("invalid proof order"));
    }
    if hard_fork {
        if !check_post_filter(challenge, &entry.meta, post_filter) {
            return Err(Error::Invalid("post filter failed"));
        }
        let mut coordinates = Sha256::new();
        for x in xs {
            coordinates.update(x.to_le_bytes());
        }
        let mut hash = Sha256::new();
        hash.update(challenge);
        hash.update(coordinates.finalize());
        Ok(hash.finalize().into())
    } else {
        Ok(calc_quality(challenge, &entry.meta))
    }
}
