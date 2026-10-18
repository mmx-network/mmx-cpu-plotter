use crate::{
    encoding::{line_point, symbol_bits, write_bits, write_symbol, LiveSet},
    io::{PlotWriter, Policy},
    mmx::PlotHeader,
    pos,
    sort::{Entry, EntryReader, SortedTable, Sorter, Table},
    storage::{Mode, Storage},
    Hash, VnxType,
};
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use sha2::{Digest, Sha256, Sha512};
use std::{
    io::{Read, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

const PARK: usize = 2048;
// Coarse batches amortize worker-pool scheduling on many-core CPUs. Keeping the
// work and result buffers below ~14 MiB still leaves disk-mode memory bounded.
const BATCH: usize = 65536;
const HEADER_SIZE: u64 = 4096;

#[derive(Clone)]
pub struct Options {
    pub ksize: u8,
    pub farmer_key: [u8; 33],
    pub contract: Option<Hash>,
    pub seed: Hash,
    pub mode: Mode,
    pub ssd: bool,
    pub output: PathBuf,
    pub temp_dir: PathBuf,
    pub threads: usize,
    pub direct_io: bool,
    /// Small plots are useful for compatibility tests, but cannot be farmed on mainnet.
    pub allow_small_k: bool,
    pub cancelled: Arc<AtomicBool>,
}
#[derive(Debug)]
pub struct PlotResult {
    pub path: PathBuf,
    pub header: PlotHeader,
}

pub fn plot_id(ksize: u8, seed: &Hash, farmer_key: &[u8; 33], contract: Option<&Hash>) -> Hash {
    let mut hash = Sha256::new();
    hash.update(if contract.is_some() {
        b"MMX/PLOTID/NFT".as_slice()
    } else {
        b"MMX/PLOTID/OG".as_slice()
    });
    hash.update([ksize]);
    hash.update(seed);
    hash.update(farmer_key);
    if let Some(contract) = contract {
        hash.update(contract);
    }
    hash.finalize().into()
}

pub fn plot(options: &Options) -> Result<PlotResult> {
    ensure!(
        (8..=32).contains(&options.ksize),
        "ksize must be between 8 and 32"
    );
    ensure!(
        options.ksize >= 29 || options.allow_small_k,
        "ksize below 29 requires --allow-small-k (test plots only)"
    );
    ensure!(options.threads > 0, "threads must be positive");
    k256::PublicKey::from_sec1_bytes(&options.farmer_key)
        .context("invalid compressed secp256k1 farmer key")?;
    ensure!(
        options.output.is_dir(),
        "output directory does not exist: {}",
        options.output.display()
    );
    check_cancel(options)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()?;
    pool.install(|| run(options))
}
fn check_cancel(options: &Options) -> Result<()> {
    ensure!(
        !options.cancelled.load(Ordering::Relaxed),
        "plotting cancelled"
    );
    Ok(())
}

fn run(options: &Options) -> Result<PlotResult> {
    let started = Instant::now();
    let id = plot_id(
        options.ksize,
        &options.seed,
        &options.farmer_key,
        options.contract.as_ref(),
    );
    let path = options.output.join(format!(
        "plot-{}-k{}-{}.plot",
        if options.ssd { "ssd" } else { "hdd" },
        options.ksize,
        hex::encode(id)
    ));
    ensure!(!path.exists(), "plot already exists: {}", path.display());
    eprintln!(
        "Plot ID: {}\nMode: {:?}, k{}, {} CPU threads",
        hex::encode(id),
        options.mode,
        options.ksize,
        options.threads
    );
    let storage = Storage::new(options.mode, &options.temp_dir, options.direct_io)
        .context("create plotting temporary storage")?;
    let mut sorter = Sorter::new(&storage);
    let domain = 1u64 << options.ksize;
    sorter.reserve_expected(domain)?;
    let mut report = Instant::now();
    for first in (0..domain).step_by(BATCH) {
        check_cancel(options)?;
        let count = (domain - first).min(BATCH as u64);
        let entries: Vec<_> = (0..count as usize)
            .into_par_iter()
            .with_min_len(256)
            .map(|offset| {
                let x = (first + offset as u64) as u32;
                let (y, meta) = pos::compute_f1(x, &id, options.ksize).expect("validated ksize");
                Entry {
                    y,
                    meta,
                    left: x as u64,
                    right: 0,
                }
            })
            .collect();
        for entry in entries {
            sorter.push(entry)?;
        }
        if report.elapsed().as_secs() >= 15 {
            eprintln!(
                "T1: {:.1}% ({:.0} X/s)",
                (first + count) as f64 * 100. / domain as f64,
                (first + count) as f64 / started.elapsed().as_secs_f64()
            );
            report = Instant::now();
        }
    }
    let mut current = sorter.finish()?;
    let mut links: Vec<Option<Table>> = (0..10).map(|_| None).collect();
    eprintln!(
        "T1: {} entries ({:.1}s elapsed)",
        current.count,
        started.elapsed().as_secs_f64()
    );
    for table in 2..=9 {
        check_cancel(options)?;
        // Consume the preceding metadata before sorting/writing its successor.
        // This avoids retaining three whole tables at once in RAM mode.
        let (next, previous_links) = match_table(&storage, current, table, options)?;
        links[table - 1] = previous_links;
        ensure!(
            next.count > 0,
            "zero matches at table {table}; choose another seed"
        );
        current = next;
        eprintln!(
            "T{table}: {} entries ({:.1}s elapsed)",
            current.count,
            started.elapsed().as_secs_f64()
        );
    }
    links[9] = Some(extract_links(&storage, &current)?);
    let final_count = current.count;
    let mut compact: Vec<Table> = Vec::new();
    let mut active: Option<LiveSet> = None;
    let mut max_pd_bytes = 0;
    for table in (3..=9).rev() {
        check_cancel(options)?;
        let source = links[table].take().unwrap();
        let previous_count = links[table - 1].as_ref().unwrap().count;
        let mut needed = LiveSet::new(previous_count)?;
        let mut reader = source.data.reader()?;
        for index in 0..source.count {
            if index % (1 << 20) == 0 {
                check_cancel(options)?;
            }
            let (l, r) = read_pair(&mut reader)?;
            if active.as_ref().is_none_or(|live| live.contains(index)) {
                needed.insert(l)?;
                needed.insert(r)?;
            }
        }
        let count = needed.build_index();
        ensure!(
            count <= domain,
            "T{} has too many live entries for {}-bit pointers",
            table - 1,
            options.ksize
        );
        let (data, mut writer) = storage.create_sized(source.count * 16)?;
        let mut reader = source.data.reader()?;
        let mut kept = 0;
        let mut delta_bits = 0;
        for index in 0..source.count {
            let (l, r) = read_pair(&mut reader)?;
            if active.as_ref().is_some_and(|live| !live.contains(index)) {
                continue;
            }
            let l = needed.index(l);
            let r = needed.index(r);
            ensure!(r > l, "non-increasing match pointers");
            delta_bits += symbol_bits(r - l)? as usize;
            write_pair(&mut writer, l, r)?;
            kept += 1;
            if kept % PARK as u64 == 0 {
                max_pd_bytes =
                    max_pd_bytes.max((PARK * options.ksize as usize + delta_bits).div_ceil(64) * 8);
                delta_bits = 0;
                check_cancel(options)?;
            }
        }
        if kept % PARK as u64 != 0 {
            delta_bits += (PARK - kept as usize % PARK) * 2;
            max_pd_bytes =
                max_pd_bytes.max((PARK * options.ksize as usize + delta_bits).div_ceil(64) * 8);
        }
        writer.flush()?;
        compact.push(Table { data, count: kept });
        active = Some(needed);
        eprintln!(
            "Pruned T{table}: {kept} entries, T{}: {count} live",
            table - 1
        );
    }
    // One common park stride is required for every pointer table.
    let mut header = PlotHeader {
        version: 1,
        ksize: options.ksize as i32,
        xbits: options.ksize as i32,
        has_meta: !options.ssd,
        seed: options.seed.into(),
        plot_id: id.into(),
        farmer_key: options.farmer_key.into(),
        contract: options.contract.map(Into::into),
        park_size_x: PARK as i32,
        park_size_y: PARK as i32,
        park_size_pd: PARK as i32,
        park_size_meta: PARK as i32,
        park_bytes_x: (PARK * (2 * options.ksize as usize - 1)).div_ceil(64) as i32 * 8,
        park_bytes_pd: max_pd_bytes as i32,
        park_bytes_meta: (PARK * 12 * options.ksize as usize).div_ceil(64) as i32 * 8,
        entry_bits_x: 2 * options.ksize as i32 - 1,
        num_entries_y: final_count,
        ..Default::default()
    };
    // Keep the incomplete final plot private; publishing must never overwrite another plot.
    let output_file = tempfile::NamedTempFile::new_in(&options.output)?;
    let policy = Policy::new(options.direct_io);
    let mut output = PlotWriter::new(policy.writer(output_file.path(), true)?);
    output.write_all(&[0; HEADER_SIZE as usize])?;
    header.table_offset_y = HEADER_SIZE;
    write_y_and_meta(&storage, &current, &mut header, &mut output, options)?;
    drop(current);
    for table in compact {
        check_cancel(options)?;
        header.table_offset_pd.push(output.position());
        write_pd(&table, &header, &mut output, options)?;
    }
    header.table_offset_x = output.position();
    write_x(
        links[2].take().unwrap(),
        active.as_ref().unwrap(),
        &header,
        &mut output,
        options,
    )?;
    header.plot_size = output.position();
    let bytes = header.encode(&crate::vnx_registry()?)?;
    ensure!(
        bytes.len() as u64 <= HEADER_SIZE,
        "VNX header exceeds reserved space"
    );
    output.flush()?;
    drop(output);
    let mut prefix = [0; HEADER_SIZE as usize];
    prefix[..bytes.len()].copy_from_slice(&bytes);
    let mut header_writer = policy.writer(output_file.path(), false)?;
    header_writer.write_all(&prefix)?;
    header_writer.flush()?;
    drop(header_writer);
    output_file.as_file().sync_all()?;
    check_cancel(options)?;
    output_file
        .persist_noclobber(&path)
        .map_err(|e| e.error)
        .context("publish completed plot")?;
    std::fs::File::open(&options.output)?.sync_all()?;
    eprintln!(
        "Wrote {} bytes in {:.1}s: {}",
        header.plot_size,
        started.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(PlotResult { path, header })
}

#[derive(Clone)]
struct Match {
    left: [u32; 14],
    right: [u32; 14],
    l: u64,
    r: u64,
}
fn hash_matches(batch: &mut Vec<Match>, sorter: &mut Sorter<'_>, ksize: u8) -> Result<()> {
    let mask = ((1u64 << ksize) - 1) as u32;
    let entries: Vec<_> = batch
        .par_iter()
        .with_min_len(256)
        .map(|pair| {
            let mut message = [0; 112];
            for (dst, word) in message
                .chunks_exact_mut(4)
                .zip(pair.left.iter().chain(&pair.right))
            {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            let hash = Sha512::digest(message);
            let mut y = 0;
            let mut meta = [0; 14];
            for (dst, bytes) in meta.iter_mut().zip(hash.chunks_exact(4)) {
                let word = u32::from_le_bytes(bytes.try_into().unwrap());
                y ^= word;
                *dst = word & mask;
            }
            Entry {
                y: y & mask,
                meta,
                left: pair.l,
                right: pair.r,
            }
        })
        .collect();
    for entry in entries {
        sorter.push(entry)?;
    }
    batch.clear();
    Ok(())
}
fn match_table(
    storage: &Storage,
    previous: SortedTable,
    table: usize,
    options: &Options,
) -> Result<(SortedTable, Option<Table>)> {
    let mut reader = previous.reader()?;
    let mut pending = reader.next()?;
    let mut index = 0;
    let mut sorter = Sorter::new(storage);
    // Uniform Y buckets yield n² / 2^k matches. Reserve a little headroom
    // without retaining twice the required table memory.
    let expected = (previous.count as u128 * previous.count as u128 * 105)
        .div_ceil((1u128 << options.ksize) * 100);
    sorter.reserve_expected(expected.try_into()?)?;
    let mut batch = Vec::with_capacity(BATCH);
    // Preserve backtraces during the matching read instead of rereading the
    // entire metadata table just to extract its two pointers.
    let (link_data, mut link_writer) = if table > 2 {
        let (data, writer) = storage.create_sized(previous.count * 16)?;
        (Some(data), Some(writer))
    } else {
        (None, None)
    };
    let mut left = Vec::new();
    let mut right = Vec::new();
    read_group(
        &mut reader,
        &mut pending,
        &mut index,
        &mut link_writer,
        &mut left,
    )?;
    let mut report = Instant::now();
    while !left.is_empty() {
        check_cancel(options)?;
        read_group(
            &mut reader,
            &mut pending,
            &mut index,
            &mut link_writer,
            &mut right,
        )?;
        if !right.is_empty() && left[0].1.y.checked_add(1) == Some(right[0].1.y) {
            for &(li, l) in &left {
                for &(ri, r) in &right {
                    batch.push(Match {
                        left: l.meta,
                        right: r.meta,
                        l: if table == 2 { l.left } else { li },
                        r: if table == 2 { r.left } else { ri },
                    });
                    if batch.len() == BATCH {
                        hash_matches(&mut batch, &mut sorter, options.ksize)?;
                        check_cancel(options)?;
                    }
                }
            }
        }
        std::mem::swap(&mut left, &mut right);
        if report.elapsed().as_secs() >= 15 {
            eprintln!(
                "T{table}: matched {:.1}% of input",
                index as f64 * 100. / previous.count as f64
            );
            report = Instant::now();
        }
    }
    hash_matches(&mut batch, &mut sorter, options.ksize)?;
    if let Some(writer) = &mut link_writer {
        writer.flush()?;
    }
    drop(link_writer);
    let links = link_data.map(|data| Table {
        data,
        count: previous.count,
    });
    drop(reader);
    drop(previous);
    Ok((sorter.finish()?, links))
}
fn read_group(
    reader: &mut EntryReader<'_>,
    pending: &mut Option<Entry>,
    index: &mut u64,
    links: &mut Option<Box<dyn Write>>,
    group: &mut Vec<(u64, Entry)>,
) -> Result<()> {
    group.clear();
    let Some(first) = pending.take() else {
        return Ok(());
    };
    group.push((*index, first));
    if let Some(writer) = links {
        write_pair(writer, first.left, first.right)?;
    }
    *index += 1;
    loop {
        *pending = reader.next()?;
        match pending {
            Some(entry) if entry.y == first.y => {
                group.push((*index, *entry));
                if let Some(writer) = links {
                    write_pair(writer, entry.left, entry.right)?;
                }
                *index += 1;
            }
            _ => break,
        }
    }
    Ok(())
}
fn extract_links(storage: &Storage, table: &SortedTable) -> Result<Table> {
    let (data, mut writer) = storage.create_sized(table.count * 16)?;
    let mut reader = table.reader()?;
    while let Some(entry) = reader.next()? {
        write_pair(&mut writer, entry.left, entry.right)?;
    }
    writer.flush()?;
    Ok(Table {
        data,
        count: table.count,
    })
}
fn write_pair(writer: &mut dyn Write, left: u64, right: u64) -> Result<()> {
    writer.write_all(&left.to_le_bytes())?;
    writer.write_all(&right.to_le_bytes())?;
    Ok(())
}
fn read_pair(reader: &mut dyn Read) -> Result<(u64, u64)> {
    let mut bytes = [0; 16];
    reader.read_exact(&mut bytes)?;
    Ok((
        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
    ))
}

fn write_y_and_meta(
    storage: &Storage,
    table: &SortedTable,
    header: &mut PlotHeader,
    output: &mut PlotWriter,
    options: &Options,
) -> Result<()> {
    // Y parks must have a common stride. Spool their encoded bodies, then pad to
    // the longest one. The final partial park repeats its last Y.
    let (ys, mut yw) = storage.create()?;
    let mut reader = table.reader()?;
    let mut max_bytes = 0;
    let mut entries = Vec::with_capacity(PARK);
    while let Some(entry) = reader.next()? {
        entries.push(entry);
        if entries.len() == PARK {
            spool_y(&entries, &mut yw, &mut max_bytes)?;
            entries.clear();
            check_cancel(options)?;
        }
    }
    if !entries.is_empty() {
        spool_y(&entries, &mut yw, &mut max_bytes)?;
    }
    yw.flush()?;
    drop(yw);
    header.park_bytes_y = max_bytes as i32;
    let mut reader = ys.reader()?;
    for _ in 0..table.count.div_ceil(PARK as u64) {
        let mut len = [0; 4];
        reader.read_exact(&mut len)?;
        let len = u32::from_le_bytes(len) as usize;
        let mut park = vec![0; max_bytes];
        reader.read_exact(&mut park[..len])?;
        output.write_all(&park)?;
    }
    if !header.has_meta {
        return Ok(());
    }
    header.table_offset_meta = output.position();
    let mut reader = table.reader()?;
    for first in (0..table.count).step_by(PARK) {
        check_cancel(options)?;
        let mut park = vec![0; header.park_bytes_meta as usize];
        let mut offset = 0;
        for _ in 0..(table.count - first).min(PARK as u64) {
            let entry = reader.next()?.context("truncated metadata table")?;
            for word in &entry.meta[..12] {
                write_bits(&mut park, &mut offset, *word as u64, header.ksize as usize);
            }
        }
        output.write_all(&park)?;
    }
    Ok(())
}
fn spool_y(entries: &[Entry], writer: &mut dyn Write, max_bytes: &mut usize) -> Result<()> {
    let mut bits = 0;
    for pair in entries.windows(2) {
        bits += symbol_bits((pair[1].y - pair[0].y) as u64)? as usize;
    }
    bits += (PARK - entries.len()) * 2;
    let len = 4 + bits.div_ceil(64) * 8;
    *max_bytes = (*max_bytes).max(len);
    let mut bytes = vec![0; len];
    bytes[..4].copy_from_slice(&entries[0].y.to_le_bytes());
    let mut offset = 32;
    for pair in entries.windows(2) {
        write_symbol(&mut bytes, &mut offset, (pair[1].y - pair[0].y) as u64)?;
    }
    writer.write_all(&(len as u32).to_le_bytes())?;
    writer.write_all(&bytes)?;
    Ok(())
}
fn write_pd(
    table: &Table,
    header: &PlotHeader,
    output: &mut dyn Write,
    options: &Options,
) -> Result<()> {
    let mut reader = table.data.reader()?;
    for first in (0..table.count).step_by(PARK) {
        check_cancel(options)?;
        let mut bytes = vec![0; header.park_bytes_pd as usize];
        let mut pointer_offset = 0;
        let mut delta_offset = PARK * header.ksize as usize;
        for _ in 0..(table.count - first).min(PARK as u64) {
            let (l, r) = read_pair(&mut reader)?;
            write_bits(&mut bytes, &mut pointer_offset, l, header.ksize as usize);
            write_symbol(&mut bytes, &mut delta_offset, r - l)?;
        }
        output.write_all(&bytes)?;
    }
    Ok(())
}
fn write_x(
    table: Table,
    active: &LiveSet,
    header: &PlotHeader,
    output: &mut dyn Write,
    options: &Options,
) -> Result<()> {
    let mut reader = table.data.reader()?;
    let mut bytes = vec![0; header.park_bytes_x as usize];
    let mut offset = 0;
    let mut kept = 0;
    for index in 0..table.count {
        let (l, r) = read_pair(&mut reader)?;
        if !active.contains(index) {
            continue;
        }
        if l == r {
            bail!("duplicate X in T2 match");
        }
        write_bits(
            &mut bytes,
            &mut offset,
            line_point(l, r),
            header.entry_bits_x as usize,
        );
        kept += 1;
        if kept % PARK == 0 {
            check_cancel(options)?;
            output.write_all(&bytes)?;
            bytes.fill(0);
            offset = 0;
        }
    }
    if offset != 0 {
        output.write_all(&bytes)?;
    }
    Ok(())
}
