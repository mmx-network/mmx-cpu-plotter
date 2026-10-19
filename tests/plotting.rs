use mmx_cpu_plotter::{mmx::PlotHeader, plot, pos, Mode, Options, VnxType};
use std::{
    io::Cursor,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

fn options(output: &Path, mode: Mode) -> Options {
    Options {
        ksize: 12,
        farmer_key: hex::decode(
            "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        )
        .unwrap()
        .try_into()
        .unwrap(),
        contract: None,
        seed: [0; 32],
        mode,
        ssd: false,
        output: output.to_owned(),
        temp_dir: output.to_owned(),
        threads: 2,
        allow_small_k: true,
        direct_io: true,
        opencl_device: None,
        cancelled: Arc::new(AtomicBool::new(false)),
    }
}
fn bits(bytes: &[u8], offset: &mut usize, count: usize) -> u64 {
    let mut value = 0;
    for i in 0..count {
        value |= (((bytes[*offset / 8] >> (*offset % 8)) & 1) as u64) << i;
        *offset += 1;
    }
    value
}
fn delta(bytes: &[u8], offset: &mut usize) -> u64 {
    for group in 0..16 {
        let digit = bits(bytes, offset, 2);
        if digit < 3 {
            return group * 3 + digit;
        }
    }
    panic!("invalid native delta code")
}
fn park(bytes: &[u8], base: u64, size: i32, count: i32, index: u64) -> &[u8] {
    let start = base as usize + index as usize / count as usize * size as usize;
    &bytes[start..start + size as usize]
}
fn coordinates(bytes: &[u8], header: &PlotHeader, index: u64) -> Vec<u32> {
    let mut pointers = vec![index];
    for &base in &header.table_offset_pd {
        pointers = pointers
            .into_iter()
            .flat_map(|index| {
                let bytes = park(
                    bytes,
                    base,
                    header.park_bytes_pd,
                    header.park_size_pd,
                    index,
                );
                let inside = index as usize % header.park_size_pd as usize;
                let mut offset = inside * header.ksize as usize;
                let left = bits(bytes, &mut offset, header.ksize as usize);
                offset = header.park_size_pd as usize * header.ksize as usize;
                let right = left
                    + (0..=inside)
                        .map(|_| delta(bytes, &mut offset))
                        .last()
                        .unwrap();
                [left, right]
            })
            .collect();
    }
    pointers
        .into_iter()
        .flat_map(|index| {
            let bytes = park(
                bytes,
                header.table_offset_x,
                header.park_bytes_x,
                header.park_size_x,
                index,
            );
            let mut offset =
                index as usize % header.park_size_x as usize * header.entry_bits_x as usize;
            let point = bits(bytes, &mut offset, header.entry_bits_x as usize);
            // Independently invert x*(x-1)/2+y without floating-point rounding.
            let mut low = 1u64;
            let mut high = 1u64 << header.ksize;
            while low + 1 < high {
                let mid = (low + high) / 2;
                if mid * (mid - 1) / 2 <= point {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            [low as u32, (point - low * (low - 1) / 2) as u32]
        })
        .collect()
}
fn decoded_header(bytes: &[u8]) -> PlotHeader {
    let mut reader = vnx::binary::Reader::new(Cursor::new(bytes), vnx::Registry::new());
    PlotHeader::from_value(reader.read().unwrap()).unwrap()
}
fn check_plot(bytes: &[u8], header: &PlotHeader) {
    assert_eq!(&decoded_header(bytes), header);
    assert_eq!(bytes.len() as u64, header.plot_size);
    assert_eq!(header.table_offset_pd.len(), 7);
    let reference = pos::compute(
        &(0..1 << header.ksize).collect::<Vec<_>>(),
        &header.plot_id.0,
        header.ksize as u8,
    )
    .unwrap();
    assert_eq!(header.num_entries_y as usize, reference.len());
    let mut actual_y = Vec::new();
    for first in (0..reference.len()).step_by(header.park_size_y as usize) {
        let bytes = park(
            bytes,
            header.table_offset_y,
            header.park_bytes_y,
            header.park_size_y,
            first as u64,
        );
        let mut offset = 0;
        let mut y = bits(bytes, &mut offset, 32);
        actual_y.push(y as u32);
        for _ in 1..(reference.len() - first).min(header.park_size_y as usize) {
            y += delta(bytes, &mut offset);
            actual_y.push(y as u32);
        }
    }
    assert_eq!(actual_y, reference.iter().map(|e| e.y).collect::<Vec<_>>());
    if header.has_meta {
        for (i, entry) in reference.iter().enumerate() {
            let bytes = park(
                bytes,
                header.table_offset_meta,
                header.park_bytes_meta,
                header.park_size_meta,
                i as u64,
            );
            let mut offset = i % header.park_size_meta as usize * header.ksize as usize * 12;
            for word in entry.meta.chunks_exact(4) {
                assert_eq!(
                    bits(bytes, &mut offset, header.ksize as usize),
                    u32::from_le_bytes(word.try_into().unwrap()) as u64
                );
            }
        }
    } else {
        assert_eq!(header.table_offset_meta, u64::MAX);
    }
    // Decode every link level and the triangular X encoding before recomputing
    // sampled full proofs from their 256 leaf coordinates.
    for i in [
        0,
        reference.len() / 3,
        reference.len() / 2,
        reference.len() - 1,
    ] {
        let xs = coordinates(bytes, header, i as u64);
        assert_eq!(xs.len(), 256);
        let proofs = pos::compute(&xs, &header.plot_id.0, header.ksize as u8).unwrap();
        assert_eq!(proofs.len(), 1);
        assert_eq!(proofs[0].y, reference[i].y);
        assert_eq!(proofs[0].meta, reference[i].meta);
    }
}

#[test]
fn both_modes_match_complete_reference_tables_and_roundtrip_proofs() {
    let mut previous = None;
    for mode in [Mode::Ram, Mode::Disk] {
        let dir = tempfile::tempdir().unwrap();
        let mut options = options(dir.path(), mode);
        // RAM mode must never access a temporary directory.
        if mode == Mode::Ram {
            options.temp_dir = dir.path().join("does-not-exist");
        }
        let result = plot(&options).unwrap();
        let bytes = std::fs::read(&result.path).unwrap();
        check_plot(&bytes, &result.header);
        if let Some(previous) = previous {
            assert_eq!(previous, bytes);
        }
        previous = Some(bytes.clone());
        assert!(plot(&options)
            .unwrap_err()
            .to_string()
            .contains("already exists"));
        assert_eq!(std::fs::read(&result.path).unwrap(), bytes);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
#[test]
fn ssd_nft_plots_roundtrip_and_cancellation_leaves_no_output() {
    let dir = tempfile::tempdir().unwrap();
    let mut options = options(dir.path(), Mode::Disk);
    options.contract = Some([7; 32]);
    options.seed[0] = 2;
    options.ssd = true;
    let result = plot(&options).unwrap();
    check_plot(&std::fs::read(&result.path).unwrap(), &result.header);
    options.seed = [1; 32];
    options.cancelled.store(true, Ordering::Relaxed);
    assert!(plot(&options)
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    options.cancelled.store(false, Ordering::Relaxed);
    options.allow_small_k = false;
    assert!(plot(&options)
        .unwrap_err()
        .to_string()
        .contains("allow-small-k"));
}

#[test]
fn failure_does_not_leave_temporary_tables_or_publish_a_plot() {
    let dir = tempfile::tempdir().unwrap();
    let mut options = options(dir.path(), Mode::Disk);
    options.contract = Some([7; 32]);
    // This development seed grows beyond what k12 pointers can represent.
    assert!(plot(&options)
        .unwrap_err()
        .to_string()
        .contains("too many live entries"));
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn opencl_memhash_plots_are_byte_identical_in_both_modes_and_formats() {
    if std::env::var("MMX_OPENCL_TESTS").as_deref() != Ok("1") {
        return;
    }
    for ssd in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut cpu = options(dir.path(), Mode::Ram);
        cpu.ssd = ssd;
        let result = plot(&cpu).unwrap();
        let expected = std::fs::read(result.path).unwrap();
        for mode in [Mode::Ram, Mode::Disk] {
            let dir = tempfile::tempdir().unwrap();
            let mut gpu = options(dir.path(), mode);
            gpu.ssd = ssd;
            gpu.opencl_device = Some(0);
            let result = plot(&gpu).unwrap();
            let actual = std::fs::read(result.path).unwrap();
            assert_eq!(actual, expected, "{mode:?}, SSD={ssd}");
            check_plot(&actual, &result.header);
        }
    }
}
#[test]
fn explicitly_requested_unavailable_gpu_fails_before_creating_data() {
    let dir = tempfile::tempdir().unwrap();
    let mut options = options(dir.path(), Mode::Disk);
    options.opencl_device = Some(usize::MAX);
    assert!(plot(&options)
        .unwrap_err()
        .to_string()
        .contains("initialize OpenCL memhash"));
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}
