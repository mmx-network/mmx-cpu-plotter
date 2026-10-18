use mmx_cpu_plotter::{mmx::PlotHeader, plot_id, pos, VnxType};
use vnx::serde_json::Value;

fn hash(s: &Value) -> [u8; 32] {
    hex::decode(s.as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap()
}
#[test]
fn cpp_memory_hash_and_f1_vectors_including_production_ksizes() {
    let mut count = 0;
    for row in include_str!("fixtures/leaves.jsonl").lines() {
        let row: Value = vnx::serde_json::from_str(row).unwrap();
        let (y, meta) = pos::compute_f1(
            row["x"].as_u64().unwrap() as u32,
            &hash(&row["id"]),
            row["ksize"].as_u64().unwrap() as u8,
        )
        .unwrap();
        assert_eq!(y as u64, row["y"].as_u64().unwrap());
        let expected: Vec<u32> = row["meta"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(meta.as_slice(), expected);
        let key = hex::decode(row["key"].as_str().unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(
            hex::encode(pos::memory_hash(&key)),
            row["memory_hash"].as_str().unwrap()
        );
        count += 1;
    }
    assert_eq!(count, 32);
}
#[test]
fn header_uses_exact_native_identity_and_plot_id_domains() {
    assert_eq!(PlotHeader::TYPE_HASH, 0x299c5790983c47b6);
    assert_eq!(PlotHeader::CODE_HASH, 0x7b8ddf3d002be0f0);
    assert_ne!(
        plot_id(29, &[0; 32], &[0; 33], None),
        plot_id(29, &[0; 32], &[0; 33], Some(&[0; 32]))
    );
    assert_ne!(
        plot_id(29, &[0; 32], &[0; 33], None),
        plot_id(30, &[0; 32], &[0; 33], None)
    );
}
