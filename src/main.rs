use anyhow::{ensure, Context, Result};
use bech32::{FromBase32, Variant};
use clap::Parser;
use mmx_cpu_plotter::{plot, Mode, Options};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Parser, Debug)]
#[command(version, about = "CPU plotter for native MMX C0 HDD and SSD plots")]
struct Args {
    /// Compressed secp256k1 farmer public key (66 hex digits)
    #[arg(short = 'f', long, required_unless_present = "list_opencl_devices")]
    farmer_key: Option<String>,
    /// Plot NFT contract: mmx1... Bech32m address or 64 hex digits
    #[arg(short = 'c', long)]
    contract: Option<String>,
    #[arg(short = 'k', long, default_value_t = 29)]
    ksize: u8,
    /// Keep all temporary data in RAM, or externally sort through temporary files
    #[arg(long, value_enum, default_value_t = Mode::Disk)]
    mode: Mode,
    /// Omit final metadata (requires SSD IOPS when farming)
    #[arg(long)]
    ssd: bool,
    /// Existing destination directory for completed .plot files
    #[arg(short = 'd', long, default_value = ".")]
    output: PathBuf,
    /// Existing temporary directory for disk mode (defaults to the destination)
    #[arg(short = 't', long)]
    temp_dir: Option<PathBuf>,
    /// CPU worker threads (defaults to available parallelism)
    #[arg(short = 'r', long)]
    threads: Option<usize>,
    /// Use OpenCL for table-1 memhash (all SHA-512 hashing stays on CPU)
    #[arg(long)]
    opencl: bool,
    /// OpenCL GPU index from --list-opencl-devices (defaults to 0)
    #[arg(long, requires = "opencl")]
    opencl_device: Option<usize>,
    /// List OpenCL GPUs and exit
    #[arg(long)]
    list_opencl_devices: bool,
    #[arg(short = 'n', long, default_value_t = 1)]
    count: usize,
    /// Reproducible 32-byte seed in hex; allowed only with --count 1
    #[arg(long)]
    seed: Option<String>,
    /// Permit k8..k28 development plots, which cannot be farmed on mainnet
    #[arg(long)]
    allow_small_k: bool,
    /// Use aligned direct I/O where supported; disable with --direct-io=false
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, require_equals = true, default_missing_value = "true")]
    direct_io: bool,
}
fn decode_hex<const N: usize>(s: &str) -> Result<[u8; N]> {
    let bytes = hex::decode(s.strip_prefix("0x").unwrap_or(s)).context("invalid hex")?;
    bytes.try_into().map_err(|v: Vec<u8>| {
        anyhow::anyhow!("expected {} hex digits, received {}", N * 2, v.len() * 2)
    })
}
fn contract(s: &str) -> Result<[u8; 32]> {
    if s == "MMX" {
        return Ok([0; 32]);
    }
    if s.to_ascii_lowercase().starts_with("mmx1") {
        let (hrp, words, variant) = bech32::decode(s).context("invalid contract address")?;
        ensure!(
            hrp == "mmx" && variant == Variant::Bech32m,
            "contract must be an MMX Bech32m address"
        );
        let mut bytes =
            Vec::<u8>::from_base32(&words).context("invalid contract address padding")?;
        ensure!(bytes.len() == 32, "contract address must encode 32 bytes");
        // MMX addresses encode a big-endian integer; native addr_t stores LE bytes.
        bytes.reverse();
        return Ok(bytes.try_into().unwrap());
    }
    decode_hex(s)
}
fn run() -> Result<()> {
    let args = Args::parse();
    if args.list_opencl_devices {
        let devices = mmx_cpu_plotter::opencl::devices()?;
        ensure!(!devices.is_empty(), "no OpenCL GPUs found");
        for device in devices {
            println!(
                "GPU {}: {} ({})",
                device.index, device.name, device.platform
            );
        }
        return Ok(());
    }
    ensure!(args.count > 0, "count must be positive");
    ensure!(
        args.seed.is_none() || args.count == 1,
        "--seed requires --count 1 to avoid duplicate plots"
    );
    let farmer_key = decode_hex(
        args.farmer_key
            .as_deref()
            .context("--farmer-key is required")?,
    )
    .context("farmer key")?;
    let contract = args.contract.as_deref().map(contract).transpose()?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let handler_cancelled = cancelled.clone();
    ctrlc::set_handler(move || {
        handler_cancelled.store(true, Ordering::Relaxed);
    })
    .context("install interrupt handler")?;
    let mut options = Options {
        ksize: args.ksize,
        farmer_key,
        contract,
        seed: [0; 32],
        mode: args.mode,
        ssd: args.ssd,
        temp_dir: args.temp_dir.unwrap_or_else(|| args.output.clone()),
        output: args.output,
        threads: args
            .threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from)),
        allow_small_k: args.allow_small_k,
        direct_io: args.direct_io,
        opencl_device: args.opencl.then_some(args.opencl_device.unwrap_or(0)),
        cancelled,
    };
    for index in 0..args.count {
        if let Some(seed) = &args.seed {
            options.seed = decode_hex(seed).context("seed")?;
        } else {
            getrandom::getrandom(&mut options.seed)
                .map_err(|e| anyhow::anyhow!("obtain plot seed: {e}"))?;
        }
        eprintln!("Plot {} / {}", index + 1, args.count);
        let result = plot(&options)?;
        println!("{}", result.path.display());
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bech32::ToBase32;
    #[test]
    fn opencl_is_opt_in_and_device_listing_needs_no_farmer_key() {
        assert!(!Args::parse_from(["plotter", "--farmer-key", "key"]).opencl);
        let args = Args::parse_from([
            "plotter",
            "--farmer-key",
            "key",
            "--opencl",
            "--opencl-device",
            "2",
        ]);
        assert!(args.opencl);
        assert_eq!(args.opencl_device, Some(2));
        assert!(
            Args::try_parse_from(["plotter", "--farmer-key", "key", "--opencl-device", "2"])
                .is_err()
        );
        assert!(Args::parse_from(["plotter", "--list-opencl-devices"]).list_opencl_devices);
        assert!(Args::try_parse_from(["plotter"]).is_err());
    }
    #[test]
    fn direct_io_defaults_on_and_can_be_disabled() {
        assert!(Args::parse_from(["plotter", "--farmer-key", "key"]).direct_io);
        assert!(Args::parse_from(["plotter", "--farmer-key", "key", "--direct-io"]).direct_io);
        assert!(
            !Args::parse_from(["plotter", "--farmer-key", "key", "--direct-io=false"]).direct_io
        );
    }
    #[test]
    fn address_decoding_preserves_native_endianness_and_validates_checksums() {
        let mut bytes = [0; 32];
        bytes[0] = 0xab;
        bytes[31] = 0xcd;
        let address = bech32::encode("mmx", bytes.to_base32(), Variant::Bech32m).unwrap();
        bytes.reverse();
        assert_eq!(contract(&address).unwrap(), bytes);
        assert_eq!(contract(&hex::encode(bytes)).unwrap(), bytes);
        assert!(
            contract(&bech32::encode("mmx", bytes.to_base32(), Variant::Bech32).unwrap()).is_err()
        );
        assert!(
            contract(&bech32::encode("other", bytes.to_base32(), Variant::Bech32m).unwrap())
                .is_err()
        );
        assert!(contract("mmx1invalid").is_err());
        assert!(decode_hex::<32>("00").is_err());
    }
}
