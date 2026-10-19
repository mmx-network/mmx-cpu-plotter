//! Native C0 MMX CPU plotting, using RAM or external merge sorting on disk.
mod encoding;
mod io;
pub mod opencl;
mod plot;
pub mod pos;
mod sort;
mod storage;

pub type Hash = [u8; 32];
pub use ::vnx::VnxType;
pub use plot::{plot, plot_id, Options, PlotResult};
pub use storage::Mode;
#[allow(clippy::all)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/header.rs"));
}
pub use generated::*;
