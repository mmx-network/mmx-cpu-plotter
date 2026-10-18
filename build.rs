fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=schema/PlotHeader.vni");
    let mut compiler = vnx_codegen::Compiler::new();
    // Native wire layouts for the three external MMX byte-array types.
    compiler.add_source("externs", "package mmx; extern hash_t = {11,32,1}; extern addr_t = {11,32,1}; extern pubkey_t = {11,33,1};")?;
    compiler.add_path("schema/PlotHeader.vni")?;
    let compiled = compiler.compile()?;
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(output.join("header.rs"), vnx_codegen::generate(&compiled)?)?;
    Ok(())
}
