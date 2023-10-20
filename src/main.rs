use librism::{
    driver::{RISMDriver, Verbosity},
    thermo::TDDriver,
    writer::RISMWriter,
};
use std::path::PathBuf;

struct Args {
    input_file: PathBuf,
    verbosity: Verbosity,
    compress: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut verbosity = Verbosity::Quiet;
    let mut input_file: Option<PathBuf> = None;
    let mut compress: bool = false;
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Short('q') | Long("quiet") => verbosity = Verbosity::Quiet,
            Short('v') | Long("verbose") => verbosity = Verbosity::Verbose,
            Short('l') | Long("loud") => verbosity = Verbosity::VeryVerbose,
            Short('c') | Long("compress") => compress = true,
            Value(val) => input_file = Some(val.into()),
            _ => return Err(arg.unexpected()),
        }
    }

    Ok(Args {
        input_file: input_file.ok_or("Missing input file")?,
        verbosity,
        compress,
    })
}

fn main() -> Result<(), lexopt::Error> {
    let args = parse_args()?;
    let mut driver = RISMDriver::from_toml(&args.input_file);
    let solutions = driver.execute(args.verbosity, args.compress);
    let writer = RISMWriter::new(
        &args
            .input_file
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string(),
        &solutions,
    );
    writer.write().unwrap();
    let td = TDDriver::new(solutions);
    println!(
        "Isothermal Compressibility: {}",
        td.isothermal_compressibility()
    );
    println!(
        "Molecular KB theory PMV: {} (A^3)",
        td.kb_partial_molar_volume()
    );
    println!(
        "Molecular KB theory PMV: {} (cm^3 / mol)",
        td.kb_partial_molar_volume() / 1e24 * 6.022e23
    );
    println!(
        "RISM KB theory PMV: {} (A^3)",
        td.rism_kb_partial_molar_volume()
    );
    println!(
        "RISM KB theory PMV: {} (cm^3 / mol)",
        td.rism_kb_partial_molar_volume() / 1e24 * 6.022e23
    );
    println!(
        "Dimensionless Molecular KB theory PMV: {} (A^3)",
        td.dimensionless_partial_molar_volume()
    );
    Ok(())
}
