use librism::drivers::rism::Compress;
use librism::Calculator;
use librism::{
    drivers::rism::{RISMDriver, Verbosity},
    grids::radial_grid::Grid,
    io::writer::RISMWriter,
    thermodynamics::thermo::{SFEs, TDDriver},
};
use std::path::PathBuf;

struct Args {
    input_file: PathBuf,
    verbosity: Verbosity,
    compress: Compress,
}

fn print_usage_instructions() {
    println!("Usage: rism [OPTIONS] <inputfile.toml>\n");
    println!("\tRuns the RISM solver and solves the problem defined in <inputfile.toml>\n");
    println!("Options:");
    println!(" [-h|--help]          Show this help message");
    println!(" [-c|--compress]      Compress the solvent-solvent problem for future use");
    println!(" Verbosity:");
    println!("  [-q|--quiet]        Suppress all output from solver (DEFAULT)");
    println!("  [-v|--verbose]      Print basic information from solver");
    println!("  [-l|--loud]         Print all information from solver");
    std::process::exit(0);
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut verbosity = Verbosity::Quiet;
    let mut input_file: Option<PathBuf> = None;
    let mut compress: Compress = Compress::NoCompress;
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Short('q') | Long("quiet") => verbosity = Verbosity::Quiet,
            Short('v') | Long("verbose") => verbosity = Verbosity::Verbose,
            Short('l') | Long("loud") => verbosity = Verbosity::VeryVerbose,
            Short('c') | Long("compress") => compress = Compress::Compress,
            Short('h') | Long("help") => print_usage_instructions(),

            Value(val) => input_file = Some(val.into()),
            _ => return Err(arg.unexpected()),
        }
    }

    Ok(Args {
        input_file: input_file.ok_or("Missing input file. Use `rism -h` to see usage.")?,
        verbosity,
        compress,
    })
}

fn main() -> Result<(), lexopt::Error> {
    #[cfg(feature = "dhat-on")]
    let _dhat = dhat::Profiler::new_heap();
    let args = parse_args()?;
    Calculator::new(args.input_file, args.verbosity, args.compress).execute();
    Ok(())
}
