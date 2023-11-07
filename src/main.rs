use librism::drivers::rism::Compress;
use librism::Calculator;
use librism::{
    drivers::rism::{RISMDriver, Verbosity},
    grids::radial_grid::Grid,
    io::writer::RISMWriter,
    thermodynamics::thermo::{SFEs, TDDriver},
};
use ndarray::{Array, Array2};
use std::path::PathBuf;

struct Args {
    input_file: PathBuf,
    verbosity: Verbosity,
    compress: Compress,
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
            Short('h') | Long("help") => {
                println!("Usage: rism [OPTIONS] <inputfile.toml>\n");
                println!(
                    "\tRuns the RISM solver and solves the problem defined in <inputfile.toml>i\n"
                );
                println!("Options:");
                println!(" [-h|--help]\tShow this help message");
                println!(" [-c|--compress]\tCompress the solvent-solvent problem for future use");
                println!(" Verbosity:");
                println!("  [-q|--quiet]\tSuppress all output from solver (DEFAULT)");
                println!("  [-v|--verbose]\tPrint basic information from solver");
                println!("  [-l|--loud]\tPrint all information from solver");
                std::process::exit(0);
            }
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
    #[cfg(feature = "dhat-on")]
    let _dhat = dhat::Profiler::new_heap();
    let args = parse_args()?;
    // let mut driver = RISMDriver::from_toml(&args.input_file);
    // let solutions = driver.execute(args.verbosity, args.compress);
    //
    // let wv = driver.solvent.borrow().wk.clone();
    // let wu = {
    //     match driver.solute {
    //         Some(v) => Some(v.borrow().wk.clone()),
    //         None => None,
    //     }
    // };
    // let td = TDDriver::new(&solutions, wv, wu);
    // let thermo = td.execute();
    // let writer = RISMWriter::new(
    //     &args
    //         .input_file
    //         .file_stem()
    //         .unwrap()
    //         .to_str()
    //         .unwrap()
    //         .to_string(),
    //     &solutions,
    //     &thermo,
    // );
    //writer.write().unwrap();
    Calculator::new(args.input_file, args.verbosity, args.compress).execute();
    Ok(())
}
