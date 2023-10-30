use librism::{
    data::Grid,
    driver::{RISMDriver, Verbosity},
    thermo::{SFEs, TDDriver},
    writer::RISMWriter,
};
use ndarray::{Array, Array2};
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
    let vv = &solutions.vv;
    let uv = solutions.uv.as_ref().unwrap();
    let density = {
        let mut dens_vec: Vec<f64> = Vec::new();
        for i in vv.data_config.solvent_species.clone().into_iter() {
            for _j in i.atom_sites {
                dens_vec.push(i.dens);
            }
        }
        Array2::from_diag(&Array::from_vec(dens_vec))
    };
    let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
    SFEs::new(
        1.0 / vv.data_config.kt / vv.data_config.temp,
        vv.data_config.ku,
        &uv.correlations,
        &density,
        &grid.rgrid,
    );
    let td = TDDriver::new(solutions);
    td.print_thermo();

    Ok(())
}
