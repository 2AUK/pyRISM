use librism::driver::{RISMDriver, Verbosity};
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
            Short('q') => verbosity = Verbosity::Quiet,
            Short('v') => verbosity = Verbosity::Verbose,
            Short('l') => verbosity = Verbosity::VeryVerbose,
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
    let mut driver = RISMDriver::from_toml(args.input_file);
    let solutions = driver.execute(args.verbosity, args.compress);
    Ok(())
}
