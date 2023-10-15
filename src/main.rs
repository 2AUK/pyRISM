use librism::driver::RISMDriver;
use std::path::PathBuf;

struct Args {
    input_file: PathBuf,
    verbosity: String,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut verbosity: String = "quiet".to_string();
    let mut input_file: Option<PathBuf> = None;
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Short('q') => verbosity = "quiet".to_string(),
            Short('v') => verbosity = "verbose".to_string(),
            Short('l') => verbosity = "vverbose".to_string(),
            Value(val) => input_file = Some(val.into()),
            _ => return Err(arg.unexpected()),
        }
    }

    Ok(Args {
        input_file: input_file.ok_or("Missing input file")?,
        verbosity,
    })
}

fn main() -> Result<(), lexopt::Error> {
    let args = parse_args()?;
    println!("{}", args.verbosity);
    println!("{}", args.input_file.display());
    let mut driver = RISMDriver::from_toml(args.input_file);
    driver.execute();
    Ok(())
}
