use std::path::Path;

use crate::{data::Grid, solution::Solutions};
use csv::{QuoteStyle, WriterBuilder};

pub struct RISMWriter<'a> {
    pub name: String,
    pub data: &'a Solutions,
}

impl<'a> RISMWriter<'a> {
    pub fn new(name: &String, data: &'a Solutions) -> Self {
        RISMWriter {
            name: name.clone(),
            data,
        }
    }

    pub fn write(&self) -> std::io::Result<()> {
        let vv = &self.data.vv;
        let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
        let path_str = format!("{}_rust.csv", self.name);
        let path = Path::new(&path_str);
        let mut wtr = WriterBuilder::new()
            .comment(Some(b'#'))
            .flexible(true)
            .quote_style(QuoteStyle::Never)
            .from_path(path)?;
        wtr.write_record(&["#input"])?;
        let mut header_row = Vec::new();
        header_row.push("r(A)".to_string());
        // generate header row
        for iat in vv.data_config.solvent_atoms.iter() {
            for jat in vv.data_config.solvent_atoms.iter() {
                let header_string = format!("{}-{}", &iat.atom_type, &jat.atom_type);
                header_row.push(header_string);
            }
        }
        wtr.write_record(header_row.as_slice())?;
        let gr = &vv.correlations.tr + &vv.correlations.cr + 1.0;
        // tabulate data
        for i in 0..vv.data_config.npts {
            let mut data_row = Vec::new();
            data_row.push(grid.rgrid[[i]].to_string());
            for j in 0..vv.data_config.nsv {
                for k in 0..vv.data_config.nsv {
                    data_row.push(gr[[i, j, k]].to_string());
                }
            }
            wtr.write_record(&data_row[..])?;
        }
        wtr.flush()?;
        Ok(())
    }
}
