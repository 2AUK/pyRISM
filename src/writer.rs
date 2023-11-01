use crate::{data::Grid, solution::Solutions};
use csv::{QuoteStyle, WriterBuilder};
use ndarray::Array3;
use std::path::Path;

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
        let gr = &vv.correlations.tr + &vv.correlations.cr + 1.0;
        self.write_file_vv(&gr, "gvv".to_string())?;
        self.write_file_vv(&vv.correlations.tr, "tvv".to_string())?;
        self.write_file_vv(&vv.correlations.cr, "cvv".to_string())?;

        match self.data.uv {
            Some(ref uv) => {
                let gr = &uv.correlations.tr + &uv.correlations.cr + 1.0;
                self.write_file_uv(&gr, "guv".to_string())?;
                self.write_file_uv(&uv.correlations.tr, "tuv".to_string())?;
                self.write_file_uv(&uv.correlations.cr, "cuv".to_string())?;
                Ok(())
            }
            None => Ok(()),
        }
    }

    fn write_file_uv(&self, func: &Array3<f64>, ext: String) -> std::io::Result<()> {
        let vv = &self.data.vv;
        let uv = self.data.uv.as_ref().unwrap();
        let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
        let path_str = format!("{}_rust.{}", self.name, ext);
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
        for iat in uv.data_config.solvent_atoms.iter() {
            for jat in vv.data_config.solvent_atoms.iter() {
                let header_string = format!("{}-{}", &iat.atom_type, &jat.atom_type);
                header_row.push(header_string);
            }
        }
        wtr.write_record(header_row.as_slice())?;
        // tabulate data
        for i in 0..uv.data_config.npts {
            let mut data_row = Vec::new();
            data_row.push(grid.rgrid[[i]].to_string());
            for j in 0..uv.data_config.nsu.unwrap() {
                for k in 0..vv.data_config.nsv {
                    data_row.push(func[[i, j, k]].to_string());
                }
            }
            wtr.write_record(&data_row[..])?;
        }
        wtr.flush()?;
        Ok(())
    }

    fn write_file_vv(&self, func: &Array3<f64>, ext: String) -> std::io::Result<()> {
        let vv = &self.data.vv;
        let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
        let path_str = format!("{}_rust.{}", self.name, ext);
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
        // tabulate data
        for i in 0..vv.data_config.npts {
            let mut data_row = Vec::new();
            data_row.push(grid.rgrid[[i]].to_string());
            for j in 0..vv.data_config.nsv {
                for k in 0..vv.data_config.nsv {
                    data_row.push(func[[i, j, k]].to_string());
                }
            }
            wtr.write_record(&data_row[..])?;
        }
        wtr.flush()?;
        Ok(())
    }
}
