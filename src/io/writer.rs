use crate::grids::radial_grid::Grid;
use crate::{data::solution::Solutions, thermodynamics::thermo::Thermodynamics};
use csv::{QuoteStyle, WriterBuilder};
use itertools::Itertools;
use ndarray::{Array1, Array3};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

pub struct RISMWriter<'a> {
    pub name: String,
    pub data: &'a Solutions,
    pub thermo: &'a Thermodynamics,
}

impl<'a> RISMWriter<'a> {
    pub fn new(name: &String, data: &'a Solutions, thermo: &'a Thermodynamics) -> Self {
        RISMWriter {
            name: name.clone(),
            data,
            thermo,
        }
    }

    pub fn write(&self) -> std::io::Result<()> {
        let vv = &self.data.vv;
        let tr = &vv.correlations.tr;
        let cr = &vv.correlations.cr;
        let gr = &vv.correlations.tr + &vv.correlations.cr + 1.0;
        // Set up grid for dumping data with
        let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
        // Construct header row for .csv file from atom-site names
        let mut vv_headers: Vec<String> = vv
            .data_config
            .solvent_atoms
            .iter()
            .cartesian_product(vv.data_config.solvent_atoms.iter())
            .map(|(iat, jat)| format!("{}-{}", &iat.atom_type, &jat.atom_type))
            .collect();
        vv_headers.insert(0, "r(A)".to_string());
        // Write correlation functions
        self.write_correlation(&gr, &grid.rgrid, &vv_headers, "gvv".to_string(), gr.dim())?;
        self.write_correlation(cr, &grid.rgrid, &vv_headers, "cvv".to_string(), gr.dim())?;
        self.write_correlation(tr, &grid.rgrid, &vv_headers, "tvv".to_string(), gr.dim())?;

        // Check for a uv problem and write files if present
        match self.data.uv {
            Some(ref uv) => {
                let tr = &uv.correlations.tr;
                let cr = &uv.correlations.cr;
                let gr = tr + cr + 1.0;
                let mut uv_headers: Vec<String> = uv
                    .data_config
                    .solvent_atoms
                    .iter()
                    .cartesian_product(vv.data_config.solvent_atoms.iter())
                    .map(|(iat, jat)| format!("{}-{}", &iat.atom_type, &jat.atom_type))
                    .collect();

                uv_headers.insert(0, "r(A)".to_string());
                self.write_correlation(&gr, &grid.rgrid, &uv_headers, "gvv".to_string(), gr.dim())?;
                self.write_correlation(cr, &grid.rgrid, &uv_headers, "cvv".to_string(), gr.dim())?;
                self.write_correlation(tr, &grid.rgrid, &uv_headers, "tvv".to_string(), gr.dim())?;

                //Write thermodynamics to file
                let td_path = format!("{}.td", self.name);
                let mut file = File::create(td_path)?;
                file.write_all(self.thermo.to_string().as_bytes())?;

                //Write densities to file
                self.write_density(&grid.rgrid, gr.dim())?;
                Ok(())
            }
            None => {
                //Write solvent-solvent thermodynamics only
                let td_path = format!("{}.td", self.name);
                let mut file = File::create(td_path)?;
                file.write_all(self.thermo.to_string().as_bytes())?;
                Ok(())
            }
        }
    }

    fn write_correlation(
        &self,
        func: &Array3<f64>,
        grid: &Array1<f64>,
        header: &[String],
        ext: String,
        _shape @ (npts, ns1, ns2): (usize, usize, usize),
    ) -> std::io::Result<()> {
        let path_str = &(format!("{}.{}", self.name, ext)).to_string();
        let path = Path::new(path_str);
        let mut wtr = WriterBuilder::new()
            .comment(Some(b'#'))
            .flexible(true)
            .quote_style(QuoteStyle::Never)
            .from_path(path)?;
        wtr.write_record(header)?;
        for i in 0..npts {
            let mut data = Vec::new();
            data.push(grid[[i]].to_string());
            for j in 0..ns1 {
                for k in 0..ns2 {
                    data.push(func[[i, j, k]].to_string());
                }
            }
            wtr.write_record(data)?;
        }
        Ok(())
    }

    fn write_density(
        &self,
        grid: &Array1<f64>,
        _shape @ (npts, _, _): (usize, usize, usize),
    ) -> std::io::Result<()> {
        let path_str = &(format!("{}.duv", self.name)).to_string();
        let path = Path::new(path_str);
        let mut wtr = WriterBuilder::new()
            .comment(Some(b'#'))
            .flexible(true)
            .quote_style(QuoteStyle::Never)
            .from_path(path)?;
        let header = vec![
            "r(A)".to_string(),
            "HNC".to_string(),
            "KH".to_string(),
            "GF".to_string(),
            "PW".to_string(),
            //"PMV".to_string(),
        ];
        wtr.write_record(header.as_slice())?;
        let densities = self.thermo.sfed.as_ref().unwrap();
        for i in 0..npts {
            let data = vec![
                grid[[i]].to_string(),
                densities.hypernettedchain[[i]].to_string(),
                densities.kovalenko_hirata[[i]].to_string(),
                densities.gaussian_fluctuations[[i]].to_string(),
                densities.partial_wave[[i]].to_string(),
                //densities.partial_molar_volume[[i]].to_string(),
            ];
            wtr.write_record(data.as_slice())?;
        }
        Ok(())
    }

    // fn write_file_uv(&self, func: &Array3<f64>, ext: String) -> std::io::Result<()> {
    //     let vv = &self.data.vv;
    //     let uv = self.data.uv.as_ref().unwrap();
    //     let path_str = format!("{}_rust.{}", self.name, ext);
    //     let path = Path::new(&path_str);
    //     let mut wtr = WriterBuilder::new()
    //         .comment(Some(b'#'))
    //         .flexible(true)
    //         .quote_style(QuoteStyle::Never)
    //         .from_path(path)?;
    //     wtr.write_record(&["#input"])?;
    //     let mut header_row = Vec::new();
    //     header_row.push("r(A)".to_string());
    //     // generate header row
    //     for iat in uv.data_config.solvent_atoms.iter() {
    //         for jat in vv.data_config.solvent_atoms.iter() {
    //             let header_string = format!("{}-{}", &iat.atom_type, &jat.atom_type);
    //             header_row.push(header_string);
    //         }
    //     }
    //     wtr.write_record(header_row.as_slice())?;
    //     // tabulate data
    //     for i in 0..uv.data_config.npts {
    //         let mut data_row = Vec::new();
    //         data_row.push(grid.rgrid[[i]].to_string());
    //         for j in 0..uv.data_config.nsu.unwrap() {
    //             for k in 0..vv.data_config.nsv {
    //                 data_row.push(func[[i, j, k]].to_string());
    //             }
    //         }
    //         wtr.write_record(&data_row[..])?;
    //     }
    //     wtr.flush()?;
    //     Ok(())
    // }
    //
    // fn write_file_vv(&self, func: &Array3<f64>, ext: String) -> std::io::Result<()> {
    //     let vv = &self.data.vv;
    //     let grid = Grid::new(vv.data_config.npts, vv.data_config.radius);
    //     let path_str = format!("{}_rust.{}", self.name, ext);
    //     let path = Path::new(&path_str);
    //     let mut wtr = WriterBuilder::new()
    //         .comment(Some(b'#'))
    //         .flexible(true)
    //         .quote_style(QuoteStyle::Never)
    //         .from_path(path)?;
    //     wtr.write_record(&["#input"])?;
    //     let mut header_row = Vec::new();
    //     header_row.push("r(A)".to_string());
    //     // generate header row
    //     for iat in vv.data_config.solvent_atoms.iter() {
    //         for jat in vv.data_config.solvent_atoms.iter() {
    //             let header_string = format!("{}-{}", &iat.atom_type, &jat.atom_type);
    //             header_row.push(header_string);
    //         }
    //     }
    //     wtr.write_record(header_row.as_slice())?;
    //     // tabulate data
    //     for i in 0..vv.data_config.npts {
    //         let mut data_row = Vec::new();
    //         data_row.push(grid.rgrid[[i]].to_string());
    //         for j in 0..vv.data_config.nsv {
    //             for k in 0..vv.data_config.nsv {
    //                 data_row.push(func[[i, j, k]].to_string());
    //             }
    //         }
    //         wtr.write_record(&data_row[..])?;
    //     }
    //     wtr.flush()?;
    //     Ok(())
    // }
}
