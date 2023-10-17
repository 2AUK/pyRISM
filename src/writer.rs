use crate::solution::Solutions;
use hdf5::{File, Result};
use std::path::PathBuf;

pub struct RISMWriter {
    pub name: String,
    pub data: Solutions,
}

impl RISMWriter {
    pub fn new(name: &String, data: Solutions) -> Self {
        RISMWriter {
            name: name.clone(),
            data,
        }
    }

    pub fn write(&self) -> Result<()> {
        let file = File::create(&self.name)?;
        let correlations_group = file.create_group("correlations")?;
        let direct_group = correlations_group.create_group("direct")?;

        let cr_builder = direct_group.new_dataset_builder();
        let cr = cr_builder
            .with_data(&self.data.vv.correlations.cr)
            .create("direct correlation function")?;
        println!("{:?}", file);
        Ok(())
    }
}
