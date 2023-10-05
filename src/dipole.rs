use crate::data::{Site, Species};
use crate::quaternion::{cross_product, Quaternion};
use ndarray::{arr1, s, Array, Array1, Array2, Slice};
use ndarray_linalg::{Eigh, IntoTriangular, UPLO};
use std::cmp::{max, min};
use std::f64::consts::PI;
use std::fmt;

#[derive(Debug, Clone)]
pub struct DipoleError;

impl fmt::Display for DipoleError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "0 dipole moment for current system")
    }
}

pub fn total_charge(species: &[Species]) -> f64 {
    species.iter().fold(0.0, |acc, x| {
        acc + x
            .atom_sites
            .iter()
            .fold(0.0, |acc_inner, y| acc_inner + y.params.last().unwrap())
    })
}

#[inline(always)]
pub fn centre_of_charge(species: &[Species]) -> Array1<f64> {
    species.iter().fold(Array::zeros(3), |acc: Array1<f64>, x| {
        acc + x
            .atom_sites
            .iter()
            .fold(Array::zeros(3), |acc_inner: Array1<f64>, y| {
                acc_inner
                    + (Array::from_shape_vec(3, y.coords.clone()).unwrap()
                        * y.params.last().unwrap().abs())
            })
    })
}

#[inline(always)]
pub fn translate(species: &mut [Species], coords: &Array1<f64>) {
    for elem in species.iter_mut() {
        for site in elem.atom_sites.iter_mut() {
            let site_coords = Array::from_iter(site.coords.clone().into_iter());
            let new_coords = site_coords - coords;
            site.coords = new_coords.to_vec();
        }
    }
}

pub fn dipole_moment(species: &[Species]) -> Result<(Array1<f64>, f64), DipoleError> {
    let dmvec = species.iter().fold(Array::zeros(3), |acc: Array1<f64>, x| {
        acc + x
            .atom_sites
            .iter()
            .fold(Array::zeros(3), |acc_inner: Array1<f64>, y| {
                acc_inner
                    + (Array::from_shape_vec(3, y.coords.clone()).unwrap()
                        * y.params.last().unwrap().abs())
            })
    });
    let dm = dmvec.mapv(|x| x.powf(2.0)).sum().sqrt();
    match dm < 1e-16 {
        true => Err(DipoleError),
        _ => Ok((dmvec, dm)),
    }
}

pub fn reorient(species: &mut [Species]) -> Result<(), DipoleError> {
    // Compute dipole moment and corrresponding vector
    let (dmvec, dm) = dipole_moment(species)?;
    let rdmdm = &dmvec / dm;
    let xaxis = arr1(&[1.0, 0.0, 0.0]);
    let yaxis = arr1(&[0.0, 1.0, 0.0]);
    let zaxis = arr1(&[0.0, 0.0, 1.0]);

    // Find quaternion to rotate molecule by
    let rotvec = cross_product(&zaxis, &rdmdm);
    let checkvec = cross_product(&rotvec, &rdmdm);
    let angle = rdmdm.dot(&zaxis).acos().copysign(checkvec.dot(&zaxis));
    let quat = Quaternion::from_axis_angle(angle, &rotvec);

    // Perform rotation of dipole to z-axis
    for elem in species.iter_mut() {
        for site in elem.atom_sites.iter_mut() {
            let site_coords = Array::from_iter(site.coords.clone().into_iter());
            let new_coords = quat.rotate(&site_coords);
            site.coords = new_coords.to_vec();
        }
    }

    // Compute moment of inertia matrix based on charges
    let moi_matrix = moment_of_inertia(species);

    // Find the principal axes
    let (_, mut principal_axes) = moi_matrix
        .into_triangular(UPLO::Upper)
        .eigh(UPLO::Upper)
        .expect("eigendecomposition of moment of inertia matrix");

    // Orient the first principal axis to the x-axis
    let x_pa: Array1<f64> = principal_axes.slice(s![0..3, 0]).to_owned();
    let mut x_angle = (f64::min(1.0, f64::max(-1.0, x_pa.dot(&xaxis)))).acos();
    let dir: Array1<f64>;
    if x_angle < PI - 1e-6 && x_angle > -PI + 1e-6 {
        dir = cross_product(&x_pa, &xaxis);
    } else {
        dir = yaxis.clone();
    }
    let x_checkvec = cross_product(&dir, &x_pa);
    if x_checkvec.dot(&xaxis).is_sign_negative() {
        x_angle = -x_angle;
    }
    let pa_x_quat = Quaternion::from_axis_angle(x_angle, &dir);

    for elem in species.iter_mut() {
        for site in elem.atom_sites.iter_mut() {
            let site_coords = Array::from_iter(site.coords.clone().into_iter());
            let new_coords = pa_x_quat.rotate(&site_coords);
            site.coords = new_coords.to_vec();
        }
    }
    for i in 0..3 {
        let pa_slice = principal_axes.slice(s![0..3, i]).clone().to_owned();
        principal_axes
            .slice_mut(s![0..3, i])
            .assign(&pa_x_quat.rotate(&pa_slice));
    }

    // Orient the second (and third) principal axis to the y-axis
    let y_pa: Array1<f64> = principal_axes.slice(s![0..3, 2]).to_owned();
    let mut y_angle = (f64::min(1.0, f64::max(-1.0, y_pa.dot(&yaxis)))).acos();
    let y_checkvec = cross_product(&xaxis, &y_pa);

    if y_checkvec.dot(&yaxis).is_sign_negative() {
        y_angle = -y_angle;
    }

    let pa_y_quat = Quaternion::from_axis_angle(y_angle, &xaxis);
    for elem in species.iter_mut() {
        for site in elem.atom_sites.iter_mut() {
            let site_coords = Array::from_iter(site.coords.clone().into_iter());
            let new_coords = pa_y_quat.rotate(&site_coords);
            site.coords = new_coords.to_vec();
        }
    }
    for i in 0..3 {
        let pa_slice = principal_axes.slice(s![0..3, i]).clone().to_owned();
        principal_axes
            .slice_mut(s![0..3, i])
            .assign(&pa_y_quat.rotate(&pa_slice));
    }
    Ok(())
}

#[inline(always)]
fn moment_of_inertia(species: &[Species]) -> Array2<f64> {
    let mut out_arr = Array::zeros((3, 3));
    let atoms: Vec<Site> = species.iter().flat_map(|x| x.atom_sites.clone()).collect();
    let charges = Array::from_shape_vec(
        atoms.len(),
        atoms
            .iter()
            .map(|atom| atom.params.last().unwrap())
            .collect(),
    )
    .unwrap();
    for (i, iat) in atoms.iter().enumerate() {
        let ci = *charges[[i]];
        out_arr[[0, 0]] += ci * iat.coords[1] * iat.coords[1] + iat.coords[2] * iat.coords[2];
        out_arr[[1, 1]] += ci * iat.coords[0] * iat.coords[0] + iat.coords[2] * iat.coords[2];
        out_arr[[2, 2]] += ci * iat.coords[1] * iat.coords[1] + iat.coords[0] * iat.coords[0];
        out_arr[[0, 1]] -= ci * iat.coords[0] * iat.coords[1];
        out_arr[[0, 2]] -= ci * iat.coords[0] * iat.coords[2];
        out_arr[[1, 2]] -= ci * iat.coords[1] * iat.coords[2];
    }

    out_arr[[1, 0]] = out_arr[[0, 1]];
    out_arr[[2, 0]] = out_arr[[0, 2]];
    out_arr[[2, 1]] = out_arr[[2, 1]];
    out_arr
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::data::Site;

    #[test]
    fn test_total_charge() {
        let ow = Site {
            atom_type: "Ow".to_string(),
            params: vec![0.0, 0.0, -1.0],
            coords: vec![0.0, 0.0, 0.0],
        };
        let h1w = Site {
            atom_type: "H1w".to_string(),
            params: vec![0.0, 0.0, 0.5],
            coords: vec![0.0, 1.0, 0.0],
        };
        let h2w = Site {
            atom_type: "H2w".to_string(),
            params: vec![0.0, 0.0, 0.5],
            coords: vec![0.0, 0.0, 1.0],
        };
        let na = Site {
            atom_type: "Na".to_string(),
            params: vec![0.0, 0.0, 1.0],
            coords: vec![0.0, 0.0, 0.0],
        };
        let cl = Site {
            atom_type: "Cl".to_string(),
            params: vec![0.0, 0.0, -1.0],
            coords: vec![0.0, 0.0, 0.0],
        };
        let water = Species {
            species_name: "Water".to_string(),
            dens: 0.0,
            ns: 3,
            atom_sites: vec![ow, h1w, h2w],
        };
        let sodium_ion = Species {
            species_name: "Sodium".to_string(),
            dens: 0.0,
            ns: 1,
            atom_sites: vec![na],
        };
        let chlorine_ion = Species {
            species_name: "Chlorine".to_string(),
            dens: 0.0,
            ns: 1,
            atom_sites: vec![cl],
        };
        let mut species = vec![water, sodium_ion, chlorine_ion];

        println!(
            "Dipole moment vector pre-alignment:\n{:?}",
            dipole_moment(&species).unwrap()
        );

        let tot_charge = total_charge(&species);
        let mut coc = centre_of_charge(&species);
        println!("total: {}\ncentre: {}", tot_charge, coc);

        coc /= tot_charge;

        translate(&mut species, &coc);

        reorient(&mut species);

        println!(
            "Dipole moment vector post-alignment:\n{:?}",
            dipole_moment(&species).unwrap()
        );
    }
}
