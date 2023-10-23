use crate::data::{Site, Species};
use crate::quaternion::{cross_product, Quaternion};
use ndarray::{arr1, s, Array, Array1, Array2, Zip};
use ndarray_linalg::Eig;
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

pub fn total_charge_species(species: &[Species]) -> f64 {
    species.iter().fold(0.0, |acc, x| {
        acc + x
            .atom_sites
            .iter()
            .fold(0.0, |acc_inner, y| acc_inner + y.params.last().unwrap())
    })
}

#[inline(always)]
pub fn total_charge(atoms: &[Site]) -> f64 {
    atoms
        .iter()
        .fold(0.0, |acc, x| acc + x.params.last().unwrap().abs())
}

pub fn centre_of_charge_species(species: &[Species]) -> Array1<f64> {
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
pub fn centre_of_charge(atoms: &[Site]) -> Array1<f64> {
    atoms.iter().fold(Array::zeros(3), |acc, x| {
        acc + (Array::from_shape_vec(3, x.coords.clone()).unwrap() * x.params.last().unwrap().abs())
    })
}

#[inline(always)]
pub fn translate(atoms: &mut [Site], coords: &Array1<f64>) {
    for site in atoms.iter_mut() {
        let site_coords = Array::from_iter(site.coords.clone().into_iter());
        let new_coords = site_coords - coords;
        site.coords = new_coords.to_vec();
    }
}

pub fn dipole_moment(atoms: &[Site]) -> Result<(Array1<f64>, f64), DipoleError> {
    let dmvec = atoms
        .iter()
        .fold(Array::zeros(3), |acc_inner: Array1<f64>, y| {
            acc_inner
                + (Array::from_shape_vec(3, y.coords.clone()).unwrap() * *y.params.last().unwrap())
        });
    let dm = dmvec.mapv(|x| x.powf(2.0)).sum().sqrt();
    match dm < 1e-16 {
        true => Err(DipoleError),
        _ => Ok((dmvec, dm)),
    }
}

pub fn reorient(atoms: &mut [Site]) -> Result<(), DipoleError> {
    // Compute dipole moment and corrresponding vector
    let (dmvec, dm) = dipole_moment(atoms)?;
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
    for site in atoms.iter_mut() {
        let site_coords = Array::from_iter(site.coords.clone().into_iter());
        let new_coords = quat.rotate(&site_coords);
        site.coords = new_coords.to_vec();
    }

    let mut temp_atoms = atoms.to_owned().clone();

    // zero the z-axis, so that only the x and y axes are aligned
    for site in temp_atoms.iter_mut() {
        site.coords = vec![site.coords[0], site.coords[1], 0.0];
    }

    // Compute moment of inertia matrix based on charges
    let moi_matrix = moment_of_inertia(&temp_atoms);

    let (_, pa) = moi_matrix
        .clone()
        .eig()
        .expect("eigendecomposition of moment of inertia matrix");

    let mut principal_axes: Array2<f64> = Array::zeros(pa.raw_dim());
    Zip::from(&mut principal_axes)
        .and(&pa)
        .for_each(|elem, &i| {
            *elem = i.re;
        });

    // // Find the principal axes
    // let (v, mut principal_axes) = moi_matrix
    //     .into_triangular(UPLO::Upper)
    //     .eigh(UPLO::Upper)
    //     .expect("eigendecomposition of moment of inertia matrix");
    //
    // println!("principal axes:\n{}\nvalues:\n{}", principal_axes, v);

    // Orient the first principal axis to the x-axis
    let x_pa: Array1<f64> = principal_axes.slice(s![0..3, 0]).to_owned();
    let mut x_angle = (f64::min(1.0, f64::max(-1.0, x_pa.dot(&xaxis)))).acos();
    let dir: Array1<f64>;
    if x_angle < (PI - 1e-6) && x_angle > (-PI + 1e-6) {
        dir = cross_product(&x_pa, &xaxis);
    } else {
        dir = yaxis.clone();
    }
    let x_checkvec = cross_product(&dir, &x_pa);
    if x_checkvec.dot(&xaxis).is_sign_negative() {
        x_angle = -x_angle;
    }
    let pa_x_quat = Quaternion::from_axis_angle(x_angle, &dir);

    for site in temp_atoms.iter_mut() {
        let site_coords = Array::from_iter(site.coords.clone().into_iter());
        let new_coords = pa_x_quat.rotate(&site_coords);
        site.coords = new_coords.to_vec();
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
    for site in temp_atoms.iter_mut() {
        let site_coords = Array::from_iter(site.coords.clone().into_iter());
        let new_coords = pa_y_quat.rotate(&site_coords);
        site.coords = new_coords.to_vec();
    }
    for i in 0..3 {
        let pa_slice = principal_axes.slice(s![0..3, i]).clone().to_owned();
        principal_axes
            .slice_mut(s![0..3, i])
            .assign(&pa_y_quat.rotate(&pa_slice));
    }

    // Set the original atom struct coordinates to the new ones
    for (origin_site, temp_site) in atoms.iter_mut().zip(temp_atoms.iter()) {
        origin_site.coords[0] = temp_site.coords[0];
        origin_site.coords[1] = temp_site.coords[1];
    }
    Ok(())
}

#[inline(always)]
fn moment_of_inertia(atoms: &[Site]) -> Array2<f64> {
    let mut out_arr = Array::zeros((3, 3));
    let charges = Array::from_shape_vec(
        atoms.len(),
        atoms
            .iter()
            .map(|atom| atom.params.last().unwrap())
            .collect(),
    )
    .unwrap();
    for (i, iat) in atoms.iter().enumerate() {
        let ci = charges[[i]].abs();
        out_arr[[0, 0]] += ci * (iat.coords[1] * iat.coords[1] + iat.coords[2] * iat.coords[2]);
        out_arr[[1, 1]] += ci * (iat.coords[0] * iat.coords[0] + iat.coords[2] * iat.coords[2]);
        out_arr[[2, 2]] += ci * (iat.coords[0] * iat.coords[0] + iat.coords[1] * iat.coords[1]);
        out_arr[[0, 1]] -= ci * iat.coords[0] * iat.coords[1];
        out_arr[[0, 2]] -= ci * iat.coords[0] * iat.coords[2];
        out_arr[[1, 2]] -= ci * iat.coords[1] * iat.coords[2];
    }

    out_arr[[1, 0]] = out_arr[[0, 1]];
    out_arr[[2, 0]] = out_arr[[0, 2]];
    out_arr[[2, 1]] = out_arr[[2, 1]];
    out_arr
}

pub fn dmtodebye(dm: f64) -> f64 {
    dm * 1.60217663e-29 * 2.997924581780902e29
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::data::Site;
    use approx::assert_relative_eq;

    const PRECISION: f64 = 1e-8;

    #[test]
    fn test_total_charge() {
        let ow = Site {
            atom_type: "Ow".to_string(),
            params: vec![0.0, 0.0, -0.8476000010975563],
            coords: vec![0.0, 0.0, 0.0],
        };
        let h1w = Site {
            atom_type: "H1w".to_string(),
            params: vec![0.0, 0.0, 0.4238],
            coords: vec![1.0, 0.0, 0.0],
        };
        let h2w = Site {
            atom_type: "H2w".to_string(),
            params: vec![0.0, 0.0, 0.4238],
            coords: vec![-3.33314000e-01, 9.42816000e-01, 0.00000000e+00],
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
        let mut calculated_result = arr1(&[0.0, 0.0, 0.0]);
        for spec in species.iter_mut() {
            match dipole_moment(&spec.atom_sites) {
                Ok(_) => {
                    let tot_charge = total_charge(&spec.atom_sites);
                    let mut coc = centre_of_charge(&spec.atom_sites);
                    coc /= tot_charge;
                    translate(&mut spec.atom_sites, &coc);
                    reorient(&mut spec.atom_sites).unwrap();
                    (calculated_result, _) = dipole_moment(&spec.atom_sites).unwrap();
                }
                Err(_) => {
                    println!("{} has no dipole moment", spec.species_name);
                }
            }
        }
        let known_result = arr1(&[0.0, 0.0, 0.4893692265207899]);
        assert_relative_eq!(
            known_result,
            calculated_result,
            max_relative = PRECISION,
            epsilon = PRECISION
        );

        // println!(
        //     "Dipole moment vector pre-alignment:\n{:?}",
        //     dipole_moment(&species).unwrap()
        // );
        //
        // let tot_charge = total_charge(&species);
        // let mut coc = centre_of_charge(&species);
        // println!("total: {}\ncentre: {}", tot_charge, coc);
        //
        // coc /= tot_charge;
        //
        // translate(&mut species, &coc);
        //
        // reorient(&mut species);
        //
        // println!(
        //     "Dipole moment vector post-alignment:\n{:?}",
        //     dipole_moment(&species).unwrap()
        // );
    }
}
