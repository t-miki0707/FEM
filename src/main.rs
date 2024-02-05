extern crate nalgebra as na;
use std::io::BufRead;
use std::io::Write;

const NODE_NUM: usize = 4;
const NODE_DOF: usize = 2;

//Object of elastic D matix
struct Elastic {
    young: f64,
    poisson: f64,
}

impl Elastic {
    fn make_dematrix(&self) -> na::Matrix3<f64> {
        let tmp: f64 = self.young / (1.0 + self.poisson) / (1.0 - 2.0 * self.poisson);
        let mut mat_d = na::Matrix3::<f64>::zeros();
        mat_d[(0, 0)] = 1.0 - self.poisson;
        mat_d[(0, 1)] = self.poisson;
        mat_d[(1, 0)] = self.poisson;
        mat_d[(1, 1)] = 1.0 - self.poisson;
        mat_d[(2, 2)] = 0.5 * (1.0 - 2.0 * self.poisson);
        mat_d *= tmp;
        return mat_d;
    }
}

//Object of global stiffness matrix configured 2-dimentional and 4-nodes
struct C2D4 {
    nodes: na::Matrix2xX<f64>,
    elements: na::Matrix6xX<usize>,
    materials: na::Matrix3xX<f64>,
}

impl C2D4 {
    fn make_global_kmatrix(&self) -> na::DMatrix<f64> {
        let mut mat_k =
            na::DMatrix::<f64>::zeros(NODE_DOF * self.nodes.ncols(), NODE_DOF * self.nodes.ncols());
        let mut ir = na::DVector::<usize>::zeros(NODE_DOF * NODE_NUM);

        for ele in 0..self.elements.ncols() {
            let mat_ke = self.make_kematrix(ele);
            let i = self.elements[(1, ele)];
            let j = self.elements[(2, ele)];
            let k = self.elements[(3, ele)];
            let l = self.elements[(4, ele)];
            //println!("{}", i);
            ir[7] = 2 * l - 1;
            ir[6] = ir[7] - 1;
            ir[5] = 2 * k - 1;
            ir[4] = ir[5] - 1;
            ir[3] = 2 * j - 1;
            ir[2] = ir[3] - 1;
            ir[1] = 2 * i - 1;
            ir[0] = ir[1] - 1;

            for ii in 0..(NODE_DOF * NODE_NUM) {
                let it = ir[ii];

                for jj in 0..(NODE_DOF * NODE_NUM) {
                    let jt = ir[jj];
                    mat_k[(it, jt)] = mat_k[(it, jt)] + mat_ke[(ii, jj)];
                }
            }
        }

        return mat_k;
    }

    fn make_kematrix(&self, ele: usize) -> na::DMatrix<f64> {
        let mut mat_ke = na::DMatrix::<f64>::zeros(NODE_DOF * NODE_NUM, NODE_DOF * NODE_NUM);
        let m = self.elements[(5, ele)] - 1;
        //println!("{}", m);
        let elastic = Elastic {
            young: self.materials[(1, m)],
            poisson: self.materials[(2, m)],
        };
        let mat_d = elastic.make_dematrix();
        //println!("{}", mat_d);

        for ii in 0..NODE_NUM {
            let (mat_b, det_j) = self.make_bmatrix(ele, ii);
            //println!("{}", mat_b);
            mat_ke += mat_b.transpose() * mat_d * mat_b * det_j;
            //println!("{}", mat_ke)
        }
        //println!("{}", mat_ke);

        return mat_ke;
    }

    fn make_bmatrix(&self, ele: usize, ii: usize) -> (na::DMatrix<f64>, f64) {
        let gauss = self.gauss_point(ii);
        let dndab = self.make_dndab(gauss.0, gauss.1);
        //println!("{}", dndab);
        let i = self.elements[(1, ele)] - 1;
        let j = self.elements[(2, ele)] - 1;
        let k = self.elements[(3, ele)] - 1;
        let l = self.elements[(4, ele)] - 1;
        let j11 = dndab[(0, 0)] * self.nodes[(0, i)]
            + dndab[(0, 1)] * self.nodes[(0, j)]
            + dndab[(0, 2)] * self.nodes[(0, k)]
            + dndab[(0, 3)] * self.nodes[(0, l)];
        let j12 = dndab[(0, 0)] * self.nodes[(1, i)]
            + dndab[(0, 1)] * self.nodes[(1, j)]
            + dndab[(0, 2)] * self.nodes[(1, k)]
            + dndab[(0, 3)] * self.nodes[(1, l)];
        let j21 = dndab[(1, 0)] * self.nodes[(0, i)]
            + dndab[(1, 1)] * self.nodes[(0, j)]
            + dndab[(1, 2)] * self.nodes[(0, k)]
            + dndab[(1, 3)] * self.nodes[(0, l)];
        let j22 = dndab[(1, 0)] * self.nodes[(1, i)]
            + dndab[(1, 1)] * self.nodes[(1, j)]
            + dndab[(1, 2)] * self.nodes[(1, k)]
            + dndab[(1, 3)] * self.nodes[(1, l)];
        let det_j = j11 * j22 - j12 * j21;
        let mut mat_b = na::DMatrix::zeros(3, 8);
        mat_b[(0, 0)] = j22 * dndab[(0, 0)] - j12 * dndab[(1, 0)];
        mat_b[(0, 2)] = j22 * dndab[(0, 1)] - j12 * dndab[(1, 1)];
        mat_b[(0, 4)] = j22 * dndab[(0, 2)] - j12 * dndab[(1, 2)];
        mat_b[(0, 6)] = j22 * dndab[(0, 3)] - j12 * dndab[(1, 3)];
        mat_b[(1, 1)] = -j21 * dndab[(0, 0)] + j11 * dndab[(1, 0)];
        mat_b[(1, 3)] = -j21 * dndab[(0, 1)] + j11 * dndab[(1, 1)];
        mat_b[(1, 5)] = -j21 * dndab[(0, 2)] + j11 * dndab[(1, 2)];
        mat_b[(1, 7)] = -j21 * dndab[(0, 3)] + j11 * dndab[(1, 3)];
        mat_b[(2, 0)] = -j21 * dndab[(0, 0)] + j11 * dndab[(1, 0)];
        mat_b[(2, 1)] = j22 * dndab[(0, 0)] - j12 * dndab[(1, 0)];
        mat_b[(2, 2)] = -j21 * dndab[(0, 1)] + j11 * dndab[(1, 1)];
        mat_b[(2, 3)] = j22 * dndab[(0, 1)] - j12 * dndab[(1, 1)];
        mat_b[(2, 4)] = -j21 * dndab[(0, 2)] + j11 * dndab[(1, 2)];
        mat_b[(2, 5)] = j22 * dndab[(0, 2)] - j12 * dndab[(1, 2)];
        mat_b[(2, 6)] = -j21 * dndab[(0, 3)] + j11 * dndab[(1, 3)];
        mat_b[(2, 7)] = j22 * dndab[(0, 3)] - j12 * dndab[(1, 3)];
        mat_b /= det_j;
        //println!("{}", mat_b);
        return (mat_b, det_j);
    }

    fn make_dndab(&self, ai: f64, bi: f64) -> na::Matrix2x4<f64> {
        let dn1da = -0.25 * (1.0 - bi);
        let dn2da = 0.25 * (1.0 - bi);
        let dn3da = 0.25 * (1.0 + bi);
        let dn4da = -0.25 * (1.0 + bi);
        let dn1db = -0.25 * (1.0 - ai);
        let dn2db = -0.25 * (1.0 + ai);
        let dn3db = 0.25 * (1.0 + ai);
        let dn4db = 0.25 * (1.0 - ai);
        let dndab = na::Matrix2x4::new(dn1da, dn2da, dn3da, dn4da, dn1db, dn2db, dn3db, dn4db);
        return dndab;
    }

    fn gauss_point(&self, ii: usize) -> (f64, f64) {
        let tmp = (1.0 / 3.0 as f64).sqrt();

        if ii == 0 {
            let ai = (-1.0) * tmp;
            let bi = (-1.0) * tmp;
            return (ai, bi);
        } else if ii == 1 {
            let ai = tmp;
            let bi = (-1.0) * tmp;
            return (ai, bi);
        } else if ii == 2 {
            let ai = tmp;
            let bi = tmp;
            return (ai, bi);
        } else {
            let ai = (-1.0) * tmp;
            let bi = tmp;
            return (ai, bi);
        }
    }

    fn calcurate_stress(
        &self,
        ele: usize,
        disp_tri: na::DVector<f64>,
    ) -> (na::Vector3<f64>, na::Vector3<f64>) {
        let mut strain = na::Vector3::zeros();
        let mut stress = na::Vector3::zeros();
        let m = self.elements[(5, ele)] - 1;
        let elastic = Elastic {
            young: self.materials[(1, m)],
            poisson: self.materials[(2, m)],
        };
        let mat_d = elastic.make_dematrix();

        for ii in 0..NODE_NUM {
            let (mat_b, _det_j) = self.make_bmatrix(ele, ii);
            let strain_tmp = mat_b * disp_tri.clone();
            strain += strain_tmp.clone();
            let stress_tmp = mat_d * strain_tmp;
            stress += stress_tmp;
        }

        //strain[(0, ele)] = strain_tri
        return (stress / 4.0, strain / 4.0);
    }
}

struct Boundary2d {
    nodes: na::Matrix2xX<f64>,
    force_node: na::DVector<usize>,
    force_node_xy: na::DMatrix<f64>,
    disp_node_no: na::DMatrix<usize>,
    disp_node_xy: na::DMatrix<f64>,
}

impl Boundary2d {
    fn make_force_vector(&self, ii: usize, timestep: usize) -> na::DVector<f64> {
        let mut vec_f = na::DVector::zeros(NODE_DOF * self.nodes.ncols());
        let tmp = NODE_DOF * self.nodes.ncols();
        let step = timestep as f64;

        for i in 0..tmp {
            if self.force_node[ii] > 0 {
                vec_f[i] = self.force_node_xy[(ii, i)] / step;
            } else {
                vec_f[i] = 0.0;
            }
        }

        return vec_f;
    }

    fn mat_boundary_setting(
        &self,
        mat_k: na::DMatrix<f64>,
        vec_f: na::DVector<f64>,
    ) -> (na::DMatrix<f64>, na::DVector<f64>) {
        let mut mat_k_c = mat_k.clone();
        let mut vec_f_c = vec_f.clone();

        for ii in 0..self.nodes.ncols() {
            for jj in 0..NODE_DOF {
                if self.disp_node_no[(jj, ii)] == 1 {
                    let iz = ii * NODE_DOF + jj;
                    vec_f_c[iz] = 0.0;
                }
            }
        }

        for ii in 0..self.nodes.ncols() {
            for jj in 0..NODE_DOF {
                if self.disp_node_no[(jj, ii)] == 1 {
                    let iz = ii * NODE_DOF + jj;
                    let vecx = mat_k.column(iz);
                    vec_f_c -= self.disp_node_xy[(jj, ii)] * vecx;

                    for kk in 0..self.nodes.ncols() * NODE_DOF {
                        mat_k_c[(kk, iz)] = 0.0;
                    }

                    mat_k_c[(iz, iz)] = 1.0;
                }
            }
        }
        return (mat_k_c, vec_f_c);
    }

    fn disp_boundary_setting(&self, disp_tri: na::DVector<f64>) -> na::DVector<f64> {
        let mut disp_tri_c = disp_tri.clone();

        for ii in 0..self.nodes.ncols() {
            for jj in 0..NODE_DOF {
                if self.disp_node_no[(jj, ii)] == 1 {
                    let iz = ii * NODE_DOF + jj;
                    disp_tri_c[iz] = self.disp_node_xy[(jj, ii)];
                }
            }
        }

        return disp_tri_c;
    }
}

//Read input data
fn input_data(
    filename: &str,
) -> (
    na::Matrix2xX<f64>,
    na::Matrix6xX<usize>,
    na::Matrix3xX<f64>,
    na::DMatrix<usize>,
    na::DMatrix<f64>,
    usize,
    na::DVector<f64>,
    na::DVector<f64>,
    na::DVector<usize>,
    na::DMatrix<f64>,
) {
    let file = std::fs::File::open(filename).unwrap();
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();
    let first_line = lines.next().unwrap().unwrap();
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let number_of_node: usize = parts[0].parse().unwrap();
    let number_of_element: usize = parts[1].parse().unwrap();
    let number_of_material: usize = parts[2].parse().unwrap();
    let number_of_pfix: usize = parts[3].parse().unwrap();
    let number_of_step: usize = parts[4].parse().unwrap();
    let mut nodes = na::Matrix2xX::zeros(number_of_node);

    //Nodes
    for i in 0..number_of_node {
        let line = lines.next().unwrap().unwrap();
        let parts: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        //println!("{}, {}", parts[1], parts[2]);
        //let tmp = parts[0] as i32;
        nodes[(0, i)] = parts[1];
        nodes[(1, i)] = parts[2];
    }

    let mut elements = na::Matrix6xX::zeros(number_of_element);

    //Elements
    for i in 0..number_of_element {
        let line = lines.next().unwrap().unwrap();
        let parts: Vec<usize> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        elements[(0, i)] = parts[0];
        elements[(1, i)] = parts[1];
        elements[(2, i)] = parts[2];
        elements[(3, i)] = parts[3];
        elements[(4, i)] = parts[4];
        elements[(5, i)] = parts[5];
    }

    let mut materials = na::Matrix3xX::zeros(number_of_material);

    //Materials
    for i in 0..number_of_material {
        let line = lines.next().unwrap().unwrap();
        let parts: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        materials[(0, i)] = parts[1];
        materials[(1, i)] = parts[2];
        materials[(2, i)] = parts[3];
    }

    let mut disp_node_no = na::DMatrix::zeros(NODE_DOF, number_of_node);
    let mut disp_node_xy = na::DMatrix::zeros(NODE_DOF, number_of_node);

    //Neunman boundary condition
    if number_of_pfix > 0 {
        for _i in 0..number_of_pfix {
            let line = lines.next().unwrap().unwrap();
            let parts: Vec<f64> = line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let tmp = parts[0] as usize;
            //println!("{}", tmp);
            disp_node_no[(0, tmp - 1)] = parts[1] as usize;
            disp_node_no[(1, tmp - 1)] = parts[2] as usize;
            disp_node_xy[(0, tmp - 1)] = parts[3];
            disp_node_xy[(1, tmp - 1)] = parts[4];
        }
    }

    //Analysis steps
    let mut total_time = na::DVector::zeros(number_of_step);
    let mut delta_time = na::DVector::zeros(number_of_step);
    let mut force = na::DVector::zeros(number_of_step);
    let mut force_node_no = na::DMatrix::zeros(number_of_step, number_of_node);
    let mut force_node_xy = na::DMatrix::zeros(number_of_step, number_of_node * NODE_DOF);

    for i in 0..number_of_step {
        let line = lines.next().unwrap().unwrap();
        let parts: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        total_time[i] = parts[1];
        delta_time[i] = parts[2];
        force[i] = parts[3] as usize;

        //Dirichlet boundary condition
        if force[i] > 0 {
            for j in 0..force[i] {
                let line = lines.next().unwrap().unwrap();
                let parts: Vec<f64> = line
                    .split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect();
                let tmp = parts[0] as usize;
                //println!("{}", tmp);
                force_node_no[(i, j)] = tmp;
                force_node_xy[(i, NODE_DOF * tmp - 2)] = parts[1];
                force_node_xy[(i, NODE_DOF * tmp - 1)] = parts[2];
            }
        }
    }
    return (
        nodes,
        elements,
        materials,
        disp_node_no,
        disp_node_xy,
        number_of_step,
        total_time,
        delta_time,
        force,
        force_node_xy,
    );
}

fn output_data(
    nodes: na::Matrix2xX<f64>,
    elements: na::Matrix6xX<usize>,
    disp: na::DMatrix<f64>,
    stress_x: na::DMatrix<f64>,
    stress_y: na::DMatrix<f64>,
    stress_xy: na::DMatrix<f64>,
    strain_x: na::DMatrix<f64>,
    strain_y: na::DMatrix<f64>,
    strain_xy: na::DMatrix<f64>,
    analysis_time: na::DVector<f64>,
) -> std::io::Result<()> {
    let mut node_file = std::fs::File::create("result/node.txt")?;
    let header = format!("{0:>8}, {1:>8}, {2:>8} \n", "NodeNo", "X", "Y");
    let _ = node_file.write(header.as_bytes());

    for ii in 0..nodes.ncols() {
        let buf = format!(
            "{0:>8}, {1:>8}, {2:>8} \n",
            ii + 1,
            nodes[(0, ii)],
            nodes[(1, ii)]
        );
        let _ = node_file.write(buf.as_bytes());
    }

    for ii in 0..analysis_time.ncols() {
        let header = format!("Time = {} \n", analysis_time[ii]);
        let _ = node_file.write(header.as_bytes());
        let header = format!("{0:>8}, {1:>8}, {2:>8} \n", "NodeNo", "dispX", "dispY");
        let _ = node_file.write(header.as_bytes());

        for jj in 0..nodes.ncols() {
            let buf = format!(
                "{0:>8}, {1:>8.4}, {2:>8.4} \n",
                jj + 1,
                disp[(ii, jj * 2)],
                disp[(ii, jj * 2 + 1)]
            );
            let _ = node_file.write(buf.as_bytes());
        }
    }

    drop(node_file);
    let mut element_file = std::fs::File::create("result/element.txt")?;
    let header = format!(
        "{0:>8}, {1:>8}, {2:>8}, {3:>8}, {4:>8} \n",
        "EleNo", "Node1", "Node2", "Node3", "Node4"
    );
    let _ = element_file.write(header.as_bytes());

    for ii in 0..elements.ncols() {
        let buf = format!(
            "{0:>8}, {1:>8}, {2:>8}, {3:>8}, {4:>8} \n",
            elements[(0, ii)],
            elements[(1, ii)],
            elements[(2, ii)],
            elements[(3, ii)],
            elements[(4, ii)],
        );
        let _ = element_file.write(buf.as_bytes());
    }

    for ii in 0..analysis_time.ncols() {
        let header = format!("Time = {} \n", analysis_time[ii]);
        let _ = element_file.write(header.as_bytes());
        let header = format!(
            "{0:>8}, {1:>8}, {2:>8}, {3:>8}, {4:>8}, {5:>8}, {6:>8} \n",
            "EleNo", "SigmaX", "SigmaY", "TauXY", "EpsX", "EpsY", "GammaXY"
        );
        let _ = element_file.write(header.as_bytes());

        for jj in 0..elements.ncols() {
            let buf = format!(
                "{0:>8}, {1:>8.4}, {2:>8.4}, {3:>8.4}, {4:>8.4}, {5:>8.4}, {6:>8.4} \n",
                jj + 1,
                stress_x[(ii, jj)],
                stress_y[(ii, jj)],
                stress_xy[(ii, jj)],
                strain_x[(ii, jj)],
                strain_y[(ii, jj)],
                strain_xy[(ii, jj)],
            );
            let _ = element_file.write(buf.as_bytes());
        }
    }
    Ok(())
}

fn fem() {
    //Read input data
    let inputdata = input_data("input/input.txt");
    let nodes = inputdata.0;
    let elements = inputdata.1;
    let materials = inputdata.2;
    let disp_node_no = inputdata.3;
    let disp_node_xy = inputdata.4;
    let number_of_step = inputdata.5;
    let total_time = inputdata.6;
    let delta_time = inputdata.7;
    let force_node = inputdata.8;
    let force_node_xy = inputdata.9;
    let matrix = C2D4 {
        nodes: nodes.clone(),
        elements: elements.clone(),
        materials: materials.clone(),
    };
    let boundary = Boundary2d {
        nodes: nodes.clone(),
        force_node: force_node.clone(),
        force_node_xy: force_node_xy.clone(),
        disp_node_no: disp_node_no.clone(),
        disp_node_xy: disp_node_xy.clone(),
    };
    let mut step = 0;
    let mut analysis_step = 0;
    let mut time_data = 0.0;

    for ii in 0..number_of_step {
        step += (total_time[ii] / delta_time[ii]) as usize;
    }

    let mut disp_all = na::DMatrix::zeros(step, nodes.ncols() * NODE_DOF);
    let mut stress_x_all = na::DMatrix::zeros(step, elements.ncols());
    let mut stress_y_all = na::DMatrix::zeros(step, elements.ncols());
    let mut stress_xy_all = na::DMatrix::zeros(step, elements.ncols());
    let mut strain_x_all = na::DMatrix::zeros(step, elements.ncols());
    let mut strain_y_all = na::DMatrix::zeros(step, elements.ncols());
    let mut strain_xy_all = na::DMatrix::zeros(step, elements.ncols());
    let mut analysis_time = na::DVector::zeros(step);

    //Analysis start
    for ii in 0..number_of_step {
        let time_step = (total_time[ii] / delta_time[ii]) as usize;

        for _jj in 0..time_step {
            time_data += delta_time[ii];
            analysis_time[analysis_step] = time_data;
            //Make global stiffness matrix
            let mat_k = matrix.make_global_kmatrix();
            //Make force vector
            let vec_f = boundary.make_force_vector(ii, time_step);
            //Set boundary conditions
            let (mat_k_c, vec_f_c) = boundary.mat_boundary_setting(mat_k, vec_f);
            //Calcurate simultaneous equations using LU decomposition
            let tmp = mat_k_c.lu();
            let disp_tri = tmp.solve(&vec_f_c).expect("Linear resolution failed.");
            //Set boundary conditions
            let disp_tri_c = boundary.disp_boundary_setting(disp_tri);
            let mut strain = na::Matrix3xX::zeros(elements.ncols());
            let mut stress = na::Matrix3xX::zeros(elements.ncols());

            for ele in 0..elements.ncols() {
                let mut disp_tmp = na::DVector::zeros(8);
                let i = elements[(1, ele)] - 1;
                let j = elements[(2, ele)] - 1;
                let k = elements[(3, ele)] - 1;
                let l = elements[(4, ele)] - 1;
                disp_tmp[0] = disp_tri_c[2 * i];
                disp_tmp[1] = disp_tri_c[2 * i + 1];
                disp_tmp[2] = disp_tri_c[2 * j];
                disp_tmp[3] = disp_tri_c[2 * j + 1];
                disp_tmp[4] = disp_tri_c[2 * k];
                disp_tmp[5] = disp_tri_c[2 * k + 1];
                disp_tmp[6] = disp_tri_c[2 * l];
                disp_tmp[7] = disp_tri_c[2 * l + 1];
                let (stress_tri, strain_tri) = matrix.calcurate_stress(ele, disp_tmp);
                strain[(0, ele)] = strain_tri[0];
                strain[(1, ele)] = strain_tri[1];
                strain[(2, ele)] = strain_tri[2];
                stress[(0, ele)] = stress_tri[0];
                stress[(1, ele)] = stress_tri[1];
                stress[(2, ele)] = stress_tri[2];
            }

            for node in 0..nodes.ncols() * NODE_DOF {
                disp_all[(analysis_step, node)] = disp_tri_c[node];
            }

            for ele in 0..elements.ncols() {
                strain_x_all[(analysis_step, ele)] = strain[(0, ele)];
                strain_y_all[(analysis_step, ele)] = strain[(1, ele)];
                strain_xy_all[(analysis_step, ele)] = strain[(2, ele)];
                stress_x_all[(analysis_step, ele)] = stress[(0, ele)];
                stress_y_all[(analysis_step, ele)] = stress[(1, ele)];
                stress_xy_all[(analysis_step, ele)] = stress[(2, ele)];
            }

            analysis_step += 1;
        }
    }

    let _output = output_data(
        nodes,
        elements,
        disp_all,
        stress_x_all,
        stress_y_all,
        stress_xy_all,
        strain_x_all,
        strain_y_all,
        strain_xy_all,
        analysis_time,
    );
}

fn main() {
    let stamp = std::fs::read_to_string("stamp.txt").unwrap();
    println!("{}", stamp);
    let _fem = fem();
}
