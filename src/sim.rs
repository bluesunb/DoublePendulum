use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use russell_lab::Vector;
use russell_ode::StrError;
use sfml::graphics::Color;
use std::f64::consts::PI;

use crate::{
    ode::{PendulumODE, PendulumParams, PendulumState},
    utils::linspace,
};

pub struct Simulation<'a> {
    pub params: PendulumParams,
    pub init_state: PendulumState,
    // pub prev_state: PendulumState,
    pub solver: PendulumODE<'a>,
    pub time: f64,
    pub running: bool,
}

impl<'a> Default for Simulation<'a> {
    fn default() -> Self {
        let params = PendulumParams::default();
        let init_state = PendulumState {
            theta1: PI * 0.5,
            theta2: PI * 0.5,
            p1: 0.0,
            p2: 0.0,
        };
        let solver = PendulumODE::new(params, init_state, 0.0).unwrap();

        Self {
            params,
            init_state,
            // prev_state: init_state,
            solver,
            time: 0.0,
            running: false,
        }
    }
}

impl<'a> Simulation<'a> {
    pub fn new(params: PendulumParams, init_state: PendulumState) -> Result<Self, StrError> {
        let solver = PendulumODE::new(params, init_state, 0.0)?;
        Ok(Self {
            params,
            init_state,
            // prev_state: init_state,
            solver,
            time: 0.0,
            running: false,
        })
    }

    pub fn reset(&mut self, state: PendulumState) {
        self.init_state = state;
        self.solver = PendulumODE::new(self.params, state, 0.0).unwrap();
        self.time = 0.0;
    }

    pub fn step(&mut self, dt: f64) -> Result<PendulumState, StrError> {
        if self.running {
            // self.prev_state = self.solver.state();
            let result = self.solver.step(dt);
            self.time += dt;
            return result;
        }
        Ok(self.solver.state())
    }

    pub fn step_sympletic(&mut self, dt: f64) -> Result<PendulumState, StrError> {
        if self.running {
            // self.prev_state = self.solver.state();
            let _ = self.solver.step_symplectic(dt);
            self.time += dt;
        }
        Ok(self.solver.state())
    }

    pub fn state(&self) -> PendulumState {
        self.solver.state()
    }

    pub fn vec_state(&self) -> &Vector {
        self.solver.vec_state()
    }
}

pub struct Simulations<'a> {
    pub sims: Vec<Simulation<'a>>,
    pub range_x: (f64, f64), // range for theta2, deviation from vertical
    pub range_y: (f64, f64), // range for theta1, deviation from horizontal
    pub n_steps_x: usize,
    pub n_steps_y: usize,
    pub running: bool,
    phys_dt: f64,
    n_substeps: usize,
    acc: f64,
}

impl<'a> Simulations<'a> {
    pub fn new(
        params: PendulumParams,
        range_x: (f64, f64),
        rnage_y: (f64, f64),
        n_steps_x: usize,
        n_steps_y: usize,
    ) -> Self {
        let thetas_1 = linspace(range_x.0, range_x.1, n_steps_x, true);
        let thetas_2 = linspace(rnage_y.0, rnage_y.1, n_steps_y, true);

        assert_eq!(n_steps_x * n_steps_y, thetas_1.len() * thetas_2.len());

        let mut sims = Vec::with_capacity(n_steps_x * n_steps_y);
        for th1 in &thetas_1 {
            for th2 in &thetas_2 {
                let init_state = PendulumState {
                    theta1: *th1,
                    theta2: *th2,
                    p1: 0.0,
                    p2: 0.0,
                };
                let mut sim = Simulation::new(params, init_state).unwrap();
                sim.running = false;
                sims.push(sim);
            }
        }

        Self {
            sims,
            range_x,
            range_y: rnage_y,
            n_steps_x,
            n_steps_y,
            phys_dt: 1.0 / 240.0,
            n_substeps: 8,
            running: false,
            acc: 0.0,
        }
    }

    pub fn get(&self, idx: usize) -> &Simulation<'a> {
        &self.sims[idx]
    }

    pub fn total_dt(mut self, total_dt: f64) -> Self {
        self.phys_dt = total_dt;
        self
    }

    pub fn n_substeps(mut self, n_substeps: usize) -> Self {
        self.n_substeps = n_substeps;
        self
    }

    pub fn len(&self) -> usize {
        self.sims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sims.is_empty()
    }

    pub fn initial_states(&self) -> Vec<PendulumState> {
        self.sims.iter().map(|sim| sim.init_state).collect()
    }

    pub fn initial_states_ref(&self) -> Vec<&PendulumState> {
        self.sims.iter().map(|sim| &sim.init_state).collect()
    }

    pub fn initial_states_mut(&mut self) -> Vec<&mut PendulumState> {
        self.sims
            .iter_mut()
            .map(|sim| &mut sim.init_state)
            .collect()
    }

    pub fn states(&self) -> Vec<PendulumState> {
        self.sims.iter().map(|sim| sim.state()).collect()
    }

    pub fn reset_all(&mut self) {
        for sim in &mut self.sims {
            sim.reset(sim.init_state);
            sim.running = false;
        }
        self.running = false;
    }

    pub fn toggle_all(&mut self) {
        self.running = !self.running;
        for sim in &mut self.sims {
            sim.running = self.running;
        }
    }

    pub fn step(&mut self, dt: f64) {
        if !self.running {
            return;
        }

        let mut steps = 0;
        self.acc += dt;
        while self.acc >= self.phys_dt && steps < 8 {
            for sim in &mut self.sims {
                if sim.running {
                    let _ = sim.step_sympletic(self.phys_dt);
                }
            }
            self.acc -= self.phys_dt;
            steps += 1;
        }
    }

    pub fn step_parallel(&mut self, dt: f64) {
        let mut steps = 0;
        self.acc += dt;
        while self.acc >= self.phys_dt && steps < 8 {
            self.sims.par_iter_mut().for_each(|sim| {
                if sim.running {
                    let _ = sim.step_sympletic(self.phys_dt);
                }
            });
            self.acc -= self.phys_dt;
            steps += 1;
        }
    }

    pub fn get_colors(&self) -> Vec<Color> {
        self.sims
            .iter()
            .map(|sim| {
                let vec_state = sim.vec_state();
                self.state_to_color(vec_state[0], vec_state[1])
            })
            .collect()
    }

    pub fn get_energies(&self) -> Vec<f64> {
        self.sims.iter().map(|sim| sim.solver.energy()).collect()
    }

    pub fn fill_colors_rgba(&self, pixels: &mut [u8]) {
        if !self.running {
            return;
        }
        assert_eq!(pixels.len(), self.sims.len() * 4);

        for (i, sim) in self.sims.iter().enumerate() {
            let v = sim.vec_state();
            let color = self.state_to_color(v[0], v[1]);

            let offset = i * 4;
            pixels[offset] = color.r as u8;
            pixels[offset + 1] = color.g as u8;
            pixels[offset + 2] = color.b as u8;
            pixels[offset + 3] = 255;
        }
    }

    pub fn fill_diff_rgba(&self, pixels: &mut [u8]) {
        if !self.running {
            return;
        }
        assert_eq!(pixels.len(), self.sims.len() * 4);

        let (th1, th2) = self
            .sims
            .iter()
            .map(|sim| {
                let vec_state = sim.vec_state();
                (vec_state[0], vec_state[1])
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let w_diag = 0.7071;
        let scale = 0.1;
        let bias = -1.0;

        let n_steps_x = self.n_steps_x;
        let n_steps_y = self.n_steps_y;

        pixels.par_chunks_mut(4).enumerate().for_each(|(i, pixel)| {
            let row = i / n_steps_x;
            let col = i % n_steps_x;

            let r0 = row.saturating_sub(1);
            let r2 = (row + 1).min(n_steps_y - 1);
            let c0 = col.saturating_sub(1);
            let c2 = (col + 1).min(n_steps_x - 1);

            let idx = |r: usize, c: usize| -> usize { r * n_steps_x + c };

            let cur_th1 = th1[i];
            let cur_th2 = th2[i];

            let mut dev = (cur_th1 - th1[idx(r0, col)]).powi(2)
                + (cur_th2 - th2[idx(r0, col)]).powi(2)
                + (cur_th1 - th1[idx(r2, col)]).powi(2)
                + (cur_th2 - th2[idx(r2, col)]).powi(2)
                + (cur_th1 - th1[idx(row, c0)]).powi(2)
                + (cur_th2 - th2[idx(row, c0)]).powi(2)
                + (cur_th1 - th1[idx(row, c2)]).powi(2)
                + (cur_th2 - th2[idx(row, c2)]).powi(2);

            dev += w_diag
                * ((cur_th1 - th1[idx(r0, c0)]).powi(2)
                    + (cur_th2 - th2[idx(r0, c0)]).powi(2)
                    + (cur_th1 - th1[idx(r0, c2)]).powi(2)
                    + (cur_th2 - th2[idx(r0, c2)]).powi(2)
                    + (cur_th1 - th1[idx(r2, c0)]).powi(2)
                    + (cur_th2 - th2[idx(r2, c0)]).powi(2)
                    + (cur_th1 - th1[idx(r2, c2)]).powi(2)
                    + (cur_th2 - th2[idx(r2, c2)]).powi(2));

            dev /= 8.0; // average deviation

            let x = 0.25 * (dev * scale + bias).clamp(0.0, 4.0);
            let s = x * x * (3.0 - 2.0 * x);
            let (r, g, b) = turbo_approx(s);
            pixel[0] = ((pixel[0] as f32 * 0.99) as u8).saturating_add(r);
            pixel[1] = ((pixel[1] as f32 * 0.99) as u8).saturating_add(g);
            pixel[2] = ((pixel[2] as f32 * 0.99) as u8).saturating_add(b);
            pixel[3] = 255;
        })
    }

    pub fn fill_energies_rgba(&self, pixels: &mut [u8]) {
        if !self.running {
            return;
        }
        assert_eq!(pixels.len(), self.sims.len() * 4);

        let energies: Vec<f64> = self.sims.iter().map(|sim| sim.solver.energy()).collect();
        let min_energy = energies
            .iter()
            .cloned()
            .fold(f64::INFINITY, |a, b| a.min(b));
        let max_energy = energies
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let range = max_energy - min_energy;

        for (i, energy) in energies.iter().enumerate() {
            let norm_energy = if range.abs() < 1e-10 {
                0.5
            } else {
                (energy - min_energy) / range
            };
            let (r, g, b) = turbo_approx(norm_energy);

            let offset = i * 4;
            pixels[offset] = r;
            pixels[offset + 1] = g;
            pixels[offset + 2] = b;
            pixels[offset + 3] = 255;
        }
    }

    #[inline]
    pub fn state_to_color(&self, theta1: f64, theta2: f64) -> Color {
        let mut r = 255.0 * (0.5 + theta2 / (self.range_x.1 - self.range_x.0));
        let mut g = 255.0 * (0.5 + theta1 / (self.range_y.1 - self.range_y.0));
        let mut b = 255.0 * (0.5 - theta2 / (self.range_x.1 - self.range_x.0));

        r = r.clamp(0.0, 255.0);
        g = g.clamp(0.0, 255.0);
        b = b.clamp(0.0, 255.0);

        Color::rgb(r as u8, g as u8, b as u8)
    }
}

fn turbo_approx(x: f64) -> (u8, u8, u8) {
    let x = x.clamp(0.0, 1.0);
    let r =
        // 0.13572138 +
        4.61539260 * x - 42.66032258 * x.powi(2) + 132.13108234 * x.powi(3)
        - 152.94239396 * x.powi(4)
        + 59.28637943 * x.powi(5);
    let g =
        // 0.09140261 +
        2.19418839 * x + 4.84296658 * x.powi(2) - 14.18503333 * x.powi(3)
        + 4.27729857 * x.powi(4)
        + 2.82956604 * x.powi(5);
    let b =
        // 0.10667330 +
        8.17095726 * x - 33.14325772 * x.powi(2) + 61.08407327 * x.powi(3)
        - 54.06743020 * x.powi(4)
        + 18.09264397 * x.powi(5);
    (
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    )
}
