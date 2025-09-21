use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use russell_lab::Vector;
use russell_ode::StrError;
use sfml::graphics::Color;
use std::f64::consts::PI;

use crate::{
    ode::{PendulumODE, PendulumParams, PendulumState, dp_dt},
    render::state_to_color,
    utils::{linspace, meshgrid, sigmoid},
};

pub struct Simulation {
    pub params: PendulumParams,
    pub init_state: PendulumState,
    pub prev_state: PendulumState,
    pub solver: PendulumODE<'static>,
    pub time: f64,
    pub running: bool,
}

impl Default for Simulation {
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
            prev_state: init_state,
            solver,
            time: 0.0,
            running: false,
        }
    }
}

impl Simulation {
    pub fn new(params: PendulumParams, init_state: PendulumState) -> Result<Self, StrError> {
        let solver = PendulumODE::new(params, init_state, 0.0)?;
        Ok(Self {
            params,
            init_state,
            prev_state: init_state,
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
            self.prev_state = self.solver.state();
            let result = self.solver.step(dt);
            self.time += dt;
            return result;
        }
        Ok(self.solver.state())
    }

    pub fn step_sympletic(&mut self, dt: f64) -> Result<PendulumState, StrError> {
        if self.running {
            self.prev_state = self.solver.state();
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

pub struct Simulations {
    pub sims: Vec<Simulation>,
    pub range_min: f64,
    pub range_max: f64,

    total_dt: f64,
    n_substeps: usize,
    acc: f64,
    running: bool,
}

impl Simulations {
    pub fn new(params: PendulumParams, num_rows: usize, range_min: f64, range_max: f64) -> Self {
        let (thetas_1, thetas_2) = meshgrid(
            &linspace(range_min, range_max, num_rows, true),
            &linspace(range_min, range_max, num_rows, true),
        );

        let init_states = thetas_1
            .iter()
            .zip(thetas_2.iter())
            .map(|(&t1, &t2)| PendulumState {
                theta1: t1,
                theta2: t2,
                p1: 0.0,
                p2: 0.0,
            })
            .collect::<Vec<_>>();

        let mut sims = init_states
            .into_iter()
            .map(|state| Simulation::new(params, state).unwrap())
            .collect::<Vec<_>>();

        for sim in &mut sims {
            sim.running = false;
        }

        Self {
            sims,
            range_min,
            range_max,
            total_dt: 1.0 / 240.0,
            n_substeps: 8,
            acc: 0.0,
            running: false,
        }
    }

    pub fn get(&self, idx: usize) -> &Simulation {
        &self.sims[idx]
    }

    pub fn total_dt(mut self, total_dt: f64) -> Self {
        self.total_dt = total_dt;
        self
    }

    pub fn n_substeps(mut self, n_substeps: usize) -> Self {
        self.n_substeps = n_substeps;
        self
    }

    pub fn reset_all(&mut self) {
        for sim in &mut self.sims {
            sim.reset(sim.init_state);
            sim.running = false;
        }
        self.acc = 0.0;
    }

    pub fn toggle_all(&mut self) {
        for sim in &mut self.sims {
            sim.running = !sim.running;
        }
        self.running = !self.running;
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

    pub fn derivatives(&self) -> Vec<PendulumState> {
        self.sims
            .iter()
            .map(|sim| {
                dp_dt(
                    &sim.params,
                    &sim.state(),
                    sim.solver.cached_a(),
                    sim.solver.cached_b(),
                )
            })
            .collect()
    }

    pub fn step(&mut self, dt: f64) {
        if !self.running {
            return;
        }

        let mut steps = 0;
        self.acc += dt;
        while self.acc >= self.total_dt && steps < self.n_substeps {
            for sim in &mut self.sims {
                if sim.running {
                    let _ = sim.step_sympletic(self.total_dt);
                }
            }
            self.acc -= self.total_dt;
            steps += 1;
        }
    }

    pub fn step_parallel(&mut self, dt: f64) {
        let mut steps = 0;
        self.acc += dt;
        while self.acc >= self.total_dt && steps < self.n_substeps {
            self.sims.par_iter_mut().for_each(|sim| {
                if sim.running {
                    let _ = sim.step_sympletic(self.total_dt);
                }
            });
            self.acc -= self.total_dt;
            steps += 1;
        }
    }

    pub fn get_colors(&self) -> Vec<Color> {
        self.sims
            .iter()
            .map(|sim| {
                let vec_state = sim.vec_state();
                state_to_color(vec_state[0], vec_state[1], self.range_min, self.range_max)
            })
            .collect()
    }

    pub fn get_diff(&self) -> Vec<f64> {
        self.sims
            .iter()
            .map(|sim| {
                let curr = sim.vec_state();
                let prev = sim.prev_state;
                (curr[0] - prev.theta1).powi(2) + (curr[1] - prev.theta2).powi(2)
            })
            .collect()
    }

    pub fn get_energies(&self) -> Vec<f64> {
        self.sims.iter().map(|sim| sim.solver.energy()).collect()
    }

    pub fn fill_colors_rgpb(&self, pixels: &mut [u8]) {
        if !self.running {
            return;
        }
        assert_eq!(pixels.len(), self.sims.len() * 4);
        let rng_min = self.range_min;
        let rng_max = self.range_max;
        for (i, sim) in self.sims.iter().enumerate() {
            let v = sim.vec_state();
            let theta1 = v[0];
            let theta2 = v[1];

            let rng = rng_max - rng_min;
            let r = (255.0 * (0.5 + theta2 / rng)).clamp(0.0, 255.0) as u8;
            let g = (255.0 * (0.5 + theta1 / rng)).clamp(0.0, 255.0) as u8;
            let b = (255.0 * (0.5 - theta2 / rng)).clamp(0.0, 255.0) as u8;

            let offset = i * 4;
            pixels[offset] = r;
            pixels[offset + 1] = g;
            pixels[offset + 2] = b;
            pixels[offset + 3] = 255;
        }
    }

    pub fn fill_diff_rgba(&self, pixels: &mut [u8], prev_diff: &mut [f64]) {
        if !self.running {
            return;
        }
        assert_eq!(pixels.len(), self.sims.len() * 4);
        assert_eq!(prev_diff.len(), self.sims.len());

        for (i, sim) in self.sims.iter().enumerate() {
            let curr = sim.vec_state();
            let prev = sim.prev_state;
            let cur_v = (curr[0] - prev.theta1).powi(2) + (curr[1] - prev.theta2).powi(2);

            let v = prev_diff[i] * 0.9 + cur_v * 0.1;
            prev_diff[i] = v;

            let s = (sigmoid(v) * 255.0) as u8;
            let dm = (sigmoid(v * 0.2) * 255.0) as u8;

            let offset = i * 4;
            pixels[offset] = dm;
            pixels[offset + 1] = dm;
            pixels[offset + 2] = s;
            pixels[offset + 3] = 255;
        }
    }

    pub fn fill_slope_rgba(&self, pixels: &mut [u8], num_rows: usize) {
        if !self.running {
            return;
        }
        assert_eq!(pixels.len(), self.sims.len() * 4);
        let mut th1 = Vec::with_capacity(num_rows * num_rows);
        let mut th2 = Vec::with_capacity(num_rows * num_rows);
        for sim in &self.sims {
            let v = sim.vec_state();
            th1.push(v[0]);
            th2.push(v[1]);
        }

        let w_diag = 0.7071;
        let scale = 0.1;
        let bias = -1.0;

        pixels.par_chunks_mut(4).enumerate().for_each(|(i, pixel)| {
            let row = i / num_rows;
            let col = i % num_rows;

            let r0 = row.saturating_sub(1);
            let r2 = (row + 1).min(num_rows - 1);
            let c0 = col.saturating_sub(1);
            let c2 = (col + 1).min(num_rows - 1);

            let idx = |r: usize, c: usize| -> usize { r * num_rows + c };

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
        });
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
