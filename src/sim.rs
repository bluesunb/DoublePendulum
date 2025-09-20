use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use russell_lab::Vector;
use russell_ode::StrError;
use sfml::graphics::Color;
use std::f64::consts::PI;

use crate::{
    ode::{PendulumODE, PendulumParams, PendulumState, dp_dt},
    render::state_to_color,
    utils::{linspace, meshgrid},
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
                ((curr[0] - prev.theta1).powi(2) + (curr[1] - prev.theta2).powi(2))
            })
            .collect()
    }

    pub fn get_energies(&self) -> Vec<f64> {
        self.sims.iter().map(|sim| sim.solver.energy()).collect()
    }
}
