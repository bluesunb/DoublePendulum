use std::f64::consts::PI;

use russell_lab::Vector;
use russell_ode::{Method, OdeSolver, Params, StrError, System};
use sfml::system::Vector2f;

#[derive(Debug, Clone, Copy)]
pub struct PendulumParams {
    pub m: f64, // mass
    pub l: f64, // length
    pub g: f64, // gravity
}

impl Default for PendulumParams {
    fn default() -> Self {
        Self {
            m: 1.0,
            l: 1.0,
            g: 9.81,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PendulumState {
    pub theta1: f64, // angle of first pendulum (rad, absolute from vertical)
    pub theta2: f64, // angle of second pendulum (rad, absolute from vertical)
    pub p1: f64,     // conjugate momentum of first pendulum
    pub p2: f64,     // conjugate momentum of second pendulum
}

impl From<PendulumState> for Vector {
    fn from(s: PendulumState) -> Self {
        Vector::from(&[s.theta1, s.theta2, s.p1, s.p2])
    }
}
impl From<&Vector> for PendulumState {
    fn from(v: &Vector) -> Self {
        Self {
            theta1: v[0],
            theta2: v[1],
            p1: v[2],
            p2: v[3],
        }
    }
}

pub struct PendulumODE<'a> {
    solver: OdeSolver<'a, PendulumParams>,
    args: PendulumParams,
    t: f64,
    y: Vector,
    // cached derived constants
    a: f64, // 6 / (m*l^2)
    b: f64, // 0.5 * m * l^2
}

fn derivative_theta(state: &PendulumState, a: f64) -> (f64, f64) {
    let delta = state.theta1 - state.theta2;
    let c = 3.0 * delta.cos();

    let denom = 16.0 - c * c;
    let eps = 1e-12;
    let denom = if denom.abs() < eps {
        eps * denom.signum()
    } else {
        denom
    };

    let theta1_dot = a * (2.0 * state.p1 - c * state.p2) / denom;
    let theta2_dot = a * (8.0 * state.p2 - c * state.p1) / denom;

    (theta1_dot, theta2_dot)
}

fn derivative(params: &PendulumParams, state: &PendulumState, a: f64, b: f64) -> PendulumState {
    let (theta1_dot, theta2_dot) = derivative_theta(state, a);
    let delta = state.theta1 - state.theta2;
    let sin_delta = delta.sin();
    let prod = theta1_dot * theta2_dot;
    let g_over_l = params.g / params.l;
    let p1_dot = -b * (prod * sin_delta + 3.0 * g_over_l * state.theta1.sin());
    let p2_dot = -b * (-prod * sin_delta + g_over_l * state.theta2.sin());

    PendulumState {
        theta1: theta1_dot,
        theta2: theta2_dot,
        p1: p1_dot,
        p2: p2_dot,
    }
}

impl<'a> PendulumODE<'a> {
    pub fn new(params: PendulumParams, initial: PendulumState, t0: f64) -> Result<Self, StrError> {
        let ndim = 4;
        let system = System::new(
            ndim,
            |f: &mut Vector, _t: f64, y: &Vector, args: &mut PendulumParams| {
                // let PendulumParams { m, l, g } = *args;

                // let theta1 = y[0];
                // let theta2 = y[1];
                // let p1 = y[2];
                // let p2 = y[3];

                // let delta = theta1 - theta2;
                // let (sin_delta, cos_delta) = delta.sin_cos();
                // let a = 6.0 / (m * l * l);
                // let c = 3.0 * cos_delta;

                // let denom = 16.0 - c * c;
                // let eps = 1e-12;
                // let denom = if denom.abs() < eps {
                //     eps * denom.signum()
                // } else {
                //     denom
                // };

                // let theta1_dot = a * (2.0 * p1 - c * p2) / denom;
                // let theta2_dot = a * (8.0 * p2 - c * p1) / denom;

                // let prod = theta1_dot * theta2_dot;
                // let b = 0.5 * m * l * l;
                // let g_over_l = g / l;

                // let p1_dot = -b * (prod * sin_delta + 3.0 * g_over_l * theta1.sin());
                // let p2_dot = -b * (-prod * sin_delta + g_over_l * theta2.sin());

                // f[0] = theta1_dot;
                // f[1] = theta2_dot;
                // f[2] = p1_dot;
                // f[3] = p2_dot;

                let a = 6.0 / (args.m * args.l * args.l);
                let b = 0.5 * args.m * args.l * args.l;
                let d = derivative(args, &y.into(), a, b);

                f[0] = d.theta1;
                f[1] = d.theta2;
                f[2] = d.p1;
                f[3] = d.p2;

                Ok(())
            },
        );

        // Dorman-Prince 8(7) method (great for non-stiff)
        let mut ode_params = Params::new(Method::DoPri8);
        ode_params.set_tolerances(1e-11, 1e-9, None);

        let solver = OdeSolver::new(ode_params, system)?;

        let a = 6.0 / (params.m * params.l * params.l);
        let b = 0.5 * params.m * params.l * params.l;

        Ok(Self {
            solver,
            args: params,
            t: t0,
            y: initial.into(),
            a,
            b,
        })
    }

    #[inline]
    pub fn time(&self) -> f64 {
        self.t
    }

    #[inline]
    pub fn state(&self) -> PendulumState {
        (&self.y).into()
    }

    #[inline]
    pub fn vec_state(&self) -> &Vector {
        &self.y
    }

    pub fn set_params(&mut self, param: PendulumParams) {
        self.args = param;
        self.a = 6.0 / (param.m * param.l * param.l);
        self.b = 0.5 * param.m * param.l * param.l;
    }

    pub fn set_state(&mut self, state: PendulumState) {
        self.y[0] = wrap_angle(state.theta1);
        self.y[1] = wrap_angle(state.theta2);
        self.y[2] = state.p1;
        self.y[3] = state.p2;
    }

    /// Advance the state by `dt` with the adaptive ODE solver
    pub fn step(&mut self, dt: f64) -> Result<PendulumState, StrError> {
        let t1 = self.t + dt;
        self.solver
            .solve(&mut self.y, self.t, t1, None, &mut self.args)?;
        self.t = t1;

        self.y[0] = wrap_angle(self.y[0]);
        self.y[1] = wrap_angle(self.y[1]);

        Ok(self.state())
    }

    /// fixed-step symplectic integrator (Velocity Verlet)
    pub fn step_symplectic(&mut self, dt: f64) -> PendulumState {
        let mut state = self.state();
        let state_dt = derivative(&self.args, &state, self.a, self.b);

        // first half-step for angles
        state.p1 += 0.5 * dt * state_dt.p1;
        state.p2 += 0.5 * dt * state_dt.p2;

        // full step for momenta
        let (w1, w2) = derivative_theta(&self.state(), self.a);
        state.theta1 = wrap_angle(state.theta1 + dt * w1);
        state.theta2 = wrap_angle(state.theta2 + dt * w2);

        // second half-step for angles
        let state_dt2 = derivative(&self.args, &state, self.a, self.b);
        state.p1 += 0.5 * dt * state_dt2.p1;
        state.p2 += 0.5 * dt * state_dt2.p2;

        self.set_state(state);
        self.t += dt;

        self.state()
    }

    pub fn energy(&self) -> f64 {
        let state = self.state();

        let delta = state.theta1 - state.theta2;
        let c = 3.0 * delta.cos();
        let denom = 16.0 - c * c;

        // angular velocities
        let w1 = self.a * (2.0 * state.p1 - c * state.p2) / denom;
        let w2 = self.a * (8.0 * state.p2 - c * state.p1) / denom;

        let tke = self.b * (w1 * w1 + w2 * w2 / 3.0);
        let vpe = -self.args.m
            * self.args.g
            * self.args.l
            * (2.0 * state.theta1.cos() + state.theta2.cos());

        tke + vpe
    }
}

fn wrap_angle(angle: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let wrapped = angle % two_pi;
    if wrapped > PI {
        wrapped - two_pi
    } else if wrapped < -PI {
        wrapped + two_pi
    } else {
        wrapped
    }
}

pub fn angles_to_xy(
    l: f64,
    theta1: f64,
    theta2: f64,
    origin: &Vector2f,
    pxl_scale: f32,
) -> (Vector2f, Vector2f) {
    let (s1, c1) = (theta1.sin() as f32, theta1.cos() as f32);
    let x1 = origin.x + (l as f32) * pxl_scale * s1;
    let y1 = origin.y + (l as f32) * pxl_scale * c1;

    let (s2, c2) = (theta2.sin() as f32, theta2.cos() as f32);
    let x2 = x1 + (l as f32) * pxl_scale * s2;
    let y2 = y1 + (l as f32) * pxl_scale * c2;

    (Vector2f::new(x1, y1), Vector2f::new(x2, y2))
}
