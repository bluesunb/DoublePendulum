use russell_lab::Vector;
use russell_ode::{Method, OdeSolver, Params, StrError, System};
use sfml::system::Vector2f;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
pub struct PendulumState {
    pub theta1: f64, // angle of first pendulum (rad, absolute from vertical)
    pub theta2: f64, // angle of second pendulum (rad, absolute from vertical)
    pub p1: f64,     // conjugate momentum of first pendulum
    pub p2: f64,     // conjugate momentum of second pendulum
}

impl From<PendulumState> for Vector {
    fn from(state: PendulumState) -> Self {
        Vector::from(&[state.theta1, state.theta2, state.p1, state.p2])
    }
}
impl From<&Vector> for PendulumState {
    fn from(value: &Vector) -> Self {
        Self {
            theta1: value[0],
            theta2: value[1],
            p1: value[2],
            p2: value[3],
        }
    }
}

pub struct PendulumODE<'a> {
    solver: OdeSolver<'a, PendulumParams>,
    args: PendulumParams,
    t: f64,
    y: Vector,
    // cached derived constants (updated when args change)
    a: f64, // 6 / (m*l^2)
    b: f64, // 0.5 * m * l^2
}

pub(crate) fn dtheta_dt(state: &PendulumState, a: f64) -> PendulumState {
    let delta = state.theta1 - state.theta2;
    let c = 3.0 * delta.cos();
    let denom_raw = 16.0 - c * c;
    let eps = 1e-12;
    let denom = if denom_raw.abs() < eps {
        denom_raw.signum() * eps
    } else {
        denom_raw
    };
    let theta1_dot = a * (2.0 * state.p1 - c * state.p2) / denom;
    let theta2_dot = a * (8.0 * state.p2 - c * state.p1) / denom;

    PendulumState {
        theta1: theta1_dot,
        theta2: theta2_dot,
        p1: 0.0,
        p2: 0.0,
    }
}

pub(crate) fn dp_dt(
    params: &PendulumParams,
    state: &PendulumState,
    a: f64,
    b: f64,
) -> PendulumState {
    let PendulumState {
        theta1: w1,
        theta2: w2,
        p1: _,
        p2: _,
    } = dtheta_dt(state, a);
    let delta = state.theta1 - state.theta2;
    let s_delta = delta.sin();
    let g_over_l = params.g / params.l;
    let p1_dot = -b * (w1 * w2 * s_delta + 3.0 * g_over_l * state.theta1.sin());
    let p2_dot = -b * (-w1 * w2 * s_delta + g_over_l * state.theta2.sin());
    PendulumState {
        theta1: w1,
        theta2: w2,
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
                let PendulumParams { m, l, g } = *args;

                let theta1 = y[0];
                let theta2 = y[1];
                let p1 = y[2];
                let p2 = y[3];

                let delta = theta1 - theta2;
                let (s_delta, c_delta) = delta.sin_cos();
                let a = 6.0 / (m * l * l);
                let c = 3.0 * c_delta;

                let denom_raw = 16.0 - c * c;
                let eps = 1e-12;
                let denom = if denom_raw.abs() < eps {
                    denom_raw.signum() * eps
                } else {
                    denom_raw
                };

                let theta1_dot = a * (2.0 * p1 - c * p2) / denom;
                let theta2_dot = a * (8.0 * p2 - c * p1) / denom;

                let prod = theta1_dot * theta2_dot;
                let b = 0.5 * m * l * l;
                let g_over_l = g / l;

                let p1_dot = -b * (prod * s_delta + 3.0 * g_over_l * theta1.sin());
                let p2_dot = -b * (-prod * s_delta + g_over_l * theta2.sin());

                f[0] = theta1_dot;
                f[1] = theta2_dot;
                f[2] = p1_dot;
                f[3] = p2_dot;

                Ok(())
            },
        );

        let mut ode_params = Params::new(Method::DoPri8);
        ode_params.set_tolerances(1e-11, 1e-9, None)?;

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

    #[inline]
    pub fn cached_a(&self) -> f64 {
        self.a
    }

    #[inline]
    pub fn cached_b(&self) -> f64 {
        self.b
    }

    pub fn set_params(&mut self, p: PendulumParams) {
        self.args = p;
        self.a = 6.0 / (p.m * p.l * p.l);
        self.b = 0.5 * p.m * p.l * p.l;
    }

    pub fn set_state(&mut self, s: PendulumState) {
        self.y[0] = wrap_angle(s.theta1);
        self.y[1] = wrap_angle(s.theta2);
        self.y[2] = s.p1;
        self.y[3] = s.p2;
    }

    /// Advance the state by `dt` with the adaptive ODE solver
    pub fn step(&mut self, dt: f64) -> Result<PendulumState, StrError> {
        let t1 = self.t + dt;
        self.solver
            .solve(&mut self.y, self.t, t1, None, &mut self.args)?;
        self.t = t1;

        self.y[0] = wrap_angle(self.y[0]);
        self.y[1] = wrap_angle(self.y[1]);

        Ok((&self.y).into())
    }

    /// fixed-step symplectic integrator (Velocity Verlet)
    pub fn step_symplectic(&mut self, dt: f64) -> PendulumState {
        let (mut theta1, mut theta2, mut p1, mut p2) = (self.y[0], self.y[1], self.y[2], self.y[3]);
        // let (theta1_dot, theta2_dot, p1_dot, p2_dot) = self.dp_dt(theta1, theta2, p1, p2);
        let PendulumState {
            theta1: _,
            theta2: _,
            p1: p1_dot,
            p2: p2_dot,
        } = dp_dt(
            &self.args,
            &PendulumState {
                theta1,
                theta2,
                p1,
                p2,
            },
            self.a,
            self.b,
        );

        // first half-step
        p1 += 0.5 * dt * p1_dot;
        p2 += 0.5 * dt * p2_dot;

        // full step for momentum
        let PendulumState {
            theta1: w1,
            theta2: w2,
            p1: _,
            p2: _,
        } = dtheta_dt(
            &PendulumState {
                theta1,
                theta2,
                p1,
                p2,
            },
            self.a,
        );
        theta1 = wrap_angle(theta1 + dt * w1);
        theta2 = wrap_angle(theta2 + dt * w2);

        // second half-step
        let PendulumState {
            theta1: _,
            theta2: _,
            p1: p1_dot2,
            p2: p2_dot2,
        } = dp_dt(
            &self.args,
            &PendulumState {
                theta1,
                theta2,
                p1,
                p2,
            },
            self.a,
            self.b,
        );

        p1 += 0.5 * dt * p1_dot2;
        p2 += 0.5 * dt * p2_dot2;

        self.y[0] = theta1;
        self.y[1] = theta2;
        self.y[2] = p1;
        self.y[3] = p2;
        self.t += dt;

        self.state()
    }

    pub fn energy(&self) -> f64 {
        let m = self.args.m;
        let l = self.args.l;
        let g = self.args.g;

        let th1 = self.y[0];
        let th2 = self.y[1];
        let p1 = self.y[2];
        let p2 = self.y[3];

        let delta = th1 - th2;
        let c = 3.0 * delta.cos();
        let denom = 16.0 - c * c;

        let w1 = self.a * (2.0 * p1 - c * p2) / denom;
        let w2 = self.a * (8.0 * p2 - c * p1) / denom;

        let tke = self.b * (w1 * w1 + w2 * w2 / 3.0);
        let v = -m * g * l * (th1.cos() + th2.cos());

        tke + v
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
    let pxl_scale = pxl_scale * 0.5; // because size is diameter

    let (s1, c1) = (theta1.sin() as f32, theta1.cos() as f32);
    let x1 = origin.x + (l as f32) * pxl_scale * s1;
    let y1 = origin.y + (l as f32) * pxl_scale * c1;

    let (s2, c2) = (theta2.sin() as f32, theta2.cos() as f32);
    let x2 = x1 + (l as f32) * pxl_scale * s2;
    let y2 = y1 + (l as f32) * pxl_scale * c2;

    (Vector2f::new(x1, y1), Vector2f::new(x2, y2))
}
