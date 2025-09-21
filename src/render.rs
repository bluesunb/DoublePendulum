use std::{collections::VecDeque, f64::consts::PI};

use sfml::{
    graphics::{
        CircleShape, Color, PrimitiveType, RenderStates, RenderTarget, Shape, Transformable, Vertex,
    },
    system::Vector2f,
};

use crate::{
    ode::angles_to_xy,
    sim::{Simulation, Simulations},
};

pub struct PendulumRenderer<'a> {
    bob: CircleShape<'a>,
    lines: [[Vertex; 2]; 2],
    trace: VecDeque<Vector2f>,
    max_trace_length: usize,
    show_pendulum: bool,
    show_trace: bool,
    origin: Vector2f,
    size: f32,

    // Extra UI elements
    background: CircleShape<'a>,
    point: CircleShape<'a>,
    pendulum_origin: Vector2f,
}

impl<'a> PendulumRenderer<'a> {
    pub fn new() -> Self {
        let origin = Vector2f::new(0.0, 0.0);
        let size = 20.0;

        let mut background = CircleShape::new(size + 20.0, 32);
        background.set_origin((background.radius(), background.radius()));
        background.set_fill_color(Color::rgba(20, 20, 20, 200));
        background.set_outline_color(Color::WHITE);
        background.set_outline_thickness(1.0);

        let point_rad = 10.0;
        let mut point = CircleShape::new(point_rad, 12);
        point.set_origin((point_rad, point_rad));
        point.set_position(origin);
        point.set_outline_color(Color::RED);
        point.set_outline_thickness(2.0);
        point.set_fill_color(Color::TRANSPARENT);

        Self {
            bob: CircleShape::new(1.0, 24),
            lines: [[Vertex::default(); 2]; 2],
            trace: VecDeque::new(),
            max_trace_length: 1000,
            show_pendulum: false,
            show_trace: false,
            origin: Vector2f::new(0.0, 0.0),
            size: 100.0,
            background,
            point,
            pendulum_origin: origin,
        }
    }

    fn update(&mut self) {
        self.bob.set_radius(self.size + 20.0);
        self.bob.set_origin((self.bob.radius(), self.bob.radius()));

        self.background.set_radius(self.size * 1.1);
        self.background
            .set_origin((self.background.radius(), self.background.radius()));
    }

    pub fn set_origin(&mut self, origin: Vector2f) {
        self.origin = origin;
        self.update();
    }

    pub fn set_size(&mut self, size: f32) {
        self.size = size;
        self.update();
    }

    pub fn show_pendulum(&mut self, show: bool) {
        self.show_pendulum = show;
    }

    pub fn show_trace(&mut self, show: bool) {
        self.show_trace = show;
    }

    pub fn clear_trace(&mut self) {
        self.trace.clear();
    }

    pub fn sample_trace(&mut self, sim: &Simulation) {
        let vec_state = sim.vec_state();
        let (_, pos2) = angles_to_xy(
            sim.params.l,
            vec_state[0],
            vec_state[1],
            &self.pendulum_origin,
            self.size,
        );

        if self.trace.len() >= self.max_trace_length {
            self.trace.pop_front();
        }
        self.trace.push_back(pos2);
    }

    pub fn draw<T: RenderTarget>(&mut self, target: &mut T, sim: &Simulation) {
        if self.show_pendulum {
            self.draw_extra(target);
            self.draw_trace(target, sim);
            self.draw_pendulum(target, sim);
        }
    }

    pub fn draw_extra<T: RenderTarget>(&mut self, target: &mut T) {
        let half_x = (target.size().x as f32) / 2.0;
        let half_y = (target.size().y as f32) / 2.0;
        let over_half_x = self.origin.x > half_x;
        let over_half_y = self.origin.y > half_y;

        let x = half_x * if over_half_x { 0.5 } else { 1.5 };
        let y = half_y * if over_half_y { 1.5 } else { 0.5 };

        self.background.set_position((x, y));
        self.pendulum_origin = Vector2f::new(x, y);
        target.draw(&self.background);

        self.point.set_position(self.origin);
        target.draw(&self.point);
    }

    pub fn draw_trace<T: RenderTarget>(&mut self, target: &mut T, sim: &Simulation) {
        if !self.show_trace {
            return;
        }

        self.sample_trace(sim);

        if self.trace.len() > 2 {
            let trace_vertices: Vec<Vertex> = self
                .trace
                .iter()
                .map(|&pos| Vertex {
                    position: pos,
                    color: Color::rgba(200, 200, 200, 100),
                    ..Default::default()
                })
                .collect();

            target.draw_primitives(
                &trace_vertices,
                PrimitiveType::LINE_STRIP,
                &RenderStates::default(),
            );
        }
    }

    pub fn draw_pendulum<T: RenderTarget>(&mut self, target: &mut T, sim: &Simulation) {
        let vec_state = sim.vec_state();
        let (pos1, pos2) = angles_to_xy(
            sim.params.l,
            vec_state[0],
            vec_state[1],
            &self.pendulum_origin,
            self.size,
        );

        // Draw lines
        let color = state_to_color(vec_state[0], vec_state[1], -PI, PI);

        self.lines[0][0].position = self.pendulum_origin;
        self.lines[0][1].position = pos1;
        self.lines[1][0].position = pos1;
        self.lines[1][1].position = pos2;

        for line in &mut self.lines {
            line[0].color = color;
            line[1].color = color;
        }

        target.draw_primitives(
            &self.lines[0],
            PrimitiveType::LINES,
            &RenderStates::default(),
        );
        target.draw_primitives(
            &self.lines[1],
            PrimitiveType::LINES,
            &RenderStates::default(),
        );

        // Draw bobs
        self.bob.set_radius(0.05 * self.size);
        self.bob.set_origin((self.bob.radius(), self.bob.radius()));
        self.bob.set_position(pos1);
        self.bob.set_fill_color(color);
        target.draw(&self.bob);

        self.bob.set_position(pos2);
        self.bob.set_fill_color(color);
        target.draw(&self.bob);
    }
}

pub struct MultiRenderer<'a> {
    bob: CircleShape<'a>,
    lines: [[Vertex; 2]; 2],
    traces: Vec<VecDeque<Vector2f>>,
    max_trace_length: usize,
    origins: Vec<Vector2f>,
    sizes: Vec<f32>,
    show_traces: bool,
}

impl<'a> MultiRenderer<'a> {
    pub fn new(origins: Vec<Vector2f>, sizes: Vec<f32>, count: usize) -> Self {
        let mut bob = CircleShape::new(1.0, 24);
        bob.set_origin((1.0, 1.0));

        let mut lines = [[Vertex::default(); 2]; 2];
        for line in &mut lines {
            line[0].color = Color::WHITE;
            line[1].color = Color::WHITE;
        }

        Self {
            bob,
            lines,
            traces: vec![VecDeque::new(); count],
            max_trace_length: 1000,
            origins,
            sizes,
            show_traces: true,
        }
    }

    pub fn show_trace(mut self, show: bool) -> Self {
        self.show_traces = show;
        self
    }

    fn sample_traces(&mut self, sims: &Simulations) {
        for (i, sim) in sims.sims.iter().enumerate() {
            let vec_state = sim.vec_state();
            let (_, pos2) = angles_to_xy(
                sim.params.l,
                vec_state[0],
                vec_state[1],
                &self.origins[i],
                self.sizes[i],
            );

            // let trace = &mut self.traces[i];
            let trace = &mut self.traces[i];
            if trace.len() >= self.max_trace_length {
                trace.pop_front();
            }
            trace.push_back(pos2);
        }
    }

    pub fn draw<T: RenderTarget>(&mut self, target: &mut T, sims: &Simulations) {
        self.draw_traces(target, sims);
        self.draw_pendulums(target, sims);
    }

    pub fn draw_traces<T: RenderTarget>(&mut self, target: &mut T, sims: &Simulations) {
        if !self.show_traces {
            return;
        }

        self.sample_traces(sims);

        for trace in self.traces.iter() {
            if trace.len() > 2 {
                let trace_vertices: Vec<Vertex> = trace
                    .iter()
                    .map(|&pos| Vertex {
                        position: pos,
                        color: Color::rgba(200, 200, 200, 100),
                        ..Default::default()
                    })
                    .collect();

                target.draw_primitives(
                    &trace_vertices,
                    PrimitiveType::LINE_STRIP,
                    &RenderStates::default(),
                );
            }
        }
    }

    pub fn draw_pendulums<T: RenderTarget>(&mut self, target: &mut T, sims: &Simulations) {
        for (i, sim) in sims.sims.iter().enumerate() {
            let vec_state = sim.vec_state();
            let (pos1, pos2) = angles_to_xy(
                sim.params.l,
                vec_state[0],
                vec_state[1],
                &self.origins[i],
                self.sizes[i],
            );

            // Draw lines
            let color = state_to_color(vec_state[0], vec_state[1], -PI, PI);

            self.lines[0][0].position = self.origins[i];
            self.lines[0][1].position = pos1;
            self.lines[1][0].position = pos1;
            self.lines[1][1].position = pos2;

            for line in &mut self.lines {
                line[0].color = color;
                line[1].color = color;
            }

            target.draw_primitives(
                &self.lines[0],
                PrimitiveType::LINES,
                &RenderStates::default(),
            );
            target.draw_primitives(
                &self.lines[1],
                PrimitiveType::LINES,
                &RenderStates::default(),
            );

            // Draw bobs
            self.bob.set_radius(0.05 * self.sizes[i]);
            self.bob.set_origin((self.bob.radius(), self.bob.radius()));
            self.bob.set_position(pos1);
            self.bob.set_fill_color(color);
            target.draw(&self.bob);

            self.bob.set_position(pos2);
            self.bob.set_fill_color(color);
            target.draw(&self.bob);
        }
    }
}

// pub struct PlotRenderer<'a> {
//     pub resolution: Vector2u,
//     pub pos: Vector2f,
//     scale: Vector2f,
//     texture: FBox<Texture>,
//     sprite: Sprite<'a>,
//     show_diff: bool,
// }

// impl<'a> PlotRenderer<'a> {
//     pub fn new<S, P>(resolution: S, position: P) -> Self
//     where
//         S: Into<Vector2u>,
//         P: Into<Vector2f>,
//     {
//         let size = resolution.into();
//         let pos = position.into();

//         let mut texture = Texture::new().expect("alloc texture");
//         texture.create(size.x, size.y).expect("create texture");
//         texture.set_smooth(false);

//         let mut sprite = Sprite::new();
//         sprite.set_position(pos);

//         Self {
//             resolution: size,
//             pos,
//             scale: Vector2f::new(1.0, 1.0),
//             texture,
//             sprite,
//             show_diff: false,
//         }
//     }

//     pub fn set_show_diff(&mut self, show: bool) {
//         self.show_diff = show;
//     }

//     pub fn toggle_show_diff(&mut self) {
//         self.show_diff = !self.show_diff;
//     }

//     pub fn set_size<T: Into<Vector2u>>(&mut self, size: T) {
//         let size = size.into();
//         let scale_x = size.x as f32 / self.resolution.x as f32;
//         let scale_y = size.y as f32 / self.resolution.y as f32;
//         self.scale = Vector2f::new(scale_x, scale_y);
//         self.sprite.set_scale(self.scale);
//     }

//     #[inline]
//     pub fn draw_pixels<T: RenderTarget>(&mut self, target: &mut T, rgba: &[u8]) {
//         self.texture
//             .update_from_pixels(rgba, self.resolution.x, self.resolution.y, 0, 0);
//         target.draw(&self.sprite);
//     }

//     fn plot<T: RenderTarget>(&mut self, target: &mut T, data: &Vec<u8>) {
//         self.texture
//             .update_from_pixels(data, self.resolution.x, self.resolution.y, 0, 0);

//         let mut sprite = sfml::graphics::Sprite::with_texture(&self.texture);
//         sprite.set_position(self.pos);
//         sprite.set_scale(self.scale);
//         target.draw(&sprite);
//     }

//     pub fn draw_color<T: RenderTarget>(&mut self, target: &mut T, sims: &Simulations) {
//         if self.show_diff {
//             return;
//         }

//         self.plot(
//             target,
//             &sims
//                 .get_colors()
//                 .iter()
//                 .flat_map(|c| [c.r, c.g, c.b, c.a])
//                 .collect::<Vec<_>>(),
//         );
//     }

//     pub fn draw_diff<T: RenderTarget>(
//         &mut self,
//         target: &mut T,
//         sims: &Simulations,
//         prev_diff: &Vec<f64>,
//     ) {
//         if !self.show_diff {
//             return;
//         }

//         self.plot(
//             target,
//             &sims
//                 .get_diff()
//                 .iter()
//                 .zip(prev_diff.iter())
//                 .flat_map(|(cur, prev)| {
//                     let v = prev * 0.9 + cur * 0.1;
//                     let s = (sigmoid(v) * 255.0) as u8;
//                     let dim = (sigmoid(0.2 * v) * 255.0) as u8;
//                     [dim, dim, s, 255u8]
//                 })
//                 .collect(),
//         );
//     }
// }

pub fn state_to_color(theta1: f64, theta2: f64, min: f64, max: f64) -> Color {
    let mut r = 255.0 * (0.5 + theta2 / (max - min));
    let mut g = 255.0 * (0.5 + theta1 / (max - min));
    let mut b = 255.0 * (0.5 - theta2 / (max - min));

    r = r.clamp(0.0, 255.0);
    g = g.clamp(0.0, 255.0);
    b = b.clamp(0.0, 255.0);

    Color::rgb(r as u8, g as u8, b as u8)
}
