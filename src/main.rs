pub mod ode;
pub mod plot;
pub mod render;
pub mod sim;
pub mod utils;

use std::f64::consts::PI;

use sfml::{
    cpp::FBox,
    graphics::{Color, Font, RenderTarget, RenderWindow, Text, Transformable},
    system::{Clock, Vector2f},
    window::{ContextSettings, Event, Key, Style, mouse::Button},
};
use strum::{EnumCount, EnumIter, IntoEnumIterator};

use crate::{ode::PendulumParams, plot::PlotRenderer, render::PendulumRenderer, sim::Simulations};

#[derive(Debug, Clone)]
pub struct AppConfig {
    // Window configuration
    pub width: u32,           // Window width
    pub height: u32,          // Window height
    pub title: String,        // Window title
    pub framerate_limit: u32, // Frame rate limit
    pub window_padding: f32,  // Padding around the window edges

    // Font configuration
    pub font_path: String, // Path to the font file
    pub font_size: u32,    // Font size for HUD text
    pub hud_color: Color,  // Color of the HUD text

    // Simulation configuration
    pub phys_dt: f64,           // Physics time step
    pub range_x: (f64, f64),    // Range for x-axis
    pub range_y: (f64, f64),    // Range for y-axis
    pub n_steps_x: usize,       // Number of slices in x direction
    pub n_steps_y: usize,       // Number of slices in y direction
    pub params: PendulumParams, // Pendulum parameters

    // App stats
    pub render_mode: RenderMode, // Current rendering mode
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumCount, EnumIter)]
pub enum RenderMode {
    #[default]
    ColorMap,
    DiffMap,
    EnergyMap,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            width: 720,
            height: 720,
            title: "Double Pendulum".to_string(),
            framerate_limit: 120,
            window_padding: 60.0,
            font_path: "/Users/bluesun/RustProject/SFML-Projects/double_pendulum/fonts/lowan.ttc"
                .to_string(),
            font_size: 12,
            hud_color: Color::WHITE,
            phys_dt: 1.0 / 240.0,
            range_x: (-PI * 0.95, PI * 0.95),
            range_y: (-PI * 0.95, PI * 0.95),
            n_steps_x: 256,
            n_steps_y: 256,
            params: PendulumParams::default(),
            render_mode: RenderMode::default(),
        }
    }
}

impl<'a> AppConfig {
    pub fn window(&self) -> FBox<RenderWindow> {
        let mut window = RenderWindow::new(
            (self.width, self.height),
            &self.title,
            Style::CLOSE,
            &ContextSettings::default(),
        )
        .expect("Could not open window");

        window.set_framerate_limit(self.framerate_limit);
        window
    }

    pub fn font(&self) -> FBox<Font> {
        Font::from_file(&self.font_path).expect("Could not load font from file")
    }

    pub fn simuations(&self) -> Simulations<'a> {
        Simulations::new(
            self.params,
            self.range_x,
            self.range_y,
            self.n_steps_x,
            self.n_steps_y,
        )
        .total_dt(self.phys_dt)
        .n_substeps(8)
    }
}

pub fn run(cfg: AppConfig) {
    let mut window = cfg.window();
    let font = cfg.font();

    // ================= Base Objects =================

    let mut hud = Text::new("", &font, cfg.font_size);
    hud.set_fill_color(Color::WHITE);
    hud.set_position((cfg.width as f32 * 0.1, cfg.height as f32 * 0.1));

    let mut sims = cfg.simuations();
    let mut pendulum_renderer = PendulumRenderer::new();
    let mut plot_renderer = PlotRenderer::new(
        (cfg.n_steps_x as u32, cfg.n_steps_y as u32),
        (cfg.window_padding, cfg.window_padding),
    );
    plot_renderer.set_size((
        (cfg.width as f32 - 2.0 * cfg.window_padding),
        (cfg.height as f32 - 2.0 * cfg.window_padding),
    ));

    let mut render_mode = RenderMode::default();

    let mut clock = Clock::start().unwrap();

    let total_pixels = cfg.n_steps_x * cfg.n_steps_y;
    let mut color_data = vec![0u8; 4 * total_pixels];
    let mut diff_data = vec![0u8; 4 * total_pixels];
    let mut energy_data = vec![0u8; 4 * total_pixels];

    // ================= Main Loop =================

    let mut fps_acc = 0.0;
    let mut sim_idx = 0;

    while window.is_open() {
        // Event handling
        while let Some(event) = window.poll_event() {
            match event {
                Event::Closed => window.close(),
                Event::KeyPressed { code, shift, .. } => match code {
                    Key::Q => window.close(),
                    Key::R => sims.reset_all(),
                    Key::Space => sims.toggle_all(),
                    Key::Tab => {
                        // Cycle through render modes
                        let cur_idx = RenderMode::iter()
                            .position(|mode| mode == render_mode)
                            .unwrap_or(0);
                        let diff = if shift { -1 } else { 1 };
                        let new_idx = (cur_idx as isize + diff)
                            .rem_euclid(RenderMode::COUNT as isize)
                            as usize;
                        render_mode = RenderMode::iter()
                            .nth(new_idx)
                            .unwrap_or(RenderMode::ColorMap);
                    }
                    _ => {}
                },
                Event::MouseButtonPressed { button, x, y } => match button {
                    Button::Left => {
                        let gap_x = (cfg.range_x.1 - cfg.range_x.0) / cfg.n_steps_x as f64;
                        let gap_y = (cfg.range_y.1 - cfg.range_y.0) / cfg.n_steps_y as f64;
                        let ix = ((x as f32 - cfg.window_padding) / gap_x as f32).floor() as isize;
                        let iy = ((y as f32 - cfg.window_padding) / gap_y as f32).floor() as isize;
                        if ix >= 0
                            && ix < cfg.n_steps_x as isize
                            && iy >= 0
                            && iy < cfg.n_steps_y as isize
                        {
                            sim_idx = (iy * cfg.n_steps_x as isize + ix) as usize;
                            pendulum_renderer.set_origin(Vector2f::new(x as f32, y as f32));
                            pendulum_renderer.show_pendulum(true);
                            pendulum_renderer.show_trace(true);
                        }
                    }
                    Button::Right => {
                        pendulum_renderer.show_pendulum(false);
                        pendulum_renderer.show_trace(false);
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        // Stepping the simulation
        let dt = clock.restart().as_seconds() as f64;
        sims.step_parallel(dt);

        // Show FPS
        fps_acc += dt;
        if fps_acc >= 1.0 || fps_acc == 0.0 {
            hud.set_string(&format!("FPS: {:.2}", 1.0 / dt));
            fps_acc = 0.0;
        }

        // Window rendering
        window.clear(Color::BLACK);
        match render_mode {
            RenderMode::ColorMap => {
                sims.fill_colors_rgba(&mut color_data);
                plot_renderer.draw(&mut window as &mut RenderWindow, &color_data);
            }
            RenderMode::DiffMap => {
                sims.fill_diff_rgba(&mut diff_data);
                plot_renderer.draw(&mut window as &mut RenderWindow, &diff_data);
            }
            RenderMode::EnergyMap => {
                sims.fill_energies_rgba(&mut energy_data);
                plot_renderer.draw(&mut window as &mut RenderWindow, &energy_data);
            }
        }
        pendulum_renderer.draw(&mut window as &mut RenderWindow, sims.get(sim_idx));
        window.draw(&hud);
        window.display();
    }
}

fn main() {
    let config = AppConfig::default();
    run(config);
}
