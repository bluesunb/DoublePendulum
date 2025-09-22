use std::f64::consts::PI;

use sfml::graphics::Color;
use strum::EnumCount;

use crate::{ode::PendulumParams, sim::Simulations, utils::linspace};

#[derive(Debug)]
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
    pub range_1: (f64, f64),    // Range for x-axis
    pub range_2: (f64, f64),    // Range for y-axis
    pub n_steps_1: usize,       // Number of slices in x direction
    pub n_steps_2: usize,       // Number of slices in y direction
    pub params: PendulumParams, // Pendulum parameters
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumCount)]
pub enum RenderMode {
    Pendulum,
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
            range_1: (-PI * 0.95, PI * 0.95),
            range_2: (-PI * 0.95, PI * 0.95),
            n_steps_1: 256,
            n_steps_2: 256,
            params: PendulumParams::default(),
        }
    }
}

impl AppConfig {
    pub fn simuations(&self) -> Simulations {
        let slices_x = linspace(self.range_1.0, self.range_1.1, self.n_steps_1, true);
        let slices_y = linspace(self.range_2.0, self.range_2.1, self.n_steps_2, true);
    }
}
