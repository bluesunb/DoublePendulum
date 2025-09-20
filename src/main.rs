pub mod ode;
// pub mod ode2;
pub mod render;
pub mod sim;
pub mod utils;

use std::f64::consts::PI;

use sfml::{
    graphics::{Color, Font, RenderTarget, RenderWindow, Text, Transformable},
    system::{Clock, Vector2f},
    window::{ContextSettings, Event, Key, Style, mouse::Button},
};

use crate::render::PendulumRenderer;
use crate::{
    ode::PendulumParams,
    render::PlotRenderer,
    sim::Simulations,
    utils::{linspace, meshgrid},
};

const PHYS_DT: f64 = 1.0 / 240.0;

fn main() {
    let mut window = RenderWindow::new(
        (1000, 1000),
        "Double Pendulum",
        Style::CLOSE,
        &ContextSettings::default(),
    )
    .unwrap();
    window.set_framerate_limit(140);

    let font =
        Font::from_file("/Users/bluesun/RustProject/SFML-Projects/double_pendulum/fonts/lowan.ttc")
            .expect("Could not load font from file");

    let mut hud = Text::new("", &font, 14);
    hud.set_fill_color(Color::WHITE);
    hud.set_position((12.0, 12.0));

    // ================= Setup Simulation =================

    let params = PendulumParams::default();
    let num_rows = 256usize;
    // let num_rows = 8usize;

    let min_theta = -PI * 0.95;
    let max_theta = PI * 0.95;

    let mut sims = Simulations::new(params, num_rows, min_theta, max_theta)
        .total_dt(PHYS_DT)
        .n_substeps(8);

    let min_window = 100.0;
    let max_window = window.size().x as f32 - min_window;
    let (origins_x, origins_y) = meshgrid(
        &linspace(min_window, max_window, num_rows, true),
        &linspace(min_window, max_window, num_rows, true),
    );
    let origins = origins_x
        .iter()
        .zip(origins_y.iter())
        .map(|(x, y)| Vector2f::new(*x, *y))
        .collect::<Vec<_>>();

    let mut renderer = PendulumRenderer::new();
    renderer.set_size(100.0);
    let mut plot_renderer =
        PlotRenderer::new((num_rows as u32, num_rows as u32), (min_window, min_window));
    plot_renderer.set_size((
        (max_window - min_window) as u32,
        (max_window - min_window) as u32,
    ));

    // ================= Main Loop =================

    let mut clock = Clock::start().unwrap(); // For fps calculation
    let mut idx = 0;
    let mut acc_time = 0.0;

    let mut prev_diff = sims.get_diff();

    while window.is_open() {
        while let Some(event) = window.poll_event() {
            match event {
                Event::Closed => window.close(),
                Event::KeyPressed { code, .. } => match code {
                    Key::Q => window.close(),
                    Key::Space => {
                        sims.toggle_all();
                        clock.restart();
                    }
                    Key::R => {
                        sims.reset_all();
                        clock.restart();
                    }
                    Key::Tab => {
                        plot_renderer.toggle_show_diff();
                    }
                    _ => {}
                },
                Event::MouseButtonPressed { button, x, y } => match button {
                    Button::Left => {
                        let gap = (max_window - min_window) / num_rows as f32;
                        let ix = ((x as f32 - min_window) / gap).floor() as isize;
                        let iy = ((y as f32 - min_window) / gap).floor() as isize;
                        if ix >= 0 && ix < num_rows as isize && iy >= 0 && iy < num_rows as isize {
                            idx = (iy * num_rows as isize + ix) as usize;
                            renderer.clear_trace();
                            renderer.set_origin(origins[idx]);
                            renderer.show_pendulum(true);
                            renderer.show_trace(true);
                        }
                    }
                    Button::Right => {
                        renderer.show_pendulum(false);
                        renderer.show_trace(false);
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        let dt = clock.restart().as_seconds() as f64;
        sims.step_parallel(dt);

        acc_time += dt;
        if acc_time > 1.0 || acc_time == 0.0 {
            hud.set_string(&format!("FPS: {:.2}", 1.0 / dt));
            acc_time = 0.0;
        }

        window.clear(Color::BLACK);
        plot_renderer.draw_color(&mut window as &mut RenderWindow, &sims);
        plot_renderer.draw_diff(&mut window as &mut RenderWindow, &sims, &prev_diff);
        renderer.draw(&mut window as &mut RenderWindow, sims.get(idx));
        window.draw(&hud);
        window.display();

        prev_diff = prev_diff.iter().map(|v| v * 0.9).collect();
    }
}
