<h1 align="center">Double Pendulum Simulation</h1>

## Overview

![demo](./demo.mp4)

A real-time **double pendulum** simulator written in Rust with [`rust-sfml`](https://github.com/jeremyletang/rust-sfml).  
It integrates the chaotic dynamics of many pendulums in parallel and visualizes them as a grid of colors or differences.

---

## Features
- Grid of pendulums with different initial angles
- Parallel stepping (using `rayon`)
- Symplectic integrator for stable motion
- Toggle between **color view** and **difference view**
- Spotlight a single pendulum with trace

---

## Controls
- `Space`: Pause/Resume  
- `R`: Reset all  
- `Tab`: Switch color/diff view  
- `Q`: Quit  
- **Left click**: select pendulum + trace  
- **Right click**: hide trace

---
