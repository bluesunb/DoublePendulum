pub fn linspace<T>(start: T, end: T, n_steps: usize, endpoint: bool) -> Vec<T>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + From<f32>,
{
    let n_steps_f = T::from(n_steps as f32);
    let diff = if endpoint {
        (end - start) / T::from((n_steps - 1) as f32)
    } else {
        (end - start) / n_steps_f
    };
    (0..n_steps)
        .map(|i| start + T::from(i as f32) * diff)
        .collect()
}

pub fn meshgrid<T>(x: &[T], y: &[T]) -> (Vec<T>, Vec<T>)
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + From<f32>,
{
    let mut xx = Vec::with_capacity(x.len() * y.len());
    let mut yy = Vec::with_capacity(x.len() * y.len());
    for &yi in y {
        for &xi in x {
            xx.push(xi);
            yy.push(yi);
        }
    }
    (xx, yy)
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn inv_sigmoid(y: f64) -> f64 {
    (y / (1.0 - y)).ln()
}
