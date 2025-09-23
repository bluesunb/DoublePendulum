use sfml::{
    cpp::FBox,
    graphics::{RenderTarget, Sprite, Texture, Transformable},
    system::{Vector2f, Vector2u},
};

pub struct PlotRenderer {
    texture: FBox<Texture>,
    pos: Vector2f,
    resolution: Vector2u,
    scale: Vector2f,
}

impl PlotRenderer {
    pub fn new<S, P>(resolution: S, position: P) -> Self
    where
        S: Into<Vector2u>,
        P: Into<Vector2f>,
    {
        let size = resolution.into();
        let pos = position.into();

        let mut texture = Texture::new().expect("Alloc texture");
        texture.create(size.x, size.y).expect("Create texture");
        texture.set_smooth(true);

        Self {
            texture,
            pos,
            resolution: size,
            scale: Vector2f::new(1.0, 1.0),
        }
    }

    pub fn set_size<T: Into<Vector2f>>(&mut self, size: T) {
        let size = size.into();
        self.scale.x = size.x / self.resolution.x as f32;
        self.scale.y = size.y / self.resolution.y as f32;
    }

    #[inline]
    pub fn draw<T: RenderTarget>(&mut self, target: &mut T, rgba: &[u8]) {
        assert_eq!(
            rgba.len(),
            4 * (self.resolution.x * self.resolution.y) as usize
        );

        self.texture
            .update_from_pixels(rgba, self.resolution.x, self.resolution.y, 0, 0);

        let mut sprite = Sprite::with_texture(&self.texture);
        sprite.set_position(self.pos);
        sprite.set_scale(self.scale);
        target.draw(&sprite);
    }
}
