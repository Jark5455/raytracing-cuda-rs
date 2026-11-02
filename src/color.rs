use crate::vec4::Vec4;
use crate::interval::Interval;

pub type Color = Vec4;

const INTENSITY: Interval = Interval { min: 0.0, max: 0.999 };

pub fn linear_to_gamma(linear: f32) -> f32 {
    if linear > 0.0 {
        return linear.powf(1.0 / 2.2);
    }

    0.0
}

pub fn write_color(color: &Color) {

    let r = linear_to_gamma(color.x());
    let g = linear_to_gamma(color.y());
    let b = linear_to_gamma(color.z());

    let ir = (INTENSITY.clamp(r) * 256.0) as u8;
    let ig = (INTENSITY.clamp(g) * 256.0) as u8;
    let ib = (INTENSITY.clamp(b) * 256.0) as u8;

    println!("{} {} {}", ir, ig, ib);
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Color {
        Vec4(r, g, b, 0.0)
    }
}