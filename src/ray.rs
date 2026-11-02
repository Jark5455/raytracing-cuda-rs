use crate::vec4::Vec4;
use crate::vec4::Point4;

pub struct Ray {
    pub origin: Point4,
    pub direction: Vec4,
}

impl Ray {

    pub fn new(origin: Point4, direction: Vec4) -> Ray {
        Ray { origin, direction }
    }

    pub fn at(&self, t: f32) -> Point4 {
        self.origin + self.direction * t
    }
}