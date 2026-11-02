use bytemuck::Zeroable;

use crate::hitable::HitRecord;
use crate::hitable::Hitable;
use crate::ray::Ray;
use crate::vec4::Point4;
use crate::vec4::Vec4;
use crate::interval::Interval;

#[derive(Copy, Clone, Debug)]
pub struct Sphere {
    pub center: Point4,
    pub radius: f32,
    pub mat: *const crate::Material0,
}

impl Sphere {
    pub fn new(center: Point4, radius: f32, mat: *const crate::Material0) -> Sphere {
        Sphere { center, radius, mat }
    }
}

impl Hitable for Sphere {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let oc = self.center - r.origin;
        let a = r.direction.length_squared();
        let h = Vec4::dot(&oc, &r.direction);
        let c = oc.length_squared() - self.radius.powi(2);

        let discriminant = h*h - a*c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range.
        let mut root = (h - sqrtd) / a;
        if root <= ray_t.min || ray_t.max <= root {
            root = (h + sqrtd) / a;
            if root <= ray_t.min || ray_t.max <= root {
                return None;
            }
        }

        let mut rec = HitRecord::zeroed();

        rec.t = root;
        rec.p = r.at(rec.t);
        rec.mat = self.mat;

        let outward_normal = (rec.p - self.center) / self.radius;
        rec.set_face_normal(r, &outward_normal);

        Some(rec)
    }
}