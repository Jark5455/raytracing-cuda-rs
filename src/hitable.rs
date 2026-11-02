use bytemuck::Zeroable;

use crate::vec4::Point4;
use crate::vec4::Vec4;
use crate::ray::Ray;
use crate::interval::Interval;

pub struct HitRecord {
    pub p: Point4,
    pub normal: Vec4,
    pub t: f32,
    pub front_face: bool,
    pub mat: *const crate::Material0,
}

unsafe impl Zeroable for HitRecord {
    fn zeroed() -> Self {
        HitRecord {
            p: Vec4(0.0, 0.0, 0.0, 0.0),
            normal: Vec4(0.0, 0.0, 0.0, 0.0),
            t: 0.0,
            front_face: false,
            mat: std::ptr::null(),
        }
    }
}

impl HitRecord {
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: &Vec4) {
        self.front_face = Vec4::dot(&r.direction, outward_normal) < 0.0;
        self.normal = if self.front_face {
            *outward_normal
        } else {
            -*outward_normal
        };
    }
}

pub trait Hitable {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord>;
}

// compile time constant hitable enum

#[macro_export]
macro_rules! hitables {
    ($name:ident { $($ty:ident),+ $(,)? }) => {
        #[repr(C, u32)]
        #[derive(Copy, Clone)]
        pub enum $name { $($ty($ty)),+ }

        impl crate::hitable::Hitable for $name {
            #[inline(always)]
            fn hit(&self, r: &crate::ray::Ray, ray_t: crate::interval::Interval) -> Option<crate::hitable::HitRecord> {
                match self {
                    $( $name::$ty(obj) => obj.hit(r, ray_t), )+
                }
            }
        }

        unsafe impl DeviceCopy for $name {}

        impl crate::hitable::Hitable for [$name] {
            fn hit(&self, r: &crate::ray::Ray, ray_t: crate::interval::Interval) -> Option<crate::hitable::HitRecord> {
                let mut closest = None;
                let mut closest_so_far = ray_t.max;
                for obj in self {
                    if let Some(rec) = obj.hit(r, crate::interval::Interval {min: ray_t.min, max: closest_so_far}) {
                        closest_so_far = rec.t;
                        closest = Some(rec);
                    }
                }
                closest
            }
        }

        $(
            impl From<$ty> for $name {
                fn from(obj: $ty) -> Self { $name::$ty(obj) }
            }
        )+
    };
}