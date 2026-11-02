use crate::ray::Ray;
use crate::hitable::HitRecord;
use crate::color::Color;
use crate::vec4::Vec4;

use rand::Rng;

pub trait Material {
    fn scatter(&self, rng: &mut impl Rng, ray_in: &Ray, rec: HitRecord) -> Option<(Color, Ray)>;
}

#[macro_export]
macro_rules! materials {
    ($name:ident { $($ty:ident),+ $(,)? }) => {
        #[repr(C, u32)]
        #[derive(Copy, Clone, Debug)]
        pub enum $name { $($ty($ty)),+ }

        impl $crate::material::Material for $name {
            #[inline(always)]
            fn scatter(&self, rng: &mut impl rand::Rng, r: &$crate::ray::Ray, rec: crate::hitable::HitRecord) -> Option<(crate::color::Color, crate::ray::Ray)> {
                match self {
                    $( $name::$ty(obj) => obj.scatter(rng, r, rec), )+
                }
            }
        }

        unsafe impl DeviceCopy for $name {}

        $(
            impl From<$ty> for $name {
                fn from(obj: $ty) -> Self { $name::$ty(obj) }
            }
        )+
    };
}

#[derive(Copy, Clone, Debug)]
pub struct Lambertian {
    pub albedo: Color,
}

impl Material for Lambertian {
    fn scatter(&self, rng: &mut impl Rng, _ray_in: &Ray, rec: HitRecord) -> Option<(Color, Ray)> {
        let mut scatter_direction = rec.normal + Vec4::random_vec3_unit(rng);

        if scatter_direction.vec3_near_zero() {
            scatter_direction = rec.normal;
        }

        Some((self.albedo, Ray::new(rec.p, scatter_direction)))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Metal {
    pub albedo: Color,
    pub fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, rng: &mut impl Rng, ray_in: &Ray, rec: HitRecord) -> Option<(Color, Ray)> {
        let reflected = Vec4::reflect(&ray_in.direction, &rec.normal).unit_vector() + (self.fuzz * Vec4::random_vec3_unit(rng));

        let scattered = Ray::new(rec.p, reflected);

        if Vec4::dot(&scattered.direction, &rec.normal) <= 0.0 {
            return None;
        }

        Some((self.albedo, Ray::new(rec.p, reflected)))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Dielectric {
    pub refractive_index: f32,
}

impl Dielectric {
    /*
    static double reflectance(double cosine, double refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
     */

    fn reflectance(cosine: f32, refraction_index: f32) -> f32 {
        let r0 = ((1.0 - refraction_index) / (1.0 + refraction_index)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, rng: &mut impl Rng, ray_in: &Ray, rec: HitRecord) -> Option<(Color, Ray)> {
        let attenuation = Color::new(1.0, 1.0, 1.0);
        let ri = if rec.front_face { 1.0 / self.refractive_index } else { self.refractive_index };

        let unit_direction = ray_in.direction.unit_vector();

        let cos_theta = Vec4::dot(&-unit_direction, &rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        

        let direction: Vec4 = if ri * sin_theta > 1.0 || Self::reflectance(cos_theta, ri) > rng.random::<f32>() {
            Vec4::reflect(&unit_direction, &rec.normal)
        } else {
            Vec4::refract(&unit_direction, &rec.normal, ri)
        };

        let scattered = Ray::new(rec.p, direction);
        Some((attenuation, scattered))
    }
}