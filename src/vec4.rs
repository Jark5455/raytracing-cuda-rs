use std::ops::{Add, Mul, Sub, Div, Neg};
use bytemuck::Zeroable;
use cuda_core::DeviceCopy;
use rand::Rng;

// why vec4? see https://leimao.github.io/blog/CUDA-Data-Alignment/

#[derive(DeviceCopy, Copy, Clone, Debug, Zeroable)]
pub struct Vec4(pub f32, pub f32, pub f32, pub f32);
pub type Point4 = Vec4;

impl Vec4 {
    pub fn x(&self) -> f32 {
        self.0
    }

    pub fn y(&self) -> f32 {
        self.1
    }

    pub fn z(&self) -> f32 {
        self.2
    }

    pub fn w(&self) -> f32 {
        self.3
    }

    pub fn dot(v: &Vec4, other: &Vec4) -> f32 {
        v.0 * other.0 + v.1 * other.1 + v.2 * other.2 + v.3 * other.3
    }

    pub fn cross(v: &Vec4, other: &Vec4) -> Vec4 {
        Vec4(
            v.1 * other.2 - v.2 * other.1,
            v.2 * other.0 - v.0 * other.2,
            v.0 * other.1 - v.1 * other.0,
            0.0,
        )
    }

    pub fn length(&self) -> f32 {
        (Self::dot(self, self)).sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        Self::dot(self, self)
    }

    pub fn unit_vector(&self) -> Vec4 {
        *self / self.length()
    }

    pub fn random_vec3(rng: &mut impl Rng) -> Vec4 {
        Vec4(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>(), 0.0)
    }

    pub fn random_vec4(rng: &mut impl Rng) -> Vec4 {
        Vec4(rng.random(), rng.random(), rng.random(), rng.random())
    }

    // random_range wont compile because it contains a panic pathway if min > max
    fn random_range_unchecked(rng: &mut impl Rng, min: f32, max: f32) -> f32 {
        min + (max - min) * rng.random::<f32>()
    }

    pub fn random_vec4_range(rng: &mut impl Rng, min: f32, max: f32) -> Vec4 {
        Vec4(
            Self::random_range_unchecked(rng, min, max),
            Self::random_range_unchecked(rng, min, max),
            Self::random_range_unchecked(rng, min, max),
            Self::random_range_unchecked(rng, min, max),
        )
    }

    pub fn random_vec3_range(rng: &mut impl Rng, min: f32, max: f32) -> Vec4 {
        Vec4(
            Self::random_range_unchecked(rng, min, max),
            Self::random_range_unchecked(rng, min, max),
            Self::random_range_unchecked(rng, min, max),
            0.0,
        )
    }

    pub fn random_vec3_unit(rng: &mut impl Rng) -> Vec4 {
        loop {
            let p = Vec4::random_vec3_range(rng, -1.0, 1.0);
            let lensq = p.length_squared();

            if 1e-160 < lensq && lensq <= 1.0 {
                return p / lensq.sqrt();
            }
        }
    }

    pub fn random_vec3_on_hemisphere(rng: &mut impl Rng, normal: Vec4) -> Vec4 {
        let on_unit_sphere = Self::random_vec3_unit(rng);

        if Vec4::dot(&on_unit_sphere, &normal) > 0.0 {
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }

    pub fn random_vec3_in_disk(rng: &mut impl Rng) -> Vec4 {
        loop {
            let p = Vec4(rng.random::<f32>(), rng.random::<f32>(), 0.0, 0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    pub fn vec3_near_zero(&self) -> bool {
        let s = 1e-8;
        self.0.abs() < s && self.1.abs() < s && self.2.abs() < s
    }

    pub fn reflect(v: &Vec4, n: &Vec4) -> Vec4 {
        v - 2.0 * Vec4::dot(v, n) * n
    }

    pub fn refract(uv: &Vec4, n: &Vec4, etai_over_etat: f32) -> Vec4 {
        let cos_theta = f32::min(Vec4::dot(&-uv, n), 1.0);
        let r_out_perp = etai_over_etat * (uv + cos_theta * n);
        let r_out_parallel = -((1.0 - r_out_perp.length_squared()).abs().sqrt()) * n;
        r_out_perp + r_out_parallel
    }
}

impl Neg for Vec4 {
    type Output = Vec4;

    #[inline(always)]
    fn neg(self) -> Vec4 {
        Vec4(-self.0, -self.1, -self.2, -self.3)
    }
}

impl Neg for &Vec4 {
    type Output = Vec4;

    #[inline(always)]
    fn neg(self) -> Vec4 {
        -*self
    }
}

macro_rules! impl_vec_op {
    ($Trait:ident, $method:ident, $op:tt) => {
        impl $Trait<Vec4> for Vec4 {
            type Output = Vec4;

            #[inline(always)]
            fn $method(self, o: Vec4) -> Vec4 {
                Vec4(self.0 $op o.0, self.1 $op o.1, self.2 $op o.2, self.3 $op o.3)
            }
        }

        impl $Trait<&Vec4> for Vec4 {
            type Output = Vec4;

            #[inline(always)]
            fn $method(self, o: &Vec4) -> Vec4 {
                self $op *o
            }
        }

        impl $Trait<Vec4> for &Vec4 {
            type Output = Vec4;

            #[inline(always)]
            fn $method(self, o: Vec4) -> Vec4 {
                *self $op o
            }
        }

        impl $Trait<&Vec4> for &Vec4 {
            type Output = Vec4;

            #[inline(always)]
            fn $method(self, o: &Vec4) -> Vec4 {
                *self $op *o
            }
        }
    };
}

macro_rules! impl_scalar_op {
    ($Trait:ident, $method:ident, $op:tt) => {
        impl $Trait<f32> for Vec4 {
            type Output = Vec4;

            #[inline(always)]
            fn $method(self, s: f32) -> Vec4 {
                Vec4(self.0 $op s, self.1 $op s, self.2 $op s, self.3 $op s)
            }
        }

        impl $Trait<f32> for &Vec4 {
            type Output = Vec4;

            #[inline(always)]
            fn $method(self, s: f32) -> Vec4 {
                *self $op s
            }
        }

        impl $Trait<Vec4> for f32 {
            type Output = Vec4;

            #[inline(always)]
            fn $method(self, v: Vec4) -> Vec4 {
                Vec4(self $op v.0, self $op v.1, self $op v.2, self $op v.3)
            }
        }

        impl $Trait<&Vec4> for f32 {
            type Output = Vec4;

            #[inline(always)]
            fn $method(self, v: &Vec4) -> Vec4 {
                self $op *v
            }
        }
    };
}

impl_vec_op!(Add, add, +);
impl_vec_op!(Sub, sub, -);
impl_vec_op!(Mul, mul, *);
impl_vec_op!(Div, div, /);

impl_scalar_op!(Mul, mul, *);
impl_scalar_op!(Div, div, /);
