mod vec4;
mod color;
mod ray;
mod camera;
mod hitable;
mod sphere;
mod interval;
mod material;

use cuda_core::{ContextLimit, CudaContext, DeviceBuffer, DeviceCopy, LaunchConfig2D};
use cuda_device::{DisjointSlice, kernel, launch_contract, thread};
use cuda_host::{cuda_module, RowWidth};
use rand_pcg::Pcg32;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;

use crate::vec4::Vec4;
use crate::color::Color;
use crate::color::write_color;
use crate::camera::Camera;
use crate::sphere::Sphere;
use crate::material::Lambertian;
use crate::material::Metal;
use crate::material::Dielectric;

hitables!( Hitable0 { Sphere } );
materials!( Material0 { Lambertian, Metal, Dielectric } );

#[cuda_module]
mod kernels {
    use cuda_device::Runtime2DIndex;

    use rand::{Rng, SeedableRng};
    use rand_pcg::{Lcg64Xsh32, Pcg32};
    use crate::material::Material;
    use super::*;
    use super::vec4::Vec4;
    use super::vec4::Point4;
    use super::color::Color;
    use super::camera::Camera;
    use super::ray::Ray;
    use super::interval::Interval;
    use super::hitable::Hitable;

    #[kernel]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn render(mut out: DisjointSlice<Color, Runtime2DIndex>, world: &[Hitable0], camera: Camera, seed: u64) {
        let col = thread::index_2d_col();
        let row = thread::index_2d_row();

        if let Some(idx) = thread::index_2d_runtime(&out) {

            let tid = (col * camera.image_height as usize + row) as u64;
            let mut rng = Pcg32::new(seed, tid);

            if let Some(px) = out.get_mut(idx) {

                let pixel_center = camera.pixel00_loc + (col as f32 * camera.pixel_delta_u + row as f32 * camera.pixel_delta_v);
                let ray_direction = pixel_center - camera.camera_center;

                let mut pixel_color = Color::new(0.0, 0.0, 0.0);

                // shitty workaround
                // (tid >> 63) is always 0
                // for some reason ptxas marks _sample as per warp not per lane
                // this causes multiple threads to decrement the shared counter rather than individual counters
                // this causes the counter to underflow which will result in collecting 4.29 billion samples
                // adding (tid >> 63) adds a dependence on tid which forces the counter to be per lane
                // nvidia pls fix
                for _sample in (0..camera.samples_per_pixel + (tid >> 63) as u32) {
                    let ray = get_ray(&mut rng, &camera, col, row);
                    pixel_color = pixel_color + ray_color(&mut rng, camera.max_depth, &ray, world);
                }

                *px = pixel_color * camera.pixel_samples_scale;
            }
        }
    }

    pub fn sample_square(rng: &mut impl Rng) -> Vec4 {
        return Vec4(rng.random::<f32>() - 0.5, rng.random::<f32>() - 0.5, 0.0, 0.0);
    }

    pub fn get_ray(rng: &mut impl Rng, camera: &Camera, col: usize, row: usize) -> Ray {
        let offset = sample_square(rng);
        let pixel_sample = camera.pixel00_loc
            + (col as f32 + offset.x()) * camera.pixel_delta_u
            + (row as f32 + offset.y()) * camera.pixel_delta_v;

        let ray_origin: Vec4;

        if camera.defocus_angle <= 0.0 {
            ray_origin = camera.camera_center;
        } else {
            ray_origin = defocus_disk_sample(rng, camera);
        }

        let ray_direction = pixel_sample - ray_origin;

        Ray::new(ray_origin, ray_direction)
    }

    pub fn defocus_disk_sample(rng: &mut impl Rng, camera: &Camera) -> Point4 {
        let p = Vec4::random_vec3_in_disk(rng);
        camera.camera_center + (p.x() * camera.defocus_disk_u) + (p.y() * camera.defocus_disk_v)
    }

    pub fn ray_color(rng: &mut impl Rng, depth: u32, ray: &Ray, world: &[Hitable0]) -> Color {

        if depth == 0 {
            return Color::new(0.0, 0.0, 0.0);
        }

        if let Some(rec) = world.hit(ray, Interval {min: 0.001, max: f32::INFINITY}) {

            unsafe {
                if let Some((attenuation, scattered)) = (*rec.mat).scatter(rng, ray, rec) {
                    return attenuation * ray_color(rng, depth - 1, &scattered, world);
                }
            }

            return Color::new(0.0, 0.0, 0.0);
        }

        let unit_direction = ray.direction.unit_vector();
        let a = 0.5 * (unit_direction.y() + 1.0);
        (1.0 - a) * Color::new(1.0, 1.0, 1.0) + a * Color::new(0.5, 0.7, 1.0)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    ctx.set_limit(ContextLimit::StackSize, 64 * 1024);

    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(&ctx)? };

    let camera = Camera::new();

    let mut material_host = Vec::<Material0>::new();
    material_host.push(Lambertian { albedo: Vec4(0.5, 0.5, 0.5, 0.0) }.into());

    let mut rng = SmallRng::from_seed([42; 32]);

    for _ in 0..529 {
        let mat_type = rng.random::<f32>();

        if mat_type < 0.8 {
            material_host.push(Lambertian { albedo: Vec4(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>(), 0.0) }.into());
        } else if mat_type < 0.95 {
            material_host.push(Metal { albedo: Vec4(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>(), 0.0), fuzz: rng.random_range(0.0..0.5) }.into());
        } else {
            material_host.push(Dielectric { refractive_index: 1.5 }.into());
        }
    }

    material_host.push(Dielectric { refractive_index: 1.5 }.into());
    material_host.push(Lambertian { albedo: Vec4(0.4, 0.2, 0.1, 0.0) }.into());
    material_host.push(Metal { albedo: Vec4(0.7, 0.6, 0.5, 0.0), fuzz: 0.0 }.into());

    let material_dev = DeviceBuffer::from_host(&stream, &material_host)?;

    let material_idx_to_addr = |idx: usize| {
        (material_dev.cu_deviceptr() as usize + (idx * size_of::<Material0>())) as *const Material0
    };

    let mut world_host = Vec::<Hitable0>::new();
    world_host.push(Sphere::new(Vec4(0.0, -1000.0, 0.0, 0.0), 1000.0, material_idx_to_addr(0)).into());

    for a in -11i32..11 {
        for b in -11i32..11 {
            let center = Vec4(a as f32 + 0.9 * rng.random::<f32>(), 0.2, b as f32 + 0.9 * rng.random::<f32>(), 0.0);

            if (center - Vec4(4.0, 0.2, 0.0, 0.0)).length() > 0.9 {
                let material_idx = 1 + (a + 11) * 22 + (b + 11);
                world_host.push(Sphere::new(center, 0.2, material_idx_to_addr(material_idx as usize)).into());
            }
        }
    }

    world_host.push(Sphere::new(Vec4(0.0, 1.0, 0.0, 0.0), 1.0, material_idx_to_addr(530)).into());
    world_host.push(Sphere::new(Vec4(-4.0, 1.0, 0.0, 0.0), 1.0, material_idx_to_addr(531)).into());
    world_host.push(Sphere::new(Vec4(4.0, 1.0, 0.0, 0.0), 1.0, material_idx_to_addr(532)).into());

    let seed = 42u64;
    let world_dev = DeviceBuffer::from_host(&stream, &world_host)?;

    let mut im_dev = DeviceBuffer::<Color>::zeroed(&stream, camera.image_width as usize * camera.image_height as usize)?;
    let im_dev2d = RowWidth::new(&mut im_dev, camera.image_width);

    let params = module.prepare_render(LaunchConfig2D::new((camera.image_width.div_ceil(16), camera.image_height.div_ceil(16)), (16, 16), 0))?;
    module.render(&stream, &params, im_dev2d, &world_dev, camera, seed)?;

    let im_host = im_dev.to_host_vec(&stream)?;

    print!("P3\n{} {}\n255\n", camera.image_width, camera.image_height);

    for px in im_host.iter() {
        write_color(&px);
    }

    Ok(())
}
