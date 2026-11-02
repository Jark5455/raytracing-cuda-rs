use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};
use cuda_host::RowWidth;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;
use rayon::prelude::*;

use crate::camera::Camera;
use crate::color::Color;
use crate::material::{Dielectric, Lambertian, Metal};
use crate::sphere::Sphere;
use crate::vec4::Vec4;
use crate::{kernels, Hitable0, Material0};

unsafe impl Sync for Hitable0 {}

const SEED: u64 = 42;

fn materials(rng: &mut SmallRng) -> Vec<Material0> {
    let mut m = Vec::<Material0>::new();
    m.push(Lambertian { albedo: Vec4(0.5, 0.5, 0.5, 0.0) }.into());

    for _ in 0..529 {
        let mat_type = rng.random::<f32>();
        if mat_type < 0.8 {
            m.push(Lambertian { albedo: Vec4(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>(), 0.0) }.into());
        } else if mat_type < 0.95 {
            m.push(Metal { albedo: Vec4(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>(), 0.0), fuzz: rng.random_range(0.0..0.5) }.into());
        } else {
            m.push(Dielectric { refractive_index: 1.5 }.into());
        }
    }

    m.push(Dielectric { refractive_index: 1.5 }.into());
    m.push(Lambertian { albedo: Vec4(0.4, 0.2, 0.1, 0.0) }.into());
    m.push(Metal { albedo: Vec4(0.7, 0.6, 0.5, 0.0), fuzz: 0.0 }.into());
    m
}

fn world(rng: &mut SmallRng, base: usize) -> Vec<Hitable0> {
    let mat = |idx: usize| (base + idx * size_of::<Material0>()) as *const Material0;

    let mut w = Vec::<Hitable0>::new();
    w.push(Sphere::new(Vec4(0.0, -1000.0, 0.0, 0.0), 1000.0, mat(0)).into());

    for a in -11i32..11 {
        for b in -11i32..11 {
            let center = Vec4(a as f32 + 0.9 * rng.random::<f32>(), 0.2, b as f32 + 0.9 * rng.random::<f32>(), 0.0);
            if (center - Vec4(4.0, 0.2, 0.0, 0.0)).length() > 0.9 {
                w.push(Sphere::new(center, 0.2, mat((1 + (a + 11) * 22 + (b + 11)) as usize)).into());
            }
        }
    }

    w.push(Sphere::new(Vec4(0.0, 1.0, 0.0, 0.0), 1.0, mat(530)).into());
    w.push(Sphere::new(Vec4(-4.0, 1.0, 0.0, 0.0), 1.0, mat(531)).into());
    w.push(Sphere::new(Vec4(4.0, 1.0, 0.0, 0.0), 1.0, mat(532)).into());
    w
}

fn render_cpu(camera: &Camera, world: &[Hitable0]) -> Vec<Color> {
    let w = camera.image_width as usize;
    let h = camera.image_height as usize;

    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (col, row) = (i % w, i / w);
            let mut rng = Pcg32::new(SEED, (col * h + row) as u64);
            let mut c = Color::new(0.0, 0.0, 0.0);
            for _ in 0..camera.samples_per_pixel {
                let ray = kernels::get_ray(&mut rng, camera, col, row);
                c = c + kernels::ray_color(&mut rng, camera.max_depth, &ray, world);
            }
            c * camera.pixel_samples_scale
        })
        .collect()
}

#[test]
#[ignore]
fn bench() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(&ctx).unwrap() };

    let mut rng = SmallRng::from_seed([42; 32]);
    let material_host = materials(&mut rng);
    let material_dev = DeviceBuffer::from_host(&stream, &material_host).unwrap();

    let world_host = world(&mut rng.clone(), material_host.as_ptr() as usize);
    let world_dev = DeviceBuffer::from_host(&stream, &world(&mut rng, material_dev.cu_deviceptr() as usize)).unwrap();

    let base = Camera::new();
    let (w, h) = (base.image_width, base.image_height);
    let mut im_dev = DeviceBuffer::<Color>::zeroed(&stream, (w * h) as usize).unwrap();
    let params = module.prepare_render(LaunchConfig2D::new((w.div_ceil(16), h.div_ceil(16)), (16, 16), 0)).unwrap();

    let mut c = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(20));

    let mut group = c.benchmark_group(format!("render_{w}x{h}"));

    for spp in [1, 4, 8, 16] {
        let mut camera = Camera::new();
        camera.samples_per_pixel = spp;
        camera.pixel_samples_scale = 1.0 / spp as f32;

        group.throughput(Throughput::Elements(u64::from(w * h * spp)));

        group.bench_with_input(BenchmarkId::new("gpu", spp), &camera, |b, camera| {
            b.iter(|| {
                let out = RowWidth::new(&mut im_dev, w);
                module.render(&stream, &params, out, &world_dev, *camera, SEED).unwrap();
                stream.synchronize().unwrap();
            })
        });

        group.bench_with_input(BenchmarkId::new("cpu_rayon", spp), &camera, |b, camera| {
            b.iter(|| render_cpu(camera, &world_host))
        });
    }

    group.finish();
    c.final_summary();
}
