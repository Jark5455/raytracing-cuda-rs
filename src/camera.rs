use crate::vec4::Vec4;
use crate::vec4::Point4;

use cuda_core::DeviceCopy;

#[derive(DeviceCopy, Copy, Clone, Debug)]
pub struct Camera {
    pub image_width: u32,
    pub image_height: u32,
    pub viewport_height: f32,
    pub viewport_width: f32,
    pub camera_center: Vec4,
    pub viewport_u: Vec4,
    pub viewport_v: Vec4,
    pub pixel_delta_u: Vec4,
    pub pixel_delta_v: Vec4,
    pub viewport_upper_left: Vec4,
    pub pixel00_loc: Vec4,
    pub samples_per_pixel: u32,
    pub pixel_samples_scale: f32,
    pub max_depth: u32,
    pub vfov: f32,
    pub look_from: Point4,
    pub look_at: Point4,
    pub vup: Vec4,
    pub u: Vec4,
    pub v: Vec4,
    pub w: Vec4,
    pub defocus_angle: f32,
    pub focus_dist: f32,
    pub defocus_disk_u: Vec4,
    pub defocus_disk_v: Vec4,
}

impl Camera {
    pub fn new() -> Self {
        let image_width = 1920u32;
        let image_height = 1080u32;

        let look_from = Vec4(13.0, 2.0, 3.0, 0.0);
        let look_at = Vec4(0.0, 0.0, 0.0, 0.0);
        let vup = Vec4(0.0, 1.0, 0.0, 0.0);

        let camera_center = look_from;

        let w = (look_from - look_at).unit_vector();
        let u = Vec4::cross(&vup, &w).unit_vector();
        let v = Vec4::cross(&w, &u);

        let vfov = 20.0f32;
        let theta = vfov.to_radians();
        let h = (theta / 2.0).tan();

        let defocus_angle = 0.6;
        let focus_dist = 10.0;

        let viewport_height = 2.0 * h * focus_dist;
        let viewport_width = viewport_height * image_width as f32 / image_height as f32;

        let viewport_u = viewport_width * u;
        let viewport_v = viewport_height * -v;

        let pixel_delta_u = viewport_u / image_width as f32;
        let pixel_delta_v = viewport_v / image_height as f32;

        let viewport_upper_left = camera_center - (focus_dist * w) - viewport_u/2.0 - viewport_v/2.0;
        let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        let samples_per_pixel = 500;
        let pixel_samples_scale = 1.0 / samples_per_pixel as f32;
        let max_depth = 50;

        let defocus_radius = focus_dist * (defocus_angle / 2.0f32).to_radians().tan();
        let defocus_disk_u = u * defocus_radius;
        let defocus_disk_v = v * defocus_radius;

        Self {
            image_width,
            image_height,
            viewport_height,
            viewport_width,
            camera_center,
            viewport_u,
            viewport_v,
            pixel_delta_u,
            pixel_delta_v,
            viewport_upper_left,
            pixel00_loc,
            samples_per_pixel,
            pixel_samples_scale,
            max_depth,
            vfov,
            look_from,
            look_at,
            vup,
            u,
            v,
            w,
            defocus_angle,
            focus_dist,
            defocus_disk_u,
            defocus_disk_v
        }
    }
}