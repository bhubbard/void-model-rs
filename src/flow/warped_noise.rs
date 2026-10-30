use crate::flow::optical_flow::OpticalFlowField;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct LatentNoiseTensor {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>, // Shape: [channels * height * width]
}

impl LatentNoiseTensor {
    pub fn new_gaussian(channels: usize, height: usize, width: usize, seed: Option<u64>) -> Self {
        let size = channels * height * width;
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        use rand::SeedableRng;

        let normal = Normal::new(0.0f32, 1.0f32).unwrap();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng)).collect();

        Self {
            channels,
            height,
            width,
            data,
        }
    }

    pub fn get_sample(&self, c: usize, y: usize, x: usize) -> f32 {
        if c < self.channels && y < self.height && x < self.width {
            self.data[c * (self.height * self.width) + y * self.width + x]
        } else {
            0.0
        }
    }

    pub fn set_sample(&mut self, c: usize, y: usize, x: usize, val: f32) {
        if c < self.channels && y < self.height && x < self.width {
            self.data[c * (self.height * self.width) + y * self.width + x] = val;
        }
    }

    /// Sample with bilinear interpolation at sub-pixel coordinates
    pub fn sample_bilinear(&self, c: usize, x: f32, y: f32) -> f32 {
        if x < 0.0 || x >= (self.width - 1) as f32 || y < 0.0 || y >= (self.height - 1) as f32 {
            let cx = x.clamp(0.0, (self.width - 1) as f32) as usize;
            let cy = y.clamp(0.0, (self.height - 1) as f32) as usize;
            return self.get_sample(c, cy, cx);
        }

        let x0 = x.floor() as usize;
        let x1 = x0 + 1;
        let y0 = y.floor() as usize;
        let y1 = y0 + 1;

        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let s00 = self.get_sample(c, y0, x0);
        let s10 = self.get_sample(c, y0, x1);
        let s01 = self.get_sample(c, y1, x0);
        let s11 = self.get_sample(c, y1, x1);

        (1.0 - dx) * (1.0 - dy) * s00
            + dx * (1.0 - dy) * s10
            + (1.0 - dx) * dy * s01
            + dx * dy * s11
    }
}

pub struct WarpedNoiseEngine;

impl WarpedNoiseEngine {
    /// Warps a latent noise tensor forward along an optical flow field:
    /// z_t(x, y) = sample(z_{t-1}, x - u(x, y), y - v(x, y))
    pub fn warp_noise(
        prev_latent: &LatentNoiseTensor,
        flow: &OpticalFlowField,
    ) -> LatentNoiseTensor {
        let mut warped = LatentNoiseTensor {
            channels: prev_latent.channels,
            height: prev_latent.height,
            width: prev_latent.width,
            data: vec![0.0; prev_latent.data.len()],
        };

        for y in 0..prev_latent.height {
            for x in 0..prev_latent.width {
                let f = flow.get(x, y);
                let src_x = x as f32 - f.u;
                let src_y = y as f32 - f.v;

                for c in 0..prev_latent.channels {
                    let val = prev_latent.sample_bilinear(c, src_x, src_y);
                    warped.set_sample(c, y, x, val);
                }
            }
        }

        warped
    }

    /// Blends warped noise with fresh Gaussian noise using degradation factor lambda (default 0.3):
    /// z_final = sqrt(1 - lambda^2) * z_warped + lambda * z_fresh
    pub fn blend_with_degradation(
        warped: &LatentNoiseTensor,
        fresh: &LatentNoiseTensor,
        degradation: f32,
    ) -> LatentNoiseTensor {
        let lambda = degradation.clamp(0.0, 1.0);
        let weight_warped = (1.0 - lambda * lambda).max(0.0).sqrt();

        let mut out = warped.clone();
        for i in 0..out.data.len() {
            out.data[i] = weight_warped * warped.data[i] + lambda * fresh.data[i];
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_noise_generation() {
        let tensor = LatentNoiseTensor::new_gaussian(16, 60, 90, Some(42));
        assert_eq!(tensor.channels, 16);
        assert_eq!(tensor.height, 60);
        assert_eq!(tensor.width, 90);
        assert_eq!(tensor.data.len(), 16 * 60 * 90);
    }

    #[test]
    fn test_warped_noise_identity_flow() {
        let tensor = LatentNoiseTensor::new_gaussian(2, 10, 10, Some(123));
        let flow = OpticalFlowField::new(10, 10); // Zero flow

        let warped = WarpedNoiseEngine::warp_noise(&tensor, &flow);
        for i in 0..tensor.data.len() {
            assert!((warped.data[i] - tensor.data[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_noise_blending_degradation() {
        let t1 = LatentNoiseTensor::new_gaussian(1, 4, 4, Some(1));
        let t2 = LatentNoiseTensor::new_gaussian(1, 4, 4, Some(2));

        let blended = WarpedNoiseEngine::blend_with_degradation(&t1, &t2, 0.3);
        assert_eq!(blended.data.len(), 16);
    }
}
