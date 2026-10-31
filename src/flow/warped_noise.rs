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

    /// Mean of all latent elements
    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    /// Sample variance of all latent elements
    pub fn variance(&self) -> f32 {
        if self.data.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let sum_sq: f32 = self.data.iter().map(|&x| (x - m) * (x - m)).sum();
        sum_sq / self.data.len() as f32
    }

    /// Serialize tensor data as little-endian `f32` bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for &val in &self.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Deserialize tensor data from little-endian `f32` bytes
    pub fn from_bytes(
        bytes: &[u8],
        channels: usize,
        height: usize,
        width: usize,
    ) -> anyhow::Result<Self> {
        let expected_len = channels * height * width * 4;
        if bytes.len() != expected_len {
            anyhow::bail!(
                "LatentNoiseTensor byte buffer mismatch: got {}, expected {}",
                bytes.len(),
                expected_len
            );
        }
        let mut data = Vec::with_capacity(channels * height * width);
        for i in 0..(channels * height * width) {
            let off = i * 4;
            data.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
        }
        Ok(Self {
            channels,
            height,
            width,
            data,
        })
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

    /// Warps a latent noise tensor backward along an optical flow field:
    /// z_{t-1}(x, y) = sample(z_t, x + u(x, y), y + v(x, y))
    pub fn warp_noise_backward(
        next_latent: &LatentNoiseTensor,
        flow: &OpticalFlowField,
    ) -> LatentNoiseTensor {
        Self::warp_noise(next_latent, &flow.invert())
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

    /// Generates a full multi-frame sequence of temporally consistent Pass 2 latent noise tensors.
    /// At each step `t`, with probability `warp_probability`, advects `z_{t-1}` along `flows[t-1]`
    /// and blends with fresh Gaussian noise using variance-preserving `degradation` ($\lambda$);
    /// otherwise injects fresh Gaussian noise.
    pub fn warp_sequence(
        initial_latent: &LatentNoiseTensor,
        flows: &[OpticalFlowField],
        degradation: f32,
        warp_probability: f32,
        seed: u64,
    ) -> Vec<LatentNoiseTensor> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let prob = warp_probability.clamp(0.0, 1.0);

        let mut sequence = Vec::with_capacity(flows.len() + 1);
        sequence.push(initial_latent.clone());

        for (step, flow) in flows.iter().enumerate() {
            let fresh = LatentNoiseTensor::new_gaussian(
                initial_latent.channels,
                initial_latent.height,
                initial_latent.width,
                Some(seed.wrapping_add(step as u64 + 1)),
            );

            let roll: f32 = rng.gen();
            let next_tensor = if roll <= prob {
                let prev = sequence.last().unwrap();
                let warped = Self::warp_noise(prev, flow);
                Self::blend_with_degradation(&warped, &fresh, degradation)
            } else {
                fresh
            };
            sequence.push(next_tensor);
        }

        sequence
    }

    /// Emits the WGPU WGSL compute shader for subpixel bilinear latent noise advection
    /// and variance-preserving degradation blending (`sqrt(1 - lambda^2) * z_warp + lambda * z_fresh`).
    pub fn wgsl_warp_shader() -> &'static str {
        r#"@group(0) @binding(0) var<storage, read> prev_latent: array<f32>;
@group(0) @binding(1) var<storage, read> fresh_latent: array<f32>;
@group(0) @binding(2) var<storage, read> flow_uv: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> out_latent: array<f32>;

struct Params {
    channels: u32,
    height: u32,
    width: u32,
    lambda: f32,
};
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) { return; }
    let uv = flow_uv[y * params.width + x];
    let sx = clamp(f32(x) - uv.x, 0.0, f32(params.width - 1u));
    let sy = clamp(f32(y) - uv.y, 0.0, f32(params.height - 1u));
    let x0 = u32(floor(sx));
    let y0 = u32(floor(sy));
    let x1 = min(x0 + 1u, params.width - 1u);
    let y1 = min(y0 + 1u, params.height - 1u);
    let dx = sx - f32(x0);
    let dy = sy - f32(y0);
    let w_warp = sqrt(max(1.0 - params.lambda * params.lambda, 0.0));
    let plane = params.height * params.width;
    for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
        let base = c * plane;
        let s00 = prev_latent[base + y0 * params.width + x0];
        let s10 = prev_latent[base + y0 * params.width + x1];
        let s01 = prev_latent[base + y1 * params.width + x0];
        let s11 = prev_latent[base + y1 * params.width + x1];
        let warped = (1.0 - dx) * (1.0 - dy) * s00 + dx * (1.0 - dy) * s10 + (1.0 - dx) * dy * s01 + dx * dy * s11;
        let idx = base + y * params.width + x;
        out_latent[idx] = w_warp * warped + params.lambda * fresh_latent[idx];
    }
}"#
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

    #[test]
    fn test_warp_sequence_and_bytes_roundtrip() {
        let init = LatentNoiseTensor::new_gaussian(4, 16, 16, Some(77));
        assert!((init.mean()).abs() < 0.25);
        assert!((init.variance() - 1.0).abs() < 0.25);

        let bytes = init.to_bytes();
        let restored = LatentNoiseTensor::from_bytes(&bytes, 4, 16, 16).unwrap();
        assert_eq!(restored.data, init.data);

        let flows = vec![OpticalFlowField::new(16, 16), OpticalFlowField::new(16, 16)];
        let seq = WarpedNoiseEngine::warp_sequence(&init, &flows, 0.3, 1.0, 99);
        assert_eq!(seq.len(), 3);
        // Variance should stay near 1.0 across variance-preserving degradation steps
        assert!((seq[2].variance() - 1.0).abs() < 0.25);
    }
}
