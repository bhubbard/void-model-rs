use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlowVector {
    pub u: f32, // Horizontal displacement (dx)
    pub v: f32, // Vertical displacement (dy)
}

impl FlowVector {
    pub fn magnitude(&self) -> f32 {
        (self.u * self.u + self.v * self.v).sqrt()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpticalFlowField {
    pub width: usize,
    pub height: usize,
    pub vectors: Vec<FlowVector>,
}

impl OpticalFlowField {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            vectors: vec![FlowVector { u: 0.0, v: 0.0 }; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> FlowVector {
        if x < self.width && y < self.height {
            self.vectors[y * self.width + x]
        } else {
            FlowVector { u: 0.0, v: 0.0 }
        }
    }

    pub fn set(&mut self, x: usize, y: usize, u: f32, v: f32) {
        if x < self.width && y < self.height {
            self.vectors[y * self.width + x] = FlowVector { u, v };
        }
    }

    /// Mean motion vector magnitude across the optical flow field
    pub fn mean_magnitude(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.vectors.iter().map(FlowVector::magnitude).sum();
        sum / self.vectors.len() as f32
    }

    /// Estimate dense 2D optical flow between two 8-bit luma frames (`prev`, `curr`)
    /// using regularized spatial-temporal Lucas-Kanade / Horn-Schunck single-step gradients.
    pub fn estimate_from_luma(prev: &[u8], curr: &[u8], width: usize, height: usize) -> Self {
        let mut field = Self::new(width, height);
        if prev.len() != width * height || curr.len() != width * height || width < 3 || height < 3 {
            return field;
        }
        let alpha_sq = 25.0f32;
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = y * width + x;
                let ix = (prev[idx + 1] as f32 - prev[idx - 1] as f32) * 0.5;
                let iy = (prev[idx + width] as f32 - prev[idx - width] as f32) * 0.5;
                let it = curr[idx] as f32 - prev[idx] as f32;
                let denom = ix * ix + iy * iy + alpha_sq;
                let u = (-ix * it) / denom;
                let v = (-iy * it) / denom;
                field.set(x, y, u, v);
            }
        }
        field
    }

    /// Downsample optical flow field by an integer stride (e.g. 8 for latent resolution)
    pub fn downsample(&self, stride: usize) -> Self {
        let new_w = self.width / stride;
        let new_h = self.height / stride;
        let mut out = Self::new(new_w, new_h);

        for ny in 0..new_h {
            for nx in 0..new_w {
                let orig_x = nx * stride;
                let orig_y = ny * stride;
                let f = self.get(orig_x, orig_y);
                // Scale flow vectors proportionally to new resolution
                out.set(nx, ny, f.u / stride as f32, f.v / stride as f32);
            }
        }

        out
    }
}
