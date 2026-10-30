use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowVector {
    pub u: f32, // Horizontal displacement (dx)
    pub v: f32, // Vertical displacement (dy)
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
            self.vectors[y * self.width + x].clone()
        } else {
            FlowVector { u: 0.0, v: 0.0 }
        }
    }

    pub fn set(&mut self, x: usize, y: usize, u: f32, v: f32) {
        if x < self.width && y < self.height {
            self.vectors[y * self.width + x] = FlowVector { u, v };
        }
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
