use anyhow::{bail, Result};
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

    /// Subpixel bilinear interpolation of the flow field at `(x, y)`
    pub fn sample_bilinear(&self, x: f32, y: f32) -> FlowVector {
        if self.width == 0 || self.height == 0 {
            return FlowVector { u: 0.0, v: 0.0 };
        }
        let max_x = (self.width - 1) as f32;
        let max_y = (self.height - 1) as f32;
        let cx = x.clamp(0.0, max_x);
        let cy = y.clamp(0.0, max_y);

        let x0 = cx.floor() as usize;
        let y0 = cy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let dx = cx - x0 as f32;
        let dy = cy - y0 as f32;

        let f00 = self.get(x0, y0);
        let f10 = self.get(x1, y0);
        let f01 = self.get(x0, y1);
        let f11 = self.get(x1, y1);

        let u = (1.0 - dx) * (1.0 - dy) * f00.u
            + dx * (1.0 - dy) * f10.u
            + (1.0 - dx) * dy * f01.u
            + dx * dy * f11.u;
        let v = (1.0 - dx) * (1.0 - dy) * f00.v
            + dx * (1.0 - dy) * f10.v
            + (1.0 - dx) * dy * f01.v
            + dx * dy * f11.v;

        FlowVector { u, v }
    }

    /// Downsample optical flow field by an integer stride (e.g. 8 for latent resolution)
    pub fn downsample(&self, stride: usize) -> Self {
        let stride = stride.max(1);
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

    /// Invert flow field direction (`(-u, -v)`) for backward <-> forward warping conversion
    pub fn invert(&self) -> Self {
        let mut out = self.clone();
        for vec in &mut out.vectors {
            vec.u = -vec.u;
            vec.v = -vec.v;
        }
        out
    }

    /// Compute per-pixel motion magnitude $\sqrt{u^2 + v^2}$
    pub fn magnitude_map(&self) -> Vec<f32> {
        self.vectors.iter().map(FlowVector::magnitude).collect()
    }

    /// Maximum motion displacement magnitude across the frame
    pub fn max_magnitude(&self) -> f32 {
        self.vectors
            .iter()
            .map(FlowVector::magnitude)
            .fold(0.0f32, f32::max)
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

    /// Serialize horizontal (`u`) and vertical (`v`) components into little-endian `f32` byte buffers
    pub fn to_planar_bytes(&self) -> (Vec<u8>, Vec<u8>) {
        let mut u_bytes = Vec::with_capacity(self.vectors.len() * 4);
        let mut v_bytes = Vec::with_capacity(self.vectors.len() * 4);
        for vec in &self.vectors {
            u_bytes.extend_from_slice(&vec.u.to_le_bytes());
            v_bytes.extend_from_slice(&vec.v.to_le_bytes());
        }
        (u_bytes, v_bytes)
    }

    /// Deserialize horizontal (`u`) and vertical (`v`) little-endian `f32` byte buffers
    pub fn from_planar_bytes(
        u_bytes: &[u8],
        v_bytes: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Self> {
        let expected_bytes = width * height * 4;
        if u_bytes.len() != expected_bytes || v_bytes.len() != expected_bytes {
            bail!(
                "OpticalFlow planar buffer size mismatch: u={}, v={}, expected={}",
                u_bytes.len(),
                v_bytes.len(),
                expected_bytes
            );
        }
        let mut vectors = Vec::with_capacity(width * height);
        for i in 0..(width * height) {
            let offset = i * 4;
            let u = f32::from_le_bytes(u_bytes[offset..offset + 4].try_into().unwrap());
            let v = f32::from_le_bytes(v_bytes[offset..offset + 4].try_into().unwrap());
            vectors.push(FlowVector { u, v });
        }
        Ok(Self {
            width,
            height,
            vectors,
        })
    }

    /// Estimate 2D dense optical flow between `prev_frame` and `next_frame` using
    /// Sum of Absolute Differences (SAD) block matching.
    /// `u` and `v` represent the motion of pixels from `prev_frame` to `next_frame`:
    /// a feature at `(x - u, y - v)` in `prev_frame` moved to `(x, y)` in `next_frame`.
    pub fn estimate_block_matching(
        prev_frame: &[u8],
        next_frame: &[u8],
        width: usize,
        height: usize,
        block_radius: usize,
        search_radius: usize,
    ) -> Result<Self> {
        let size = width * height;
        if prev_frame.len() != size || next_frame.len() != size {
            bail!(
                "Frame size mismatch in block matching: prev={}, next={}, expected={}",
                prev_frame.len(),
                next_frame.len(),
                size
            );
        }

        let mut flow = Self::new(width, height);
        let br = block_radius as isize;
        let sr = search_radius as isize;
        let w = width as isize;
        let h = height as isize;

        for y in 0..h {
            for x in 0..w {
                let mut best_sad = u64::MAX;
                let mut best_u = 0isize;
                let mut best_v = 0isize;

                // Evaluate zero displacement first so ties prefer stationary flow
                for dv in -sr..=sr {
                    for du in -sr..=sr {
                        let prev_cx = x - du;
                        let prev_cy = y - dv;
                        if prev_cx < 0 || prev_cx >= w || prev_cy < 0 || prev_cy >= h {
                            continue;
                        }

                        let mut sad = 0u64;
                        for by in -br..=br {
                            let ny = (y + by).clamp(0, h - 1) as usize;
                            let py = (prev_cy + by).clamp(0, h - 1) as usize;
                            for bx in -br..=br {
                                let nx = (x + bx).clamp(0, w - 1) as usize;
                                let px = (prev_cx + bx).clamp(0, w - 1) as usize;
                                let diff = (next_frame[ny * width + nx] as i32
                                    - prev_frame[py * width + px] as i32)
                                    .unsigned_abs()
                                    as u64;
                                sad += diff;
                            }
                        }

                        let dist_sq = (du * du + dv * dv) as u64;
                        if sad < best_sad
                            || (sad == best_sad
                                && dist_sq < (best_u * best_u + best_v * best_v) as u64)
                        {
                            best_sad = sad;
                            best_u = du;
                            best_v = dv;
                        }
                    }
                }

                flow.set(x as usize, y as usize, best_u as f32, best_v as f32);
            }
        }

        Ok(flow)
    }

    /// Estimate subpixel dense 2D optical flow via Lucas-Kanade spatial-temporal structure tensor
    /// with Tikhonov regularization (`regularization` prevents singular matrices in flat regions).
    pub fn estimate_lucas_kanade(
        prev_frame: &[u8],
        next_frame: &[u8],
        width: usize,
        height: usize,
        window_radius: usize,
    ) -> Result<Self> {
        let size = width * height;
        if prev_frame.len() != size || next_frame.len() != size {
            bail!(
                "Frame size mismatch in Lucas-Kanade: prev={}, next={}, expected={}",
                prev_frame.len(),
                next_frame.len(),
                size
            );
        }

        let mut ix = vec![0.0f32; size];
        let mut iy = vec![0.0f32; size];
        let mut it = vec![0.0f32; size];

        let w = width as isize;
        let h = height as isize;

        for y in 0..h {
            for x in 0..w {
                let xl = (x - 1).max(0) as usize;
                let xr = (x + 1).min(w - 1) as usize;
                let yu = (y - 1).max(0) as usize;
                let yd = (y + 1).min(h - 1) as usize;
                let cx = x as usize;
                let cy = y as usize;

                let gx_prev =
                    (prev_frame[cy * width + xr] as f32 - prev_frame[cy * width + xl] as f32) * 0.5;
                let gx_next =
                    (next_frame[cy * width + xr] as f32 - next_frame[cy * width + xl] as f32) * 0.5;
                ix[cy * width + cx] = 0.5 * (gx_prev + gx_next);

                let gy_prev =
                    (prev_frame[yd * width + cx] as f32 - prev_frame[yu * width + cx] as f32) * 0.5;
                let gy_next =
                    (next_frame[yd * width + cx] as f32 - next_frame[yu * width + cx] as f32) * 0.5;
                iy[cy * width + cx] = 0.5 * (gy_prev + gy_next);

                it[cy * width + cx] =
                    next_frame[cy * width + cx] as f32 - prev_frame[cy * width + cx] as f32;
            }
        }

        let mut flow = Self::new(width, height);
        let r = window_radius as isize;
        let reg = 1e-2f32;

        for y in 0..h {
            for x in 0..w {
                let mut s_xx = reg;
                let mut s_yy = reg;
                let mut s_xy = 0.0f32;
                let mut s_xt = 0.0f32;
                let mut s_yt = 0.0f32;

                for dy in -r..=r {
                    let ny = (y + dy).clamp(0, h - 1) as usize;
                    for dx in -r..=r {
                        let nx = (x + dx).clamp(0, w - 1) as usize;
                        let idx = ny * width + nx;
                        let gx = ix[idx];
                        let gy = iy[idx];
                        let gt = it[idx];

                        s_xx += gx * gx;
                        s_yy += gy * gy;
                        s_xy += gx * gy;
                        s_xt += gx * gt;
                        s_yt += gy * gt;
                    }
                }

                let det = s_xx * s_yy - s_xy * s_xy;
                if det.abs() > 1e-5 {
                    let u = (-s_yy * s_xt + s_xy * s_yt) / det;
                    let v = (s_xy * s_xt - s_xx * s_yt) / det;
                    flow.set(x as usize, y as usize, u, v);
                }
            }
        }

        Ok(flow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_matching_translation_estimation() {
        let w = 16;
        let h = 16;
        let mut f0 = vec![0u8; w * h];
        let mut f1 = vec![0u8; w * h];

        // Place a bright 3x3 patch at (5, 6) in f0, shifted by (u=+2, v=+1) to (7, 7) in f1
        for dy in 0..3 {
            for dx in 0..3 {
                f0[(6 + dy) * w + (5 + dx)] = 250;
                f1[(7 + dy) * w + (7 + dx)] = 250;
            }
        }

        let flow = OpticalFlowField::estimate_block_matching(&f0, &f1, w, h, 1, 3).unwrap();
        let center_vec = flow.get(8, 8);
        assert_eq!(center_vec.u, 2.0);
        assert_eq!(center_vec.v, 1.0);
    }

    #[test]
    fn test_flow_planar_bytes_roundtrip_and_invert() {
        let mut flow = OpticalFlowField::new(4, 4);
        flow.set(1, 2, 3.5, -1.25);

        let (u_bytes, v_bytes) = flow.to_planar_bytes();
        let restored = OpticalFlowField::from_planar_bytes(&u_bytes, &v_bytes, 4, 4).unwrap();
        assert_eq!(restored, flow);

        let inv = flow.invert();
        assert_eq!(inv.get(1, 2), FlowVector { u: -3.5, v: 1.25 });
        assert!((flow.max_magnitude() - (3.5f32.hypot(-1.25))).abs() < 1e-5);
    }
}
