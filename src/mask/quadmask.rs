use serde::{Deserialize, Serialize};

/// Canonical VOID QuadMask Pixel Values
pub const PRIMARY_OBJECT: u8 = 0; // Primary object to be removed
pub const OVERLAP: u8 = 63; // Overlap of primary and affected objects
pub const AFFECTED_OBJECT: u8 = 127; // Affected objects (shadows, reflections, held items)
pub const BACKGROUND: u8 = 255; // Background — keep as-is

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuadMaskStats {
    pub primary_pixels: usize,
    pub overlap_pixels: usize,
    pub affected_pixels: usize,
    pub background_pixels: usize,
    pub total_pixels: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuadMaskFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl QuadMaskFrame {
    pub fn new(width: usize, height: usize, initial_val: u8) -> Self {
        Self {
            width,
            height,
            data: vec![initial_val; width * height],
        }
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            BACKGROUND
        }
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, val: u8) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = val;
        }
    }

    /// Exact combination algorithm from upstream Stage 4 (`stage4_combine_masks.py`)
    ///
    /// Rules:
    /// - black=0, grey=255   -> 0   (primary object only)
    /// - black=255, grey=127 -> 127 (affected object only)
    /// - black=0, grey=127   -> 63  (overlap)
    /// - black=255, grey=255 -> 255 (background)
    pub fn combine(
        black_frame: &[u8],
        grey_frame: &[u8],
        width: usize,
        height: usize,
    ) -> anyhow::Result<Self> {
        let size = width * height;
        if black_frame.len() != size || grey_frame.len() != size {
            anyhow::bail!(
                "Frame buffer size mismatch: black={}, grey={}, expected={}",
                black_frame.len(),
                grey_frame.len(),
                size
            );
        }

        let mut output = vec![BACKGROUND; size];

        for i in 0..size {
            let b = black_frame[i];
            let g = grey_frame[i];

            if b == 0 && g == 255 {
                output[i] = PRIMARY_OBJECT;
            } else if b == 255 && g == 127 {
                output[i] = AFFECTED_OBJECT;
            } else if b == 0 && g == 127 {
                output[i] = OVERLAP;
            } else {
                output[i] = BACKGROUND;
            }
        }

        Ok(Self {
            width,
            height,
            data: output,
        })
    }

    /// Combine primary, affected, and optional explicit overlap masks into a QuadMask.
    /// Any pixel `< 128` in `primary_frame` is treated as primary; any pixel `< 200` in
    /// `affected_frame` is treated as affected; any pixel `< 128` in `overlap_frame` forces `OVERLAP`.
    pub fn combine_with_overlap(
        primary_frame: &[u8],
        affected_frame: &[u8],
        overlap_frame: Option<&[u8]>,
        width: usize,
        height: usize,
    ) -> anyhow::Result<Self> {
        let size = width * height;
        if primary_frame.len() != size || affected_frame.len() != size {
            anyhow::bail!(
                "Frame buffer size mismatch: primary={}, affected={}, expected={}",
                primary_frame.len(),
                affected_frame.len(),
                size
            );
        }
        if let Some(ov) = overlap_frame {
            if ov.len() != size {
                anyhow::bail!(
                    "Overlap buffer size mismatch: overlap={}, expected={}",
                    ov.len(),
                    size
                );
            }
        }

        let mut output = vec![BACKGROUND; size];
        for i in 0..size {
            let is_ov = overlap_frame.map(|ov| ov[i] < 128).unwrap_or(false);
            let is_primary = primary_frame[i] < 128;
            let is_affected = affected_frame[i] < 200;

            output[i] = if is_ov || (is_primary && is_affected) {
                OVERLAP
            } else if is_primary {
                PRIMARY_OBJECT
            } else if is_affected {
                AFFECTED_OBJECT
            } else {
                BACKGROUND
            };
        }

        Ok(Self {
            width,
            height,
            data: output,
        })
    }

    /// Morphological dilation of active foreground regions (`PRIMARY_OBJECT`, `OVERLAP`, `AFFECTED_OBJECT`)
    /// into `BACKGROUND` by `radius` pixels (Chebyshev window).
    /// Since lower pixel values have higher removal priority (`0 < 63 < 127 < 255`),
    /// foreground dilation corresponds to a local minimum filter.
    pub fn dilate(&self, radius: usize) -> Self {
        if radius == 0 || self.width == 0 || self.height == 0 {
            return self.clone();
        }
        let mut out = self.clone();
        let r = radius as isize;
        let w = self.width as isize;
        let h = self.height as isize;

        for y in 0..h {
            for x in 0..w {
                let mut min_val = BACKGROUND;
                for dy in -r..=r {
                    let ny = y + dy;
                    if ny < 0 || ny >= h {
                        continue;
                    }
                    for dx in -r..=r {
                        let nx = x + dx;
                        if nx < 0 || nx >= w {
                            continue;
                        }
                        let v = self.data[(ny as usize) * self.width + (nx as usize)];
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
                out.data[(y as usize) * self.width + (x as usize)] = min_val;
            }
        }
        out
    }

    /// Morphological erosion of active foreground regions by `radius` pixels.
    /// Since background is `255` and foreground regions are `< 255`, foreground erosion
    /// corresponds to a local maximum filter.
    pub fn erode(&self, radius: usize) -> Self {
        if radius == 0 || self.width == 0 || self.height == 0 {
            return self.clone();
        }
        let mut out = self.clone();
        let r = radius as isize;
        let w = self.width as isize;
        let h = self.height as isize;

        for y in 0..h {
            for x in 0..w {
                let mut max_val = PRIMARY_OBJECT;
                for dy in -r..=r {
                    let ny = y + dy;
                    if ny < 0 || ny >= h {
                        max_val = BACKGROUND;
                        continue;
                    }
                    for dx in -r..=r {
                        let nx = x + dx;
                        if nx < 0 || nx >= w {
                            max_val = BACKGROUND;
                            continue;
                        }
                        let v = self.data[(ny as usize) * self.width + (nx as usize)];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                out.data[(y as usize) * self.width + (x as usize)] = max_val;
            }
        }
        out
    }

    /// Converts the QuadMask into a normalized floating-point inpainting alpha mask in `[0.0, 1.0]`
    /// where `1.0` indicates full synthesis (`PRIMARY_OBJECT` and `OVERLAP`),
    /// `0.85` indicates `AFFECTED_OBJECT` harmonization, and `0.0` indicates preserved `BACKGROUND`.
    pub fn inpaint_alpha_mask(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&p| match p {
                PRIMARY_OBJECT | OVERLAP => 1.0,
                AFFECTED_OBJECT => 0.85,
                BACKGROUND => 0.0,
                other => 1.0 - (other as f32 / 255.0),
            })
            .collect()
    }

    /// Applies separable 2D Gaussian feathering (`sigma` in pixels) to the inpainting alpha mask,
    /// returning soft blending weights in `[0.0, 1.0]` for seamless boundary transitions.
    pub fn gaussian_feather(&self, sigma: f32) -> Vec<f32> {
        let base = self.inpaint_alpha_mask();
        if sigma <= 0.0 || self.width == 0 || self.height == 0 {
            return base;
        }

        let radius = ((sigma * 3.0).ceil() as usize).max(1);
        let mut kernel = Vec::with_capacity(2 * radius + 1);
        let mut sum = 0.0f32;
        let two_sigma_sq = 2.0 * sigma * sigma;

        for i in 0..=(2 * radius) {
            let d = (i as isize - radius as isize) as f32;
            let w = (-d * d / two_sigma_sq).exp();
            kernel.push(w);
            sum += w;
        }
        for w in &mut kernel {
            *w /= sum;
        }

        let r = radius as isize;
        let w = self.width as isize;
        let h = self.height as isize;

        // Horizontal pass
        let mut temp = vec![0.0f32; base.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f32;
                for k in -r..=r {
                    let nx = (x + k).clamp(0, w - 1) as usize;
                    let kw = kernel[(k + r) as usize];
                    acc += base[(y as usize) * self.width + nx] * kw;
                }
                temp[(y as usize) * self.width + (x as usize)] = acc;
            }
        }

        // Vertical pass
        let mut out = vec![0.0f32; base.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f32;
                for k in -r..=r {
                    let ny = (y + k).clamp(0, h - 1) as usize;
                    let kw = kernel[(k + r) as usize];
                    acc += temp[ny * self.width + (x as usize)] * kw;
                }
                out[(y as usize) * self.width + (x as usize)] = acc.clamp(0.0, 1.0);
            }
        }

        out
    }

    /// Calculate statistical distribution of QuadMask regions
    pub fn stats(&self) -> QuadMaskStats {
        let mut primary = 0;
        let mut overlap = 0;
        let mut affected = 0;
        let mut bg = 0;

        for &p in &self.data {
            match p {
                PRIMARY_OBJECT => primary += 1,
                OVERLAP => overlap += 1,
                AFFECTED_OBJECT => affected += 1,
                _ => bg += 1,
            }
        }

        QuadMaskStats {
            primary_pixels: primary,
            overlap_pixels: overlap,
            affected_pixels: affected,
            background_pixels: bg,
            total_pixels: self.data.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuadMaskVideo {
    pub frames: Vec<QuadMaskFrame>,
    pub width: usize,
    pub height: usize,
}

impl QuadMaskVideo {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            frames: Vec::new(),
            width,
            height,
        }
    }

    pub fn add_frame(&mut self, frame: QuadMaskFrame) {
        self.frames.push(frame);
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_masks_truth_table() {
        let black = vec![0, 255, 0, 255];
        let grey = vec![255, 127, 127, 255];

        let combined = QuadMaskFrame::combine(&black, &grey, 2, 2).unwrap();
        assert_eq!(combined.data[0], PRIMARY_OBJECT); // 0
        assert_eq!(combined.data[1], AFFECTED_OBJECT); // 127
        assert_eq!(combined.data[2], OVERLAP); // 63
        assert_eq!(combined.data[3], BACKGROUND); // 255
    }

    #[test]
    fn test_quadmask_stats() {
        let mut frame = QuadMaskFrame::new(10, 10, BACKGROUND);
        frame.set_pixel(1, 1, PRIMARY_OBJECT);
        frame.set_pixel(1, 2, OVERLAP);
        frame.set_pixel(1, 3, AFFECTED_OBJECT);

        let stats = frame.stats();
        assert_eq!(stats.primary_pixels, 1);
        assert_eq!(stats.overlap_pixels, 1);
        assert_eq!(stats.affected_pixels, 1);
        assert_eq!(stats.background_pixels, 97);
        assert_eq!(stats.total_pixels, 100);
    }

    #[test]
    fn test_quadmask_dilate_and_erode() {
        let mut frame = QuadMaskFrame::new(5, 5, BACKGROUND);
        frame.set_pixel(2, 2, PRIMARY_OBJECT);

        let dilated = frame.dilate(1);
        // 3x3 neighborhood around (2, 2) should become PRIMARY_OBJECT (9 pixels)
        assert_eq!(dilated.stats().primary_pixels, 9);

        let eroded = dilated.erode(1);
        // Eroding the 3x3 block by 1 restores the single center pixel (2, 2)
        assert_eq!(eroded.stats().primary_pixels, 1);
        assert_eq!(eroded.get_pixel(2, 2), PRIMARY_OBJECT);
    }

    #[test]
    fn test_quadmask_gaussian_feather() {
        let mut frame = QuadMaskFrame::new(7, 7, BACKGROUND);
        for y in 2..=4 {
            for x in 2..=4 {
                frame.set_pixel(x, y, PRIMARY_OBJECT);
            }
        }
        let feathered = frame.gaussian_feather(1.0);
        assert_eq!(feathered.len(), 49);
        // Center pixel should be highest (> 0.6), far corner should be ~0.0, edge should be intermediate
        let center = feathered[3 * 7 + 3];
        let edge = feathered[3 * 7 + 1];
        let corner = feathered[0];
        assert!(center > edge);
        assert!(edge > corner);
        assert!(corner < 0.05);
    }
}
