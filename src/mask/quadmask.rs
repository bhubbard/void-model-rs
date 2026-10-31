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

    /// Dilate non-background mask regions (`PRIMARY_OBJECT`, `OVERLAP`, `AFFECTED_OBJECT`)
    /// by `radius` pixels using Chebyshev (box) neighborhood priority.
    pub fn dilate(&self, radius: usize) -> Self {
        if radius == 0 || self.width == 0 || self.height == 0 {
            return self.clone();
        }
        let mut out = self.clone();
        let r = radius as isize;
        for y in 0..self.height {
            for x in 0..self.width {
                let mut min_val = self.get_pixel(x, y);
                if min_val == PRIMARY_OBJECT {
                    continue;
                }
                for dy in -r..=r {
                    for dx in -r..=r {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        if nx >= 0
                            && (nx as usize) < self.width
                            && ny >= 0
                            && (ny as usize) < self.height
                        {
                            let neighbor = self.get_pixel(nx as usize, ny as usize);
                            if neighbor < min_val {
                                min_val = neighbor;
                            }
                        }
                    }
                }
                out.set_pixel(x, y, min_val);
            }
        }
        out
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
}
