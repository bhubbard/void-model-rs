use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointPrompt {
    pub x: f64,
    pub y: f64,
    pub frame_idx: usize,
    pub is_positive: bool,
}

impl PointPrompt {
    pub fn positive(x: f64, y: f64, frame_idx: usize) -> Self {
        Self {
            x,
            y,
            frame_idx,
            is_positive: true,
        }
    }

    pub fn negative(x: f64, y: f64, frame_idx: usize) -> Self {
        Self {
            x,
            y,
            frame_idx,
            is_positive: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl BoundingBox {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self {
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    pub fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x_min && px <= self.x_max && py >= self.y_min && py <= self.y_max
    }

    pub fn width(&self) -> f64 {
        (self.x_max - self.x_min).max(0.0)
    }

    pub fn height(&self) -> f64 {
        (self.y_max - self.y_min).max(0.0)
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Compute Intersection-over-Union (IoU) between two bounding boxes
    pub fn iou(&self, other: &BoundingBox) -> f64 {
        let inter_xmin = self.x_min.max(other.x_min);
        let inter_ymin = self.y_min.max(other.y_min);
        let inter_xmax = self.x_max.min(other.x_max);
        let inter_ymax = self.y_max.min(other.y_max);

        let inter_w = (inter_xmax - inter_xmin).max(0.0);
        let inter_h = (inter_ymax - inter_ymin).max(0.0);
        let inter_area = inter_w * inter_h;

        let union_area = self.area() + other.area() - inter_area;
        if union_area <= 1e-9 {
            0.0
        } else {
            inter_area / union_area
        }
    }

    /// Expand bounding box by `margin` pixels on all sides, clamped to frame dimensions
    pub fn expand(&self, margin: f64, max_width: f64, max_height: f64) -> Self {
        Self {
            x_min: (self.x_min - margin).clamp(0.0, max_width),
            y_min: (self.y_min - margin).clamp(0.0, max_height),
            x_max: (self.x_max + margin).clamp(0.0, max_width),
            y_max: (self.y_max + margin).clamp(0.0, max_height),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoAnnotationConfig {
    pub video_path: String,
    pub output_dir: String,
    pub primary_points: Vec<PointPrompt>,
    pub boxes: Vec<BoundingBox>,
}

impl VideoAnnotationConfig {
    /// Rasterizes annotated bounding boxes into a binary mask (`0` = inside box, `255` = background)
    pub fn rasterize_box_mask(&self, width: usize, height: usize) -> Vec<u8> {
        let mut mask = vec![255u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let px = x as f64;
                let py = y as f64;
                if self.boxes.iter().any(|b| b.contains(px, py)) {
                    mask[y * width + x] = 0;
                }
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_iou_and_expand() {
        let b1 = BoundingBox::new(10.0, 10.0, 50.0, 50.0);
        let b2 = BoundingBox::new(30.0, 30.0, 70.0, 70.0);
        // Intersection is [30..50, 30..50] = 20x20 = 400
        // Union is 1600 + 1600 - 400 = 2800 -> IoU = 400 / 2800 = 1/7
        assert!((b1.iou(&b2) - (1.0 / 7.0)).abs() < 1e-6);

        let expanded = b1.expand(15.0, 100.0, 100.0);
        assert_eq!(expanded.x_min, 0.0);
        assert_eq!(expanded.y_min, 0.0);
        assert_eq!(expanded.x_max, 65.0);
        assert_eq!(expanded.y_max, 65.0);
    }

    #[test]
    fn test_rasterize_box_mask() {
        let cfg = VideoAnnotationConfig {
            video_path: "demo.mp4".to_string(),
            output_dir: "out".to_string(),
            primary_points: vec![PointPrompt::positive(5.0, 5.0, 0)],
            boxes: vec![BoundingBox::new(2.0, 2.0, 4.0, 4.0)],
        };
        let mask = cfg.rasterize_box_mask(8, 8);
        assert_eq!(mask[3 * 8 + 3], 0);
        assert_eq!(mask[0], 255);
    }
}
