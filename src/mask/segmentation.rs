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
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoAnnotationConfig {
    pub video_path: String,
    pub output_dir: String,
    pub primary_points: Vec<PointPrompt>,
    pub boxes: Vec<BoundingBox>,
}
