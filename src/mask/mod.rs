pub mod quadmask;
pub mod segmentation;

pub use quadmask::{
    QuadMaskFrame, QuadMaskStats, QuadMaskVideo, AFFECTED_OBJECT, BACKGROUND, OVERLAP,
    PRIMARY_OBJECT,
};
pub use segmentation::{BoundingBox, PointPrompt, VideoAnnotationConfig};
