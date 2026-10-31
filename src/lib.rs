//! # VOID (Rust Edition) 🦀🎬
//!
//! A high-performance, native pure Rust port of Netflix's [VOID: Video Object and Interaction Deletion](https://github.com/netflix/void-model) —
//! interaction-aware video inpainting, QuadMask reasoning, optical flow warped noise, and physical scene dynamics.

pub mod flow;
pub mod mask;
pub mod physics;
pub mod pipeline;
pub mod vlm;

pub use flow::{FlowVector, LatentNoiseTensor, OpticalFlowField, WarpedNoiseEngine};
pub use mask::{
    BoundingBox, PointPrompt, QuadMaskFrame, QuadMaskStats, QuadMaskVideo, VideoAnnotationConfig,
    AFFECTED_OBJECT, BACKGROUND, OVERLAP, PRIMARY_OBJECT,
};
pub use physics::{PhysicsConfig, PhysicsDynamics, PhysicsScenario, TrajectoryPoint};
pub use pipeline::{InpaintConfig, InpaintPipeline, TwoPassReport};
pub use vlm::{
    ContactRelation, InteractionType, SceneEntity, SceneInteractionGraph, VlmObjectAnalysis,
    VlmReasoner, VlmSceneAnalysis,
};
