use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InpaintConfig {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub use_warped_noise: bool,
    pub warped_noise_degradation: f32,
    pub warped_noise_probability: f32,
    pub steps: usize,
    pub guidance_scale: f32,
}

impl Default for InpaintConfig {
    fn default() -> Self {
        Self {
            width: 720,
            height: 480,
            frames: 49,
            use_warped_noise: true,
            warped_noise_degradation: 0.3,
            warped_noise_probability: 0.8,
            steps: 50,
            guidance_scale: 6.0,
        }
    }
}

pub struct InpaintPipeline;

impl InpaintPipeline {
    /// Builds the FFmpeg side-by-side review grid command:
    /// Displays [Input Video | QuadMask | Result]
    pub fn build_comparison_grid_cmd(
        input_video: &Path,
        quadmask_video: &Path,
        result_video: &Path,
        output_grid: &Path,
    ) -> Vec<String> {
        vec![
            "ffmpeg".to_string(),
            "-y".to_string(),
            "-i".to_string(),
            input_video.to_string_lossy().to_string(),
            "-i".to_string(),
            quadmask_video.to_string_lossy().to_string(),
            "-i".to_string(),
            result_video.to_string_lossy().to_string(),
            "-filter_complex".to_string(),
            "[0:v][1:v][2:v]hstack=inputs=3[grid]".to_string(),
            "-map".to_string(),
            "[grid]".to_string(),
            "-c:v".to_string(),
            "libx264".to_string(),
            "-crf".to_string(),
            "18".to_string(),
            "-preset".to_string(),
            "fast".to_string(),
            output_grid.to_string_lossy().to_string(),
        ]
    }

    /// Execute FFmpeg command
    pub fn run_ffmpeg(cmd: &[String]) -> Result<()> {
        if cmd.is_empty() {
            bail!("Empty FFmpeg command");
        }
        let status = Command::new(&cmd[0]).args(&cmd[1..]).status()?;
        if !status.success() {
            bail!("FFmpeg exited with error status: {}", status);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_grid_cmd() {
        let cmd = InpaintPipeline::build_comparison_grid_cmd(
            Path::new("in.mp4"),
            Path::new("mask.mp4"),
            Path::new("res.mp4"),
            Path::new("grid.mp4"),
        );
        assert_eq!(cmd[0], "ffmpeg");
        assert!(cmd.iter().any(|c| c.contains("hstack=inputs=3")));
    }
}
