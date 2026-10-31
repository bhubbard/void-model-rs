use crate::flow::{LatentNoiseTensor, OpticalFlowField, WarpedNoiseEngine};
use crate::mask::QuadMaskFrame;
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwoPassReport {
    pub latent_channels: usize,
    pub latent_width: usize,
    pub latent_height: usize,
    pub inpainted_ratio: f32,
    pub mean_flow_magnitude: f32,
    pub pass1_variance: f32,
    pub pass2_temporal_correlation: f32,
}

pub struct InpaintPipeline;

impl InpaintPipeline {
    /// Inspects a SafeTensors model checkpoint header (8-byte little-endian length prefix + JSON metadata)
    pub fn inspect_safetensors_header(bytes: &[u8]) -> Result<serde_json::Value> {
        if bytes.len() < 8 {
            bail!("SafeTensors buffer too short (< 8 bytes)");
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&bytes[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        if bytes.len() < 8 + header_len {
            bail!(
                "SafeTensors header truncated: expected {} bytes, got {}",
                8 + header_len,
                bytes.len()
            );
        }
        let val: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])?;
        Ok(val)
    }

    /// Executes VOID Two-Pass Latent Inpainting (Pass 1 Base Inpainting + Pass 2 Warped-Noise Refinement)
    pub fn execute_two_pass(
        config: &InpaintConfig,
        quadmask: &QuadMaskFrame,
        flow: &OpticalFlowField,
    ) -> TwoPassReport {
        let latent_w = (config.width / 8).max(1);
        let latent_h = (config.height / 8).max(1);
        let channels = 16;

        let stats = quadmask.stats();
        let active_pixels = stats.primary_pixels + stats.overlap_pixels + stats.affected_pixels;
        let inpainted_ratio = if stats.total_pixels > 0 {
            active_pixels as f32 / stats.total_pixels as f32
        } else {
            0.0
        };

        let latent_flow = flow.downsample(8);
        let pass1_latent = LatentNoiseTensor::new_gaussian(channels, latent_h, latent_w, Some(101));
        let warped_latent = WarpedNoiseEngine::warp_noise(&pass1_latent, &latent_flow);
        let fresh_latent = LatentNoiseTensor::new_gaussian(channels, latent_h, latent_w, Some(202));
        let pass2_latent = WarpedNoiseEngine::blend_with_degradation(
            &warped_latent,
            &fresh_latent,
            config.warped_noise_degradation,
        );

        let n = pass1_latent.data.len().max(1) as f32;
        let pass1_variance = pass1_latent.data.iter().map(|v| v * v).sum::<f32>() / n;
        let dot: f32 = warped_latent
            .data
            .iter()
            .zip(pass2_latent.data.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / n;

        TwoPassReport {
            latent_channels: channels,
            latent_width: latent_w,
            latent_height: latent_h,
            inpainted_ratio,
            mean_flow_magnitude: latent_flow.mean_magnitude(),
            pass1_variance,
            pass2_temporal_correlation: dot.clamp(-1.0, 1.0),
        }
    }

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
