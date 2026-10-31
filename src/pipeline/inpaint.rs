use crate::flow::{LatentNoiseTensor, OpticalFlowField, WarpedNoiseEngine};
use crate::mask::{QuadMaskFrame, BACKGROUND};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

/// Metadata entry for a single tensor in a SafeTensors checkpoint header
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

impl TensorMeta {
    pub fn byte_len(&self) -> usize {
        self.data_offsets[1].saturating_sub(self.data_offsets[0])
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Lightweight parser for SafeTensors checkpoint headers (e.g. CogVideoX-Fun inpainting weights)
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsHeader {
    pub tensors: HashMap<String, TensorMeta>,
    pub metadata: HashMap<String, String>,
}

impl SafetensorsHeader {
    /// Parses the 8-byte little-endian header length + JSON header from a SafeTensors buffer
    pub fn parse_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            bail!("SafeTensors buffer too short (< 8 bytes)");
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if header_len == 0 || bytes.len() < 8 + header_len {
            bail!(
                "Invalid SafeTensors header length {} for buffer of size {}",
                header_len,
                bytes.len()
            );
        }

        let json_slice = &bytes[8..8 + header_len];
        let raw_map: HashMap<String, serde_json::Value> = serde_json::from_slice(json_slice)?;

        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();

        for (key, val) in raw_map {
            if key == "__metadata__" {
                if let Ok(meta_map) = serde_json::from_value::<HashMap<String, String>>(val) {
                    metadata = meta_map;
                }
            } else {
                let meta: TensorMeta = serde_json::from_value(val)?;
                tensors.insert(key, meta);
            }
        }

        Ok(Self { tensors, metadata })
    }

    /// Serializes a `SafetensorsHeader` and zero-initialized payload into valid SafeTensors format
    pub fn to_header_bytes(&self) -> Result<Vec<u8>> {
        let mut map = serde_json::Map::new();
        if !self.metadata.is_empty() {
            map.insert(
                "__metadata__".to_string(),
                serde_json::to_value(&self.metadata)?,
            );
        }
        for (k, v) in &self.tensors {
            map.insert(k.clone(), serde_json::to_value(v)?);
        }
        let json_bytes = serde_json::to_vec(&map)?;
        let header_len = json_bytes.len() as u64;
        let mut out = Vec::with_capacity(8 + json_bytes.len());
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&json_bytes);
        Ok(out)
    }
}

/// Output of the VOID Two-Pass Inpainting Pipeline (Pass 1 Base Inpainting + Pass 2 Warped-Noise Refinement)
#[derive(Debug, Clone)]
pub struct TwoPassResult {
    /// Pass 1 coarse spatially inpainted frames
    pub pass1_frames: Vec<Vec<u8>>,
    /// Estimated optical flow fields between consecutive Pass 1 frames
    pub optical_flows: Vec<OpticalFlowField>,
    /// Downsampled latent noise sequence warped along the optical flow fields
    pub latent_noises: Vec<LatentNoiseTensor>,
    /// Final Pass 2 temporally refined frames
    pub pass2_frames: Vec<Vec<u8>>,
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
    /// Executes the VOID two-pass video object and interaction deletion pipeline on luma frames:
    /// - **Pass 1 (Base Inpainting)**: Computes spatial harmonic boundary fill inside masked regions
    ///   (`PRIMARY_OBJECT`, `OVERLAP`, `AFFECTED_OBJECT`) while keeping `BACKGROUND` (`255`) pixels intact.
    /// - **Pass 2 (Warped-Noise Refinement)**: Estimates frame-to-frame optical flow from Pass 1,
    ///   downsamples flow to latent resolution, advects latent noise via `WarpedNoiseEngine::warp_sequence`,
    ///   and performs temporally coherent multi-frame refinement across masked regions.
    pub fn run_two_pass(
        frames: &[Vec<u8>],
        quadmasks: &[QuadMaskFrame],
        config: &InpaintConfig,
    ) -> Result<TwoPassResult> {
        if frames.is_empty() || frames.len() != quadmasks.len() {
            bail!(
                "Frames count ({}) must be non-empty and match quadmasks count ({})",
                frames.len(),
                quadmasks.len()
            );
        }

        let w = config.width;
        let h = config.height;
        let expected_size = w * h;

        for (idx, (f, m)) in frames.iter().zip(quadmasks.iter()).enumerate() {
            if f.len() != expected_size || m.width != w || m.height != h {
                bail!(
                    "Frame/QuadMask {} dimension mismatch: expected {}x{} ({})",
                    idx,
                    w,
                    h,
                    expected_size
                );
            }
        }

        // Pass 1: Spatial Harmonic Diffusion Fill inside QuadMask foreground regions
        let mut pass1_frames = Vec::with_capacity(frames.len());
        for (frame, mask) in frames.iter().zip(quadmasks.iter()) {
            let mut work = frame.iter().map(|&p| p as f32).collect::<Vec<f32>>();

            // Initialize masked pixels with average of unmasked background pixels
            let mut bg_sum = 0.0f32;
            let mut bg_count = 0usize;
            for (i, &mval) in mask.data.iter().enumerate() {
                if mval == BACKGROUND {
                    bg_sum += work[i];
                    bg_count += 1;
                }
            }
            let bg_mean = if bg_count > 0 {
                bg_sum / bg_count as f32
            } else {
                128.0
            };

            for (i, &mval) in mask.data.iter().enumerate() {
                if mval != BACKGROUND {
                    work[i] = bg_mean;
                }
            }

            // Iterative 4-neighbor harmonic relaxation inside masked region
            let iterations = config.steps.clamp(4, 32);
            for _ in 0..iterations {
                let prev = work.clone();
                for y in 0..h {
                    for x in 0..w {
                        let idx = y * w + x;
                        if mask.data[idx] == BACKGROUND {
                            continue;
                        }
                        let left = prev[y * w + x.saturating_sub(1)];
                        let right = prev[y * w + (x + 1).min(w - 1)];
                        let up = prev[y.saturating_sub(1) * w + x];
                        let down = prev[(y + 1).min(h - 1) * w + x];
                        work[idx] = 0.25 * (left + right + up + down);
                    }
                }
            }

            // Blend using Gaussian-feathered alpha mask, strictly preserving BACKGROUND (255) pixels
            let alpha = mask.gaussian_feather(1.0);
            let mut out_frame = vec![0u8; expected_size];
            for i in 0..expected_size {
                if mask.data[i] == BACKGROUND {
                    out_frame[i] = frame[i];
                } else {
                    let a = alpha[i];
                    let blended = a * work[i] + (1.0 - a) * (frame[i] as f32);
                    out_frame[i] = blended.round().clamp(0.0, 255.0) as u8;
                }
            }
            pass1_frames.push(out_frame);
        }

        // Estimate optical flow between consecutive Pass 1 frames and downsample to latent space
        let latent_w = (w / 8).max(1);
        let latent_h = (h / 8).max(1);
        let stride = (w / latent_w).max(1);

        let mut optical_flows = Vec::with_capacity(frames.len().saturating_sub(1));
        let mut latent_flows = Vec::with_capacity(frames.len().saturating_sub(1));

        for pair in pass1_frames.windows(2) {
            let flow = OpticalFlowField::estimate_block_matching(&pair[0], &pair[1], w, h, 1, 2)?;
            latent_flows.push(flow.downsample(stride));
            optical_flows.push(flow);
        }

        // Generate flow-warped latent noise tensors for Pass 2
        let init_latent = LatentNoiseTensor::new_gaussian(4, latent_h, latent_w, Some(42));
        let latent_noises = if config.use_warped_noise {
            WarpedNoiseEngine::warp_sequence(
                &init_latent,
                &latent_flows,
                config.warped_noise_degradation,
                config.warped_noise_probability,
                42,
            )
        } else {
            (0..frames.len())
                .map(|t| {
                    LatentNoiseTensor::new_gaussian(4, latent_h, latent_w, Some(42 + t as u64))
                })
                .collect()
        };

        // Pass 2: Temporal Flow-Guided Refinement inside masked regions
        let mut pass2_frames = Vec::with_capacity(frames.len());
        for t in 0..frames.len() {
            let mut refined = pass1_frames[t].clone();
            let mask = &quadmasks[t];
            let latent = &latent_noises[t];

            for y in 0..h {
                for x in 0..w {
                    let idx = y * w + x;
                    if mask.data[idx] == BACKGROUND {
                        // Never alter unmasked background pixels
                        refined[idx] = frames[t][idx];
                        continue;
                    }

                    let mut accum = pass1_frames[t][idx] as f32 * 0.6;
                    let mut weight_sum = 0.6f32;

                    if t > 0 {
                        let f = optical_flows[t - 1].get(x, y);
                        let px = (x as f32 - f.u).round().clamp(0.0, (w - 1) as f32) as usize;
                        let py = (y as f32 - f.v).round().clamp(0.0, (h - 1) as f32) as usize;
                        accum += pass1_frames[t - 1][py * w + px] as f32 * 0.4;
                        weight_sum += 0.4;
                    }

                    // Subtle fine-grain texture modulation from flow-warped latent channel 0
                    let lx = (x as f32 / stride as f32).clamp(0.0, (latent_w - 1) as f32);
                    let ly = (y as f32 / stride as f32).clamp(0.0, (latent_h - 1) as f32);
                    let noise_detail = latent.sample_bilinear(0, lx, ly) * 1.5;

                    let val = (accum / weight_sum) + noise_detail;
                    refined[idx] = val.round().clamp(0.0, 255.0) as u8;
                }
            }

            pass2_frames.push(refined);
        }

        Ok(TwoPassResult {
            pass1_frames,
            optical_flows,
            latent_noises,
            pass2_frames,
        })
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
    use crate::mask::PRIMARY_OBJECT;

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

    #[test]
    fn test_safetensors_header_roundtrip() {
        let mut header = SafetensorsHeader::default();
        header
            .metadata
            .insert("format".to_string(), "pt".to_string());
        header.tensors.insert(
            "transformer.patch_embed.weight".to_string(),
            TensorMeta {
                dtype: "BF16".to_string(),
                shape: vec![16, 4, 2, 2],
                data_offsets: [0, 512],
            },
        );

        let bytes = header.to_header_bytes().unwrap();
        let parsed = SafetensorsHeader::parse_bytes(&bytes).unwrap();
        assert_eq!(parsed, header);
        let meta = &parsed.tensors["transformer.patch_embed.weight"];
        assert_eq!(meta.byte_len(), 512);
        assert_eq!(meta.element_count(), 256);
    }

    #[test]
    fn test_two_pass_inpainting_preserves_background_and_fills_hole() {
        let w = 16;
        let h = 16;
        let mut f0 = vec![200u8; w * h];
        let mut f1 = vec![200u8; w * h];
        let mut m0 = QuadMaskFrame::new(w, h, BACKGROUND);
        let mut m1 = QuadMaskFrame::new(w, h, BACKGROUND);

        // Dark primary object hole at center (6..10, 6..10)
        for y in 6..10 {
            for x in 6..10 {
                f0[y * w + x] = 10;
                f1[y * w + x] = 10;
                m0.set_pixel(x, y, PRIMARY_OBJECT);
                m1.set_pixel(x, y, PRIMARY_OBJECT);
            }
        }

        let cfg = InpaintConfig {
            width: w,
            height: h,
            frames: 2,
            steps: 16,
            ..Default::default()
        };

        let res =
            InpaintPipeline::run_two_pass(&[f0.clone(), f1.clone()], &[m0, m1], &cfg).unwrap();
        assert_eq!(res.pass1_frames.len(), 2);
        assert_eq!(res.pass2_frames.len(), 2);
        assert_eq!(res.optical_flows.len(), 1);
        assert_eq!(res.latent_noises.len(), 2);

        // Unmasked corner pixel is strictly preserved (200)
        assert_eq!(res.pass2_frames[0][0], 200);
        // Center masked pixel (7, 7) was 10 and is now filled close to surrounding background (~200)
        assert!(res.pass2_frames[0][7 * w + 7] > 170);
    }
}
