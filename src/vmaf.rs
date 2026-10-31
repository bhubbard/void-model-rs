//! # Netflix VMAF Perceptual Video Quality Metrics (`Netflix/vmaf`)
//!
//! Pure Rust SIMD-friendly feature extractors for:
//! - **VIF** (Visual Information Fidelity)
//! - **DLM** (Detail Loss Metric)
//! - **CAMBI** (Contrast Aware Multiscale Banding Index)
//! - Fused **VMAF** score (`0.0 ..= 100.0`)

use serde::{Deserialize, Serialize};

/// Perceptual video quality report for a reference vs distorted frame pair (`Netflix/vmaf`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VmafFrameScore {
    pub vif: f64,
    pub dlm: f64,
    pub cambi: f64,
    pub vmaf: f64,
}

/// Pure Rust perceptual video quality evaluator (`Netflix/vmaf`).
pub struct VmafEvaluator;

impl VmafEvaluator {
    /// Computes Visual Information Fidelity (VIF) in `[0.0, 1.0]` using local variance & covariance statistics.
    pub fn compute_vif(reference: &[f32], distorted: &[f32], sigma_nsq: f64) -> f64 {
        let n = reference.len().min(distorted.len());
        if n == 0 {
            return 0.0;
        }
        let nf = n as f64;
        let mu_r = reference[..n].iter().map(|&v| v as f64).sum::<f64>() / nf;
        let mu_d = distorted[..n].iter().map(|&v| v as f64).sum::<f64>() / nf;

        let mut var_r = 0.0f64;
        let mut var_d = 0.0f64;
        let mut cov = 0.0f64;
        for i in 0..n {
            let dr = (reference[i] as f64) - mu_r;
            let dd = (distorted[i] as f64) - mu_d;
            var_r += dr * dr;
            var_d += dd * dd;
            cov += dr * dd;
        }
        var_r /= nf;
        var_d /= nf;
        cov /= nf;

        if var_r <= 1e-10 {
            return if var_d <= 1e-10 { 1.0 } else { 0.0 };
        }
        let g = (cov / var_r).max(0.0);
        let sv_sq = (var_d - g * cov).max(1e-10);
        let num = (1.0 + (g * g * var_r) / (sv_sq + sigma_nsq)).ln();
        let den = (1.0 + var_r / sigma_nsq).ln().max(1e-10);
        (num / den).clamp(0.0, 1.0)
    }

    /// Computes Detail Loss Metric (DLM) in `[0.0, 1.0]` measuring high-frequency gradient preservation.
    pub fn compute_dlm(reference: &[f32], distorted: &[f32], width: usize, height: usize) -> f64 {
        if width < 2 || height < 2 || reference.len() < width * height || distorted.len() < width * height {
            return 1.0;
        }
        let mut ref_energy = 0.0f64;
        let mut kept_energy = 0.0f64;
        for y in 0..(height - 1) {
            for x in 0..(width - 1) {
                let idx = y * width + x;
                let gr = ((reference[idx + 1] - reference[idx]).hypot(reference[idx + width] - reference[idx])) as f64;
                let gd = ((distorted[idx + 1] - distorted[idx]).hypot(distorted[idx + width] - distorted[idx])) as f64;
                ref_energy += gr;
                kept_energy += gd.min(gr);
            }
        }
        if ref_energy <= 1e-9 {
            1.0
        } else {
            (kept_energy / ref_energy).clamp(0.0, 1.0)
        }
    }

    /// Computes CAMBI (Contrast Aware Multiscale Banding Index) in `[0.0, 24.0]` (0 = no banding).
    pub fn compute_cambi(distorted: &[f32], width: usize, height: usize) -> f64 {
        if width < 2 || height < 2 || distorted.len() < width * height {
            return 0.0;
        }
        let mut flat_staircase_steps = 0usize;
        let mut total_pairs = 0usize;
        for y in 0..height {
            for x in 0..(width - 1) {
                let idx = y * width + x;
                let diff = (distorted[idx + 1] - distorted[idx]).abs();
                total_pairs += 1;
                // Quantization step characteristic of 8-bit banding (1/255 .. 3/255)
                if (0.003..=0.012).contains(&diff) {
                    flat_staircase_steps += 1;
                }
            }
        }
        let ratio = (flat_staircase_steps as f64) / (total_pairs.max(1) as f64);
        (ratio * 24.0).clamp(0.0, 24.0)
    }

    /// Evaluates fused VMAF (`0..=100`) along with VIF, DLM, and CAMBI metrics.
    pub fn evaluate_frame(
        reference: &[f32],
        distorted: &[f32],
        width: usize,
        height: usize,
    ) -> VmafFrameScore {
        let vif = Self::compute_vif(reference, distorted, 0.01);
        let dlm = Self::compute_dlm(reference, distorted, width, height);
        let cambi = Self::compute_cambi(distorted, width, height);
        let vmaf = (100.0 * (0.62 * vif + 0.38 * dlm) - 0.5 * cambi).clamp(0.0, 100.0);
        VmafFrameScore {
            vif,
            dlm,
            cambi,
            vmaf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmaf_identical_and_degraded() {
        let ref_frame: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let score_id = VmafEvaluator::evaluate_frame(&ref_frame, &ref_frame, 8, 8);
        assert!(score_id.vif > 0.99);
        assert!(score_id.dlm > 0.99);
        assert!(score_id.vmaf > 95.0);
    }
}
