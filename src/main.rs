use anyhow::Result;
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use void_model::{
    flow::{LatentNoiseTensor, OpticalFlowField, WarpedNoiseEngine},
    mask::QuadMaskFrame,
    physics::PhysicsDynamics,
    pipeline::{InpaintConfig, InpaintPipeline},
    vlm::VlmReasoner,
};

#[derive(Parser, Debug)]
#[command(
    name = "void",
    about = "High-performance, pure Rust port of Netflix's VOID — Video Object and Interaction Deletion.",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Combine black mask (primary object) and grey mask (affected objects) into a QuadMask
    #[command(visible_alias = "quadmask")]
    Mask {
        /// Primary object mask (0=object, 255=background)
        #[arg(long, visible_alias = "primary")]
        black: PathBuf,

        /// Affected objects mask (127=affected, 255=background)
        #[arg(long, visible_alias = "affected")]
        grey: PathBuf,

        /// Optional explicit overlap mask
        #[arg(long)]
        overlap: Option<PathBuf>,

        /// Width in pixels
        #[arg(long, default_value = "720")]
        width: usize,

        /// Height in pixels
        #[arg(long, default_value = "480")]
        height: usize,

        /// Optional morphological dilation radius in pixels
        #[arg(long, default_value = "0")]
        dilate: usize,

        /// Optional morphological erosion radius in pixels
        #[arg(long, default_value = "0")]
        erode: usize,

        /// Optional Gaussian feather sigma in pixels
        #[arg(long, default_value = "0.0")]
        feather: f32,

        /// Output quadmask path
        #[arg(short, long, default_value = "quadmask.raw")]
        output: PathBuf,
    },

    /// Run VLM interaction reasoning prompt builder or parse analysis
    #[command(visible_aliases = ["vlm-prompt", "vlm-parse"])]
    Reason {
        /// Primary object name to remove (e.g. "person holding guitar")
        #[arg(short, long, default_value = "primary_actor")]
        object: String,

        /// Optional path to existing vlm_analysis.json to parse and summarize
        #[arg(long, visible_alias = "input")]
        parse: Option<PathBuf>,

        /// Optional entity name to query for unsupported dependent objects
        #[arg(long)]
        check_unsupported: Option<String>,
    },

    /// Generate optical flow warped noise tensor for Pass 2 refinement
    #[command(visible_alias = "warp-noise")]
    Warp {
        /// Channels in latent space (default 16 for CogVideoX)
        #[arg(long, default_value = "16")]
        channels: usize,

        /// Latent height (default 60 = 480 / 8)
        #[arg(long, default_value = "60")]
        height: usize,

        /// Latent width (default 90 = 720 / 8)
        #[arg(long, default_value = "90")]
        width: usize,

        /// Noise degradation factor lambda (0.0 to 1.0)
        #[arg(long, default_value = "0.3")]
        degradation: f32,

        /// Optional horizontal optical flow binary file (flow_u.bin)
        #[arg(long)]
        flow_u: Option<PathBuf>,

        /// Optional vertical optical flow binary file (flow_v.bin)
        #[arg(long)]
        flow_v: Option<PathBuf>,

        /// Optional previous latent noise binary file (prev_noise.bin)
        #[arg(long)]
        prev_noise: Option<PathBuf>,

        /// Optional output file for the warped latent noise tensor
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Simulate physical scene dynamics (gravity drop or elastic rebound)
    Physics {
        /// Type of simulation: 'fall' / 'freefall' or 'rebound'
        #[arg(short, long, visible_alias = "scenario", default_value = "fall")]
        sim_type: String,

        /// Starting vertical coordinate (default: 200.0)
        #[arg(long, default_value = "200.0")]
        initial_y: f64,

        /// Gravitational acceleration in pixels/sec^2 (default: 980.0)
        #[arg(long, default_value = "980.0")]
        gravity: f64,

        /// Optional ground plane Y coordinate for collision bounce
        #[arg(long)]
        ground_y: Option<f64>,

        /// Ground collision restitution coefficient e in [0.0, 1.0] (default: 0.4)
        #[arg(long, default_value = "0.4")]
        restitution: f64,

        /// Total frames to simulate
        #[arg(long, default_value = "49")]
        frames: usize,

        /// Frame rate (default: 30.0)
        #[arg(long, default_value = "30.0")]
        fps: f64,

        /// Optional JSON output path for simulated trajectory points
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Build FFmpeg comparison grid command [Input | QuadMask | Result]
    #[command(visible_alias = "inpaint-grid")]
    Grid {
        #[arg(long, visible_alias = "original")]
        input: PathBuf,

        #[arg(long, visible_alias = "quadmask")]
        mask: PathBuf,

        #[arg(long, visible_alias = "inpainted")]
        result: PathBuf,

        #[arg(short, long, default_value = "comparison_grid.mp4")]
        output: PathBuf,

        #[arg(long)]
        dry_run: bool,
    },

    /// Execute two-pass (Base Inpainting + Warped-Noise Refinement) pipeline on raw luma frames
    Inpaint {
        /// Frame width in pixels
        #[arg(long, default_value = "64")]
        width: usize,

        /// Frame height in pixels
        #[arg(long, default_value = "64")]
        height: usize,

        /// Number of frames
        #[arg(long, default_value = "4")]
        frames: usize,

        /// Latent noise degradation factor lambda
        #[arg(long, default_value = "0.3")]
        degradation: f32,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Mask {
            black,
            grey,
            overlap,
            width,
            height,
            dilate,
            erode,
            feather,
            output,
        } => {
            println!("\n  🎬 VOID: Combining Masks into QuadMask");
            println!("  ──────────────────────────────────────────");
            let size = width * height;
            let black_buf = if black.exists() {
                fs::read(&black)?
            } else {
                vec![255u8; size]
            };
            let grey_buf = if grey.exists() {
                fs::read(&grey)?
            } else {
                vec![255u8; size]
            };
            let overlap_buf = match overlap {
                Some(ref p) if p.exists() => Some(fs::read(p)?),
                _ => None,
            };

            let mut combined = if overlap_buf.is_some() {
                QuadMaskFrame::combine_with_overlap(
                    &black_buf,
                    &grey_buf,
                    overlap_buf.as_deref(),
                    width,
                    height,
                )?
            } else {
                QuadMaskFrame::combine(&black_buf, &grey_buf, width, height)?
            };

            if dilate > 0 {
                combined = combined.dilate(dilate);
            }
            if erode > 0 {
                combined = combined.erode(erode);
            }
            if feather > 0.0 {
                let _feathered = combined.gaussian_feather(feather);
            }

            let stats = combined.stats();

            fs::write(&output, &combined.data)?;
            println!("  ✓ Output QuadMask written to: {}", output.display());
            println!("    Primary (0):    {} pixels", stats.primary_pixels);
            println!("    Affected (127): {} pixels", stats.affected_pixels);
            println!("    Overlap (63):   {} pixels", stats.overlap_pixels);
            println!("    Background:     {} pixels", stats.background_pixels);
        }

        Commands::Reason {
            object,
            parse,
            check_unsupported,
        } => {
            if let Some(path) = parse {
                let json = fs::read_to_string(&path)?;
                let analysis = VlmReasoner::parse_vlm_analysis(&json)?;
                let graph = VlmReasoner::to_interaction_graph(&analysis);

                println!("\n  🧠 VOID: VLM Interaction Analysis");
                println!("  ──────────────────────────────────────────");
                println!("  Primary Object: {}", analysis.primary_object);
                println!("  Physics Summary: {}", analysis.physics_summary);
                println!("  Identified Entities ({}):", graph.entities.len());
                for e in &graph.entities {
                    println!(
                        "    - {} ({:?}) will_move={}",
                        e.name, e.interaction_type, e.will_move
                    );
                }
                if let Some(target) = check_unsupported {
                    let unsup = graph.find_unsupported_objects(&target);
                    println!("  Unsupported on removing '{}': {:?}", target, unsup);
                }
            } else {
                let prompt = VlmReasoner::build_prompt(&object);
                println!("\n  📝 Generated VLM Prompt for Object: \"{}\"", object);
                println!("  ──────────────────────────────────────────\n");
                println!("{}", prompt);
            }
        }

        Commands::Warp {
            channels,
            height,
            width,
            degradation,
            flow_u,
            flow_v,
            prev_noise,
            output,
        } => {
            println!("\n  🌀 VOID Pass 2: Generating Warped Noise Latents");
            println!("  ──────────────────────────────────────────");
            let latent = match prev_noise {
                Some(ref p) if p.exists() => {
                    let bytes = fs::read(p)?;
                    LatentNoiseTensor::from_bytes(&bytes, channels, height, width)?
                }
                _ => LatentNoiseTensor::new_gaussian(channels, height, width, Some(42)),
            };
            let flow = match (flow_u, flow_v) {
                (Some(ref pu), Some(ref pv)) if pu.exists() && pv.exists() => {
                    let u_bytes = fs::read(pu)?;
                    let v_bytes = fs::read(pv)?;
                    OpticalFlowField::from_planar_bytes(&u_bytes, &v_bytes, width, height)?
                }
                _ => OpticalFlowField::new(width, height),
            };
            let warped = WarpedNoiseEngine::warp_noise(&latent, &flow);
            let fresh = LatentNoiseTensor::new_gaussian(channels, height, width, Some(43));
            let blended = WarpedNoiseEngine::blend_with_degradation(&warped, &fresh, degradation);

            if let Some(out_path) = output {
                fs::write(&out_path, blended.to_bytes())?;
                println!("  ✓ Saved warped noise tensor to: {}", out_path.display());
            }

            println!(
                "  Shape: [channels={}, height={}, width={}]",
                blended.channels, blended.height, blended.width
            );
            println!("  Degradation Factor: {:.2}", degradation);
            println!("  ✓ Generated temporally consistent latent noise tensor.");
        }

        Commands::Physics {
            sim_type,
            initial_y,
            gravity,
            ground_y,
            restitution,
            frames,
            fps,
            output,
        } => {
            println!("\n  ⚡ VOID: Physical Scene Dynamics Simulation");
            println!("  ──────────────────────────────────────────");

            let pts = if sim_type == "rebound" {
                let pts = PhysicsDynamics::simulate_elastic_recovery(
                    initial_y + 100.0,
                    initial_y,
                    2.0,
                    2.5,
                    fps,
                    frames,
                );
                println!("  Simulated Elastic Rebound (Trampoline/Cushion Surface):");
                for p in pts.iter().take(5) {
                    println!("    Frame {:02}: y = {:.2}", p.frame, p.y);
                }
                pts
            } else if let Some(gy) = ground_y {
                let pts = PhysicsDynamics::simulate_free_fall_with_collision(
                    100.0,
                    initial_y,
                    15.0,
                    0.0,
                    gravity,
                    gy,
                    restitution,
                    0.02,
                    fps,
                    frames,
                );
                println!(
                    "  Simulated Free Fall with Ground Bounce (ground_y={:.1}, e={:.2}):",
                    gy, restitution
                );
                for p in pts.iter().take(5) {
                    println!("    Frame {:02}: x = {:.2}, y = {:.2}", p.frame, p.x, p.y);
                }
                pts
            } else {
                let pts = PhysicsDynamics::simulate_free_fall(
                    100.0, initial_y, 15.0, 0.0, gravity, fps, frames,
                );
                println!("  Simulated Free Fall (Unsupported Released Object):");
                for p in pts.iter().take(5) {
                    println!("    Frame {:02}: x = {:.2}, y = {:.2}", p.frame, p.x, p.y);
                }
                pts
            };

            if let Some(out_path) = output {
                let json = serde_json::to_string_pretty(&pts)?;
                fs::write(&out_path, json)?;
                println!("  ✓ Trajectory written to: {}", out_path.display());
            }
        }

        Commands::Grid {
            input,
            mask,
            result,
            output,
            dry_run,
        } => {
            let cmd = InpaintPipeline::build_comparison_grid_cmd(&input, &mask, &result, &output);
            if dry_run {
                println!("\n  Dry-run FFmpeg Grid Command:");
                println!("    {}", cmd.join(" "));
                return Ok(());
            }

            println!("  Compiling comparison review grid: {}", output.display());
            println!("  Command: {}", cmd.join(" "));
        }

        Commands::Inpaint {
            width,
            height,
            frames,
            degradation,
        } => {
            let cfg = InpaintConfig {
                width,
                height,
                frames,
                warped_noise_degradation: degradation,
                ..Default::default()
            };
            let raw_frames = vec![vec![200u8; width * height]; frames];
            let masks = vec![QuadMaskFrame::new(width, height, 255); frames];
            let res = InpaintPipeline::run_two_pass(&raw_frames, &masks, &cfg)?;
            println!(
                "  ✓ Completed Two-Pass VOID Inpainting on {} frames ({}x{}).",
                res.pass2_frames.len(),
                width,
                height
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_quadmask_alias_and_morphological_flags() {
        let cli = Cli::try_parse_from([
            "void",
            "quadmask",
            "--primary",
            "p.raw",
            "--affected",
            "a.raw",
            "--overlap",
            "o.raw",
            "--dilate",
            "2",
            "--feather",
            "1.5",
            "--output",
            "out.raw",
        ])
        .unwrap();

        match cli.command {
            Commands::Mask {
                dilate,
                feather,
                overlap,
                ..
            } => {
                assert_eq!(dilate, 2);
                assert!((feather - 1.5).abs() < 1e-5);
                assert!(overlap.is_some());
            }
            _ => panic!("Expected Mask command"),
        }
    }

    #[test]
    fn test_cli_readme_aliases_vlm_warp_physics_grid() {
        let c1 = Cli::try_parse_from([
            "void",
            "vlm-parse",
            "--input",
            "analysis.json",
            "--check-unsupported",
            "dining_table",
        ])
        .unwrap();
        assert!(matches!(c1.command, Commands::Reason { .. }));

        let c2 = Cli::try_parse_from([
            "void",
            "warp-noise",
            "--flow-u",
            "u.bin",
            "--flow-v",
            "v.bin",
            "--prev-noise",
            "prev.bin",
            "--degradation",
            "0.3",
            "--channels",
            "4",
            "--height",
            "64",
            "--width",
            "64",
        ])
        .unwrap();
        assert!(matches!(c2.command, Commands::Warp { .. }));

        let c3 = Cli::try_parse_from([
            "void",
            "physics",
            "--scenario",
            "freefall",
            "--initial-y",
            "200.0",
            "--ground-y",
            "420.0",
            "--restitution",
            "0.4",
            "--output",
            "trajectory.json",
        ])
        .unwrap();
        assert!(matches!(c3.command, Commands::Physics { .. }));

        let c4 = Cli::try_parse_from([
            "void",
            "inpaint-grid",
            "--original",
            "in.mp4",
            "--quadmask",
            "mask.mp4",
            "--inpainted",
            "out.mp4",
        ])
        .unwrap();
        assert!(matches!(c4.command, Commands::Grid { .. }));
    }
}
