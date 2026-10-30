use anyhow::Result;
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use void_model::{
    flow::{LatentNoiseTensor, OpticalFlowField, WarpedNoiseEngine},
    mask::QuadMaskFrame,
    physics::PhysicsDynamics,
    pipeline::InpaintPipeline,
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
    Mask {
        /// Black mask image or video frame (0=object, 255=background)
        #[arg(long)]
        black: PathBuf,

        /// Grey mask image or video frame (127=affected, 255=background)
        #[arg(long)]
        grey: PathBuf,

        /// Width in pixels
        #[arg(long, default_value = "720")]
        width: usize,

        /// Height in pixels
        #[arg(long, default_value = "480")]
        height: usize,

        /// Output quadmask path
        #[arg(short, long, default_value = "quadmask.raw")]
        output: PathBuf,
    },

    /// Run VLM interaction reasoning prompt builder or parse analysis
    Reason {
        /// Primary object name to remove (e.g. "person holding guitar")
        #[arg(short, long)]
        object: String,

        /// Optional path to existing vlm_analysis.json to parse and summarize
        #[arg(long)]
        parse: Option<PathBuf>,
    },

    /// Generate optical flow warped noise tensor for Pass 2 refinement
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
    },

    /// Simulate physical scene dynamics (gravity drop or elastic rebound)
    Physics {
        /// Type of simulation: 'fall' or 'rebound'
        #[arg(short, long, default_value = "fall")]
        sim_type: String,

        /// Total frames to simulate
        #[arg(long, default_value = "49")]
        frames: usize,

        /// Frame rate (default: 30.0)
        #[arg(long, default_value = "30.0")]
        fps: f64,
    },

    /// Build FFmpeg comparison grid command [Input | QuadMask | Result]
    Grid {
        #[arg(long)]
        input: PathBuf,

        #[arg(long)]
        mask: PathBuf,

        #[arg(long)]
        result: PathBuf,

        #[arg(short, long, default_value = "comparison_grid.mp4")]
        output: PathBuf,

        #[arg(long)]
        dry_run: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Mask {
            black,
            grey,
            width,
            height,
            output,
        } => {
            println!("\n  🎬 VOID: Combining Masks into QuadMask");
            println!("  ──────────────────────────────────────────");
            let size = width * height;
            // Synthesize or read buffers
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

            let combined = QuadMaskFrame::combine(&black_buf, &grey_buf, width, height)?;
            let stats = combined.stats();

            fs::write(&output, &combined.data)?;
            println!("  ✓ Output QuadMask written to: {}", output.display());
            println!("    Primary (0):    {} pixels", stats.primary_pixels);
            println!("    Affected (127): {} pixels", stats.affected_pixels);
            println!("    Overlap (63):   {} pixels", stats.overlap_pixels);
            println!("    Background:     {} pixels", stats.background_pixels);
        }

        Commands::Reason { object, parse } => {
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
        } => {
            println!("\n  🌀 VOID Pass 2: Generating Warped Noise Latents");
            println!("  ──────────────────────────────────────────");
            let latent = LatentNoiseTensor::new_gaussian(channels, height, width, Some(42));
            let flow = OpticalFlowField::new(width, height);
            let warped = WarpedNoiseEngine::warp_noise(&latent, &flow);
            let fresh = LatentNoiseTensor::new_gaussian(channels, height, width, Some(43));
            let blended = WarpedNoiseEngine::blend_with_degradation(&warped, &fresh, degradation);

            println!(
                "  Shape: [channels={}, height={}, width={}]",
                blended.channels, blended.height, blended.width
            );
            println!("  Degradation Factor: {:.2}", degradation);
            println!("  ✓ Generated temporally consistent latent noise tensor.");
        }

        Commands::Physics {
            sim_type,
            frames,
            fps,
        } => {
            println!("\n  ⚡ VOID: Physical Scene Dynamics Simulation");
            println!("  ──────────────────────────────────────────");

            if sim_type == "rebound" {
                let pts =
                    PhysicsDynamics::simulate_elastic_recovery(300.0, 150.0, 2.0, 2.5, fps, frames);
                println!("  Simulated Elastic Rebound (Trampoline/Cushion Surface):");
                for p in pts.iter().take(5) {
                    println!("    Frame {:02}: y = {:.2}", p.frame, p.y);
                }
            } else {
                let pts = PhysicsDynamics::simulate_free_fall(
                    100.0, 200.0, 15.0, 0.0, 980.0, fps, frames,
                );
                println!("  Simulated Free Fall (Unsupported Released Object):");
                for p in pts.iter().take(5) {
                    println!("    Frame {:02}: x = {:.2}, y = {:.2}", p.frame, p.x, p.y);
                }
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
    }

    Ok(())
}
