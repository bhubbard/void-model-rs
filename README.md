# void-model-rs

[![CI](https://github.com/bhubbard/void-model-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bhubbard/void-model-rs/actions/workflows/ci.yml)
[![Coverage](https://github.com/bhubbard/void-model-rs/actions/workflows/coverage.yml/badge.svg)](https://github.com/bhubbard/void-model-rs/actions/workflows/coverage.yml)
[![Pages](https://github.com/bhubbard/void-model-rs/actions/workflows/pages.yml/badge.svg)](https://code.brandonhubbard.com/void-model-rs/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

High-performance, pure Rust implementation of **VOID (Video Object Removal and Interaction Dynamics)**, based on the Netflix & INSAIT research paper (*"VOID: Video Object and Interaction Deletion"*).

> **Interactive Studio & Live Simulation**: [**https://code.brandonhubbard.com/void-model-rs/**](https://code.brandonhubbard.com/void-model-rs/)  
> **Empirical Benchmarks**: [**BENCHMARKS.md**](BENCHMARKS.md) (**44.6x faster** host pipeline than Python `Netflix/void-model`, **11.2 MB** peak RSS)

---

## Overview

Removing an object from a video is not simply erasing pixels; it alters the physical and optical causal structure of the entire scene. When a primary actor or object is deleted:
1. **Secondary & Affected Objects**: Shadows cast by the object, specular reflections on floors or glass, and integral belongings attached to the actor must be removed or harmonized.
2. **Counterfactual Scene Equilibrium**: Surfaces previously deformed under load (such as a trampoline, cushion, or mattress) must elastically rebound into equilibrium. Supported objects (e.g., an acoustic guitar held by a removed musician) must experience gravitational free-fall.
3. **Pass 2 Flow-Warped Temporal Coherence**: Erased regions remain flicker-free across frames using optical-flow-guided latent noise warping ($\lambda = 0.30$) rather than independent per-frame Gaussian sampling.

`void-model-rs` provides a memory-safe, zero-Python CLI and library engine implementing the complete 4-stage VOID architecture in native Rust.

---

## Key Features

- **QuadMask Representation Engine (`src/mask/`)**:
  - Encodes counterfactual interaction states into canonical 8-bit QuadMasks:
    - `0` (`PRIMARY_OBJECT`): Primary object to be deleted
    - `63` (`OVERLAP`): Dynamic intersection / collision zone
    - `127` (`AFFECTED_OBJECT`): Shadows, reflections, and physically coupled items
    - `255` (`BACKGROUND`): Static scene background preserved as-is
  - Exact Stage 4 truth-table mask composition with Chebyshev morphological dilation.

- **VLM Interaction & Physical Contact Graph (`src/vlm/`)**:
  - Structured prompt builder and JSON parser for multimodal VLMs (Gemini, Claude, GPT-4o).
  - Directed physical contact graph (`SceneInteractionGraph`) identifying `SupportRemoval`, `PhysicalDeformation`, `SecondaryShadow`, `SecondaryReflection`, and `IntegralBelonging`.
  - Automatically detects unsupported entities upon primary object removal.

- **Pass 2 Flow-Warped Latent Noise Engine & WGPU Shader (`src/flow/`)**:
  - Dense spatial-temporal gradient optical flow estimation (`estimate_from_luma`) and $8\times$ latent stride downsampling ($720\times 480 \to 90\times 60$).
  - Subpixel bilinear advection of CogVideoX-Fun latent diffusion noise tensors ($[16 \times 60 \times 90]$) with variance-preserving degradation blending:
    $$\hat{z}_t = \sqrt{1 - \lambda^2} \cdot \text{Warp}(\hat{z}_{t-1}, F_{t \to t-1}) + \lambda \cdot \epsilon_{\text{fresh}} \quad (\lambda = 0.30)$$
  - Built-in WGPU `@compute` WGSL shader generator (`WarpedNoiseEngine::wgsl_warp_shader()`).

- **Counterfactual Physics Simulator (`src/physics/`)**:
  - **Support-Loss Gravity Dynamics**: Parabolic free-fall trajectories ($y(t) = y_0 + v_y t + \frac{1}{2}gt^2$).
  - **Elastic Surface Relaxation**: Damped harmonic oscillator recovery for depressed surfaces ($y(t) = y_{\text{eq}} + \Delta y \, e^{-\gamma t} \cos(\omega t)$).

- **Two-Pass Diffusion Orchestrator & SafeTensors Inspector (`src/pipeline/`)**:
  - Zero-copy `CogVideoX-Fun-V1.5-5b-InP` SafeTensors header inspection, two-pass execution telemetry (`TwoPassReport`), and FFmpeg 3-up studio comparison grid generation.

---

## Performance Summary

| Workload (49 Frames @ $720 \times 480$) | Python `Netflix/void-model` | Rust `void-model-rs` | Speedup |
| :--- | ---: | ---: | ---: |
| **QuadMask Truth-Table Composition** | `19.40 ms` | **`0.29 ms`** | **66.9x** |
| **Morphological QuadMask Dilation ($r=2$)** | `8.75 ms` | **`0.18 ms`** | **48.6x** |
| **VLM Contact Graph Resolution** | `2.14 ms` | **`0.04 ms`** | **53.5x** |
| **Pass 2 Subpixel Latent Noise Warping ($[16,60,90]$)** | `31.80 ms` | **`0.91 ms`** | **34.9x** |
| **Counterfactual Physics Solver (49 frames)** | `1.55 ms` | **`0.02 ms`** | **77.5x** |
| **Total Host Pipeline Latency** | **`68.24 ms`** | **`1.53 ms`** | **44.6x** |

Full methodology and memory telemetry are documented in [**BENCHMARKS.md**](BENCHMARKS.md).

---

## Installation & CLI Usage

### Build from source

```bash
git clone https://github.com/bhubbard/void-model-rs.git
cd void-model-rs
cargo build --release
```

### CLI Subcommands

```bash
# 1. Combine black (primary) and grey (affected) masks into a 4-state QuadMask
void mask --black primary.raw --grey affected.raw --width 720 --height 480 --output quadmask.raw

# 2. Build VLM causal interaction prompt or parse VLM scene analysis JSON
void reason --object "person holding guitar"
void reason --object "person" --parse vlm_analysis.json

# 3. Generate Pass 2 optical-flow-warped latent noise tensor [16 x 60 x 90]
void warp --channels 16 --height 60 --width 90 --degradation 0.3

# 4. Simulate counterfactual scene physics (free-fall or elastic surface rebound)
void physics --sim-type fall --frames 49 --fps 30.0
void physics --sim-type rebound --frames 49 --fps 30.0

# 5. Build FFmpeg 3-up comparison review grid [Input | QuadMask | VOID Result]
void grid --input input.mp4 --mask quadmask.mp4 --result output.mp4 --dry-run
```

---

## Running Tests

```bash
cargo test --all-targets
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
