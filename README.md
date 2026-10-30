# void-model-rs

High-performance, pure Rust implementation of **VOID (Video Object Removal and Interaction Dynamics)**, based on the ECCV 2026 paper by Netflix and INSAIT (*"VOID: Video Object Removal and Interaction Dynamics"*).

## Overview

Removing an object from a video is not simply erasing pixels; it alters the physical and optical context of the entire scene. When an object is deleted:
1. **Secondary & Affected Objects**: Shadows cast by the object, specular reflections on floors or glass, and items held by or interacting with the object must be removed or harmonized.
2. **Dynamic Scene Equilibrium**: Surfaces previously deformed under load (such as a mattress, cushion, or trampoline) must elastically rebound into relaxation. Supported objects (e.g. books stacked on a removed table) must experience gravitational free-fall or physical repositioning.
3. **Temporal Coherence**: Erased regions must remain flicker-free across frames using flow-guided noise warping rather than independent per-frame generation.

`void-model-rs` provides a memory-safe, ultra-fast CLI and library engine implementing the entire VOID architecture in native Rust.

---

## Key Features

- **QuadMask Representation Engine**:
  - Encodes fine-grained interaction states into standard 8-bit masks:
    - `0`: Primary object to be removed
    - `63`: Overlap / dynamic intersection zone
    - `127`: Affected objects (shadows, reflections, attached belongings)
    - `255`: Static background
  - Truth-table mask composition with edge dilation, erosion, and Gaussian feathering.

- **VLM Interaction & Physical Contact Graph**:
  - Automated structured prompt builder for Multimodal VLMs (Gemini, Claude, GPT-4o).
  - Graph-based physical contact relationship analyzer (`SupportedBy`, `AttachedTo`, `PressingOn`, `ShadowCaster`, `ReflectedBy`).
  - Detects unsupported entities upon primary object removal and triggers physical re-simulation.

- **Flow-Warped Latent Noise Engine**:
  - Advects latent diffusion noise tensors ($C \times H \times W$) forward and backward along 2D optical flow fields $(u, v)$ via bilinear subpixel interpolation.
  - Latent noise degradation blending:
    $$\hat{\epsilon}_t = \text{Warp}(\hat{\epsilon}_{t-1}, F_{t \to t-1}) \cdot (1 - \lambda) + \epsilon_{\text{rand}} \cdot \lambda \quad (\lambda = 0.3)$$
  - Eliminates temporal shimmering and inpainting flicker.

- **Interaction Physics Simulator**:
  - **Support-Loss Gravity Dynamics**: Simulates realistic free-fall parabolic trajectories with ground collision damping ($e = 0.4$) and air resistance.
  - **Elastic Surface Relaxation**: Computes damped harmonic oscillator decay for indentations and deformed cushions ($y(t) = A_0 e^{-\gamma t} \cos(\omega t)$).

- **Diffusion & Inpainting Orchestration Pipeline**:
  - Generates FFmpeg-ready inpainting commands, side-by-side comparison grids, and metadata exports.
  - Zero Python dependencies, pure Rust type safety, and blazing fast execution.

---

## Installation & CLI Usage

### Build from source

```bash
git clone https://github.com/bhubbard/void-model-rs.git
cd void-model-rs
cargo build --release
```

### CLI Subcommands

#### 1. Generate QuadMask from Primary & Secondary Segmentations
```bash
void-model quadmask \
  --primary primary_person.png \
  --affected shadow_and_cane.png \
  --overlap intersection.png \
  --output quadmask.png \
  --dilate 2 \
  --feather 1.5
```

#### 2. Analyze Video Scene Interactions with VLM
```bash
# Print the VLM prompt designed to extract physical interaction graphs:
void-model vlm-prompt

# Parse and validate the VLM response:
void-model vlm-parse \
  --input vlm_analysis.json \
  --check-unsupported "dining_table"
```

#### 3. Warp Latent Noise with Optical Flow
```bash
void-model warp-noise \
  --flow-u flow_u.bin \
  --flow-v flow_v.bin \
  --prev-noise prev_noise.bin \
  --output warped_noise.bin \
  --degradation 0.3 \
  --channels 4 \
  --height 64 \
  --width 64
```

#### 4. Simulate Scene Physics on Support Loss
```bash
void-model physics \
  --scenario freefall \
  --initial-y 200.0 \
  --gravity 980.0 \
  --frames 30 \
  --fps 30.0 \
  --output trajectory.json
```

#### 5. Generate Multi-View Inpaint Comparison Grid
```bash
void-model inpaint-grid \
  --original input.mp4 \
  --quadmask quadmask.mp4 \
  --inpainted clean_output.mp4 \
  --output comparison_grid.mp4
```

---

## Architecture

```
void-model-rs/
├── src/
│   ├── mask/          # QuadMask encoder, truth table rules, morphological filters
│   │   ├── quadmask.rs
│   │   ├── segmentation.rs
│   │   └── mod.rs
│   ├── vlm/           # VLM interaction graph, physical dependency resolver
│   │   ├── interaction.rs
│   │   ├── reasoner.rs
│   │   └── mod.rs
│   ├── flow/          # Latent noise advection, optical flow field operations
│   │   ├── optical_flow.rs
│   │   ├── warped_noise.rs
│   │   └── mod.rs
│   ├── physics/       # Damped harmonic surface relaxation, gravity free-fall
│   │   ├── dynamics.rs
│   │   ├── config.rs
│   │   └── mod.rs
│   ├── pipeline/      # Orchestrator, FFmpeg generation, metadata tracking
│   │   ├── inpaint.rs
│   │   └── mod.rs
│   ├── lib.rs         # Clean public library interface
│   └── main.rs        # Production CLI binary
└── tests/
    └── upstream_parity.rs  # Full integration test suite matching Netflix paper
```

---

## Running Tests

All unit and integration tests run with standard cargo:

```bash
cargo test
```

---

## License

Licensed under the Apache License, Version 2.0 (matching the upstream Netflix VOID repository). See [LICENSE](LICENSE) for details.
