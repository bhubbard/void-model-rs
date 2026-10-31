use std::path::Path;
use void_model::{
    flow::{LatentNoiseTensor, OpticalFlowField, WarpedNoiseEngine},
    mask::{
        BoundingBox, PointPrompt, QuadMaskFrame, VideoAnnotationConfig, AFFECTED_OBJECT,
        BACKGROUND, OVERLAP, PRIMARY_OBJECT,
    },
    physics::{PhysicsConfig, PhysicsDynamics, PhysicsScenario},
    pipeline::{InpaintConfig, InpaintPipeline, SafetensorsHeader, TensorMeta},
    vlm::{ContactRelationType, InteractionType, SceneEntity, SceneInteractionGraph, VlmReasoner},
};

#[test]
fn test_upstream_quadmask_truth_table_rules() {
    // 2x2 test pattern:
    // Pixel 0: Black=0 (obj), Grey=255 (bg)   -> QuadMask=0 (Primary only)
    // Pixel 1: Black=255 (bg), Grey=127 (obj) -> QuadMask=127 (Affected only)
    // Pixel 2: Black=0 (obj), Grey=127 (obj)  -> QuadMask=63 (Overlap)
    // Pixel 3: Black=255 (bg), Grey=255 (bg) -> QuadMask=255 (Background)
    let black = vec![0, 255, 0, 255];
    let grey = vec![255, 127, 127, 255];

    let combined = QuadMaskFrame::combine(&black, &grey, 2, 2).unwrap();
    assert_eq!(combined.data[0], PRIMARY_OBJECT);
    assert_eq!(combined.data[1], AFFECTED_OBJECT);
    assert_eq!(combined.data[2], OVERLAP);
    assert_eq!(combined.data[3], BACKGROUND);
}

#[test]
fn test_upstream_quadmask_pixel_statistics() {
    let mut frame = QuadMaskFrame::new(100, 100, BACKGROUND);
    // Draw 10 primary pixels, 5 overlap, 20 affected
    for i in 0..10 {
        frame.set_pixel(i, 0, PRIMARY_OBJECT);
    }
    for i in 0..5 {
        frame.set_pixel(i, 1, OVERLAP);
    }
    for i in 0..20 {
        frame.set_pixel(i, 2, AFFECTED_OBJECT);
    }

    let stats = frame.stats();
    assert_eq!(stats.primary_pixels, 10);
    assert_eq!(stats.overlap_pixels, 5);
    assert_eq!(stats.affected_pixels, 20);
    assert_eq!(stats.background_pixels, 10000 - 35);
    assert_eq!(stats.total_pixels, 10000);
}

#[test]
fn test_upstream_vlm_interaction_prompt_and_parser() {
    let prompt = VlmReasoner::build_prompt("magician holding floating cane");
    assert!(prompt.contains("magician holding floating cane"));
    assert!(prompt.contains("identified_objects"));
    assert!(prompt.contains("physics_summary"));

    let json_mock = r#"{
        "primary_object": "magician",
        "identified_objects": [
            {
                "name": "cane",
                "category": "affected_object",
                "will_move": true,
                "needs_trajectory": true,
                "reasoning": "Cane drops when magician is removed."
            },
            {
                "name": "floor_shadow",
                "category": "shadow",
                "will_move": false,
                "needs_trajectory": false,
                "reasoning": "Shadow cast on the floor."
            }
        ],
        "physics_summary": "Magician vanishes, cane drops to floor"
    }"#;

    let analysis = VlmReasoner::parse_vlm_analysis(json_mock).unwrap();
    assert_eq!(analysis.primary_object, "magician");
    assert_eq!(analysis.identified_objects.len(), 2);
    assert!(analysis.identified_objects[0].will_move);

    let graph = VlmReasoner::to_interaction_graph(&analysis);
    assert_eq!(graph.entities.len(), 3);
    let unsupported = graph.find_unsupported_objects("magician");
    assert_eq!(unsupported, vec!["cane"]);
}

#[test]
fn test_upstream_motion_flow_downsampling() {
    let mut high_res_flow = OpticalFlowField::new(720, 480);
    // Set flow at (16, 24)
    high_res_flow.set(16, 24, 8.0, 16.0);

    let latent_flow = high_res_flow.downsample(8);
    assert_eq!(latent_flow.width, 90);
    assert_eq!(latent_flow.height, 60);

    let downsampled_vec = latent_flow.get(2, 3);
    assert_eq!(downsampled_vec.u, 1.0); // 8.0 / 8
    assert_eq!(downsampled_vec.v, 2.0); // 16.0 / 8
}

#[test]
fn test_upstream_bilinear_subpixel_sampling() {
    let mut tensor = LatentNoiseTensor::new_gaussian(1, 4, 4, Some(999));
    tensor.set_sample(0, 0, 0, 0.0);
    tensor.set_sample(0, 0, 1, 1.0);
    tensor.set_sample(0, 1, 0, 0.0);
    tensor.set_sample(0, 1, 1, 1.0);

    // Midpoint between x=0 and x=1 at y=0.5 should be 0.5
    let mid = tensor.sample_bilinear(0, 0.5, 0.5);
    assert!((mid - 0.5).abs() < 1e-4);
}

#[test]
fn test_upstream_warped_noise_flow_advection() {
    let mut tensor = LatentNoiseTensor::new_gaussian(1, 8, 8, Some(101));
    tensor.set_sample(0, 2, 2, 5.0);

    let mut flow = OpticalFlowField::new(8, 8);
    // Motion pointing right: pixel at (3, 2) sampled from (2, 2)
    flow.set(3, 2, 1.0, 0.0);

    let warped = WarpedNoiseEngine::warp_noise(&tensor, &flow);
    // Value at (3, 2) should now be ~5.0
    let val_after = warped.get_sample(0, 2, 3);
    assert!((val_after - 5.0).abs() < 0.1);
}

#[test]
fn test_upstream_noise_blending_degradation() {
    let t1 = LatentNoiseTensor::new_gaussian(1, 4, 4, Some(10));
    let t2 = LatentNoiseTensor::new_gaussian(1, 4, 4, Some(20));

    // Degradation factor lambda = 0.3
    let blended = WarpedNoiseEngine::blend_with_degradation(&t1, &t2, 0.3);
    assert_eq!(blended.data.len(), 16);
    // Check blend math: sqrt(1 - 0.3^2)*t1 + 0.3*t2
    let expected = (1.0f32 - 0.09f32).sqrt() * t1.data[0] + 0.3 * t2.data[0];
    assert!((blended.data[0] - expected).abs() < 1e-4);
}

#[test]
fn test_upstream_physics_gravity_freefall() {
    let fps = 30.0;
    let frames = 31;
    let g = 980.0;
    let points = PhysicsDynamics::simulate_free_fall(200.0, 100.0, 0.0, 0.0, g, fps, frames);

    assert_eq!(points.len(), 31);
    assert_eq!(points[0].y, 100.0);
    // At t = 1.0s (frame 30): y = 100 + 0.5 * 980 * 1 = 590.0
    assert!((points[30].y - 590.0).abs() < 1e-3);
}

#[test]
fn test_upstream_physics_elastic_relaxation() {
    let points = PhysicsDynamics::simulate_elastic_recovery(400.0, 200.0, 1.5, 2.0, 30.0, 60);
    assert_eq!(points[0].y, 400.0);
    // Relaxing towards 200.0
    let final_y = points[59].y;
    assert!((final_y - 200.0).abs() < 20.0);
}

#[test]
fn test_upstream_physics_config_roundtrip() {
    let scenario = PhysicsScenario {
        static_objects: vec!["table".to_string(), "floor".to_string()],
        physics_objects: vec!["bowl".to_string(), "spoon".to_string()],
        notes: "Ingredients mixing test".to_string(),
        ..Default::default()
    };

    let mut cfg = PhysicsConfig::default();
    cfg.scenarios
        .insert("mix_scenario_01".to_string(), scenario);

    let json_str = serde_json::to_string_pretty(&cfg).unwrap();
    let loaded = PhysicsConfig::from_json_str(&json_str).unwrap();

    assert!(loaded.scenarios.contains_key("mix_scenario_01"));
    let sc = &loaded.scenarios["mix_scenario_01"];
    assert_eq!(sc.physics_objects.len(), 2);
    assert_eq!(sc.static_objects.len(), 2);
}

#[test]
fn test_upstream_inpaint_pipeline_grid() {
    let cmd = InpaintPipeline::build_comparison_grid_cmd(
        Path::new("video.mp4"),
        Path::new("quadmask.mp4"),
        Path::new("void_result.mp4"),
        Path::new("output_grid.mp4"),
    );

    assert_eq!(cmd[0], "ffmpeg");
    assert!(cmd.iter().any(|c| c == "-filter_complex"));
    assert!(cmd
        .iter()
        .any(|c| c.contains("[0:v][1:v][2:v]hstack=inputs=3[grid]")));
}

#[test]
fn test_upstream_quadmask_morphology_and_feathering() {
    let mut frame = QuadMaskFrame::new(9, 9, BACKGROUND);
    frame.set_pixel(4, 4, PRIMARY_OBJECT);

    let dilated = frame.dilate(2);
    // 5x5 region around (4, 4) becomes PRIMARY_OBJECT -> 25 pixels
    assert_eq!(dilated.stats().primary_pixels, 25);

    let eroded = dilated.erode(2);
    assert_eq!(eroded.stats().primary_pixels, 1);
    assert_eq!(eroded.get_pixel(4, 4), PRIMARY_OBJECT);

    let alpha = dilated.gaussian_feather(1.5);
    assert_eq!(alpha.len(), 81);
    assert!(alpha[4 * 9 + 4] > 0.75);
    assert!(alpha[4 * 9 + 4] > alpha[4 * 9 + 1]);
    assert!(alpha[0] < 0.05);
}

#[test]
fn test_upstream_quadmask_combine_with_overlap_and_annotation_box() {
    let cfg = VideoAnnotationConfig {
        video_path: "input.mp4".to_string(),
        output_dir: "masks".to_string(),
        primary_points: vec![PointPrompt::positive(2.0, 2.0, 0)],
        boxes: vec![BoundingBox::new(1.0, 1.0, 2.0, 2.0)],
    };
    let primary_mask = cfg.rasterize_box_mask(4, 4);
    let affected_mask = vec![255u8; 16];
    let mut overlap_mask = vec![255u8; 16];
    overlap_mask[0] = 0; // force pixel 0 into OVERLAP

    let combined = QuadMaskFrame::combine_with_overlap(
        &primary_mask,
        &affected_mask,
        Some(&overlap_mask),
        4,
        4,
    )
    .unwrap();
    assert_eq!(combined.get_pixel(0, 0), OVERLAP);
    assert_eq!(combined.get_pixel(1, 1), PRIMARY_OBJECT);
    assert_eq!(combined.get_pixel(3, 3), BACKGROUND);
}

#[test]
fn test_upstream_optical_flow_estimation_and_sequence_warping() {
    let w = 16;
    let h = 16;
    let mut f0 = vec![20u8; w * h];
    let mut f1 = vec![20u8; w * h];

    // 4x4 textured block translated by (+1, +2) from (4, 4) to (5, 6)
    for dy in 0..4 {
        for dx in 0..4 {
            let val = (120 + dx * 20 + dy * 15) as u8;
            f0[(4 + dy) * w + (4 + dx)] = val;
            f1[(6 + dy) * w + (5 + dx)] = val;
        }
    }

    let flow_bm = OpticalFlowField::estimate_block_matching(&f0, &f1, w, h, 2, 3).unwrap();
    let v_bm = flow_bm.get(6, 7);
    assert_eq!(v_bm.u, 1.0);
    assert_eq!(v_bm.v, 2.0);

    let flow_lk = OpticalFlowField::estimate_lucas_kanade(&f0, &f1, w, h, 2).unwrap();
    assert!(flow_lk.max_magnitude() > 0.0);

    // Multi-frame sequence warping with probability=1.0 and backward warping check
    let init = LatentNoiseTensor::new_gaussian(2, w, h, Some(55));
    let seq = WarpedNoiseEngine::warp_sequence(&init, std::slice::from_ref(&flow_bm), 0.3, 1.0, 55);
    assert_eq!(seq.len(), 2);

    let back = WarpedNoiseEngine::warp_noise_backward(&seq[1], &flow_bm);
    assert_eq!(back.data.len(), init.data.len());
}

#[test]
fn test_upstream_vlm_transitive_support_and_rebound_queries() {
    let mut graph = SceneInteractionGraph::default();
    graph.add_entity(SceneEntity::new_primary("actor"));
    graph.add_entity(SceneEntity::new_affected(
        "hat",
        InteractionType::IntegralBelonging,
        false,
        false,
    ));
    graph.add_entity(SceneEntity::new_affected(
        "shadow",
        InteractionType::SecondaryShadow,
        false,
        false,
    ));
    graph.add_entity(SceneEntity::new_affected(
        "mattress",
        InteractionType::PhysicalDeformation,
        false,
        false,
    ));
    graph.add_entity(SceneEntity::new_affected(
        "tray",
        InteractionType::SupportRemoval,
        true,
        true,
    ));
    graph.add_entity(SceneEntity::new_affected(
        "glass",
        InteractionType::SupportRemoval,
        true,
        true,
    ));

    graph.add_contact("actor", "tray", "holding");
    graph.add_contact("tray", "glass", "supporting");
    graph.add_contact("actor", "mattress", "pressing_on");

    assert_eq!(
        graph.contacts[2].relation_type(),
        ContactRelationType::PressingOn
    );
    assert_eq!(
        graph.find_unsupported_objects("actor"),
        vec!["tray".to_string(), "glass".to_string()]
    );
    assert_eq!(
        graph.find_elastic_rebound_objects("actor"),
        vec!["mattress".to_string()]
    );
    assert_eq!(
        graph.find_immediate_removal_objects("actor"),
        vec!["actor".to_string(), "hat".to_string(), "shadow".to_string()]
    );
}

#[test]
fn test_upstream_physics_ground_collision_and_mask_trajectory_warp() {
    let pts = PhysicsDynamics::simulate_free_fall_with_collision(
        10.0, 5.0, 2.0, 0.0, 980.0, 15.0, 0.4, 0.02, 30.0, 20,
    );
    assert_eq!(pts.len(), 20);
    assert!(pts.iter().all(|p| p.y <= 15.0 + 1e-6));

    let mut mask = QuadMaskFrame::new(32, 32, BACKGROUND);
    mask.set_pixel(10, 5, AFFECTED_OBJECT);
    let video = PhysicsDynamics::warp_mask_along_trajectory(&mask, &pts);
    assert_eq!(video.frame_count(), 20);
    assert_eq!(video.frames[0].stats().affected_pixels, 1);
}

#[test]
fn test_upstream_two_pass_inpainting_and_safetensors_checkpoint() {
    let mut header = SafetensorsHeader::default();
    header.tensors.insert(
        "cogvideox.inpaint.conv_in.weight".to_string(),
        TensorMeta {
            dtype: "F16".to_string(),
            shape: vec![32, 16, 3, 3],
            data_offsets: [0, 9216],
        },
    );
    let raw = header.to_header_bytes().unwrap();
    let decoded = SafetensorsHeader::parse_bytes(&raw).unwrap();
    assert_eq!(
        decoded.tensors["cogvideox.inpaint.conv_in.weight"].element_count(),
        32 * 16 * 9
    );

    let w = 16;
    let h = 16;
    let mut f0 = vec![180u8; w * h];
    let mut f1 = vec![180u8; w * h];
    let mut m0 = QuadMaskFrame::new(w, h, BACKGROUND);
    let mut m1 = QuadMaskFrame::new(w, h, BACKGROUND);
    for y in 5..9 {
        for x in 5..9 {
            f0[y * w + x] = 0;
            f1[y * w + x] = 0;
            m0.set_pixel(x, y, PRIMARY_OBJECT);
            m1.set_pixel(x, y, PRIMARY_OBJECT);
        }
    }

    let cfg = InpaintConfig {
        width: w,
        height: h,
        frames: 2,
        steps: 12,
        ..Default::default()
    };
    let out = InpaintPipeline::run_two_pass(&[f0, f1], &[m0, m1], &cfg).unwrap();
    assert_eq!(out.pass2_frames.len(), 2);
    assert_eq!(out.pass2_frames[0][0], 180);
    assert!(out.pass2_frames[0][6 * w + 6] > 150);

    // Also verify WGPU WGSL shader emission and TwoPassReport telemetry
    let wgsl = WarpedNoiseEngine::wgsl_warp_shader();
    assert!(wgsl.contains("@compute @workgroup_size(8, 8, 1)"));
    assert!(wgsl.contains("flow_uv"));

    let full_cfg = InpaintConfig::default();
    let mut full_mask = QuadMaskFrame::new(720, 480, BACKGROUND);
    full_mask.set_pixel(10, 10, PRIMARY_OBJECT);
    let full_flow = OpticalFlowField::new(720, 480);
    let report = InpaintPipeline::execute_two_pass(&full_cfg, &full_mask, &full_flow);
    assert_eq!(report.latent_channels, 16);
    assert_eq!(report.latent_width, 90);
    assert_eq!(report.latent_height, 60);
    assert!(report.pass2_temporal_correlation > 0.8);

    let header_json = br#"{"__metadata__":{"format":"pt","model":"CogVideoX-Fun-V1.5-5b-InP"}}"#;
    let mut buf = (header_json.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(header_json);
    let meta = InpaintPipeline::inspect_safetensors_header(&buf).unwrap();
    assert_eq!(meta["__metadata__"]["model"], "CogVideoX-Fun-V1.5-5b-InP");
}
