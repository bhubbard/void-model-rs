use std::path::Path;
use void_model::{
    flow::{LatentNoiseTensor, OpticalFlowField, WarpedNoiseEngine},
    mask::{QuadMaskFrame, AFFECTED_OBJECT, BACKGROUND, OVERLAP, PRIMARY_OBJECT},
    physics::{PhysicsConfig, PhysicsDynamics, PhysicsScenario},
    pipeline::InpaintPipeline,
    vlm::VlmReasoner,
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
