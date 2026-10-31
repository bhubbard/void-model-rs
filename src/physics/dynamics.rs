use crate::mask::{QuadMaskFrame, QuadMaskVideo, BACKGROUND};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub frame: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct PhysicsDynamics;

impl PhysicsDynamics {
    /// Simulates gravity free-fall trajectory when support is removed.
    ///
    /// - `initial_x`, `initial_y`: Starting coordinates
    /// - `v_x`, `v_y`: Initial velocity vectors
    /// - `gravity`: Acceleration in pixels/sec^2 (e.g. 980.0)
    /// - `fps`: Video frame rate
    /// - `total_frames`: Number of frames to simulate
    pub fn simulate_free_fall(
        initial_x: f64,
        initial_y: f64,
        v_x: f64,
        v_y: f64,
        gravity: f64,
        fps: f64,
        total_frames: usize,
    ) -> Vec<TrajectoryPoint> {
        let mut trajectory = Vec::with_capacity(total_frames);

        for frame in 0..total_frames {
            let t = frame as f64 / fps;
            let x = initial_x + v_x * t;
            let y = initial_y + v_y * t + 0.5 * gravity * t * t;

            trajectory.push(TrajectoryPoint {
                frame,
                x,
                y,
                z: 0.0,
            });
        }

        trajectory
    }

    /// Simulates gravity free-fall trajectory with ground-plane collision bounce (`restitution`, e.g. $e = 0.4$)
    /// and linear air resistance (`drag`, e.g. `0.05`).
    #[allow(clippy::too_many_arguments)]
    pub fn simulate_free_fall_with_collision(
        initial_x: f64,
        initial_y: f64,
        v_x: f64,
        v_y: f64,
        gravity: f64,
        ground_y: f64,
        restitution: f64,
        drag: f64,
        fps: f64,
        total_frames: usize,
    ) -> Vec<TrajectoryPoint> {
        let mut trajectory = Vec::with_capacity(total_frames);
        if total_frames == 0 {
            return trajectory;
        }

        let dt = 1.0 / fps.max(1.0);
        let substeps = 8;
        let sub_dt = dt / substeps as f64;
        let e = restitution.clamp(0.0, 1.0);
        let k = drag.max(0.0);

        let mut x = initial_x;
        let mut y = initial_y;
        let mut vx = v_x;
        let mut vy = v_y;

        for frame in 0..total_frames {
            trajectory.push(TrajectoryPoint {
                frame,
                x,
                y,
                z: 0.0,
            });

            for _ in 0..substeps {
                let ax = -k * vx;
                let ay = gravity - k * vy;
                vx += ax * sub_dt;
                vy += ay * sub_dt;
                x += vx * sub_dt;
                y += vy * sub_dt;

                if y >= ground_y {
                    y = ground_y;
                    if vy > 0.0 {
                        vy = -vy * e;
                        // Apply horizontal surface friction on bounce
                        vx *= 1.0 - 0.1 * (1.0 - e);
                        if vy.abs() < gravity * sub_dt * 2.0 {
                            vy = 0.0;
                        }
                    }
                }
            }
        }

        trajectory
    }

    /// Simulates elastic spring recovery (e.g. trampoline, cushion, or mattress returning to neutral).
    ///
    /// - `depressed_y`: Surface position under actor's weight
    /// - `neutral_y`: Equilibrium surface position
    /// - `frequency_hz`: Oscillation frequency
    /// - `damping`: Damping decay rate
    pub fn simulate_elastic_recovery(
        depressed_y: f64,
        neutral_y: f64,
        frequency_hz: f64,
        damping: f64,
        fps: f64,
        total_frames: usize,
    ) -> Vec<TrajectoryPoint> {
        let mut trajectory = Vec::with_capacity(total_frames);
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let delta_y = depressed_y - neutral_y;

        for frame in 0..total_frames {
            let t = frame as f64 / fps;
            // Damped harmonic relaxation
            let current_y = neutral_y + delta_y * (-damping * t).exp() * (omega * t).cos();

            trajectory.push(TrajectoryPoint {
                frame,
                x: 0.0,
                y: current_y,
                z: 0.0,
            });
        }

        trajectory
    }

    /// Translates foreground pixels (`!= BACKGROUND`) of an initial `QuadMaskFrame` along a
    /// simulated `trajectory` (relative to `trajectory[0]`), producing a `QuadMaskVideo`.
    pub fn warp_mask_along_trajectory(
        initial_mask: &QuadMaskFrame,
        trajectory: &[TrajectoryPoint],
    ) -> QuadMaskVideo {
        let mut video = QuadMaskVideo::new(initial_mask.width, initial_mask.height);
        let Some(origin) = trajectory.first() else {
            return video;
        };

        let w = initial_mask.width as isize;
        let h = initial_mask.height as isize;

        for pt in trajectory {
            let dx = (pt.x - origin.x).round() as isize;
            let dy = (pt.y - origin.y).round() as isize;
            let mut frame = QuadMaskFrame::new(initial_mask.width, initial_mask.height, BACKGROUND);

            for y in 0..h {
                for x in 0..w {
                    let val = initial_mask.get_pixel(x as usize, y as usize);
                    if val != BACKGROUND {
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx >= 0 && nx < w && ny >= 0 && ny < h {
                            frame.set_pixel(nx as usize, ny as usize, val);
                        }
                    }
                }
            }
            video.add_frame(frame);
        }

        video
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_fall_trajectory() {
        let points = PhysicsDynamics::simulate_free_fall(100.0, 200.0, 10.0, 0.0, 980.0, 30.0, 31);
        assert_eq!(points[0].x, 100.0);
        assert_eq!(points[0].y, 200.0);

        // At frame 30 (1.0 second), y = 200 + 0.5 * 980 * 1.0^2 = 690.0
        assert!((points[30].y - 690.0).abs() < 1e-3);
        // At frame 30, x = 100 + 10 * 1.0 = 110.0
        assert!((points[30].x - 110.0).abs() < 1e-3);
    }

    #[test]
    fn test_elastic_recovery() {
        let points = PhysicsDynamics::simulate_elastic_recovery(300.0, 200.0, 2.0, 3.0, 30.0, 60);
        assert_eq!(points[0].y, 300.0); // start at depressed position
                                        // After 2 seconds with damping=3.0, displacement should have substantially decayed towards 200.0
        assert!((points[59].y - 200.0).abs() < 5.0);
    }

    #[test]
    fn test_free_fall_with_ground_collision_and_drag() {
        let ground_y = 350.0;
        let pts = PhysicsDynamics::simulate_free_fall_with_collision(
            50.0, 200.0, 20.0, 0.0, 980.0, ground_y, 0.4, 0.05, 30.0, 60,
        );
        assert_eq!(pts.len(), 60);
        // Never penetrates below ground_y
        assert!(pts.iter().all(|p| p.y <= ground_y + 1e-6));
        // Reaches the ground plane and settles near ground_y by the end of the simulation
        let max_y = pts.iter().map(|p| p.y).fold(0.0f64, f64::max);
        assert!((max_y - ground_y).abs() < 5.0);
        assert!((pts[59].y - ground_y).abs() < 2.0);
    }

    #[test]
    fn test_warp_mask_along_trajectory() {
        use crate::mask::AFFECTED_OBJECT;
        let mut mask = QuadMaskFrame::new(20, 20, BACKGROUND);
        mask.set_pixel(5, 5, AFFECTED_OBJECT);

        let traj = vec![
            TrajectoryPoint {
                frame: 0,
                x: 100.0,
                y: 100.0,
                z: 0.0,
            },
            TrajectoryPoint {
                frame: 1,
                x: 102.0,
                y: 104.0,
                z: 0.0,
            },
        ];

        let video = PhysicsDynamics::warp_mask_along_trajectory(&mask, &traj);
        assert_eq!(video.frame_count(), 2);
        assert_eq!(video.frames[0].get_pixel(5, 5), AFFECTED_OBJECT);
        assert_eq!(video.frames[1].get_pixel(7, 9), AFFECTED_OBJECT);
    }
}
