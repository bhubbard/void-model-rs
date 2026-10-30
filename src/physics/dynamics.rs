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
}
