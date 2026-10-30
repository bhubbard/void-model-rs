use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsScenario {
    #[serde(default)]
    pub static_objects: Vec<String>,
    #[serde(default)]
    pub physics_objects: Vec<String>,
    #[serde(default)]
    pub all_objects: Vec<String>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicsConfig {
    #[serde(flatten)]
    pub scenarios: HashMap<String, PhysicsScenario>,
}

impl std::str::FromStr for PhysicsConfig {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let config: PhysicsConfig = serde_json::from_str(s)?;
        Ok(config)
    }
}

impl PhysicsConfig {
    pub fn from_json_str(json_str: &str) -> Result<Self> {
        let config: PhysicsConfig = serde_json::from_str(json_str)?;
        Ok(config)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read physics config from {}", path.display()))?;
        Self::from_json_str(&content)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let json_str = serde_json::to_string_pretty(self)?;
        fs::write(path, json_str)
            .with_context(|| format!("Failed to write physics config to {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_config_json() {
        let json = r#"{
            "guitar_fall_scenario": {
                "static_objects": ["wall", "floor"],
                "physics_objects": ["guitar"],
                "all_objects": ["wall", "floor", "guitar"],
                "notes": "Guitar falls to floor after player deletion"
            }
        }"#;

        let cfg = PhysicsConfig::from_json_str(json).unwrap();
        assert!(cfg.scenarios.contains_key("guitar_fall_scenario"));
        let sc = &cfg.scenarios["guitar_fall_scenario"];
        assert_eq!(sc.physics_objects, vec!["guitar"]);
        assert_eq!(sc.static_objects.len(), 2);
    }
}
