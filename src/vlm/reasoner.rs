use crate::vlm::interaction::{InteractionType, SceneEntity, SceneInteractionGraph};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmObjectAnalysis {
    pub name: String,
    pub category: String, // "integral_belonging", "affected_object", "shadow", "reflection"
    pub will_move: bool,
    pub needs_trajectory: bool,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmSceneAnalysis {
    pub primary_object: String,
    pub identified_objects: Vec<VlmObjectAnalysis>,
    pub physics_summary: String,
}

pub struct VlmReasoner;

impl VlmReasoner {
    /// Builds the structured Gemini/VLM prompt for stage 2 interaction analysis
    pub fn build_prompt(primary_object: &str) -> String {
        format!(
            r#"You are an expert computer vision and physics reasoning assistant for the VOID video inpainting pipeline.
The primary object being deleted from the video is: "{primary_object}".

Analyze the video carefully and identify:
1. Integral belongings attached to "{primary_object}" that must be removed together (e.g. shoes, hat, watch).
2. Affected objects that physically interact with "{primary_object}" (e.g. held items that will drop, cushions that will decompress, trampolines that spring up).
3. Secondary optical effects (cast shadows, floor/water reflections).

Respond strictly with a JSON object adhering to this schema:
{{
  "primary_object": "{primary_object}",
  "identified_objects": [
    {{
      "name": "guitar",
      "category": "affected_object",
      "will_move": true,
      "needs_trajectory": true,
      "reasoning": "Guitar is supported by the person and will fall when the person is deleted."
    }},
    {{
      "name": "floor_shadow",
      "category": "shadow",
      "will_move": false,
      "needs_trajectory": false,
      "reasoning": "Shadow cast by the primary actor on the hardwood floor."
    }}
  ],
  "physics_summary": "Removal of person triggers free-fall of held guitar and elimination of contact shadow."
}}"#
        )
    }

    /// Parses the JSON response from the VLM
    pub fn parse_vlm_analysis(json_str: &str) -> Result<VlmSceneAnalysis> {
        let cleaned = json_str
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let analysis: VlmSceneAnalysis = serde_json::from_str(cleaned)?;
        Ok(analysis)
    }

    /// Converts VLM analysis into a `SceneInteractionGraph`
    pub fn to_interaction_graph(analysis: &VlmSceneAnalysis) -> SceneInteractionGraph {
        let mut graph = SceneInteractionGraph::default();
        graph.add_entity(SceneEntity::new_primary(&analysis.primary_object));

        for obj in &analysis.identified_objects {
            let int_type = match obj.category.as_str() {
                "shadow" => InteractionType::SecondaryShadow,
                "reflection" => InteractionType::SecondaryReflection,
                "integral_belonging" => InteractionType::IntegralBelonging,
                _ => {
                    if obj.will_move {
                        InteractionType::SupportRemoval
                    } else {
                        InteractionType::PhysicalDeformation
                    }
                }
            };

            let mut entity =
                SceneEntity::new_affected(&obj.name, int_type, obj.will_move, obj.needs_trajectory);
            entity.description = obj.reasoning.clone();

            if obj.will_move {
                graph.add_contact(&analysis.primary_object, &obj.name, "supporting");
            }

            graph.add_entity(entity);
        }

        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vlm_json_parsing() {
        let json = r#"{
            "primary_object": "person",
            "identified_objects": [
                {
                    "name": "trampoline",
                    "category": "affected_object",
                    "will_move": false,
                    "needs_trajectory": false,
                    "reasoning": "Surface is depressed under weight and will spring back."
                },
                {
                    "name": "body_shadow",
                    "category": "shadow",
                    "will_move": false,
                    "needs_trajectory": false,
                    "reasoning": "Shadow cast on the trampoline mat."
                }
            ],
            "physics_summary": "Person jumping on trampoline"
        }"#;

        let parsed = VlmReasoner::parse_vlm_analysis(json).unwrap();
        assert_eq!(parsed.primary_object, "person");
        assert_eq!(parsed.identified_objects.len(), 2);

        let graph = VlmReasoner::to_interaction_graph(&parsed);
        assert_eq!(graph.entities.len(), 3); // 1 primary + 2 affected
    }
}
