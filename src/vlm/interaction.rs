use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionType {
    PrimaryRemoval,
    SupportRemoval,
    PhysicalDeformation,
    SecondaryShadow,
    SecondaryReflection,
    IntegralBelonging,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntity {
    pub name: String,
    pub is_primary: bool,
    pub interaction_type: InteractionType,
    pub will_move: bool,
    pub needs_trajectory: bool,
    pub description: String,
}

impl SceneEntity {
    pub fn new_primary(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            is_primary: true,
            interaction_type: InteractionType::PrimaryRemoval,
            will_move: false,
            needs_trajectory: false,
            description: "Primary actor to be deleted from scene".to_string(),
        }
    }

    pub fn new_affected(
        name: impl Into<String>,
        interaction_type: InteractionType,
        will_move: bool,
        needs_trajectory: bool,
    ) -> Self {
        Self {
            name: name.into(),
            is_primary: false,
            interaction_type,
            will_move,
            needs_trajectory,
            description: String::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContactRelation {
    pub supporter: String, // e.g. "person"
    pub supported: String, // e.g. "guitar"
    pub relation: String,  // "holding", "standing_on", "leaning_against"
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SceneInteractionGraph {
    pub entities: Vec<SceneEntity>,
    pub contacts: Vec<ContactRelation>,
}

impl SceneInteractionGraph {
    pub fn add_entity(&mut self, entity: SceneEntity) {
        self.entities.push(entity);
    }

    pub fn add_contact(
        &mut self,
        supporter: impl Into<String>,
        supported: impl Into<String>,
        relation: impl Into<String>,
    ) {
        self.contacts.push(ContactRelation {
            supporter: supporter.into(),
            supported: supported.into(),
            relation: relation.into(),
        });
    }

    /// Identify which objects lose physical support when the primary object is removed
    pub fn find_unsupported_objects(&self, primary_name: &str) -> Vec<String> {
        self.contacts
            .iter()
            .filter(|c| c.supporter == primary_name)
            .map(|c| c.supported.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_graph_unsupported() {
        let mut graph = SceneInteractionGraph::default();
        graph.add_entity(SceneEntity::new_primary("person"));
        graph.add_entity(SceneEntity::new_affected(
            "guitar",
            InteractionType::SupportRemoval,
            true,
            true,
        ));
        graph.add_contact("person", "guitar", "holding");

        let unsupported = graph.find_unsupported_objects("person");
        assert_eq!(unsupported, vec!["guitar"]);
    }
}
