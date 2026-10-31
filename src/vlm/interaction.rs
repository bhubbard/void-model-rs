use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

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

/// Physical contact relationship categories between scene entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContactRelationType {
    SupportedBy,
    AttachedTo,
    PressingOn,
    ShadowCaster,
    ReflectedBy,
}

impl ContactRelationType {
    pub fn from_label(label: &str) -> Self {
        match label.trim().to_lowercase().as_str() {
            "attached_to" | "wearing" | "integral" => Self::AttachedTo,
            "pressing_on" | "standing_on" | "sitting_on" | "deforming" => Self::PressingOn,
            "shadow" | "shadow_caster" | "casts_shadow" => Self::ShadowCaster,
            "reflection" | "reflected_by" => Self::ReflectedBy,
            _ => Self::SupportedBy,
        }
    }
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

impl ContactRelation {
    pub fn relation_type(&self) -> ContactRelationType {
        ContactRelationType::from_label(&self.relation)
    }
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

    /// Identify all objects that lose physical support (directly or transitively)
    /// when `primary_name` is removed.
    ///
    /// Performs a cycle-safe breadth-first traversal of support chains (e.g. `person -> tray -> cup`),
    /// while keeping objects that still have an independent static supporter outside the removed set.
    pub fn find_unsupported_objects(&self, primary_name: &str) -> Vec<String> {
        let mut lost_support: HashSet<String> = HashSet::new();
        lost_support.insert(primary_name.to_string());

        let mut unsupported_order: Vec<String> = Vec::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(primary_name.to_string());

        while let Some(current) = queue.pop_front() {
            for contact in &self.contacts {
                if contact.supporter != current {
                    continue;
                }
                let rtype = contact.relation_type();
                if rtype != ContactRelationType::SupportedBy
                    && rtype != ContactRelationType::AttachedTo
                {
                    continue;
                }

                let candidate = &contact.supported;
                if lost_support.contains(candidate) {
                    continue;
                }

                // Check whether candidate has any remaining supporter that has NOT lost support
                let has_other_support = self.contacts.iter().any(|c| {
                    c.supported == *candidate
                        && (c.relation_type() == ContactRelationType::SupportedBy
                            || c.relation_type() == ContactRelationType::AttachedTo)
                        && !lost_support.contains(&c.supporter)
                });

                if !has_other_support {
                    lost_support.insert(candidate.clone());
                    unsupported_order.push(candidate.clone());
                    queue.push_back(candidate.clone());
                }
            }
        }

        unsupported_order
    }

    /// Identify deformed surfaces (e.g. cushions, trampolines, mattresses) that should
    /// elastically rebound when `primary_name` is deleted.
    pub fn find_elastic_rebound_objects(&self, primary_name: &str) -> Vec<String> {
        let mut rebound = Vec::new();
        for entity in &self.entities {
            if entity.interaction_type == InteractionType::PhysicalDeformation
                && !rebound.contains(&entity.name)
            {
                rebound.push(entity.name.clone());
            }
        }
        for contact in &self.contacts {
            if contact.supporter == primary_name
                && contact.relation_type() == ContactRelationType::PressingOn
                && !rebound.contains(&contact.supported)
            {
                rebound.push(contact.supported.clone());
            }
        }
        rebound
    }

    /// Identify all entities that must be erased simultaneously with the primary object
    /// (the primary object itself, integral belongings, cast shadows, and reflections).
    pub fn find_immediate_removal_objects(&self, primary_name: &str) -> Vec<String> {
        let mut out = vec![primary_name.to_string()];
        for entity in &self.entities {
            if matches!(
                entity.interaction_type,
                InteractionType::IntegralBelonging
                    | InteractionType::SecondaryShadow
                    | InteractionType::SecondaryReflection
            ) && !out.contains(&entity.name)
            {
                out.push(entity.name.clone());
            }
        }
        out
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

    #[test]
    fn test_transitive_and_multi_supporter_contact_graph() {
        let mut graph = SceneInteractionGraph::default();
        graph.add_entity(SceneEntity::new_primary("waiter"));
        // waiter -> tray -> wine_glass (both lose support)
        graph.add_contact("waiter", "tray", "holding");
        graph.add_contact("tray", "wine_glass", "supporting");

        // centerpiece is supported by both waiter (touching) and table (static) -> stays supported!
        graph.add_contact("waiter", "centerpiece", "supporting");
        graph.add_contact("table", "centerpiece", "supporting");

        let unsupported = graph.find_unsupported_objects("waiter");
        assert_eq!(unsupported, vec!["tray", "wine_glass"]);
    }
}
