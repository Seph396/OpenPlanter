/// Serializable event types for Tauri IPC.
///
/// These events are emitted by the engine and consumed by the frontend.
use serde::{Deserialize, Serialize};

/// A trace message from the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub message: String,
}

/// An engine step completion event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepEvent {
    pub depth: u32,
    pub step: u32,
    pub tool_name: Option<String>,
    pub tokens: TokenUsage,
    pub elapsed_ms: u64,
    pub is_final: bool,
}

/// Token usage counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Tokens written to the prompt cache this step (Anthropic only; 0 if not reported).
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    /// Tokens read from the prompt cache this step (Anthropic only; 0 if not reported).
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

/// Streaming delta — partial text from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaEvent {
    pub kind: DeltaKind,
    pub text: String,
}

/// The kind of streaming delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaKind {
    Text,
    Thinking,
    ToolCallStart,
    ToolCallArgs,
}

/// Agent solve completed successfully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteEvent {
    pub result: String,
}

/// Agent encountered an error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    pub message: String,
}

/// Background wiki curator completed an update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratorUpdateEvent {
    pub summary: String,
    pub files_changed: u32,
}

/// Wiki knowledge graph data for the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// The tier of a node in the wiki knowledge graph hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    Source,
    Section,
    Fact,
}

/// A node in the wiki knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub category: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_type: Option<NodeType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// An edge in the wiki knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub label: Option<String>,
}

/// All events the engine can emit to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Trace(TraceEvent),
    Step(StepEvent),
    Delta(DeltaEvent),
    Complete(CompleteEvent),
    Error(ErrorEvent),
    WikiUpdated(GraphData),
}

/// Configuration view sent to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigView {
    pub provider: String,
    pub model: String,
    pub reasoning_effort: Option<String>,
    pub workspace: String,
    pub session_id: Option<String>,
    pub recursive: bool,
    pub max_depth: i64,
    pub max_steps_per_call: i64,
    pub demo: bool,
    /// Model used for `subtask` children. `None` = inherit the parent's model.
    #[serde(default)]
    pub subtask_model: Option<String>,
    /// Model used for `execute` (leaf) children. Falls back to `subtask_model`,
    /// then the parent's model, when `None`.
    #[serde(default)]
    pub execute_model: Option<String>,
    /// Hard cap on `exa_agent` calls per run, shared across depth-0 and all children.
    #[serde(default)]
    pub max_exa_agent_calls: u32,
}

/// Partial configuration update from the frontend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PartialConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
}

/// Model information for the model list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: Option<String>,
    pub provider: String,
}

/// Session information for the session list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub created_at: String,
    pub turn_count: u32,
    pub last_objective: Option<String>,
}

/// Slash command result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashResult {
    pub output: String,
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_event_serialization() {
        let event = AgentEvent::Trace(TraceEvent {
            message: "Starting solve".into(),
        });
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"trace\""));
        assert!(json.contains("Starting solve"));
    }

    #[test]
    fn test_delta_kind_serialization() {
        let delta = DeltaEvent {
            kind: DeltaKind::ToolCallStart,
            text: "read_file".into(),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains("\"kind\":\"tool_call_start\""));
    }

    #[test]
    fn test_graph_data_serialization() {
        let graph = GraphData {
            nodes: vec![GraphNode {
                id: "fec".into(),
                label: "FEC Federal".into(),
                category: "campaign-finance".into(),
                path: "wiki/campaign-finance/fec-federal.md".into(),
                node_type: None,
                parent_id: None,
                content: None,
            }],
            edges: vec![GraphEdge {
                source: "fec".into(),
                target: "sec-edgar".into(),
                label: Some("cross-ref".into()),
            }],
        };
        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("fec"));
        assert!(json.contains("sec-edgar"));
        // Optional fields should be omitted when None
        assert!(!json.contains("node_type"));
        assert!(!json.contains("parent_id"));
        assert!(!json.contains("content"));
    }

    #[test]
    fn test_graph_node_with_type_serialization() {
        let node = GraphNode {
            id: "fec::summary".into(),
            label: "Summary".into(),
            category: "campaign-finance".into(),
            path: "wiki/campaign-finance/fec-federal.md".into(),
            node_type: Some(NodeType::Section),
            parent_id: Some("fec".into()),
            content: Some("The FEC maintains data...".into()),
        };
        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains("\"node_type\":\"section\""));
        assert!(json.contains("\"parent_id\":\"fec\""));
        assert!(json.contains("\"content\":"));
    }

    #[test]
    fn test_node_type_serialization() {
        let source = NodeType::Source;
        let section = NodeType::Section;
        let fact = NodeType::Fact;
        assert_eq!(serde_json::to_string(&source).unwrap(), "\"source\"");
        assert_eq!(serde_json::to_string(&section).unwrap(), "\"section\"");
        assert_eq!(serde_json::to_string(&fact).unwrap(), "\"fact\"");
    }

    #[test]
    fn test_token_usage_serde_roundtrip_with_cache_fields() {
        let usage = TokenUsage {
            input_tokens: 1000,
            output_tokens: 200,
            cache_creation_input_tokens: 5000,
            cache_read_input_tokens: 12000,
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("\"cache_creation_input_tokens\":5000"));
        assert!(json.contains("\"cache_read_input_tokens\":12000"));
        let roundtripped: TokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtripped.input_tokens, 1000);
        assert_eq!(roundtripped.output_tokens, 200);
        assert_eq!(roundtripped.cache_creation_input_tokens, 5000);
        assert_eq!(roundtripped.cache_read_input_tokens, 12000);
    }

    #[test]
    fn test_token_usage_deserializes_without_cache_fields() {
        // Old payloads (or non-Anthropic providers) that never had the new
        // fields must still deserialize, defaulting the new counters to 0.
        let json = r#"{"input_tokens": 42, "output_tokens": 7}"#;
        let usage: TokenUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn test_step_event_serialization() {
        let step = StepEvent {
            depth: 0,
            step: 3,
            tool_name: Some("read_file".into()),
            tokens: TokenUsage {
                input_tokens: 1234,
                output_tokens: 567,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
            elapsed_ms: 2345,
            is_final: false,
        };
        let json = serde_json::to_string(&step).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["depth"], 0);
        assert_eq!(parsed["step"], 3);
        assert_eq!(parsed["tool_name"], "read_file");
        assert_eq!(parsed["tokens"]["input_tokens"], 1234);
    }
}
