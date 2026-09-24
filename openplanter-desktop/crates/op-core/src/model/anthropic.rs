// Anthropic model implementation.
//
// Uses the Anthropic Messages API with SSE streaming.

use anyhow::{anyhow, Context};
use reqwest_eventsource::{Event, RequestBuilderExt};
use tokio_util::sync::CancellationToken;

use crate::events::{DeltaEvent, DeltaKind};
use super::{BaseModel, Message, ModelTurn, ToolCall};

pub struct AnthropicModel {
    client: reqwest::Client,
    model: String,
    base_url: String,
    api_key: String,
    reasoning_effort: Option<String>,
    max_tokens: u64,
}

impl AnthropicModel {
    pub fn new(
        model: String,
        base_url: String,
        api_key: String,
        reasoning_effort: Option<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            model,
            base_url,
            api_key,
            reasoning_effort,
            max_tokens: 16384,
        }
    }

    /// Whether this model uses the newer "adaptive" thinking format
    /// (`thinking: {type: "adaptive"}` + `output_config: {effort}`) instead of
    /// the legacy `thinking: {type: "enabled", budget_tokens}` format.
    ///
    /// Claude Opus 4.6 and every Claude 5-generation model (opus/sonnet/haiku/fable
    /// with a `-5` version segment, e.g. `claude-opus-5`, `claude-sonnet-5`,
    /// `claude-fable-5-1`) use adaptive thinking; older models reject it.
    fn uses_adaptive_thinking(&self) -> bool {
        let lower = self.model.to_lowercase();
        if lower.contains("opus-4-6") || lower.contains("opus-4.6") {
            return true;
        }
        for family in ["opus", "sonnet", "haiku", "fable"] {
            let needle = format!("claude-{family}-5");
            if lower.contains(&needle) {
                return true;
            }
        }
        false
    }

    /// Extract the system prompt from messages (Anthropic uses a top-level `system` field).
    fn extract_system(messages: &[Message]) -> Option<String> {
        for msg in messages {
            if let Message::System { content } = msg {
                return Some(content.clone());
            }
        }
        None
    }

    /// Convert messages to Anthropic format, excluding system messages.
    ///
    /// Consecutive `Tool` messages are merged into a single `user` message
    /// with multiple `tool_result` content blocks, since Anthropic rejects
    /// consecutive same-role messages.
    fn convert_messages(messages: &[Message]) -> Vec<serde_json::Value> {
        let mut result: Vec<serde_json::Value> = Vec::new();

        for msg in messages {
            match msg {
                Message::System { .. } => {}
                Message::User { content } => {
                    result.push(serde_json::json!({
                        "role": "user",
                        "content": content,
                    }));
                }
                Message::Assistant { content, tool_calls } => {
                    let mut blocks: Vec<serde_json::Value> = Vec::new();
                    if !content.is_empty() {
                        blocks.push(serde_json::json!({
                            "type": "text",
                            "text": content,
                        }));
                    }
                    if let Some(tcs) = tool_calls {
                        for tc in tcs {
                            let input: serde_json::Value =
                                serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                            blocks.push(serde_json::json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": input,
                            }));
                        }
                    }
                    result.push(serde_json::json!({
                        "role": "assistant",
                        "content": blocks,
                    }));
                }
                Message::Tool { tool_call_id, content } => {
                    let block = serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                    });
                    // Merge into previous user message if it contains tool_result blocks
                    if let Some(last) = result.last_mut() {
                        if last.get("role").and_then(|r| r.as_str()) == Some("user") {
                            if let Some(arr) = last.get_mut("content").and_then(|c| c.as_array_mut()) {
                                if arr.iter().any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result")) {
                                    arr.push(block);
                                    continue;
                                }
                            }
                        }
                    }
                    result.push(serde_json::json!({
                        "role": "user",
                        "content": [block],
                    }));
                }
            }
        }

        result
    }

    /// Mark the last content block of the last user/tool-result message as a
    /// cache breakpoint, so the growing conversation prefix is cached step to
    /// step. Converts a plain-string `content` field to a content-block array
    /// if needed (Anthropic requires blocks to attach `cache_control`).
    fn add_cache_control_to_last_message(messages: &mut [serde_json::Value]) {
        let Some(last_msg) = messages.last_mut() else {
            return;
        };
        if last_msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            return;
        }
        let Some(content) = last_msg.get_mut("content") else {
            return;
        };

        if let Some(text) = content.as_str() {
            let text = text.to_string();
            *content = serde_json::json!([{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }]);
            return;
        }

        if let Some(arr) = content.as_array_mut() {
            if let Some(last_block) = arr.last_mut() {
                if let Some(obj) = last_block.as_object_mut() {
                    obj.insert(
                        "cache_control".to_string(),
                        serde_json::json!({"type": "ephemeral"}),
                    );
                }
            }
        }
    }

    fn build_payload(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
    ) -> serde_json::Value {
        let effort = self
            .reasoning_effort
            .as_deref()
            .map(|e| e.trim().to_lowercase())
            .unwrap_or_default();
        let use_thinking = matches!(effort.as_str(), "low" | "medium" | "high");

        let mut converted_messages = Self::convert_messages(messages);
        Self::add_cache_control_to_last_message(&mut converted_messages);

        let mut payload = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted_messages,
            "stream": true,
        });

        if let Some(system) = Self::extract_system(messages) {
            // System must be an array of content blocks (not a plain string) so we
            // can mark it as a cache breakpoint.
            payload["system"] = serde_json::json!([{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]);
        }

        if !tools.is_empty() {
            let mut tools_vec = tools.to_vec();
            if let Some(last_tool) = tools_vec.last_mut() {
                if let Some(obj) = last_tool.as_object_mut() {
                    obj.insert(
                        "cache_control".to_string(),
                        serde_json::json!({"type": "ephemeral"}),
                    );
                }
            }
            payload["tools"] = serde_json::Value::Array(tools_vec);
        }

        if use_thinking {
            if self.uses_adaptive_thinking() {
                payload["thinking"] = serde_json::json!({"type": "adaptive"});
                payload["output_config"] = serde_json::json!({"effort": effort});
            } else {
                let budget: u64 = match effort.as_str() {
                    "low" => 1024,
                    "medium" => 4096,
                    _ => 8192,
                };
                let max_tokens = if self.max_tokens <= budget {
                    budget + 8192
                } else {
                    self.max_tokens
                };
                payload["max_tokens"] = serde_json::json!(max_tokens);
                payload["thinking"] = serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget,
                });
            }
            // Thinking is incompatible with temperature
        } else {
            payload["temperature"] = serde_json::json!(0.0);
        }

        payload
    }
}

#[async_trait::async_trait]
impl BaseModel for AnthropicModel {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
    ) -> anyhow::Result<ModelTurn> {
        let noop = |_: DeltaEvent| {};
        let cancel = CancellationToken::new();
        self.chat_stream(messages, tools, &noop, &cancel).await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
        on_delta: &(dyn Fn(DeltaEvent) + Send + Sync),
        cancel: &CancellationToken,
    ) -> anyhow::Result<ModelTurn> {
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let payload = self.build_payload(messages, tools);

        let request = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json");

        let mut es = request.json(&payload).eventsource()?;

        let mut text = String::new();
        let mut thinking = String::new();
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;
        let mut cache_creation_input_tokens: Option<u64> = None;
        let mut cache_read_input_tokens: Option<u64> = None;

        // Track content blocks by index for tool calls
        struct BlockState {
            block_type: String,
            tool_id: String,
            tool_name: String,
            input_json: String,
        }
        let mut blocks: std::collections::HashMap<u64, BlockState> = std::collections::HashMap::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        use futures::StreamExt;
        loop {
            if cancel.is_cancelled() {
                es.close();
                return Err(anyhow!("Cancelled"));
            }

            let event = tokio::select! {
                _ = cancel.cancelled() => {
                    es.close();
                    return Err(anyhow!("Cancelled"));
                }
                ev = tokio::time::timeout(
                    std::time::Duration::from_secs(120),
                    es.next(),
                ) => match ev {
                    Ok(inner) => inner,
                    Err(_) => {
                        es.close();
                        return Err(anyhow!("SSE stream timed out (no data for 120s)"));
                    }
                },
            };

            let event = match event {
                Some(Ok(ev)) => ev,
                Some(Err(reqwest_eventsource::Error::StreamEnded)) => break,
                Some(Err(e)) => {
                    es.close();
                    return Err(anyhow!("SSE stream error: {e}"));
                }
                None => break,
            };

            match event {
                Event::Open => {}
                Event::Message(msg) => {
                    let data: serde_json::Value = serde_json::from_str(&msg.data)
                        .with_context(|| format!("Failed to parse SSE chunk: {}", &msg.data))?;

                    let msg_type = data
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or(&msg.event);

                    match msg_type {
                        "message_start" => {
                            if let Some(usage) = data.pointer("/message/usage") {
                                if let Some(it) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                                    input_tokens = it;
                                }
                                if let Some(ct) = usage
                                    .get("cache_creation_input_tokens")
                                    .and_then(|v| v.as_u64())
                                {
                                    cache_creation_input_tokens = Some(ct);
                                }
                                if let Some(rt) = usage
                                    .get("cache_read_input_tokens")
                                    .and_then(|v| v.as_u64())
                                {
                                    cache_read_input_tokens = Some(rt);
                                }
                            }
                        }

                        "content_block_start" => {
                            let idx = data.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                            let block = data.get("content_block").unwrap_or(&serde_json::Value::Null);
                            let btype = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");

                            let state = match btype {
                                "tool_use" => {
                                    let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                                    let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                                    if !name.is_empty() {
                                        on_delta(DeltaEvent {
                                            kind: DeltaKind::ToolCallStart,
                                            text: name.clone(),
                                        });
                                    }
                                    BlockState {
                                        block_type: "tool_use".to_string(),
                                        tool_id: id,
                                        tool_name: name,
                                        input_json: String::new(),
                                    }
                                }
                                "thinking" => BlockState {
                                    block_type: "thinking".to_string(),
                                    tool_id: String::new(),
                                    tool_name: String::new(),
                                    input_json: String::new(),
                                },
                                _ => BlockState {
                                    block_type: "text".to_string(),
                                    tool_id: String::new(),
                                    tool_name: String::new(),
                                    input_json: String::new(),
                                },
                            };
                            blocks.insert(idx, state);
                        }

                        "content_block_delta" => {
                            let idx = data.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                            let delta = match data.get("delta") {
                                Some(d) => d,
                                None => continue,
                            };
                            let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");

                            match delta_type {
                                "text_delta" => {
                                    if let Some(t) = delta.get("text").and_then(|t| t.as_str()) {
                                        if !t.is_empty() {
                                            text.push_str(t);
                                            on_delta(DeltaEvent {
                                                kind: DeltaKind::Text,
                                                text: t.to_string(),
                                            });
                                        }
                                    }
                                }
                                "thinking_delta" => {
                                    if let Some(t) = delta.get("thinking").and_then(|t| t.as_str()) {
                                        if !t.is_empty() {
                                            thinking.push_str(t);
                                            on_delta(DeltaEvent {
                                                kind: DeltaKind::Thinking,
                                                text: t.to_string(),
                                            });
                                        }
                                    }
                                }
                                "input_json_delta" => {
                                    if let Some(chunk) = delta.get("partial_json").and_then(|j| j.as_str()) {
                                        if !chunk.is_empty() {
                                            if let Some(block) = blocks.get_mut(&idx) {
                                                block.input_json.push_str(chunk);
                                            }
                                            on_delta(DeltaEvent {
                                                kind: DeltaKind::ToolCallArgs,
                                                text: chunk.to_string(),
                                            });
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }

                        "content_block_stop" => {
                            let idx = data.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                            if let Some(block) = blocks.get(&idx) {
                                if block.block_type == "tool_use" {
                                    tool_calls.push(ToolCall {
                                        id: block.tool_id.clone(),
                                        name: block.tool_name.clone(),
                                        arguments: block.input_json.clone(),
                                    });
                                }
                            }
                        }

                        "message_delta" => {
                            if let Some(usage) = data.get("usage") {
                                if let Some(ot) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                                    output_tokens = ot;
                                }
                                if let Some(ct) = usage
                                    .get("cache_creation_input_tokens")
                                    .and_then(|v| v.as_u64())
                                {
                                    cache_creation_input_tokens = Some(ct);
                                }
                                if let Some(rt) = usage
                                    .get("cache_read_input_tokens")
                                    .and_then(|v| v.as_u64())
                                {
                                    cache_read_input_tokens = Some(rt);
                                }
                            }
                        }

                        "error" => {
                            let err_msg = data
                                .pointer("/error/message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown API error");
                            es.close();
                            return Err(anyhow!("Anthropic API error: {err_msg}"));
                        }

                        _ => {}
                    }
                }
            }
        }

        Ok(ModelTurn {
            text,
            thinking: if thinking.is_empty() { None } else { Some(thinking) },
            tool_calls,
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(model: &str, reasoning_effort: Option<&str>) -> AnthropicModel {
        AnthropicModel::new(
            model.to_string(),
            "https://api.anthropic.com/v1".to_string(),
            "sk-ant-test".to_string(),
            reasoning_effort.map(|s| s.to_string()),
        )
    }

    // ── uses_adaptive_thinking ──

    #[test]
    fn test_uses_adaptive_thinking_opus_46() {
        assert!(make_model("claude-opus-4-6", None).uses_adaptive_thinking());
        assert!(make_model("claude-opus-4.6-20250610", None).uses_adaptive_thinking());
        assert!(!make_model("claude-sonnet-4-5", None).uses_adaptive_thinking());
    }

    #[test]
    fn test_uses_adaptive_thinking_claude_5_family() {
        assert!(make_model("claude-opus-5", None).uses_adaptive_thinking());
        assert!(make_model("claude-sonnet-5", None).uses_adaptive_thinking());
        assert!(make_model("claude-fable-5-1", None).uses_adaptive_thinking());
        assert!(make_model("claude-haiku-5-20260101", None).uses_adaptive_thinking());
    }

    #[test]
    fn test_uses_adaptive_thinking_legacy_models() {
        assert!(!make_model("claude-sonnet-4-5", None).uses_adaptive_thinking());
        assert!(!make_model("claude-haiku-4-5", None).uses_adaptive_thinking());
        assert!(!make_model("claude-opus-4-1", None).uses_adaptive_thinking());
    }

    // ── extract_system ──

    #[test]
    fn test_extract_system_present() {
        let msgs = vec![
            Message::System { content: "Be helpful.".to_string() },
            Message::User { content: "Hi".to_string() },
        ];
        assert_eq!(AnthropicModel::extract_system(&msgs), Some("Be helpful.".to_string()));
    }

    #[test]
    fn test_extract_system_absent() {
        let msgs = vec![Message::User { content: "Hi".to_string() }];
        assert_eq!(AnthropicModel::extract_system(&msgs), None);
    }

    // ── convert_messages ──

    #[test]
    fn test_convert_filters_system() {
        let msgs = vec![
            Message::System { content: "System prompt".to_string() },
            Message::User { content: "Hello".to_string() },
        ];
        let converted = AnthropicModel::convert_messages(&msgs);
        assert_eq!(converted.len(), 1); // System is filtered out
        assert_eq!(converted[0]["role"], "user");
    }

    #[test]
    fn test_convert_assistant_with_tool_calls() {
        let msgs = vec![Message::Assistant {
            content: "I'll check.".to_string(),
            tool_calls: Some(vec![ToolCall {
                id: "toolu_1".to_string(),
                name: "read_file".to_string(),
                arguments: r#"{"path":"test.txt"}"#.to_string(),
            }]),
        }];
        let converted = AnthropicModel::convert_messages(&msgs);
        assert_eq!(converted[0]["role"], "assistant");
        let content = converted[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2); // text block + tool_use block
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["name"], "read_file");
        assert_eq!(content[1]["input"]["path"], "test.txt");
    }

    #[test]
    fn test_convert_tool_result() {
        let msgs = vec![Message::Tool {
            tool_call_id: "toolu_1".to_string(),
            content: "file contents here".to_string(),
        }];
        let converted = AnthropicModel::convert_messages(&msgs);
        assert_eq!(converted[0]["role"], "user");
        let content = converted[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn test_convert_merges_consecutive_tool_messages() {
        let msgs = vec![
            Message::Assistant {
                content: "Using tools.".to_string(),
                tool_calls: Some(vec![
                    ToolCall { id: "t1".into(), name: "read_file".into(), arguments: "{}".into() },
                    ToolCall { id: "t2".into(), name: "list_files".into(), arguments: "{}".into() },
                ]),
            },
            Message::Tool { tool_call_id: "t1".into(), content: "file1 contents".into() },
            Message::Tool { tool_call_id: "t2".into(), content: "file list".into() },
        ];
        let converted = AnthropicModel::convert_messages(&msgs);
        // Should be 2 messages: assistant + one merged user
        assert_eq!(converted.len(), 2, "consecutive Tool messages should merge into one user message");
        let user_content = converted[1]["content"].as_array().unwrap();
        assert_eq!(user_content.len(), 2, "merged user message should have 2 tool_result blocks");
        assert_eq!(user_content[0]["tool_use_id"], "t1");
        assert_eq!(user_content[1]["tool_use_id"], "t2");
    }

    // ── build_payload ──

    #[test]
    fn test_payload_no_thinking_has_temperature() {
        let model = make_model("claude-sonnet-4-5", None);
        let msgs = vec![
            Message::System { content: "System".to_string() },
            Message::User { content: "Hi".to_string() },
        ];
        let payload = model.build_payload(&msgs, &[]);
        assert_eq!(payload["temperature"], 0.0);
        assert_eq!(payload["system"][0]["text"], "System");
        assert_eq!(payload["stream"], true);
        assert!(payload.get("thinking").is_none());
    }

    #[test]
    fn test_payload_opus_46_adaptive_thinking() {
        let model = make_model("claude-opus-4-6", Some("high"));
        let msgs = vec![Message::User { content: "Hi".to_string() }];
        let payload = model.build_payload(&msgs, &[]);
        assert!(payload.get("temperature").is_none()); // No temperature with thinking
        assert_eq!(payload["thinking"]["type"], "adaptive");
        assert_eq!(payload["output_config"]["effort"], "high");
    }

    #[test]
    fn test_payload_claude_opus_5_adaptive_thinking() {
        let model = make_model("claude-opus-5", Some("high"));
        let msgs = vec![Message::User { content: "Hi".to_string() }];
        let payload = model.build_payload(&msgs, &[]);
        assert_eq!(payload["thinking"]["type"], "adaptive");
        assert_eq!(payload["output_config"]["effort"], "high");
        assert!(payload["thinking"].get("budget_tokens").is_none());
    }

    #[test]
    fn test_payload_older_model_enabled_thinking() {
        let model = make_model("claude-sonnet-4-5", Some("medium"));
        let msgs = vec![Message::User { content: "Hi".to_string() }];
        let payload = model.build_payload(&msgs, &[]);
        assert_eq!(payload["thinking"]["type"], "enabled");
        assert_eq!(payload["thinking"]["budget_tokens"], 4096);
    }

    #[test]
    fn test_payload_system_extracted_to_top_level() {
        let model = make_model("claude-sonnet-4-5", None);
        let msgs = vec![
            Message::System { content: "You are helpful.".to_string() },
            Message::User { content: "Test".to_string() },
        ];
        let payload = model.build_payload(&msgs, &[]);
        // System should be top-level, not in messages array
        assert_eq!(payload["system"][0]["text"], "You are helpful.");
        let messages = payload["messages"].as_array().unwrap();
        assert!(messages.iter().all(|m| m["role"] != "system"));
    }

    // ── prompt caching ──

    #[test]
    fn test_payload_system_has_cache_control() {
        let model = make_model("claude-sonnet-4-5", None);
        let msgs = vec![
            Message::System { content: "Be helpful.".to_string() },
            Message::User { content: "Hi".to_string() },
        ];
        let payload = model.build_payload(&msgs, &[]);
        let system = payload["system"].as_array().expect("system should be an array");
        assert_eq!(system.len(), 1);
        assert_eq!(system[0]["type"], "text");
        assert_eq!(system[0]["text"], "Be helpful.");
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_payload_last_tool_has_cache_control() {
        let model = make_model("claude-sonnet-4-5", None);
        let msgs = vec![Message::User { content: "Hi".to_string() }];
        let tools = vec![
            serde_json::json!({"name": "read_file", "description": "reads a file"}),
            serde_json::json!({"name": "list_files", "description": "lists files"}),
        ];
        let payload = model.build_payload(&msgs, &tools);
        let tools_out = payload["tools"].as_array().unwrap();
        assert_eq!(tools_out.len(), 2);
        assert!(tools_out[0].get("cache_control").is_none(), "only the last tool should be marked");
        assert_eq!(tools_out[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_payload_last_user_message_has_cache_control() {
        let model = make_model("claude-sonnet-4-5", None);
        let msgs = vec![Message::User { content: "Hi".to_string() }];
        let payload = model.build_payload(&msgs, &[]);
        let messages = payload["messages"].as_array().unwrap();
        let last = messages.last().unwrap();
        assert_eq!(last["role"], "user");
        let content = last["content"].as_array().expect("plain string content should be converted to blocks");
        let last_block = content.last().unwrap();
        assert_eq!(last_block["type"], "text");
        assert_eq!(last_block["text"], "Hi");
        assert_eq!(last_block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_payload_last_tool_result_message_has_cache_control() {
        let model = make_model("claude-sonnet-4-5", None);
        let msgs = vec![
            Message::Assistant {
                content: "Using tools.".to_string(),
                tool_calls: Some(vec![ToolCall {
                    id: "t1".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                }]),
            },
            Message::Tool { tool_call_id: "t1".into(), content: "file contents".into() },
        ];
        let payload = model.build_payload(&msgs, &[]);
        let messages = payload["messages"].as_array().unwrap();
        let last = messages.last().unwrap();
        assert_eq!(last["role"], "user");
        let content = last["content"].as_array().unwrap();
        let last_block = content.last().unwrap();
        assert_eq!(last_block["type"], "tool_result");
        assert_eq!(last_block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_payload_only_last_message_has_cache_control() {
        let model = make_model("claude-sonnet-4-5", None);
        let msgs = vec![
            Message::User { content: "First".to_string() },
            Message::Assistant { content: "Reply".to_string(), tool_calls: None },
            Message::User { content: "Second".to_string() },
        ];
        let payload = model.build_payload(&msgs, &[]);
        let messages = payload["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        // First user message should remain a plain string (untouched).
        assert!(messages[0]["content"].is_string());
        // Last message (second user turn) should be converted and cache-tagged.
        let content = messages[2]["content"].as_array().unwrap();
        assert_eq!(content.last().unwrap()["cache_control"]["type"], "ephemeral");
    }

    // ── model_name / provider_name ──

    #[test]
    fn test_model_name_and_provider() {
        let model = make_model("claude-opus-4-6", None);
        assert_eq!(model.model_name(), "claude-opus-4-6");
        assert_eq!(model.provider_name(), "anthropic");
    }
}
