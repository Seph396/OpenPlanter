/// Web tools: Exa search, fetch_url.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use serde_json::json;

use super::ToolResult;

fn clip(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let end = text.floor_char_boundary(max_chars);
    let omitted = text.len() - end;
    format!(
        "{}\n\n...[truncated {omitted} chars]...",
        &text[..end]
    )
}

pub async fn web_search(
    exa_api_key: Option<&str>,
    exa_base_url: &str,
    query: &str,
    num_results: i64,
    include_text: bool,
    max_file_chars: usize,
    timeout_sec: u64,
) -> ToolResult {
    let query = query.trim();
    if query.is_empty() {
        return ToolResult::error("web_search requires non-empty query".into());
    }

    let api_key = match exa_api_key {
        Some(k) if !k.trim().is_empty() => k,
        _ => return ToolResult::error("EXA_API_KEY not configured".into()),
    };

    let clamped = num_results.max(1).min(20);
    let mut payload = json!({
        "query": query,
        "type": "auto",
        "numResults": clamped,
        "contents": {"highlights": true},
    });
    if include_text {
        payload["contents"]["text"] = json!({"maxCharacters": 4000});
    }

    let url = format!("{}/search", exa_base_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("Content-Type", "application/json")
        .header("User-Agent", "exa-py 1.0.18")
        .timeout(std::time::Duration::from_secs(timeout_sec))
        .json(&payload)
        .send()
        .await;

    let resp = match response {
        Ok(r) => r,
        Err(e) => return ToolResult::error(format!("Web search failed: {e}")),
    };

    let body: serde_json::Value = match resp.json().await {
        Ok(b) => b,
        Err(e) => return ToolResult::error(format!("Web search response parse error: {e}")),
    };

    let mut out_results: Vec<serde_json::Value> = Vec::new();
    if let Some(results) = body.get("results").and_then(|r| r.as_array()) {
        for row in results {
            let mut item = json!({
                "url": row.get("url").and_then(|u| u.as_str()).unwrap_or(""),
                "title": row.get("title").and_then(|t| t.as_str()).unwrap_or(""),
                "snippet": row.get("highlights").and_then(|h| h.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(" ... "))
                    .or_else(|| row.get("highlight").and_then(|h| h.as_str()).map(String::from))
                    .or_else(|| row.get("snippet").and_then(|s| s.as_str()).map(String::from))
                    .unwrap_or_default(),
            });
            if include_text {
                if let Some(text) = row.get("text").and_then(|t| t.as_str()) {
                    item["text"] = json!(clip(text, 4000));
                }
            }
            out_results.push(item);
        }
    }

    let output = json!({
        "query": query,
        "results": out_results,
        "total": out_results.len(),
    });
    ToolResult::ok(clip(
        &serde_json::to_string_pretty(&output).unwrap_or_default(),
        max_file_chars,
    ))
}

pub async fn fetch_url(
    exa_api_key: Option<&str>,
    exa_base_url: &str,
    urls: &[String],
    max_file_chars: usize,
    timeout_sec: u64,
) -> ToolResult {
    if urls.is_empty() {
        return ToolResult::error("fetch_url requires at least one valid URL".into());
    }

    let api_key = match exa_api_key {
        Some(k) if !k.trim().is_empty() => k,
        _ => return ToolResult::error("EXA_API_KEY not configured".into()),
    };

    let normalized: Vec<&str> = urls
        .iter()
        .map(|u| u.trim())
        .filter(|u| !u.is_empty())
        .take(10)
        .collect();

    if normalized.is_empty() {
        return ToolResult::error("fetch_url requires at least one valid URL".into());
    }

    let payload = json!({
        "ids": normalized,
        "text": { "maxCharacters": 8000 },
    });

    let url = format!("{}/contents", exa_base_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("Content-Type", "application/json")
        .header("User-Agent", "exa-py 1.0.18")
        .timeout(std::time::Duration::from_secs(timeout_sec))
        .json(&payload)
        .send()
        .await;

    let resp = match response {
        Ok(r) => r,
        Err(e) => return ToolResult::error(format!("Fetch URL failed: {e}")),
    };

    let body: serde_json::Value = match resp.json().await {
        Ok(b) => b,
        Err(e) => return ToolResult::error(format!("Fetch URL response parse error: {e}")),
    };

    let mut pages: Vec<serde_json::Value> = Vec::new();
    if let Some(results) = body.get("results").and_then(|r| r.as_array()) {
        for row in results {
            pages.push(json!({
                "url": row.get("url").and_then(|u| u.as_str()).unwrap_or(""),
                "title": row.get("title").and_then(|t| t.as_str()).unwrap_or(""),
                "text": clip(
                    row.get("text").and_then(|t| t.as_str()).unwrap_or(""),
                    8000,
                ),
            }));
        }
    }

    let output = json!({
        "pages": pages,
        "total": pages.len(),
    });
    ToolResult::ok(clip(
        &serde_json::to_string_pretty(&output).unwrap_or_default(),
        max_file_chars,
    ))
}

// ---------------------------------------------------------------------
// exa_agent — Exa Connect "Agent" runs (POST /agent/runs, poll GET .../{id})
//
// Verified against https://exa.ai/docs/exa-spec.yaml (CreateAgentRunRequest,
// AgentRun, AgentRunOutput, AgentDataSource schemas):
//   - dataSources items are objects `{"provider": "<name>"}`, not bare
//     strings — the spec's AgentDataSourceProvider enum lists valid names
//     (fiber, financial_datasets, similarweb, baselayer, affiliate,
//     particle, jinko, polymarket, macrobond).
//   - effort is a free string from AgentEffort's enum (minimal/low/medium/
//     high/xhigh/auto/max); passed through as given, not validated here.
//   - AgentRun.status is one of queued/running/completed/failed/cancelled.
//   - AgentRun.output = { text, structured, grounding: [{field, citations:
//     [{url, title}], confidence}] }.
// Both `x-api-key` and `Authorization: Bearer` are valid per the spec's
// `securitySchemes` (apiKey + bearer); this tool uses Bearer per the brief.
// ---------------------------------------------------------------------

/// Build the POST /agent/runs request body. Pure/unit-testable — no network.
pub fn build_exa_agent_payload(
    query: &str,
    data_sources: Option<&[String]>,
    output_schema: Option<&serde_json::Value>,
    effort: Option<&str>,
) -> serde_json::Value {
    let mut payload = json!({ "query": query });
    if let Some(sources) = data_sources {
        if !sources.is_empty() {
            let arr: Vec<serde_json::Value> = sources
                .iter()
                .map(|s| json!({ "provider": s }))
                .collect();
            payload["dataSources"] = json!(arr);
        }
    }
    if let Some(schema) = output_schema {
        payload["outputSchema"] = schema.clone();
    }
    if let Some(e) = effort {
        if !e.trim().is_empty() {
            payload["effort"] = json!(e);
        }
    }
    payload
}

/// Extract the observation text from a terminal `AgentRun` JSON object:
/// `output.text` plus a deduplicated list of grounding citation URLs.
/// Pure/unit-testable — no network.
pub fn format_exa_agent_output(run: &serde_json::Value) -> String {
    let text = run
        .get("output")
        .and_then(|o| o.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    let mut urls: Vec<String> = Vec::new();
    if let Some(grounding) = run.get("output").and_then(|o| o.get("grounding")).and_then(|g| g.as_array()) {
        for g in grounding {
            if let Some(citations) = g.get("citations").and_then(|c| c.as_array()) {
                for c in citations {
                    if let Some(url) = c.get("url").and_then(|u| u.as_str()) {
                        if !urls.iter().any(|u| u == url) {
                            urls.push(url.to_string());
                        }
                    }
                }
            }
        }
    }

    if urls.is_empty() {
        text.to_string()
    } else {
        format!("{text}\n\nSources:\n{}", urls.iter().map(|u| format!("- {u}")).collect::<Vec<_>>().join("\n"))
    }
}

/// Run an Exa Connect Agent: create the run, poll until it reaches a
/// terminal status (or `timeout_sec` elapses), and return `output.text`
/// plus grounding citation URLs as the observation.
#[allow(clippy::too_many_arguments)]
/// Atomically check-and-increment `call_counter` against `max_calls`. Returns
/// `Ok(call_number)` (1-based) when the call is allowed and the counter has
/// already been incremented to reflect it; `Err(current_count)` when the
/// budget is exhausted and the counter was left unchanged. Shared across
/// depth-0 and every `subtask`/`execute` child via the same `Arc`, so
/// concurrent callers never overrun `max_calls`.
fn try_reserve_exa_agent_call(call_counter: &AtomicU32, max_calls: u32) -> Result<u32, u32> {
    call_counter
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |c| {
            if c < max_calls {
                Some(c + 1)
            } else {
                None
            }
        })
        .map(|prev| prev + 1)
        .map_err(|current| current)
}

pub async fn exa_agent(
    exa_api_key: Option<&str>,
    exa_base_url: &str,
    query: &str,
    data_sources: Option<&[String]>,
    output_schema: Option<&serde_json::Value>,
    effort: Option<&str>,
    max_observation_chars: usize,
    timeout_sec: u64,
    call_counter: &Arc<AtomicU32>,
    max_calls: u32,
) -> ToolResult {
    let query = query.trim();
    if query.is_empty() {
        return ToolResult::error("exa_agent requires non-empty query".into());
    }

    let api_key = match exa_api_key {
        Some(k) if !k.trim().is_empty() => k,
        _ => return ToolResult::error("EXA_API_KEY not configured".into()),
    };

    // Reserve budget right before issuing the HTTP request — only calls that
    // actually proceed to the request count against `max_calls`.
    if let Err(current) = try_reserve_exa_agent_call(call_counter, max_calls) {
        return ToolResult::error(format!(
            "exa_agent budget exhausted ({current}/{max_calls}). Use web_search or local records."
        ));
    }

    let payload = build_exa_agent_payload(query, data_sources, output_schema, effort);
    let base = exa_base_url.trim_end_matches('/');
    let client = reqwest::Client::new();

    let create_resp = client
        .post(format!("{base}/agent/runs"))
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(30))
        .json(&payload)
        .send()
        .await;

    let resp = match create_resp {
        Ok(r) => r,
        Err(e) => return ToolResult::error(format!("exa_agent run creation failed: {e}")),
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return ToolResult::error(format!("exa_agent run creation failed ({status}): {body}"));
    }

    let mut run: serde_json::Value = match resp.json().await {
        Ok(b) => b,
        Err(e) => return ToolResult::error(format!("exa_agent response parse error: {e}")),
    };

    let run_id = match run.get("id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => return ToolResult::error(format!("exa_agent response missing id: {run}")),
    };

    // Poll GET /agent/runs/{id} with backoff until terminal or timeout.
    let deadline_sec = timeout_sec.min(600).max(1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(deadline_sec);
    let mut backoff_ms: u64 = 1000;

    loop {
        let status = run.get("status").and_then(|v| v.as_str()).unwrap_or("");
        if matches!(status, "completed" | "failed" | "cancelled") {
            break;
        }
        if std::time::Instant::now() >= deadline {
            return ToolResult::error(format!(
                "exa_agent run {run_id} did not reach a terminal status within {deadline_sec}s \
                 (last status: {status}). It may still complete server-side; check again later."
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms * 2).min(5000);

        let poll_resp = client
            .get(format!("{base}/agent/runs/{run_id}"))
            .header("Authorization", format!("Bearer {api_key}"))
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await;

        run = match poll_resp {
            Ok(r) if r.status().is_success() => match r.json().await {
                Ok(b) => b,
                Err(e) => return ToolResult::error(format!("exa_agent poll parse error: {e}")),
            },
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                return ToolResult::error(format!("exa_agent poll failed ({status}): {body}"));
            }
            Err(e) => return ToolResult::error(format!("exa_agent poll failed: {e}")),
        };
    }

    let status = run.get("status").and_then(|v| v.as_str()).unwrap_or("");
    if status != "completed" {
        let reason = run.get("stopReason").and_then(|v| v.as_str()).unwrap_or("unknown");
        return ToolResult::error(format!("exa_agent run {run_id} ended with status={status} stopReason={reason}"));
    }

    ToolResult::ok(clip(&format_exa_agent_output(&run), max_observation_chars))
}

#[cfg(test)]
mod exa_agent_tests {
    use super::*;

    #[test]
    fn test_payload_minimal_query_only() {
        let payload = build_exa_agent_payload("find officers of Acme LLC", None, None, None);
        assert_eq!(payload["query"], "find officers of Acme LLC");
        assert!(payload.get("dataSources").is_none());
        assert!(payload.get("outputSchema").is_none());
        assert!(payload.get("effort").is_none());
    }

    #[test]
    fn test_payload_data_sources_wrapped_as_provider_objects() {
        let sources = vec!["baselayer".to_string(), "fiber".to_string()];
        let payload = build_exa_agent_payload("q", Some(&sources), None, None);
        assert_eq!(
            payload["dataSources"],
            json!([{ "provider": "baselayer" }, { "provider": "fiber" }])
        );
    }

    #[test]
    fn test_payload_empty_data_sources_omitted() {
        let sources: Vec<String> = vec![];
        let payload = build_exa_agent_payload("q", Some(&sources), None, None);
        assert!(payload.get("dataSources").is_none());
    }

    #[test]
    fn test_payload_output_schema_and_effort_passthrough() {
        let schema = json!({"type": "object", "required": ["companies"]});
        let payload = build_exa_agent_payload("q", None, Some(&schema), Some("high"));
        assert_eq!(payload["outputSchema"], schema);
        assert_eq!(payload["effort"], "high");
    }

    #[test]
    fn test_payload_blank_effort_omitted() {
        let payload = build_exa_agent_payload("q", None, None, Some("   "));
        assert!(payload.get("effort").is_none());
    }

    #[test]
    fn test_format_output_text_only_no_grounding() {
        let run = json!({
            "output": { "text": "Acme LLC has 3 officers.", "structured": null, "grounding": [] }
        });
        let out = format_exa_agent_output(&run);
        assert_eq!(out, "Acme LLC has 3 officers.");
    }

    #[test]
    fn test_format_output_includes_deduped_citation_urls() {
        let run = json!({
            "output": {
                "text": "Acme LLC has 3 officers.",
                "structured": null,
                "grounding": [
                    { "field": "text", "citations": [{"url": "https://sos.ca.gov/a", "title": "A"}] },
                    { "field": "text", "citations": [
                        {"url": "https://sos.ca.gov/a", "title": "A dup"},
                        {"url": "https://sos.ca.gov/b", "title": "B"}
                    ] }
                ]
            }
        });
        let out = format_exa_agent_output(&run);
        assert!(out.contains("Acme LLC has 3 officers."));
        assert!(out.contains("Sources:"));
        assert!(out.contains("- https://sos.ca.gov/a"));
        assert!(out.contains("- https://sos.ca.gov/b"));
        // deduped: URL appears exactly once even though cited twice
        assert_eq!(out.matches("https://sos.ca.gov/a").count(), 1);
    }

    #[tokio::test]
    async fn test_exa_agent_requires_api_key() {
        let counter = Arc::new(AtomicU32::new(0));
        let result = exa_agent(None, "https://api.exa.ai", "query", None, None, None, 6000, 45, &counter, 12).await;
        assert!(result.is_error);
        assert!(result.content.contains("EXA_API_KEY not configured"));
        assert_eq!(counter.load(Ordering::SeqCst), 0, "no HTTP request issued, budget must not be spent");
    }

    #[tokio::test]
    async fn test_exa_agent_requires_nonempty_query() {
        let counter = Arc::new(AtomicU32::new(0));
        let result = exa_agent(Some("key"), "https://api.exa.ai", "   ", None, None, None, 6000, 45, &counter, 12).await;
        assert!(result.is_error);
        assert!(result.content.contains("non-empty query"));
        assert_eq!(counter.load(Ordering::SeqCst), 0, "no HTTP request issued, budget must not be spent");
    }

    // ── exa_agent call cap ──

    #[test]
    fn test_try_reserve_exa_agent_call_allows_up_to_max() {
        let counter = AtomicU32::new(0);
        assert_eq!(try_reserve_exa_agent_call(&counter, 2), Ok(1));
        assert_eq!(try_reserve_exa_agent_call(&counter, 2), Ok(2));
        assert_eq!(try_reserve_exa_agent_call(&counter, 2), Err(2));
        // Counter is left at the cap, not incremented past it, after a
        // rejected reservation.
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_exa_agent_errors_at_cap_without_issuing_request() {
        let counter = Arc::new(AtomicU32::new(3));
        let result = exa_agent(
            Some("key"), "https://api.exa.ai", "find officers", None, None, None, 6000, 45, &counter, 3,
        )
        .await;
        assert!(result.is_error);
        assert!(
            result.content.contains("exa_agent budget exhausted (3/3)"),
            "got: {}",
            result.content
        );
        assert!(result.content.contains("Use web_search or local records."));
        // Rejected reservation must not perturb the counter.
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_exa_agent_counter_shared_between_parent_and_child_contexts() {
        // Mirrors how engine::solve and engine::subagent::run_child thread
        // the SAME Arc<AtomicU32> into every WorkspaceTools they create.
        let shared = Arc::new(AtomicU32::new(0));
        let parent_view = shared.clone();
        let child_view = shared.clone();

        assert_eq!(try_reserve_exa_agent_call(&parent_view, 5), Ok(1));
        assert_eq!(try_reserve_exa_agent_call(&child_view, 5), Ok(2));
        assert_eq!(try_reserve_exa_agent_call(&parent_view, 5), Ok(3));

        // All three views observe the same accumulated count.
        assert_eq!(shared.load(Ordering::SeqCst), 3);
        assert_eq!(parent_view.load(Ordering::SeqCst), 3);
        assert_eq!(child_view.load(Ordering::SeqCst), 3);
    }
}
