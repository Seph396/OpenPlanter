// Real recursive sub-agents: subtask / execute / list_artifacts / read_artifact.
//
// Mirrors `agent/engine.py::_apply_tool_call` for these four action names, with
// the differences called out below (all deliberate, disclosed simplifications):
//
// - Each recursion level gets its OWN `WorkspaceTools` instance rather than
//   sharing one mutable instance with its parent (Python shares `self.tools`
//   across every depth). This avoids a global lock around every filesystem/
//   shell tool call. The cost: the "must read_file before write_file" overwrite
//   guard and the background-job table are not shared across depths.
// - Children do not run the background wiki curator or context compaction —
//   those stay wired into the depth-0 loop in `engine::solve` only.
// - `acceptance_criteria` is accepted in the tool schema (for prompt-text
//   parity with `prompts::ACCEPTANCE_CRITERIA_SECTION`) but not evaluated by
//   an LLM judge in this pass — no PASS/FAIL verdict is appended.
// - Unlike the Python reference (which only ever exposes `subtask` to the
//   model — `execute`, `list_artifacts`, `read_artifact` are implemented but
//   never included in any tool_defs list actually handed to a model), this
//   Rust port exposes all four whenever `config.recursive` is true, per
//   explicit instruction in this task's brief.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use futures::future::{join_all, FutureExt};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::builder::{build_model, select_child_model_name};
use crate::config::AgentConfig;
use crate::events::{StepEvent, TokenUsage};
use crate::model::{BaseModel, Message, ToolCall};
use crate::tools::defs::{build_tool_defs_mode, ToolMode};
use crate::tools::WorkspaceTools;

use super::SolveEmitter;

/// Why a recursion level's loop terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    Final,
    Cancelled,
    ModelError,
    BudgetExhausted,
}

/// Outcome of one `run_child` invocation.
#[derive(Debug, Clone, Default)]
pub struct LoopResult {
    pub text: String,
    pub kind: LoopKind,
    /// Number of model-turn steps this level actually took (>=1 unless
    /// cancelled before the first turn).
    pub steps: u32,
    /// Count of each tool name invoked at this level (not recursing into
    /// grandchildren's own tool calls). Includes `subtask`/`execute` calls
    /// themselves, but not what those children did internally.
    pub tool_counts: BTreeMap<String, u32>,
    /// The model's text on the step immediately before the final one, when
    /// that step also carried tool calls (i.e. the model narrated an
    /// intention while still acting). `None` if there was no such text, or
    /// if the run ended on the very first step. Distinct from `text`, which
    /// is the actual final answer (or budget/cancel/error message).
    pub last_narration: Option<String>,
}

impl Default for LoopKind {
    fn default() -> Self {
        LoopKind::BudgetExhausted
    }
}

/// Immutable context shared by every recursion level of one `solve()` call.
pub struct RecursionCtx<'a> {
    pub config: &'a AgentConfig,
    pub emitter: &'a dyn SolveEmitter,
    pub model: &'a dyn BaseModel,
    pub provider: &'a str,
    pub system_prompt: &'a str,
    pub artifacts_dir: PathBuf,
}

/// Run one recursion level's full agentic step loop (its own conversation,
/// its own `WorkspaceTools`, its own step budget) and return the outcome.
///
/// `model_override`, when `Some`, is used in place of `ctx.model` for this
/// level (and is what gets threaded into any further delegation from this
/// level — see `spawn_delegation`). It is owned (not borrowed) specifically
/// so a freshly-built per-tier model can be handed down without fighting the
/// recursive `Pin<Box<dyn Future + Send + 'a>>` lifetime, which is anchored
/// to `ctx`'s lifetime, not to any one delegation call's stack frame.
///
/// Boxed because this function is (indirectly, via `spawn_delegation`) recursive.
pub fn run_child<'a>(
    ctx: &'a RecursionCtx<'a>,
    model_override: Option<Box<dyn BaseModel>>,
    objective: String,
    depth: u32,
    mode: ToolMode,
    cancel: CancellationToken,
) -> Pin<Box<dyn Future<Output = LoopResult> + Send + 'a>> {
    Box::pin(async move {
        let mut steps: u32 = 0;
        let mut tool_counts: BTreeMap<String, u32> = BTreeMap::new();
        let mut last_narration: Option<String> = None;

        if cancel.is_cancelled() {
            return LoopResult {
                text: "Task cancelled.".into(),
                kind: LoopKind::Cancelled,
                steps,
                tool_counts,
                last_narration,
            };
        }

        let model: &dyn BaseModel = model_override.as_deref().unwrap_or(ctx.model);
        let tool_defs = build_tool_defs_mode(ctx.provider, mode);
        let mut tools = WorkspaceTools::new(ctx.config);
        let mut messages = vec![
            Message::System {
                content: ctx.system_prompt.to_string(),
            },
            Message::User {
                content: objective.clone(),
            },
        ];
        let max_steps = ctx.config.max_steps_per_call.max(1) as usize;
        let noop_delta = |_: crate::events::DeltaEvent| {};

        for step in 1..=max_steps {
            if cancel.is_cancelled() {
                tools.cleanup();
                return LoopResult {
                    text: "Task cancelled.".into(),
                    kind: LoopKind::Cancelled,
                    steps,
                    tool_counts,
                    last_narration,
                };
            }

            let turn = match model
                .chat_stream(&messages, &tool_defs, &noop_delta, &cancel)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    tools.cleanup();
                    return LoopResult {
                        text: format!("Model error at depth {depth}, step {step}: {e}"),
                        kind: LoopKind::ModelError,
                        steps,
                        tool_counts,
                        last_narration,
                    };
                }
            };
            steps += 1;

            let tool_calls_opt = if turn.tool_calls.is_empty() {
                None
            } else {
                Some(turn.tool_calls.clone())
            };
            messages.push(Message::Assistant {
                content: turn.text.clone(),
                tool_calls: tool_calls_opt,
            });

            if turn.tool_calls.is_empty() {
                if !turn.text.is_empty() {
                    ctx.emitter.emit_step(StepEvent {
                        depth,
                        step: step as u32,
                        tool_name: None,
                        tokens: TokenUsage {
                            input_tokens: turn.input_tokens,
                            output_tokens: turn.output_tokens,
                            cache_creation_input_tokens: turn.cache_creation_input_tokens.unwrap_or(0),
                            cache_read_input_tokens: turn.cache_read_input_tokens.unwrap_or(0),
                        },
                        elapsed_ms: 0,
                        is_final: true,
                    });
                    tools.cleanup();
                    return LoopResult {
                        text: turn.text,
                        kind: LoopKind::Final,
                        steps,
                        tool_counts,
                        last_narration,
                    };
                }
                messages.push(Message::Tool {
                    tool_call_id: "empty".into(),
                    content: "No tool calls and no text in response. Please use a tool or provide a final answer.".into(),
                });
                continue;
            }

            for tc in &turn.tool_calls {
                *tool_counts.entry(tc.name.clone()).or_insert(0) += 1;
            }
            if !turn.text.is_empty() {
                last_narration = Some(turn.text.clone());
            }

            // Sequential tools run in place; subtask/execute fan out concurrently.
            let mut ordered: Vec<Option<(String, String)>> = vec![None; turn.tool_calls.len()];
            let mut parallel_idx: Vec<usize> = Vec::new();

            for (i, tc) in turn.tool_calls.iter().enumerate() {
                if is_delegation_call(&tc.name) {
                    parallel_idx.push(i);
                } else {
                    let content = run_plain_tool(ctx, &mut tools, tc).await;
                    ordered[i] = Some((tc.id.clone(), content));
                }
            }

            if !parallel_idx.is_empty() {
                let parent_model_name = model.model_name().to_string();
                let futs: Vec<_> = parallel_idx
                    .iter()
                    .map(|&i| {
                        let tc = turn.tool_calls[i].clone();
                        spawn_delegation(ctx, parent_model_name.clone(), tc, depth, cancel.clone()).boxed()
                    })
                    .collect();
                let results = join_all(futs).await;
                for (k, &i) in parallel_idx.iter().enumerate() {
                    ordered[i] = Some((turn.tool_calls[i].id.clone(), results[k].clone()));
                }
            }

            let first_tool_name = turn.tool_calls.first().map(|tc| tc.name.clone());
            for entry in ordered.into_iter().flatten() {
                messages.push(Message::Tool {
                    tool_call_id: entry.0,
                    content: entry.1,
                });
            }

            ctx.emitter.emit_step(StepEvent {
                depth,
                step: step as u32,
                tool_name: first_tool_name,
                tokens: TokenUsage {
                    input_tokens: turn.input_tokens,
                    output_tokens: turn.output_tokens,
                    cache_creation_input_tokens: turn.cache_creation_input_tokens.unwrap_or(0),
                    cache_read_input_tokens: turn.cache_read_input_tokens.unwrap_or(0),
                },
                elapsed_ms: 0,
                is_final: false,
            });
        }

        tools.cleanup();
        LoopResult {
            text: format!(
                "Step budget exhausted at depth {depth} for objective: {objective}\n\
                 Please try with a more specific task, higher step budget, or deeper recursion."
            ),
            kind: LoopKind::BudgetExhausted,
            steps,
            tool_counts,
            last_narration,
        }
    })
}

fn is_delegation_call(name: &str) -> bool {
    name == "subtask" || name == "execute"
}

/// Dispatch a non-delegation tool call: filesystem/shell/web tools go through
/// `WorkspaceTools`; `list_artifacts`/`read_artifact` are handled here since
/// they need `artifacts_dir`, which `WorkspaceTools` doesn't know about.
pub async fn run_plain_tool(ctx: &RecursionCtx<'_>, tools: &mut WorkspaceTools, tc: &ToolCall) -> String {
    match tc.name.as_str() {
        "list_artifacts" => list_artifacts(&ctx.artifacts_dir),
        "read_artifact" => {
            let args: Value = serde_json::from_str(&tc.arguments).unwrap_or_default();
            let id = args.get("artifact_id").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            if id.is_empty() {
                return "read_artifact requires artifact_id".into();
            }
            let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
            read_artifact(&ctx.artifacts_dir, &id, offset, limit)
        }
        _ => tools.execute(&tc.name, &tc.arguments).await.content,
    }
}

/// Run a `subtask` or `execute` delegation call: validates depth/recursion
/// gating, resolves a per-tier child model (if `subtask_model`/`execute_model`
/// is configured), recurses into `run_child` at `depth + 1`, records an
/// artifact, and returns the formatted observation string.
///
/// `parent_model_name` is the effective model name of the *calling* level
/// (which may itself already be an override from an earlier delegation) —
/// it's what downward-only tier enforcement clamps against.
///
/// Boxed + recursive: this is what lets `subtask`/`execute` nest arbitrarily
/// deep (bounded by `config.max_depth`).
pub fn spawn_delegation<'a>(
    ctx: &'a RecursionCtx<'a>,
    parent_model_name: String,
    tc: ToolCall,
    depth: u32,
    cancel: CancellationToken,
) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(async move {
        let args: Value = serde_json::from_str(&tc.arguments).unwrap_or_default();
        let objective = args
            .get("objective")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if objective.is_empty() {
            return format!("{} requires objective", tc.name);
        }
        if !ctx.config.recursive {
            return format!(
                "{} tool not available in flat mode.",
                tc.name
            );
        }
        if depth >= ctx.config.max_depth.max(0) as u32 {
            return format!("Max recursion depth reached; cannot run {}.", tc.name);
        }

        let (mode, label) = if tc.name == "subtask" {
            (ToolMode::Recursive, "Subtask")
        } else {
            (ToolMode::ExecuteChild, "Execute")
        };
        let is_execute = tc.name == "execute";

        // Resolve a per-tier child model, if configured. Build failure (bad
        // model name/provider mismatch) is a static-config error, not a
        // runtime occurrence — disclosed simplification: on failure we fall
        // back to `None` here, which `run_child` resolves to `ctx.model`
        // (the ROOT model), not necessarily this level's actual parent model
        // if an earlier delegation already overrode it. Only reachable on a
        // build error.
        let model_override: Option<Box<dyn BaseModel>> =
            match select_child_model_name(ctx.config, is_execute, &parent_model_name) {
                Some(child_model_name) if child_model_name != parent_model_name => {
                    let mut child_cfg = ctx.config.clone();
                    child_cfg.model = child_model_name.clone();
                    child_cfg.provider = "auto".to_string();
                    match build_model(&child_cfg) {
                        Ok(m) => Some(m),
                        Err(e) => {
                            ctx.emitter.emit_trace(&format!(
                                "[d{depth}] failed to build per-tier model '{child_model_name}': {e}; using parent model"
                            ));
                            None
                        }
                    }
                }
                _ => None,
            };

        ctx.emitter
            .emit_trace(&format!("[d{depth}] >> {}: {}", tc.name, objective));

        let result = run_child(ctx, model_override, objective.clone(), depth + 1, mode, cancel).await;
        write_artifact(&ctx.artifacts_dir, depth + 1, &tc.name, &objective, &result);

        format!("{label} result for '{objective}':\n{}", result.text)
    })
}

// ---------------------------------------------------------------------
// Artifact storage — {workspace}/{session_root_dir}/artifacts/{id}.jsonl
// ---------------------------------------------------------------------

fn write_artifact(dir: &Path, depth: u32, kind: &str, objective: &str, result: &LoopResult) {
    let artifact_id = format!("d{depth}-{}", uuid::Uuid::new_v4().simple());
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    // `result` is the final answer text (kept under its original key name for
    // backward compatibility with existing artifact readers/tests). `steps`,
    // `tool_counts`, and `last_narration` are new — a reader parsing an older
    // artifact file simply won't see them; nothing here requires a Rust
    // struct with #[serde(default)] since these records are write-only JSON
    // values, not deserialized anywhere in this codebase (verified via grep).
    let record = serde_json::json!({
        "artifact_id": artifact_id,
        "kind": kind,
        "depth": depth,
        "objective": objective,
        "result": result.text,
        "steps": result.steps,
        "tool_counts": result.tool_counts,
        "last_narration": result.last_narration,
    });
    let path = dir.join(format!("{artifact_id}.jsonl"));
    let _ = std::fs::write(&path, format!("{}\n", record));
}

pub fn list_artifacts(dir: &Path) -> String {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return "No artifacts found.".into(),
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("jsonl"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return "No artifacts found.".into();
    }
    let mut lines = Vec::new();
    for p in &paths {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        match std::fs::read_to_string(p) {
            Ok(content) => match content.lines().next().map(serde_json::from_str::<Value>) {
                Some(Ok(v)) => {
                    let id = v.get("artifact_id").and_then(|x| x.as_str()).unwrap_or(&stem);
                    let objective = v.get("objective").and_then(|x| x.as_str()).unwrap_or("(no objective)");
                    let clipped = if objective.len() > 120 { &objective[..120] } else { objective };
                    lines.push(format!("- {id}: {clipped}"));
                }
                _ => lines.push(format!("- {stem}: (unreadable)")),
            },
            Err(_) => lines.push(format!("- {stem}: (unreadable)")),
        }
    }
    format!("Artifacts ({}):\n{}", lines.len(), lines.join("\n"))
}

pub fn read_artifact(dir: &Path, artifact_id: &str, offset: usize, limit: usize) -> String {
    // Guard against path traversal via a crafted artifact_id.
    if artifact_id.contains('/') || artifact_id.contains('\\') || artifact_id.contains("..") {
        return format!("Artifact '{artifact_id}' not found.");
    }
    let path = dir.join(format!("{artifact_id}.jsonl"));
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return format!("Artifact '{artifact_id}' not found."),
    };
    let all_lines: Vec<&str> = content.lines().collect();
    let total = all_lines.len();
    let end = (offset + limit).min(total);
    let selected = if offset < total { &all_lines[offset..end] } else { &[] };
    format!(
        "Artifact {artifact_id} (lines {offset}-{} of {total}):\n{}",
        offset + selected.len(),
        selected.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::DeltaEvent;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    struct NullEmitter;
    impl SolveEmitter for NullEmitter {
        fn emit_trace(&self, _message: &str) {}
        fn emit_delta(&self, _event: DeltaEvent) {}
        fn emit_step(&self, _event: StepEvent) {}
        fn emit_complete(&self, _result: &str) {}
        fn emit_error(&self, _message: &str) {}
    }

    /// A scripted mock model. Each call returns the next `ModelTurn` from
    /// `script` (cycling to the last entry once exhausted). Optionally sleeps
    /// before returning, to test concurrency.
    struct MockModel {
        script: Vec<crate::model::ModelTurn>,
        call_count: AtomicUsize,
        delay_ms: u64,
    }

    #[async_trait::async_trait]
    impl BaseModel for MockModel {
        async fn chat(&self, _messages: &[Message], _tools: &[Value]) -> anyhow::Result<crate::model::ModelTurn> {
            unreachable!("tests use chat_stream")
        }

        async fn chat_stream(
            &self,
            _messages: &[Message],
            _tools: &[Value],
            _on_delta: &(dyn Fn(DeltaEvent) + Send + Sync),
            _cancel: &CancellationToken,
        ) -> anyhow::Result<crate::model::ModelTurn> {
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let turn = self
                .script
                .get(idx)
                .cloned()
                .unwrap_or_else(|| self.script.last().cloned().unwrap());
            Ok(turn)
        }

        fn model_name(&self) -> &str {
            "mock-model"
        }

        fn provider_name(&self) -> &str {
            "openai"
        }
    }

    fn final_turn(text: &str) -> crate::model::ModelTurn {
        crate::model::ModelTurn {
            text: text.to_string(),
            ..Default::default()
        }
    }

    fn tool_call_turn(calls: Vec<(&str, Value)>) -> crate::model::ModelTurn {
        crate::model::ModelTurn {
            text: String::new(),
            tool_calls: calls
                .into_iter()
                .enumerate()
                .map(|(i, (name, args))| ToolCall {
                    id: format!("tc{i}"),
                    name: name.to_string(),
                    arguments: args.to_string(),
                })
                .collect(),
            ..Default::default()
        }
    }

    fn test_config(tmp: &Path) -> AgentConfig {
        let mut cfg = AgentConfig::default();
        cfg.workspace = tmp.to_path_buf();
        cfg.max_steps_per_call = 5;
        cfg.max_depth = 4;
        cfg.recursive = true;
        cfg
    }

    #[tokio::test]
    async fn test_depth_limit_error_at_max_depth() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_depth = 1;
        let model = MockModel {
            script: vec![final_turn("should not be reached")],
            call_count: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let emitter = NullEmitter;
        let ctx = RecursionCtx {
            config: &cfg,
            emitter: &emitter,
            model: &model,
            provider: "openai",
            system_prompt: "sys",
            artifacts_dir: tmp.path().join(".openplanter/artifacts"),
        };
        let tc = ToolCall {
            id: "t0".into(),
            name: "subtask".into(),
            arguments: serde_json::json!({"objective": "do a thing"}).to_string(),
        };
        // depth == max_depth (1) → must be rejected without calling the model.
        let observation = spawn_delegation(&ctx, "mock-model".into(), tc, 1, CancellationToken::new()).await;
        assert!(
            observation.contains("Max recursion depth reached; cannot run subtask."),
            "got: {observation}"
        );
        assert_eq!(model.call_count.load(Ordering::SeqCst), 0, "model must not be called past max_depth");
    }

    #[tokio::test]
    async fn test_two_subtasks_run_concurrently() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_config(tmp.path());
        // Each call sleeps 120ms then returns a final answer. If subtasks ran
        // sequentially this would take >= 240ms; concurrently, ~120-180ms.
        let model = MockModel {
            script: vec![
                tool_call_turn(vec![
                    ("subtask", serde_json::json!({"objective": "task A"})),
                    ("subtask", serde_json::json!({"objective": "task B"})),
                ]),
                final_turn("done A"),
                final_turn("done B"),
                final_turn("wrap up"),
            ],
            call_count: AtomicUsize::new(0),
            delay_ms: 120,
        };
        let emitter = NullEmitter;
        let ctx = RecursionCtx {
            config: &cfg,
            emitter: &emitter,
            model: &model,
            provider: "openai",
            system_prompt: "sys",
            artifacts_dir: tmp.path().join(".openplanter/artifacts"),
        };

        let start = std::time::Instant::now();
        let result = run_child(&ctx, None, "root objective".into(), 0, ToolMode::Recursive, CancellationToken::new()).await;
        let elapsed = start.elapsed();

        assert_eq!(result.kind, LoopKind::Final);
        // Timeline: root turn (120ms) -> two subtasks concurrently (120ms, not
        // 240ms, if they overlap) -> wrap-up turn (120ms) = ~360ms concurrent
        // vs. ~480ms if the two subtasks ran sequentially. Assert well below
        // the sequential figure, with margin for scheduler jitter.
        assert!(
            elapsed < std::time::Duration::from_millis(450),
            "expected concurrent subtasks (~360ms), got {elapsed:?} (sequential would be ~480ms)"
        );
    }

    #[tokio::test]
    async fn test_execute_child_tool_set_excludes_delegation() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_config(tmp.path());
        let seen_tools: Arc<Mutex<Vec<Vec<Value>>>> = Arc::new(Mutex::new(Vec::new()));

        struct RecordingModel {
            seen: Arc<Mutex<Vec<Vec<Value>>>>,
        }
        #[async_trait::async_trait]
        impl BaseModel for RecordingModel {
            async fn chat(&self, _m: &[Message], _t: &[Value]) -> anyhow::Result<crate::model::ModelTurn> {
                unreachable!()
            }
            async fn chat_stream(
                &self,
                _messages: &[Message],
                tools: &[Value],
                _on_delta: &(dyn Fn(DeltaEvent) + Send + Sync),
                _cancel: &CancellationToken,
            ) -> anyhow::Result<crate::model::ModelTurn> {
                self.seen.lock().unwrap().push(tools.to_vec());
                Ok(final_turn("leaf done"))
            }
            fn model_name(&self) -> &str {
                "mock"
            }
            fn provider_name(&self) -> &str {
                "openai"
            }
        }

        let model = RecordingModel { seen: seen_tools.clone() };
        let emitter = NullEmitter;
        let ctx = RecursionCtx {
            config: &cfg,
            emitter: &emitter,
            model: &model,
            provider: "openai",
            system_prompt: "sys",
            artifacts_dir: tmp.path().join(".openplanter/artifacts"),
        };
        let tc = ToolCall {
            id: "t0".into(),
            name: "execute".into(),
            arguments: serde_json::json!({"objective": "leaf work"}).to_string(),
        };
        let observation = spawn_delegation(&ctx, "mock".into(), tc, 0, CancellationToken::new()).await;
        assert!(observation.contains("leaf done"));

        let seen = seen_tools.lock().unwrap();
        assert_eq!(seen.len(), 1);
        let names: Vec<String> = seen[0]
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_string())
            .collect();
        for delegation in ["subtask", "execute", "list_artifacts", "read_artifact"] {
            assert!(!names.contains(&delegation.to_string()), "execute child saw {delegation}");
        }
    }

    #[test]
    fn test_list_and_read_artifact_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("artifacts");
        let result = LoopResult {
            text: "the thing was found".into(),
            kind: LoopKind::Final,
            steps: 3,
            tool_counts: BTreeMap::from([("read_file".to_string(), 2)]),
            last_narration: Some("Reading the file now.".into()),
        };
        write_artifact(&dir, 1, "subtask", "find the thing", &result);

        let listing = list_artifacts(&dir);
        assert!(listing.starts_with("Artifacts (1):"));
        assert!(listing.contains("find the thing"));

        // Extract the id from the listing line: "- d1-XXXX: find the thing"
        let id = listing
            .lines()
            .nth(1)
            .unwrap()
            .trim_start_matches("- ")
            .split(':')
            .next()
            .unwrap()
            .to_string();

        let read = read_artifact(&dir, &id, 0, 10);
        assert!(read.contains("the thing was found"));
    }

    #[test]
    fn test_write_artifact_records_steps_tool_counts_and_narration() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("artifacts");
        let result = LoopResult {
            text: "final answer text".into(),
            kind: LoopKind::Final,
            steps: 4,
            tool_counts: BTreeMap::from([
                ("read_file".to_string(), 2),
                ("write_file".to_string(), 1),
            ]),
            last_narration: Some("Now I have sufficient evidence. Let me write the files.".into()),
        };
        write_artifact(&dir, 1, "subtask", "obj", &result);

        let mut entries = std::fs::read_dir(&dir).unwrap();
        let path = entries.next().unwrap().unwrap().path();
        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_json::from_str(content.trim()).unwrap();

        assert_eq!(parsed["result"], "final answer text");
        assert_eq!(parsed["steps"], 4);
        assert_eq!(parsed["tool_counts"]["read_file"], 2);
        assert_eq!(parsed["tool_counts"]["write_file"], 1);
        assert_eq!(
            parsed["last_narration"],
            "Now I have sufficient evidence. Let me write the files."
        );
    }

    #[tokio::test]
    async fn test_run_child_tracks_steps_and_tool_counts() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = test_config(tmp.path());
        let model = MockModel {
            script: vec![
                tool_call_turn(vec![("read_file", serde_json::json!({"path": "a.txt"}))]),
                tool_call_turn(vec![("read_file", serde_json::json!({"path": "b.txt"}))]),
                final_turn("done"),
            ],
            call_count: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let emitter = NullEmitter;
        let ctx = RecursionCtx {
            config: &cfg,
            emitter: &emitter,
            model: &model,
            provider: "openai",
            system_prompt: "sys",
            artifacts_dir: tmp.path().join(".openplanter/artifacts"),
        };
        let result = run_child(&ctx, None, "objective".into(), 0, ToolMode::Recursive, CancellationToken::new()).await;
        assert_eq!(result.kind, LoopKind::Final);
        assert_eq!(result.text, "done");
        assert_eq!(result.steps, 3);
        assert_eq!(result.tool_counts.get("read_file"), Some(&2));
    }

    #[test]
    fn test_read_artifact_rejects_path_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("artifacts");
        let out = read_artifact(&dir, "../../etc/passwd", 0, 10);
        assert!(out.contains("not found"));
    }

    #[tokio::test]
    async fn test_subtask_model_build_failure_falls_back_to_parent_model() {
        // Wiring test: cfg.subtask_model is configured but no anthropic API key
        // is present, so `build_model` fails inside `spawn_delegation`. This
        // must fall back gracefully to the parent's (mock) model rather than
        // propagating the build error — never attempts a real network call,
        // since `build_model` fails before any HTTP client is used.
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.subtask_model = Some("claude-haiku-4-5".into());
        cfg.anthropic_api_key = None; // force build_model to fail
        let model = MockModel {
            script: vec![final_turn("handled by parent mock model")],
            call_count: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let emitter = NullEmitter;
        let ctx = RecursionCtx {
            config: &cfg,
            emitter: &emitter,
            model: &model,
            provider: "openai",
            system_prompt: "sys",
            artifacts_dir: tmp.path().join(".openplanter/artifacts"),
        };
        let tc = ToolCall {
            id: "t0".into(),
            name: "subtask".into(),
            arguments: serde_json::json!({"objective": "do a thing"}).to_string(),
        };
        let observation = spawn_delegation(&ctx, "gpt-5.2".into(), tc, 0, CancellationToken::new()).await;
        assert!(
            observation.contains("handled by parent mock model"),
            "expected fallback to parent model's result, got: {observation}"
        );
        assert_eq!(model.call_count.load(Ordering::SeqCst), 1, "parent mock model should have been used");
    }

    #[tokio::test]
    async fn test_recursive_false_rejects_subtask() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.recursive = false;
        let model = MockModel {
            script: vec![final_turn("unused")],
            call_count: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let emitter = NullEmitter;
        let ctx = RecursionCtx {
            config: &cfg,
            emitter: &emitter,
            model: &model,
            provider: "openai",
            system_prompt: "sys",
            artifacts_dir: tmp.path().join(".openplanter/artifacts"),
        };
        let tc = ToolCall {
            id: "t0".into(),
            name: "subtask".into(),
            arguments: serde_json::json!({"objective": "x"}).to_string(),
        };
        let observation = spawn_delegation(&ctx, "mock".into(), tc, 0, CancellationToken::new()).await;
        assert!(observation.contains("not available in flat mode"));
        assert_eq!(model.call_count.load(Ordering::SeqCst), 0);
    }
}
