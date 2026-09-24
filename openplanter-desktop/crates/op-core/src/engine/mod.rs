// Recursive language model engine.
//
// Provides the SolveEmitter trait, demo_solve, and a real solve flow
// with a multi-step agentic loop that executes tool calls.

pub mod context;
pub mod curator;
pub mod judge;
pub mod subagent;

use futures::future::FutureExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::builder::build_model;
use crate::config::AgentConfig;
use crate::events::{DeltaEvent, DeltaKind, StepEvent, TokenUsage};
use crate::model::Message;
use crate::prompts::build_system_prompt;
use crate::tools::defs::{build_tool_defs_mode, ToolMode};
use crate::tools::WorkspaceTools;

use self::curator::{extract_step_context, run_curator, CuratorResult};
use self::subagent::{run_plain_tool, spawn_delegation, RecursionCtx};

/// Outcome from a background curator task (success or error).
enum CuratorOutcome {
    Done(CuratorResult),
    Error(String),
}

/// Abort all in-flight curator tasks.
fn abort_curators(handles: &mut Vec<JoinHandle<()>>) {
    for h in handles.drain(..) {
        h.abort();
    }
}

/// Drain completed curator results from the channel, inject system messages
/// and emit events for any that changed files.
fn drain_curator_results(
    rx: &mut mpsc::UnboundedReceiver<CuratorOutcome>,
    messages: &mut Vec<Message>,
    emitter: &dyn SolveEmitter,
) {
    while let Ok(outcome) = rx.try_recv() {
        match outcome {
            CuratorOutcome::Done(result) => {
                if result.files_changed > 0 {
                    emitter.emit_trace(&format!(
                        "[curator] wiki updated: {} ({} files)",
                        result.summary, result.files_changed
                    ));
                    messages.push(Message::System {
                        content: format!("[Wiki Curator] {}", result.summary),
                    });
                    emitter.emit_curator_update(&result.summary, result.files_changed);
                }
            }
            CuratorOutcome::Error(e) => {
                emitter.emit_trace(&format!("[curator] error: {e}"));
            }
        }
    }
}

/// Wait for in-flight curators (up to timeout), drain final results, abort rest.
async fn finish_curators(
    handles: &mut Vec<JoinHandle<()>>,
    rx: &mut mpsc::UnboundedReceiver<CuratorOutcome>,
    messages: &mut Vec<Message>,
    emitter: &dyn SolveEmitter,
) {
    if handles.is_empty() {
        return;
    }
    emitter.emit_trace(&format!(
        "[curator] waiting for {} in-flight curator(s)...",
        handles.len()
    ));

    // Wait up to 30 seconds total for all curators to finish
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    for h in handles.iter_mut() {
        let remaining = deadline - tokio::time::Instant::now();
        if remaining.is_zero() {
            break;
        }
        let _ = tokio::time::timeout(remaining, h).await;
    }

    // Final drain
    drain_curator_results(rx, messages, emitter);

    // Abort any still running
    abort_curators(handles);
}

// Abstraction for emitting solve events.
//
// Implemented by TauriEmitter (op-tauri) for real event emission
// and by TestEmitter (tests) for deterministic verification.
pub trait SolveEmitter: Send + Sync {
    fn emit_trace(&self, message: &str);
    fn emit_delta(&self, event: DeltaEvent);
    fn emit_step(&self, event: StepEvent);
    fn emit_complete(&self, result: &str);
    fn emit_error(&self, message: &str);
    /// Called when a background curator finishes updating wiki files.
    /// Default no-op — override in TauriEmitter/LoggingEmitter.
    fn emit_curator_update(&self, _summary: &str, _files_changed: u32) {}
}

// Demo solve flow that echoes the objective with simulated streaming.
//
// This is a placeholder until the full engine is implemented in Phase 4.
// It emits the standard event sequence so the frontend can be developed
// and tested against a working backend.
pub async fn demo_solve(
    objective: &str,
    emitter: &dyn SolveEmitter,
    cancel: CancellationToken,
) {
    emitter.emit_trace(&format!("Solving: {objective}"));

    if cancel.is_cancelled() {
        emitter.emit_error("Cancelled");
        return;
    }

    // Simulate thinking
    emitter.emit_delta(DeltaEvent {
        kind: DeltaKind::Thinking,
        text: format!("Analyzing: {objective}"),
    });

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    if cancel.is_cancelled() {
        emitter.emit_error("Cancelled");
        return;
    }

    // Simulate streaming text response
    let response = format!("Demo response for: {objective}");
    for chunk in response.as_bytes().chunks(20) {
        if cancel.is_cancelled() {
            emitter.emit_error("Cancelled");
            return;
        }
        let text = String::from_utf8_lossy(chunk).to_string();
        emitter.emit_delta(DeltaEvent {
            kind: DeltaKind::Text,
            text,
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // Emit step summary
    emitter.emit_step(StepEvent {
        depth: 0,
        step: 1,
        tool_name: None,
        tokens: TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        },
        elapsed_ms: 350,
        is_final: true,
    });

    emitter.emit_complete(&response);
}

/// One-time nudge sent when a model turn has no tool calls and non-empty text
/// but does not end in the `DONE` completion marker (see
/// `prompts::COMPLETION_PROTOCOL_SECTION`). Deliberate deviation from the
/// Python reference (which treats any no-tool-call + text turn as final
/// unconditionally) — a live run showed a `subtask` child narrating "Now I
/// have sufficient evidence. Let me write the two deliverable files." and
/// then stopping without writing anything, forcing the parent to relaunch it.
pub(crate) const TEXT_ONLY_NUDGE: &str = "You replied without a tool call and without DONE. \
If the work is finished, reply with your final answer ending in DONE. Otherwise continue with \
the next tool call.";

/// If the last non-empty line of `text` is exactly `DONE` (case-insensitive,
/// trailing punctuation such as `.`/`!` allowed), returns `text` with that
/// trailing line removed. Returns `None` otherwise.
pub(crate) fn strip_trailing_done(text: &str) -> Option<String> {
    let trimmed_end = text.trim_end();
    if trimmed_end.is_empty() {
        return None;
    }
    let (before, last_line) = match trimmed_end.rfind('\n') {
        Some(idx) => (&trimmed_end[..idx], &trimmed_end[idx + 1..]),
        None => ("", trimmed_end),
    };
    let candidate = last_line
        .trim()
        .trim_end_matches(|c: char| c.is_ascii_punctuation());
    if candidate.eq_ignore_ascii_case("done") {
        Some(before.trim_end().to_string())
    } else {
        None
    }
}

/// Observation injected in place of a truncated turn's tool call result(s)
/// (or as a plain nudge, if there were no tool calls) when a model turn ends
/// with `ModelTurn::truncated == true` — Anthropic `stop_reason ==
/// "max_tokens"`, or OpenAI-shaped `finish_reason == "length"`.
///
/// A live run showed a depth-0 step end with `step_tokens_out == 16384`
/// (the old hard-coded `max_tokens`) mid-`write_file` tool call: the call's
/// JSON arguments were truncated, the file never landed, and the run ended
/// silently. Any tool call(s) on a truncated turn must NOT be executed
/// as-is — their `arguments` may be incomplete/corrupt JSON — so this is
/// returned as the tool result observation instead, and the loop continues
/// (this counts as one step against the budget, same as any other turn).
pub(crate) fn truncated_turn_message(max_output_tokens: u64) -> String {
    format!(
        "Your previous output was cut off at the max_tokens limit ({max_output_tokens}). \
         Write large files in chunks: write_file the first part, then edit_file/append the \
         rest. Keep each tool call under ~{} tokens.",
        max_output_tokens / 2
    )
}

/// Decide whether a text-only (no tool calls) turn is final, given whether
/// the one-time completion nudge has already been sent this loop.
///
/// Returns `Some(final_text)` when the turn should be treated as final:
/// either the text ends in a `DONE` marker (stripped from the returned text),
/// or a nudge was already sent once this loop (in which case `text` is
/// returned unchanged — see `TEXT_ONLY_NUDGE`'s "or if the work is finished"
/// framing). Returns `None` when the caller should send `TEXT_ONLY_NUDGE` and
/// continue the loop instead.
pub(crate) fn resolve_text_only_turn(text: &str, already_nudged: bool) -> Option<String> {
    if let Some(stripped) = strip_trailing_done(text) {
        Some(stripped)
    } else if already_nudged {
        Some(text.to_string())
    } else {
        None
    }
}

/// Ensure every `tool_use`/tool-call block on an assistant turn has a
/// matching tool-result message before the next non-tool message. Providers
/// (Anthropic in particular) reject a request where a `tool_use` has no
/// matching `tool_result` — a session can end up in this state if a turn's
/// tool-result never got appended (e.g. the truncation guard didn't fire on
/// an older run, or a session was resumed mid-poisoned-state). Inserts a
/// synthetic tool-result so the session can continue instead of erroring on
/// every subsequent turn.
///
/// Disclosed simplification: the synthetic result is plain text content, not
/// a `tool_result` block with `is_error: true` — `Message::Tool` has no
/// `is_error` field, and adding one would touch every construction site of
/// this enum variant across the codebase. The content text alone is enough
/// to stop the 400; the model still sees the call as unsuccessful (see the
/// wording below).
pub(crate) fn sanitize_orphaned_tool_calls(messages: &mut Vec<Message>) {
    let mut i = 0;
    while i < messages.len() {
        let tool_call_ids: Option<Vec<String>> = match &messages[i] {
            Message::Assistant {
                tool_calls: Some(tcs),
                ..
            } if !tcs.is_empty() => Some(tcs.iter().map(|tc| tc.id.clone()).collect()),
            _ => None,
        };
        let Some(tool_call_ids) = tool_call_ids else {
            i += 1;
            continue;
        };

        // Tool-result messages immediately following this assistant turn.
        let mut answered: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut j = i + 1;
        while j < messages.len() {
            match &messages[j] {
                Message::Tool { tool_call_id, .. } => {
                    answered.insert(tool_call_id.clone());
                    j += 1;
                }
                _ => break,
            }
        }

        let mut insert_pos = j;
        for id in &tool_call_ids {
            if !answered.contains(id) {
                messages.insert(
                    insert_pos,
                    Message::Tool {
                        tool_call_id: id.clone(),
                        content: "tool call was truncated".to_string(),
                    },
                );
                insert_pos += 1;
            }
        }
        i = insert_pos;
    }
}

/// Rough token estimate: ~4 chars per token.
fn estimate_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| match m {
            Message::System { content } | Message::User { content } => content.len(),
            Message::Assistant { content, tool_calls } => {
                content.len()
                    + tool_calls
                        .as_ref()
                        .map(|tcs| tcs.iter().map(|tc| tc.arguments.len() + tc.name.len()).sum())
                        .unwrap_or(0)
            }
            Message::Tool { content, .. } => content.len(),
        })
        .sum::<usize>()
        / 4
}

/// Compact conversation context when it grows too large.
///
/// Keeps the system prompt, user objective, and the most recent messages
/// intact. Truncates older Tool result content to a short placeholder.
fn compact_messages(messages: &mut Vec<Message>, max_tokens: usize) {
    if estimate_tokens(messages) <= max_tokens {
        return;
    }

    // Keep the first 2 messages (System + User) and the last `keep_recent`
    // messages intact. Truncate Tool content in between.
    let keep_recent = 10; // Keep last ~10 messages (a few steps worth)
    let protected_tail = messages.len().saturating_sub(keep_recent);

    for i in 2..protected_tail {
        if let Message::Tool { content, .. } = &mut messages[i] {
            if content.len() > 200 {
                let preview = &content[..content.len().min(150)];
                *content = format!("{preview}\n...[truncated — older tool result]");
            }
        }
    }
}

/// Real solve flow with a multi-step agentic loop.
///
/// Calls the model with tool definitions. If the model returns tool calls,
/// executes them, appends results, and loops until the model returns a
/// final text answer or the step budget is exhausted.
///
/// Falls back to demo_solve when `config.demo` is true.
pub async fn solve(
    objective: &str,
    config: &AgentConfig,
    emitter: &dyn SolveEmitter,
    cancel: CancellationToken,
) {
    if config.demo {
        return demo_solve(objective, emitter, cancel).await;
    }

    // 1. Build model
    let model = match build_model(config) {
        Ok(m) => m,
        Err(e) => {
            emitter.emit_error(&e.to_string());
            return;
        }
    };

    let provider = model.provider_name().to_string();
    emitter.emit_trace(&format!(
        "Solving with {}/{}",
        provider,
        model.model_name()
    ));

    // 2. Build tools and messages
    let tool_mode = if config.recursive { ToolMode::Recursive } else { ToolMode::Flat };
    let tool_defs = build_tool_defs_mode(&provider, tool_mode);
    // Shared across depth-0 and every subtask/execute child (see RecursionCtx
    // and WorkspaceTools::new) so the exa_agent budget applies to the whole run.
    let exa_call_counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let mut tools = WorkspaceTools::new(config, exa_call_counter.clone());

    let system_prompt = build_system_prompt(
        config.recursive,
        config.acceptance_criteria,
        config.demo,
    );
    let artifacts_dir = config.workspace.join(&config.session_root_dir).join("artifacts");
    let rctx = RecursionCtx {
        config,
        emitter,
        model: model.as_ref(),
        provider: &provider,
        system_prompt: &system_prompt,
        artifacts_dir,
        exa_call_counter,
    };
    let mut messages = vec![
        Message::System {
            content: system_prompt.clone(),
        },
        Message::User {
            content: objective.to_string(),
        },
    ];

    let max_steps = config.max_steps_per_call as usize;

    // 3. Background curator channel
    let (curator_tx, mut curator_rx) = mpsc::unbounded_channel::<CuratorOutcome>();
    let mut curator_handles: Vec<JoinHandle<()>> = Vec::new();

    // Has the one-time TEXT_ONLY_NUDGE already been sent for this loop?
    let mut text_only_nudged = false;

    // 4. Agentic loop
    for step in 1..=max_steps {
        if cancel.is_cancelled() {
            emitter.emit_error("Cancelled");
            tools.cleanup();
            abort_curators(&mut curator_handles);
            return;
        }

        // Drain completed curator results and inject as system messages
        drain_curator_results(&mut curator_rx, &mut messages, emitter);

        let step_start = std::time::Instant::now();

        // Compact context if it's grown too large (~100k token budget)
        compact_messages(&mut messages, 100_000);

        // Guard against a poisoned history (an orphaned tool_use with no
        // matching tool_result) 400ing every subsequent turn.
        sanitize_orphaned_tool_calls(&mut messages);

        // Call model with streaming
        let turn = match model
            .chat_stream(&messages, &tool_defs, &|delta| emitter.emit_delta(delta), &cancel)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                tools.cleanup();
                abort_curators(&mut curator_handles);
                if msg == "Cancelled" {
                    emitter.emit_error("Cancelled");
                } else {
                    emitter.emit_error(&msg);
                }
                return;
            }
        };

        // Append assistant message to conversation
        let tool_calls_opt = if turn.tool_calls.is_empty() {
            None
        } else {
            Some(turn.tool_calls.clone())
        };
        messages.push(Message::Assistant {
            content: turn.text.clone(),
            tool_calls: tool_calls_opt,
        });

        // Turn was cut off at the provider's output token limit. Any tool
        // call(s) may have incomplete/corrupt JSON arguments — do not execute
        // them. Inject the cut-off observation and continue instead of
        // falling into the normal tool-execution / text-only branches below.
        if turn.truncated {
            emitter.emit_step(StepEvent {
                depth: 0,
                step: step as u32,
                tool_name: turn.tool_calls.first().map(|tc| tc.name.clone()),
                tokens: TokenUsage {
                    input_tokens: turn.input_tokens,
                    output_tokens: turn.output_tokens,
                    cache_creation_input_tokens: turn.cache_creation_input_tokens.unwrap_or(0),
                    cache_read_input_tokens: turn.cache_read_input_tokens.unwrap_or(0),
                },
                elapsed_ms: step_start.elapsed().as_millis() as u64,
                is_final: false,
            });
            let notice = truncated_turn_message(config.max_output_tokens);
            if turn.tool_calls.is_empty() {
                messages.push(Message::User { content: notice });
            } else {
                for tc in &turn.tool_calls {
                    messages.push(Message::Tool {
                        tool_call_id: tc.id.clone(),
                        content: notice.clone(),
                    });
                }
            }
            continue;
        }

        // No tool calls + text present: matches agent/engine.py::_solve_recursive
        // ("No tool calls + text present = final answer", engine.py:442-463),
        // except gated by the DONE completion protocol — a deliberate deviation
        // from Python (which treats this unconditionally as final). See
        // TEXT_ONLY_NUDGE for why.
        if turn.tool_calls.is_empty() && !turn.text.is_empty() {
            match resolve_text_only_turn(&turn.text, text_only_nudged) {
                Some(final_text) => {
                    emitter.emit_step(StepEvent {
                        depth: 0,
                        step: step as u32,
                        tool_name: None,
                        tokens: TokenUsage {
                            input_tokens: turn.input_tokens,
                            output_tokens: turn.output_tokens,
                            cache_creation_input_tokens: turn.cache_creation_input_tokens.unwrap_or(0),
                            cache_read_input_tokens: turn.cache_read_input_tokens.unwrap_or(0),
                        },
                        elapsed_ms: step_start.elapsed().as_millis() as u64,
                        is_final: true,
                    });
                    emitter.emit_complete(&final_text);
                    tools.cleanup();
                    // Wait for in-flight curators before exiting
                    finish_curators(&mut curator_handles, &mut curator_rx, &mut messages, emitter).await;
                    return;
                }
                None => {
                    text_only_nudged = true;
                    emitter.emit_step(StepEvent {
                        depth: 0,
                        step: step as u32,
                        tool_name: None,
                        tokens: TokenUsage {
                            input_tokens: turn.input_tokens,
                            output_tokens: turn.output_tokens,
                            cache_creation_input_tokens: turn.cache_creation_input_tokens.unwrap_or(0),
                            cache_read_input_tokens: turn.cache_read_input_tokens.unwrap_or(0),
                        },
                        elapsed_ms: step_start.elapsed().as_millis() as u64,
                        is_final: false,
                    });
                    messages.push(Message::User {
                        content: TEXT_ONLY_NUDGE.into(),
                    });
                    continue;
                }
            }
        }

        // No tool calls AND no text = unexpected empty response. Python
        // (engine.py:465-474) and the child loop (subagent.rs) both nudge
        // here rather than ending the run; depth-0 previously fell into the
        // "no tool calls" branch above and returned an empty final answer,
        // silently ending a run on a transient/malformed empty turn. Nudge
        // and continue instead, matching both references.
        if turn.tool_calls.is_empty() {
            emitter.emit_step(StepEvent {
                depth: 0,
                step: step as u32,
                tool_name: None,
                tokens: TokenUsage {
                    input_tokens: turn.input_tokens,
                    output_tokens: turn.output_tokens,
                    cache_creation_input_tokens: turn.cache_creation_input_tokens.unwrap_or(0),
                    cache_read_input_tokens: turn.cache_read_input_tokens.unwrap_or(0),
                },
                elapsed_ms: step_start.elapsed().as_millis() as u64,
                is_final: false,
            });
            messages.push(Message::Tool {
                tool_call_id: "empty".into(),
                content: "No tool calls and no text in response. Please use a tool or provide a final answer.".into(),
            });
            continue;
        }

        // Execute tool calls: subtask/execute fan out concurrently (real
        // recursive sub-agents); everything else runs sequentially in order.
        if cancel.is_cancelled() {
            emitter.emit_error("Cancelled");
            tools.cleanup();
            abort_curators(&mut curator_handles);
            return;
        }

        let mut ordered: Vec<Option<(String, String)>> = vec![None; turn.tool_calls.len()];
        let mut parallel_idx: Vec<usize> = Vec::new();

        for (i, tc) in turn.tool_calls.iter().enumerate() {
            if tc.name == "subtask" || tc.name == "execute" {
                parallel_idx.push(i);
                continue;
            }
            emitter.emit_trace(&format!("Executing tool: {} ({})", tc.name, tc.id));
            let content = run_plain_tool(&rctx, &mut tools, tc).await;
            ordered[i] = Some((tc.id.clone(), content));
        }

        if !parallel_idx.is_empty() {
            let parent_model_name = model.model_name().to_string();
            let futs: Vec<_> = parallel_idx
                .iter()
                .map(|&i| {
                    let tc = turn.tool_calls[i].clone();
                    emitter.emit_trace(&format!("Executing tool: {} ({})", tc.name, tc.id));
                    spawn_delegation(&rctx, parent_model_name.clone(), tc, 0, cancel.clone()).boxed()
                })
                .collect();
            let results = futures::future::join_all(futs).await;
            for (k, &i) in parallel_idx.iter().enumerate() {
                ordered[i] = Some((turn.tool_calls[i].id.clone(), results[k].clone()));
            }
        }

        for entry in ordered.into_iter().flatten() {
            messages.push(Message::Tool {
                tool_call_id: entry.0,
                content: entry.1,
            });
        }

        // Emit step (non-final) AFTER tools execute so the frontend
        // can refresh the wiki graph with newly written files.
        let first_tool = turn.tool_calls.first().map(|tc| tc.name.clone());
        emitter.emit_step(StepEvent {
            depth: 0,
            step: step as u32,
            tool_name: first_tool,
            tokens: TokenUsage {
                input_tokens: turn.input_tokens,
                output_tokens: turn.output_tokens,
                cache_creation_input_tokens: turn.cache_creation_input_tokens.unwrap_or(0),
                cache_read_input_tokens: turn.cache_read_input_tokens.unwrap_or(0),
            },
            elapsed_ms: step_start.elapsed().as_millis() as u64,
            is_final: false,
        });

        // Spawn background curator after each non-final step
        let context = extract_step_context(&messages);
        if !context.is_empty() {
            let tx = curator_tx.clone();
            let curator_cfg = config.clone();
            let curator_cancel = cancel.clone();
            emitter.emit_trace(&format!("[curator] spawning for step {step}"));
            curator_handles.push(tokio::spawn(async move {
                let outcome = match run_curator(&context, &curator_cfg, curator_cancel).await {
                    Ok(result) => CuratorOutcome::Done(result),
                    Err(e) => CuratorOutcome::Error(e),
                };
                let _ = tx.send(outcome);
            }));
        }

        // Budget warnings
        let remaining = max_steps - step;
        if remaining == max_steps / 2 {
            emitter.emit_trace(&format!(
                "Step budget: {remaining}/{max_steps} steps remaining (50%)"
            ));
        } else if remaining == max_steps / 4 {
            emitter.emit_trace(&format!(
                "Step budget: {remaining}/{max_steps} steps remaining (25%)"
            ));
        }
    }

    // Budget exhausted
    tools.cleanup();
    finish_curators(&mut curator_handles, &mut curator_rx, &mut messages, emitter).await;
    emitter.emit_error(&format!(
        "Step budget exhausted after {max_steps} steps. \
         The model did not produce a final answer within the allowed steps."
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ToolCall;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    enum RecordedEvent {
        Trace(String),
        Delta(DeltaEvent),
        Step(StepEvent),
        Complete(String),
        Error(String),
    }

    struct TestEmitter {
        events: Arc<Mutex<Vec<RecordedEvent>>>,
    }

    impl TestEmitter {
        fn new() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn events(&self) -> Vec<RecordedEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl SolveEmitter for TestEmitter {
        fn emit_trace(&self, message: &str) {
            self.events
                .lock()
                .unwrap()
                .push(RecordedEvent::Trace(message.to_string()));
        }

        fn emit_delta(&self, event: DeltaEvent) {
            self.events
                .lock()
                .unwrap()
                .push(RecordedEvent::Delta(event));
        }

        fn emit_step(&self, event: StepEvent) {
            self.events
                .lock()
                .unwrap()
                .push(RecordedEvent::Step(event));
        }

        fn emit_complete(&self, result: &str) {
            self.events
                .lock()
                .unwrap()
                .push(RecordedEvent::Complete(result.to_string()));
        }

        fn emit_error(&self, message: &str) {
            self.events
                .lock()
                .unwrap()
                .push(RecordedEvent::Error(message.to_string()));
        }
    }

    #[tokio::test]
    async fn test_demo_solve_emits_complete_sequence() {
        let emitter = TestEmitter::new();
        let token = CancellationToken::new();

        demo_solve("Test objective", &emitter, token).await;

        let events = emitter.events();
        assert!(events.len() >= 4, "expected at least 4 events, got {}", events.len());

        // First event: trace
        assert!(matches!(&events[0], RecordedEvent::Trace(_)));

        // Second event: thinking delta
        assert!(
            matches!(&events[1], RecordedEvent::Delta(d) if matches!(d.kind, DeltaKind::Thinking))
        );

        // At least one text delta
        let has_text_delta = events
            .iter()
            .any(|e| matches!(e, RecordedEvent::Delta(d) if matches!(d.kind, DeltaKind::Text)));
        assert!(has_text_delta, "expected at least one text delta");

        // At least one step
        let has_step = events.iter().any(|e| matches!(e, RecordedEvent::Step(_)));
        assert!(has_step, "expected a step event");

        // Last event: complete
        assert!(
            matches!(events.last(), Some(RecordedEvent::Complete(_))),
            "expected last event to be Complete"
        );
    }

    #[tokio::test]
    async fn test_demo_solve_cancel() {
        let emitter = TestEmitter::new();
        let token = CancellationToken::new();
        token.cancel(); // Cancel before starting

        demo_solve("Test objective", &emitter, token).await;

        let events = emitter.events();

        let has_error = events
            .iter()
            .any(|e| matches!(e, RecordedEvent::Error(m) if m == "Cancelled"));
        assert!(has_error, "expected a Cancelled error event");

        let has_complete = events.iter().any(|e| matches!(e, RecordedEvent::Complete(_)));
        assert!(!has_complete, "should not have a Complete event when cancelled");
    }

    #[tokio::test]
    async fn test_demo_solve_echoes_objective() {
        let emitter = TestEmitter::new();
        let token = CancellationToken::new();

        demo_solve("Hello world", &emitter, token).await;

        let events = emitter.events();

        // Text deltas should contain the objective
        let text_content: String = events
            .iter()
            .filter_map(|e| match e {
                RecordedEvent::Delta(d) if matches!(d.kind, DeltaKind::Text) => {
                    Some(d.text.clone())
                }
                _ => None,
            })
            .collect();
        assert!(
            text_content.contains("Hello world"),
            "text deltas should contain objective, got: {text_content}"
        );

        // Complete event should contain the objective
        let complete_text = events
            .iter()
            .find_map(|e| match e {
                RecordedEvent::Complete(r) => Some(r.clone()),
                _ => None,
            })
            .expect("should have a Complete event");
        assert!(
            complete_text.contains("Hello world"),
            "complete result should contain objective, got: {complete_text}"
        );
    }

    #[tokio::test]
    async fn test_demo_solve_cancel_mid_flight() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let emitter = TestEmitter {
            events: events.clone(),
        };
        let token = CancellationToken::new();
        let cancel_handle = token.clone();

        // Spawn demo_solve on a separate task, just like agent.rs does
        let task = tokio::spawn(async move {
            demo_solve("Mid-cancel test", &emitter, token).await;
        });

        // Wait for the trace event to be emitted, then cancel
        // This proves cancellation works mid-solve, not just pre-solve
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            let current = events.lock().unwrap().len();
            if current >= 2 {
                // At least trace + thinking delta emitted; cancel now
                cancel_handle.cancel();
                break;
            }
        }

        task.await.expect("task should not panic");

        let recorded = events.lock().unwrap().clone();

        // Should have an error with "Cancelled"
        let has_error = recorded
            .iter()
            .any(|e| matches!(e, RecordedEvent::Error(m) if m == "Cancelled"));
        assert!(has_error, "expected Cancelled error after mid-flight cancel");

        // Should NOT have a Complete event
        let has_complete = recorded
            .iter()
            .any(|e| matches!(e, RecordedEvent::Complete(_)));
        assert!(
            !has_complete,
            "should not have Complete after mid-flight cancel"
        );
    }

    #[tokio::test]
    async fn test_demo_solve_spawned_task_completes() {
        // Simulates the exact pattern used in agent.rs:
        // spawn demo_solve on a task, let it run to completion
        let events = Arc::new(Mutex::new(Vec::new()));
        let emitter = TestEmitter {
            events: events.clone(),
        };
        let token = CancellationToken::new();

        let task = tokio::spawn(async move {
            demo_solve("Spawned test", &emitter, token).await;
        });

        task.await.expect("spawned task should not panic");

        let recorded = events.lock().unwrap().clone();

        // Verify full sequence completed through the spawned task
        assert!(
            matches!(recorded.first(), Some(RecordedEvent::Trace(_))),
            "first event should be Trace"
        );
        assert!(
            matches!(recorded.last(), Some(RecordedEvent::Complete(_))),
            "last event should be Complete"
        );

        // Verify the complete event contains the objective
        let complete_text = recorded
            .iter()
            .find_map(|e| match e {
                RecordedEvent::Complete(r) => Some(r.clone()),
                _ => None,
            })
            .unwrap();
        assert!(complete_text.contains("Spawned test"));
    }

    #[test]
    fn test_estimate_tokens() {
        let messages = vec![
            Message::System { content: "System prompt".into() }, // 13 chars
            Message::User { content: "Hello".into() },          // 5 chars
            Message::Tool { tool_call_id: "t1".into(), content: "x".repeat(4000) },
        ];
        let tokens = estimate_tokens(&messages);
        // (13 + 5 + 4000) / 4 = 1004
        assert_eq!(tokens, 1004);
    }

    // ── sanitize_orphaned_tool_calls ──

    #[test]
    fn test_sanitize_orphaned_tool_calls_inserts_missing_result() {
        let mut messages = vec![
            Message::User { content: "write a big file".into() },
            Message::Assistant {
                content: String::new(),
                tool_calls: Some(vec![ToolCall {
                    id: "tc1".into(),
                    name: "write_file".into(),
                    arguments: "{\"path\":\"x\"".into(), // truncated JSON
                }]),
            },
            // No matching Message::Tool — the tool_result never got appended.
            Message::User { content: "next objective".into() },
        ];

        sanitize_orphaned_tool_calls(&mut messages);

        assert_eq!(messages.len(), 4, "expected a synthetic tool_result inserted");
        match &messages[2] {
            Message::Tool { tool_call_id, content } => {
                assert_eq!(tool_call_id, "tc1");
                assert_eq!(content, "tool call was truncated");
            }
            other => panic!("expected Message::Tool at index 2, got {other:?}"),
        }
        // The message after the injected result is untouched.
        assert!(matches!(&messages[3], Message::User { content } if content == "next objective"));
    }

    #[test]
    fn test_sanitize_orphaned_tool_calls_fills_only_the_missing_one_of_several() {
        let mut messages = vec![
            Message::Assistant {
                content: String::new(),
                tool_calls: Some(vec![
                    ToolCall { id: "tc1".into(), name: "read_file".into(), arguments: "{}".into() },
                    ToolCall { id: "tc2".into(), name: "write_file".into(), arguments: "{}".into() },
                ]),
            },
            // Only tc1 was answered — tc2's result never landed.
            Message::Tool { tool_call_id: "tc1".into(), content: "file contents".into() },
        ];

        sanitize_orphaned_tool_calls(&mut messages);

        assert_eq!(messages.len(), 3);
        match &messages[2] {
            Message::Tool { tool_call_id, content } => {
                assert_eq!(tool_call_id, "tc2");
                assert_eq!(content, "tool call was truncated");
            }
            other => panic!("expected synthetic Message::Tool for tc2, got {other:?}"),
        }
    }

    #[test]
    fn test_sanitize_orphaned_tool_calls_no_op_when_fully_answered() {
        let mut messages = vec![
            Message::Assistant {
                content: String::new(),
                tool_calls: Some(vec![ToolCall { id: "tc1".into(), name: "read_file".into(), arguments: "{}".into() }]),
            },
            Message::Tool { tool_call_id: "tc1".into(), content: "ok".into() },
            Message::Assistant { content: "done\nDONE".into(), tool_calls: None },
        ];
        let before = messages.len();

        sanitize_orphaned_tool_calls(&mut messages);

        assert_eq!(messages.len(), before, "fully-answered history must be untouched");
    }

    #[test]
    fn test_sanitize_orphaned_tool_calls_no_op_on_text_only_history() {
        let mut messages = vec![
            Message::System { content: "sys".into() },
            Message::User { content: "hi".into() },
            Message::Assistant { content: "hello\nDONE".into(), tool_calls: None },
        ];
        let before = messages.clone();

        sanitize_orphaned_tool_calls(&mut messages);

        assert_eq!(messages.len(), before.len());
    }

    #[test]
    fn test_sanitize_orphaned_tool_calls_handles_consecutive_poisoned_turns() {
        // Two assistant turns in a row (not realistic in practice, but the
        // function must not confuse one turn's missing result with another's).
        let mut messages = vec![
            Message::Assistant {
                content: String::new(),
                tool_calls: Some(vec![ToolCall { id: "a1".into(), name: "read_file".into(), arguments: "{}".into() }]),
            },
            Message::Assistant {
                content: String::new(),
                tool_calls: Some(vec![ToolCall { id: "b1".into(), name: "write_file".into(), arguments: "{}".into() }]),
            },
        ];

        sanitize_orphaned_tool_calls(&mut messages);

        // Expect: [Assistant a1, Tool a1, Assistant b1, Tool b1]
        assert_eq!(messages.len(), 4);
        assert!(matches!(&messages[1], Message::Tool { tool_call_id, .. } if tool_call_id == "a1"));
        assert!(matches!(&messages[3], Message::Tool { tool_call_id, .. } if tool_call_id == "b1"));
    }

    #[test]
    fn test_compact_messages_no_op_when_under_limit() {
        let mut messages = vec![
            Message::System { content: "System".into() },
            Message::User { content: "Hello".into() },
            Message::Tool { tool_call_id: "t1".into(), content: "Short result".into() },
        ];
        compact_messages(&mut messages, 100_000);
        // Should be unchanged
        if let Message::Tool { content, .. } = &messages[2] {
            assert_eq!(content, "Short result");
        }
    }

    #[test]
    fn test_compact_messages_truncates_old_tool_results() {
        let big_result = "x".repeat(8000);
        let mut messages = vec![
            Message::System { content: "System".into() },
            Message::User { content: "Hello".into() },
        ];

        // Add 15 old steps (assistant + tool pairs) to exceed keep_recent
        for i in 0..15 {
            messages.push(Message::Assistant { content: format!("step{i}"), tool_calls: None });
            messages.push(Message::Tool { tool_call_id: format!("t{i}"), content: big_result.clone() });
        }

        // Total: ~(6 + 5 + 15*(5+8000)) / 4 ≈ 30_000 tokens
        // Set limit below that to trigger compaction
        compact_messages(&mut messages, 10_000);

        // Old tool result (index 3, early in the list) should be truncated
        if let Message::Tool { content, .. } = &messages[3] {
            assert!(content.len() < 300, "old tool result should be truncated, got {} chars", content.len());
            assert!(content.contains("truncated"));
        }

        // Recent tool result (last one) should be intact
        let last_tool = messages.iter().rev().find(|m| matches!(m, Message::Tool { .. })).unwrap();
        if let Message::Tool { content, .. } = last_tool {
            assert_eq!(content.len(), 8000, "recent tool result should be intact");
        }
    }

    // ── strip_trailing_done / resolve_text_only_turn ──

    #[test]
    fn test_strip_trailing_done_exact_match() {
        assert_eq!(strip_trailing_done("DONE"), Some(String::new()));
        assert_eq!(
            strip_trailing_done("Found the answer.\nDONE"),
            Some("Found the answer.".to_string())
        );
    }

    #[test]
    fn test_strip_trailing_done_case_insensitive_and_punctuation() {
        assert_eq!(
            strip_trailing_done("All set.\ndone."),
            Some("All set.".to_string())
        );
        assert_eq!(
            strip_trailing_done("All set.\nDoNe!!!"),
            Some("All set.".to_string())
        );
    }

    #[test]
    fn test_strip_trailing_done_trailing_whitespace_after_marker() {
        assert_eq!(
            strip_trailing_done("Wrapping up.\nDONE\n\n  "),
            Some("Wrapping up.".to_string())
        );
    }

    #[test]
    fn test_strip_trailing_done_no_match() {
        assert_eq!(strip_trailing_done("Still working on it"), None);
        // Whole-line match only — a word containing "done" doesn't count.
        assert_eq!(strip_trailing_done("This task is UNDONE"), None);
        // DONE must be the LAST non-empty line, not just present somewhere.
        assert_eq!(
            strip_trailing_done("DONE\nActually, one more thing"),
            None
        );
        assert_eq!(strip_trailing_done(""), None);
    }

    #[test]
    fn test_resolve_text_only_turn_done_marker_finalizes_and_strips() {
        let result = resolve_text_only_turn("Analysis complete.\nDONE", false);
        assert_eq!(result, Some("Analysis complete.".to_string()));
    }

    #[test]
    fn test_resolve_text_only_turn_no_marker_not_nudged_yet_returns_none() {
        // "nudge once then continue": first text-only turn without DONE must
        // NOT finalize — caller is expected to send TEXT_ONLY_NUDGE instead.
        let result = resolve_text_only_turn("Now I have sufficient evidence.", false);
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_text_only_turn_second_turn_after_nudge_finalizes() {
        // Once a nudge has already been sent this loop, any further text-only
        // turn is final even without a DONE marker.
        let result = resolve_text_only_turn("Still no marker here", true);
        assert_eq!(result, Some("Still no marker here".to_string()));
    }

    #[test]
    fn test_resolve_text_only_turn_done_marker_wins_even_if_already_nudged() {
        let result = resolve_text_only_turn("Wrapping up.\nDONE", true);
        assert_eq!(result, Some("Wrapping up.".to_string()));
    }

    /// `solve()` (depth-0) resolves its model via `build_model(config)`, which
    /// is not injectable, so its loop isn't reachable with a mock model —
    /// unlike `engine::subagent::run_child`, which IS integration-tested with
    /// a `MockModel` (see `subagent::tests::test_run_child_nudges_once_then_*`
    /// below). `solve()`'s text-only-turn branch calls the exact same
    /// `resolve_text_only_turn` function subagent's loop calls (see
    /// `mod.rs`'s "No tool calls + text present" branch vs. subagent.rs's),
    /// so this test locks down the two-turn sequence solve() performs by
    /// driving that shared function directly, in the order solve() calls it.
    #[test]
    fn test_solve_text_only_turn_nudges_once_then_finalizes_with_done_stripped() {
        let mut nudged = false;
        // Turn 1: narrating without DONE — must not finalize, must nudge.
        let turn1 = resolve_text_only_turn("Now I have sufficient evidence. Let me write the files.", nudged);
        assert_eq!(turn1, None, "first text-only turn without DONE must nudge, not finalize");
        nudged = true;

        // Turn 2: model complies and ends with DONE — finalizes, DONE stripped.
        let turn2 = resolve_text_only_turn("Wrote both deliverable files.\nDONE", nudged);
        assert_eq!(turn2, Some("Wrote both deliverable files.".to_string()));
    }
}
