# OpenPlanter fork: make everything work (2026-09-23)

Goal: desktop app is the daily tool for the operator's research workflow and later ships inside the website.
Branch: fix/caching-and-opus5 (a9e2c00 caching+Opus5, 710823c workspace+keychain, web_search highlights fix).

## Wave 1 (parallel, separate worktrees)

### 1A backend (op-core) — rust-engineer
- Real sub-agents in Rust: port `subtask`, `execute`, `list_artifacts`, `read_artifact` from agent/engine.py + tool_defs.py.
  Same-process recursion, depth limit = config.max_depth, parallel subtask/execute calls in one turn via tokio tasks,
  execute children get tool defs minus delegation tools, downward-only model tier. Events must stream to the UI
  (reuse existing engine events; tag with depth).
- New tool `exa_agent`: POST {exa_base_url}/agent/runs, Authorization: Bearer exa_api_key, body {query, dataSources?, outputSchema?},
  then poll GET /agent/runs/{id} until finished (respect timeout, run in background if >45s). Verify request/response
  shape against https://exa.ai/docs/exa-spec.yaml before coding. Return output.content + grounding citations.
- Remove the prompt text that advertises subtask/execute when recursion is off; keep it when on (now true).
- Tests for recursion depth, parallel fan-out, execute tool-set narrowing, exa_agent payload shape.

### 1B frontend + tauri commands — frontend-developer
- Sidebar becomes editable: provider <select>, model <select> (populated from known models + /model list),
  reasoning <select>, recursive toggle, max_depth number. All call update_config (exists) and persist via settings store.
- Credentials: each row gets a "Set…" button → prompt → saves via new tauri command `set_credential(provider, value)`
  into user credential store (~/.openplanter/credentials.json) AND, on macOS, `security add-generic-password -U`
  under openplanter-<provider>. Never log values. Status refreshes.
- Graph pane: live refresh during a run (listen for write_file events touching .openplanter/wiki, debounce 2s).
  Fix pane collapse: chat pane must not push side panes off-screen (min-width:0, overflow-x:auto on pre/code).
- Sessions list refreshes on session-changed and after set_workspace.
- Cost strip: show cache_read / cache_creation tokens next to in/out.

## Wave 2
- Exa Connect provider picker in sidebar (baselayer, fiber…) passed to exa_agent dataSources default.
- Research preset: workspace template, seed list loader.

## Rules
- cargo test + vitest green; tauri build produces DMG; no push; one atomic commit per worktree branch; no SUMMARY.md files.
