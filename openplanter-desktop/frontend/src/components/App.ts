/** Root layout component. */
import { open } from "@tauri-apps/plugin-dialog";
import { createStatusBar } from "./StatusBar";
import { createChatPane } from "./ChatPane";
import { createGraphPane } from "./GraphPane";
import { appState } from "../state/store";
import {
  listSessions,
  openSession,
  deleteSession,
  getCredentialsStatus,
  getSessionHistory,
  setWorkspace,
  updateConfig,
  listModels,
  setCredential,
} from "../api/invoke";
import type { ChatMessage } from "../state/store";
import type { ReplayEntry } from "../api/types";
import {
  loadLayoutState,
  saveLayoutState,
  resizeLeftWidth,
  resizeRightWidth,
  LEFT_DEFAULT,
  RIGHT_DEFAULT,
  COLLAPSED_WIDTH,
} from "../layout/paneLayout";

const PROVIDERS = ["auto", "openai", "anthropic", "openrouter", "cerebras", "ollama"];
const REASONING_LEVELS = ["none", "low", "medium", "high"];
const CRED_PROVIDERS = ["openai", "anthropic", "openrouter", "cerebras", "ollama", "exa"];

export function createApp(root: HTMLElement): void {
  // Status bar
  const statusBar = createStatusBar();
  root.appendChild(statusBar);

  // Sidebar
  const sidebar = document.createElement("div");
  sidebar.className = "sidebar";

  const sessionsHeader = document.createElement("h3");
  sessionsHeader.textContent = "Sessions";
  sidebar.appendChild(sessionsHeader);

  // New session button
  const newSessionBtn = document.createElement("div");
  newSessionBtn.className = "session-item";
  newSessionBtn.style.color = "var(--accent)";
  newSessionBtn.style.fontWeight = "600";
  newSessionBtn.textContent = "+ New Session";
  newSessionBtn.addEventListener("click", () => switchToNewSession(sessionList));
  sidebar.appendChild(newSessionBtn);

  const sessionList = document.createElement("div");
  sessionList.className = "session-list";
  sidebar.appendChild(sessionList);

  // Model section
  const modelSection = document.createElement("div");
  modelSection.className = "sidebar-section";
  const modelHeader = document.createElement("h3");
  modelHeader.textContent = "Model";
  modelSection.appendChild(modelHeader);
  const settingsDisplay = document.createElement("div");
  settingsDisplay.className = "settings-display";
  modelSection.appendChild(settingsDisplay);
  sidebar.appendChild(modelSection);

  // Agents section
  const agentsSection = document.createElement("div");
  agentsSection.className = "sidebar-section";
  const agentsHeader = document.createElement("h3");
  agentsHeader.textContent = "Agents";
  agentsSection.appendChild(agentsHeader);
  const agentsDisplay = document.createElement("div");
  agentsDisplay.className = "settings-display";
  agentsSection.appendChild(agentsDisplay);
  sidebar.appendChild(agentsSection);

  const { elements: settingsEls, render: renderSettingsControls } = buildSettingsControls(
    settingsDisplay,
    agentsDisplay
  );

  // Workspace section: current path + folder picker
  const workspaceSection = document.createElement("div");
  workspaceSection.className = "sidebar-section";
  const workspaceHeader = document.createElement("h3");
  workspaceHeader.textContent = "Workspace";
  workspaceSection.appendChild(workspaceHeader);

  const workspaceRow = document.createElement("div");
  workspaceRow.className = "form-row";
  workspaceRow.style.gridTemplateColumns = "1fr auto";

  const workspaceLabel = document.createElement("span");
  workspaceLabel.className = "workspace-path";

  const openFolderBtn = document.createElement("button");
  openFolderBtn.textContent = "Open Folder…";
  openFolderBtn.className = "btn";
  openFolderBtn.addEventListener("click", () => openWorkspacePicker(workspaceLabel, sessionList, credsDisplay));

  workspaceRow.appendChild(workspaceLabel);
  workspaceRow.appendChild(openFolderBtn);
  workspaceSection.appendChild(workspaceRow);
  sidebar.appendChild(workspaceSection);

  function renderWorkspaceLabel() {
    const ws = appState.get().workspace || "";
    const base = ws.split(/[\\/]/).filter(Boolean).pop() || ws || "—";
    workspaceLabel.textContent = "";
    const bold = document.createElement("b");
    bold.textContent = base;
    workspaceLabel.appendChild(bold);
    workspaceLabel.title = ws;
  }
  appState.subscribe(renderWorkspaceLabel);
  renderWorkspaceLabel();

  // Accounts section (credentials)
  const accountsSection = document.createElement("div");
  accountsSection.className = "sidebar-section";
  const credsHeader = document.createElement("h3");
  credsHeader.textContent = "Accounts";
  accountsSection.appendChild(credsHeader);

  const credsDisplay = document.createElement("div");
  credsDisplay.className = "cred-status";
  accountsSection.appendChild(credsDisplay);
  sidebar.appendChild(accountsSection);

  root.appendChild(sidebar);

  // Chat pane
  const chatPane = createChatPane();
  root.appendChild(chatPane);

  // Graph pane
  const graphPane = createGraphPane();
  root.appendChild(graphPane);

  // Resizable/collapsible side panes (sidebar + graph pane)
  setupPaneLayout(root, sidebar, graphPane);

  // Reactive settings controls (provider/model/reasoning/recursive/max_depth)
  appState.subscribe(renderSettingsControls);
  renderSettingsControls();
  void settingsEls; // controls are wired inside buildSettingsControls

  // Load sessions
  loadSessions(sessionList);

  // Reload session list when session changes. This is the single source of
  // truth for refreshing the list \u2014 covers every path that mutates the
  // active session, including InputBar's lazy session creation on first
  // message (which otherwise left the sidebar list stale).
  window.addEventListener("session-changed", () => {
    loadSessions(sessionList);
  });

  // Reload session list when session changes
  appState.subscribe(() => {
    highlightActiveSession(sessionList);
  });

  // Load credentials status
  loadCredentials(credsDisplay);
}

/**
 * Wire up drag-to-resize and collapse/expand for the two side panes
 * (sidebar on the left, graph pane on the right). Widths and collapsed
 * state are persisted to localStorage and restored on load. The chat pane
 * in between is untouched — it already fills the remaining `1fr` track.
 *
 * Resize/collapse dispatch a debounced "pane-layout-resize" window event so
 * GraphPane can re-fit its Cytoscape instance without this module needing
 * to import the (heavy) graph rendering code directly.
 */
function setupPaneLayout(root: HTMLElement, leftPane: HTMLElement, rightPane: HTMLElement): void {
  const state = loadLayoutState();

  const leftGutter = document.createElement("div");
  leftGutter.className = "pane-gutter";
  leftGutter.title = "Drag to resize · double-click to reset";
  leftPane.appendChild(leftGutter);

  const leftChevron = document.createElement("button");
  leftChevron.type = "button";
  leftChevron.className = "pane-chevron";
  leftPane.appendChild(leftChevron);

  const rightGutter = document.createElement("div");
  rightGutter.className = "pane-gutter";
  rightGutter.title = "Drag to resize · double-click to reset";
  rightPane.appendChild(rightGutter);

  const rightChevron = document.createElement("button");
  rightChevron.type = "button";
  rightChevron.className = "pane-chevron";
  rightPane.appendChild(rightChevron);

  let resizeNotifyTimer: ReturnType<typeof setTimeout> | null = null;
  function notifyResize(): void {
    if (resizeNotifyTimer) clearTimeout(resizeNotifyTimer);
    resizeNotifyTimer = setTimeout(() => {
      resizeNotifyTimer = null;
      window.dispatchEvent(new CustomEvent("pane-layout-resize"));
    }, 120);
  }

  function apply(): void {
    root.style.setProperty(
      "--pane-left-width",
      `${state.leftCollapsed ? COLLAPSED_WIDTH : state.leftWidth}px`
    );
    root.style.setProperty(
      "--pane-right-width",
      `${state.rightCollapsed ? COLLAPSED_WIDTH : state.rightWidth}px`
    );
    leftPane.classList.toggle("pane-collapsed", state.leftCollapsed);
    rightPane.classList.toggle("pane-collapsed", state.rightCollapsed);
    leftChevron.textContent = state.leftCollapsed ? "›" : "‹";
    leftChevron.title = state.leftCollapsed ? "Expand sidebar (⌘[)" : "Collapse sidebar (⌘[)";
    rightChevron.textContent = state.rightCollapsed ? "‹" : "›";
    rightChevron.title = state.rightCollapsed
      ? "Expand graph pane (⌘])"
      : "Collapse graph pane (⌘])";
  }
  apply();

  function toggleLeft(): void {
    state.leftCollapsed = !state.leftCollapsed;
    apply();
    saveLayoutState(state);
    notifyResize();
  }

  function toggleRight(): void {
    state.rightCollapsed = !state.rightCollapsed;
    apply();
    saveLayoutState(state);
    notifyResize();
  }

  leftChevron.addEventListener("click", toggleLeft);
  rightChevron.addEventListener("click", toggleRight);

  function bindDrag(
    gutter: HTMLElement,
    getStartWidth: () => number,
    computeWidth: (start: number, deltaX: number) => number,
    apply_: (width: number) => void,
    reset: () => void
  ): void {
    gutter.addEventListener("dblclick", () => reset());

    gutter.addEventListener("mousedown", (e: MouseEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startWidth = getStartWidth();
      gutter.classList.add("dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";

      function onMouseMove(ev: MouseEvent): void {
        apply_(computeWidth(startWidth, ev.clientX - startX));
        notifyResize();
      }
      function onMouseUp(): void {
        gutter.classList.remove("dragging");
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
        saveLayoutState(state);
      }
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    });
  }

  bindDrag(
    leftGutter,
    () => state.leftWidth,
    resizeLeftWidth,
    (w) => {
      state.leftWidth = w;
      state.leftCollapsed = false; // dragging always expands
      apply();
    },
    () => {
      state.leftWidth = LEFT_DEFAULT;
      apply();
      saveLayoutState(state);
    }
  );

  bindDrag(
    rightGutter,
    () => state.rightWidth,
    resizeRightWidth,
    (w) => {
      state.rightWidth = w;
      state.rightCollapsed = false; // dragging always expands
      apply();
      notifyResize();
    },
    () => {
      state.rightWidth = RIGHT_DEFAULT;
      apply();
      saveLayoutState(state);
      notifyResize();
    }
  );

  // Keyboard shortcuts: Cmd+[ / Cmd+] on macOS, Ctrl+[ / Ctrl+] elsewhere.
  window.addEventListener("keydown", (e: KeyboardEvent) => {
    const isMac = navigator.platform.toUpperCase().includes("MAC");
    const modifierHeld = isMac ? e.metaKey : e.ctrlKey;
    if (!modifierHeld || (e.key !== "[" && e.key !== "]")) return;
    e.preventDefault();
    if (e.key === "[") toggleLeft();
    else toggleRight();
  });
}

/**
 * Build the editable settings controls (provider/model/reasoning/recursive/max_depth)
 * inside `container`. Returns the built elements plus a `render` function that
 * syncs control values from `appState` — called on every state change, but only
 * touches the DOM for fields that actually changed so it never stomps on
 * in-progress typing (e.g. the max-depth number input).
 */
function buildSettingsControls(
  modelContainer: HTMLElement,
  agentsContainer: HTMLElement
): {
  elements: {
    providerSelect: HTMLSelectElement;
    modelSelect: HTMLSelectElement;
    reasoningSelect: HTMLSelectElement;
    recursiveCheckbox: HTMLInputElement;
    maxDepthInput: HTMLInputElement;
    maxOutputTokensInput: HTMLInputElement;
    subtaskModelSelect: HTMLSelectElement;
    executeModelSelect: HTMLSelectElement;
    maxExaAgentCallsInput: HTMLInputElement;
    exaAgentTimeoutSecInput: HTMLInputElement;
  };
  render: () => void;
} {
  modelContainer.innerHTML = "";
  agentsContainer.innerHTML = "";

  function row(labelText: string, control: HTMLElement): HTMLElement {
    const r = document.createElement("div");
    r.className = "form-row";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = labelText;
    r.append(label, control);
    return r;
  }

  const providerSelect = document.createElement("select");
  providerSelect.className = "settings-provider-select value";
  for (const p of PROVIDERS) {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p;
    providerSelect.appendChild(opt);
  }

  const modelSelect = document.createElement("select");
  modelSelect.className = "settings-model-select value";

  const modelCustomInput = document.createElement("input");
  modelCustomInput.type = "text";
  modelCustomInput.className = "settings-model-custom-input";
  modelCustomInput.placeholder = "custom model id";

  const reasoningSelect = document.createElement("select");
  reasoningSelect.className = "settings-reasoning-select value";
  for (const lvl of REASONING_LEVELS) {
    const opt = document.createElement("option");
    opt.value = lvl === "none" ? "" : lvl;
    opt.textContent = lvl;
    reasoningSelect.appendChild(opt);
  }

  const recursiveCheckbox = document.createElement("input");
  recursiveCheckbox.type = "checkbox";
  recursiveCheckbox.className = "settings-recursive-checkbox";

  const maxDepthInput = document.createElement("input");
  maxDepthInput.type = "number";
  maxDepthInput.min = "1";
  maxDepthInput.max = "20";
  maxDepthInput.className = "settings-maxdepth-input";

  const maxOutputTokensInput = document.createElement("input");
  maxOutputTokensInput.type = "number";
  maxOutputTokensInput.min = "1024";
  maxOutputTokensInput.step = "1024";
  maxOutputTokensInput.className = "settings-max-output-tokens-input";

  const subtaskModelSelect = document.createElement("select");
  subtaskModelSelect.className = "settings-subtask-model-select value";

  const executeModelSelect = document.createElement("select");
  executeModelSelect.className = "settings-execute-model-select value";

  const maxExaAgentCallsInput = document.createElement("input");
  maxExaAgentCallsInput.type = "number";
  maxExaAgentCallsInput.min = "0";
  maxExaAgentCallsInput.className = "settings-max-exa-agent-calls-input";

  const exaAgentTimeoutSecInput = document.createElement("input");
  exaAgentTimeoutSecInput.type = "number";
  exaAgentTimeoutSecInput.min = "1";
  exaAgentTimeoutSecInput.max = "600";
  exaAgentTimeoutSecInput.className = "settings-exa-agent-timeout-input";

  const tierModelHint = document.createElement("div");
  tierModelHint.className = "settings-hint";
  tierModelHint.textContent = "Recommended: sub-agent claude-sonnet-5, leaf claude-haiku-4-5";

  const modelIdRow = row("model id", modelCustomInput);
  modelIdRow.classList.add("hidden"); // shown only when "Custom…" is selected

  function syncModelIdRowVisibility(): void {
    modelIdRow.classList.toggle("hidden", modelSelect.value !== "__custom__");
  }

  modelContainer.append(
    row("provider", providerSelect),
    row("model", modelSelect),
    modelIdRow,
    row("reasoning", reasoningSelect),
    row("max output tokens", maxOutputTokensInput)
  );

  agentsContainer.append(
    row("recursive", recursiveCheckbox),
    row("max depth", maxDepthInput),
    row("sub-agent model", subtaskModelSelect),
    row("leaf model", executeModelSelect),
    tierModelHint,
    row("exa call cap", maxExaAgentCallsInput),
    row("exa timeout (s)", exaAgentTimeoutSecInput)
  );

  /** Populate the model select with known models for `provider`, keeping `currentModel` selected. */
  async function refreshModelOptions(provider: string, currentModel: string): Promise<void> {
    modelSelect.innerHTML = "";
    const customOpt = document.createElement("option");
    customOpt.value = "__custom__";
    customOpt.textContent = "Custom…";
    modelSelect.appendChild(customOpt);

    // Show the current model immediately (synchronously) so the select never
    // looks empty while the known-models fetch is in flight.
    let matched = false;
    try {
      const models = await listModels(provider === "auto" ? "all" : provider);
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.name ? `${m.name} (${m.id})` : m.id;
        if (m.id === currentModel) matched = true;
        modelSelect.appendChild(opt);
      }
    } catch (e) {
      console.error("Failed to list models:", e);
    }

    if (!matched && currentModel) {
      const opt = document.createElement("option");
      opt.value = currentModel;
      opt.textContent = currentModel;
      modelSelect.insertBefore(opt, modelSelect.firstChild!.nextSibling);
    }
    modelSelect.value = currentModel || "__custom__";
    syncModelIdRowVisibility();
  }

  /** Populate a tier-model select ("inherit" + known models for `provider`), keeping `currentValue` selected. */
  async function refreshTierModelOptions(
    select: HTMLSelectElement,
    provider: string,
    currentValue: string | null
  ): Promise<void> {
    select.innerHTML = "";
    const inheritOpt = document.createElement("option");
    inheritOpt.value = "";
    inheritOpt.textContent = "inherit";
    select.appendChild(inheritOpt);

    let matched = !currentValue;
    try {
      const models = await listModels(provider === "auto" ? "all" : provider);
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.name ? `${m.name} (${m.id})` : m.id;
        if (m.id === currentValue) matched = true;
        select.appendChild(opt);
      }
    } catch (e) {
      console.error("Failed to list models:", e);
    }

    if (!matched && currentValue) {
      const opt = document.createElement("option");
      opt.value = currentValue;
      opt.textContent = currentValue;
      select.insertBefore(opt, select.firstChild!.nextSibling);
    }
    select.value = currentValue || "";
  }

  async function applyPartial(partial: Parameters<typeof updateConfig>[0]): Promise<void> {
    try {
      const config = await updateConfig(partial);
      appState.update((s) => ({
        ...s,
        provider: config.provider,
        model: config.model,
        reasoningEffort: config.reasoning_effort,
        recursive: config.recursive,
        maxDepth: config.max_depth,
        subtaskModel: config.subtask_model,
        executeModel: config.execute_model,
        maxExaAgentCalls: config.max_exa_agent_calls,
        maxOutputTokens: config.max_output_tokens,
        exaAgentTimeoutSec: config.exa_agent_timeout_sec,
      }));
    } catch (e) {
      console.error("Failed to update config:", e);
    }
  }

  providerSelect.addEventListener("change", () => {
    const provider = providerSelect.value;
    refreshModelOptions(provider, appState.get().model);
    applyPartial({ provider });
  });

  modelSelect.addEventListener("change", () => {
    syncModelIdRowVisibility();
    if (modelSelect.value === "__custom__") {
      modelCustomInput.focus();
      return;
    }
    applyPartial({ model: modelSelect.value });
  });

  modelCustomInput.addEventListener("change", () => {
    const value = modelCustomInput.value.trim();
    if (value) applyPartial({ model: value });
  });

  reasoningSelect.addEventListener("change", () => {
    applyPartial({ reasoning_effort: reasoningSelect.value });
  });

  recursiveCheckbox.addEventListener("change", () => {
    applyPartial({ recursive: recursiveCheckbox.checked });
  });

  maxDepthInput.addEventListener("change", () => {
    const n = parseInt(maxDepthInput.value, 10);
    if (!Number.isNaN(n) && n > 0) applyPartial({ max_depth: n });
  });

  subtaskModelSelect.addEventListener("change", () => {
    applyPartial({ subtask_model: subtaskModelSelect.value });
  });

  executeModelSelect.addEventListener("change", () => {
    applyPartial({ execute_model: executeModelSelect.value });
  });

  maxExaAgentCallsInput.addEventListener("change", () => {
    const n = parseInt(maxExaAgentCallsInput.value, 10);
    if (!Number.isNaN(n) && n >= 0) applyPartial({ max_exa_agent_calls: n });
  });

  maxOutputTokensInput.addEventListener("change", () => {
    const n = parseInt(maxOutputTokensInput.value, 10);
    if (!Number.isNaN(n) && n > 0) applyPartial({ max_output_tokens: n });
  });

  exaAgentTimeoutSecInput.addEventListener("change", () => {
    const n = parseInt(exaAgentTimeoutSecInput.value, 10);
    if (!Number.isNaN(n) && n > 0) applyPartial({ exa_agent_timeout_sec: n });
  });

  // Last-rendered snapshot so unrelated appState updates (e.g. token counts
  // ticking during a run) don't overwrite in-progress edits in these controls.
  let last = {
    provider: "",
    model: "",
    reasoningEffort: null as string | null,
    recursive: true,
    maxDepth: 0,
    subtaskModel: null as string | null,
    executeModel: null as string | null,
    maxExaAgentCalls: 0,
    maxOutputTokens: 0,
    exaAgentTimeoutSec: 0,
  };
  let modelOptionsLoadedFor = "";
  let tierModelOptionsLoadedFor = "";

  function render(): void {
    const s = appState.get();
    if (s.provider !== last.provider) {
      providerSelect.value = PROVIDERS.includes(s.provider) ? s.provider : "auto";
    }
    if (s.provider !== modelOptionsLoadedFor || s.model !== last.model) {
      modelOptionsLoadedFor = s.provider;
      refreshModelOptions(s.provider, s.model);
    }
    if (
      s.provider !== tierModelOptionsLoadedFor ||
      s.subtaskModel !== last.subtaskModel ||
      s.executeModel !== last.executeModel
    ) {
      tierModelOptionsLoadedFor = s.provider;
      refreshTierModelOptions(subtaskModelSelect, s.provider, s.subtaskModel);
      refreshTierModelOptions(executeModelSelect, s.provider, s.executeModel);
    }
    if (s.reasoningEffort !== last.reasoningEffort) {
      reasoningSelect.value = s.reasoningEffort ?? "";
    }
    if (s.recursive !== last.recursive) {
      recursiveCheckbox.checked = s.recursive;
    }
    if (s.maxDepth !== last.maxDepth) {
      maxDepthInput.value = String(s.maxDepth);
    }
    if (s.maxExaAgentCalls !== last.maxExaAgentCalls) {
      maxExaAgentCallsInput.value = String(s.maxExaAgentCalls);
    }
    if (s.maxOutputTokens !== last.maxOutputTokens) {
      maxOutputTokensInput.value = String(s.maxOutputTokens);
    }
    if (s.exaAgentTimeoutSec !== last.exaAgentTimeoutSec) {
      exaAgentTimeoutSecInput.value = String(s.exaAgentTimeoutSec);
    }
    last = {
      provider: s.provider,
      model: s.model,
      reasoningEffort: s.reasoningEffort,
      recursive: s.recursive,
      maxDepth: s.maxDepth,
      subtaskModel: s.subtaskModel,
      executeModel: s.executeModel,
      maxExaAgentCalls: s.maxExaAgentCalls,
      maxOutputTokens: s.maxOutputTokens,
      exaAgentTimeoutSec: s.exaAgentTimeoutSec,
    };
  }

  return {
    elements: {
      providerSelect,
      modelSelect,
      reasoningSelect,
      recursiveCheckbox,
      maxDepthInput,
      maxOutputTokensInput,
      subtaskModelSelect,
      executeModelSelect,
      maxExaAgentCallsInput,
      exaAgentTimeoutSecInput,
    },
    render,
  };
}

/** Switch to a new session, clearing chat state. */
async function switchToNewSession(sessionList: HTMLElement): Promise<void> {
  try {
    const session = await openSession();
    appState.update((s) => ({
      ...s,
      sessionId: session.id,
      messages: [],
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheCreationTokens: 0,
      currentStep: 0,
      currentDepth: 0,
      inputQueue: [],
    }));
    // Dispatch event to clear ChatPane DOM
    window.dispatchEvent(new CustomEvent("session-changed", { detail: { isNew: true } }));
    // Add welcome message
    appState.update((s) => ({
      ...s,
      messages: [
        {
          id: crypto.randomUUID(),
          role: "system" as const,
          content: `New session: ${session.id.slice(0, 8)}`,
          timestamp: Date.now(),
        },
      ],
    }));
    // Reload session list
    loadSessions(sessionList);
  } catch (e) {
    console.error("Failed to create new session:", e);
  }
}

/** Convert a ReplayEntry to a ChatMessage for display. */
function replayEntryToMessage(entry: ReplayEntry): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role: entry.role as ChatMessage["role"],
    content: entry.content,
    toolName: entry.tool_name ?? undefined,
    timestamp: new Date(entry.timestamp).getTime() || Date.now(),
    isRendered: entry.is_rendered ?? (entry.role === "assistant"),
    stepNumber: entry.step_number ?? undefined,
    stepTokensIn: entry.step_tokens_in ?? undefined,
    stepTokensOut: entry.step_tokens_out ?? undefined,
    stepElapsed: entry.step_elapsed ?? undefined,
    stepModelPreview: entry.step_model_preview ?? undefined,
    stepToolCalls: entry.step_tool_calls?.map((tc) => ({
      name: tc.name,
      keyArg: tc.key_arg,
      elapsed: tc.elapsed,
    })),
  };
}

/** Switch to an existing session, loading message history. */
async function switchToSession(sessionId: string, sessionList: HTMLElement): Promise<void> {
  try {
    const resumed = await openSession(sessionId, true);
    appState.update((s) => ({
      ...s,
      sessionId: resumed.id,
      messages: [],
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheCreationTokens: 0,
      currentStep: 0,
      currentDepth: 0,
      inputQueue: [],
    }));
    // Dispatch event to clear ChatPane DOM
    window.dispatchEvent(new CustomEvent("session-changed", { detail: { isNew: false } }));

    // Load message history from replay.jsonl
    let messages: ChatMessage[] = [];
    try {
      const history = await getSessionHistory(resumed.id);
      messages = history.map(replayEntryToMessage);
    } catch (e) {
      console.error("Failed to load session history:", e);
    }

    // Add info message, then history
    const info = resumed.last_objective
      ? `Resumed session ${resumed.id.slice(0, 8)} \u2014 ${resumed.last_objective}`
      : `Resumed session ${resumed.id.slice(0, 8)}`;
    appState.update((s) => ({
      ...s,
      messages: [
        {
          id: crypto.randomUUID(),
          role: "system" as const,
          content: info,
          timestamp: Date.now(),
        },
        ...messages,
      ],
    }));
    highlightActiveSession(sessionList);
  } catch (e) {
    console.error("Failed to resume session:", e);
  }
}

function highlightActiveSession(container: HTMLElement): void {
  const currentId = appState.get().sessionId;
  for (const item of container.querySelectorAll(".session-item")) {
    const el = item as HTMLElement;
    if (el.title === currentId) {
      el.style.background = "var(--bg-tertiary)";
      el.style.color = "var(--accent)";
    } else {
      el.style.background = "";
      el.style.color = "";
    }
  }
}

async function loadSessions(container: HTMLElement): Promise<void> {
  try {
    const sessions = await listSessions(20);
    container.innerHTML = "";
    if (sessions.length === 0) {
      const empty = document.createElement("div");
      empty.className = "session-item";
      empty.style.color = "var(--text-muted)";
      empty.textContent = "No sessions yet";
      container.appendChild(empty);
      return;
    }
    for (const session of sessions) {
      const item = document.createElement("div");
      item.className = "session-item";
      item.title = session.id;
      item.style.display = "flex";
      item.style.alignItems = "center";
      item.style.justifyContent = "space-between";

      const label = document.createElement("span");
      label.style.overflow = "hidden";
      label.style.textOverflow = "ellipsis";
      label.style.whiteSpace = "nowrap";
      label.style.flex = "1";
      const date = new Date(session.created_at);
      const dateStr = date.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
      label.textContent = session.last_objective
        ? `${dateStr} \u2014 ${session.last_objective}`
        : dateStr;

      label.addEventListener("click", () => switchToSession(session.id, container));

      const deleteBtn = document.createElement("span");
      deleteBtn.className = "session-delete";
      deleteBtn.textContent = "\u00d7";
      deleteBtn.title = "Delete session";
      let confirmPending = false;
      let confirmTimer: ReturnType<typeof setTimeout> | null = null;
      function resetDeleteBtn() {
        confirmPending = false;
        deleteBtn.textContent = "\u00d7";
        deleteBtn.style.color = "";
        deleteBtn.style.fontWeight = "";
        deleteBtn.style.display = "";
      }
      deleteBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirmPending) {
          // First click: enter confirmation state
          confirmPending = true;
          deleteBtn.textContent = "Delete?";
          deleteBtn.style.color = "var(--error)";
          deleteBtn.style.fontWeight = "600";
          deleteBtn.style.display = "inline"; // override CSS display:none
          confirmTimer = setTimeout(resetDeleteBtn, 3000);
          return;
        }
        // Second click: actually delete
        if (confirmTimer) clearTimeout(confirmTimer);
        confirmPending = false;
        deleteBtn.textContent = "...";
        try {
          await deleteSession(session.id);
          if (appState.get().sessionId === session.id) {
            await switchToNewSession(container);
          } else {
            await loadSessions(container);
          }
        } catch (err) {
          deleteBtn.textContent = "Error!";
          console.error("Failed to delete session:", err);
          setTimeout(resetDeleteBtn, 2000);
        }
      });

      item.appendChild(label);
      item.appendChild(deleteBtn);
      container.appendChild(item);
    }
    highlightActiveSession(container);
  } catch (e) {
    console.error("Failed to load sessions:", e);
  }
}

/** Open a native folder picker and switch the active workspace to the chosen directory. */
async function openWorkspacePicker(
  workspaceLabel: HTMLElement,
  sessionList: HTMLElement,
  credsDisplay: HTMLElement
): Promise<void> {
  try {
    const selected = await open({ directory: true, multiple: false });
    if (!selected || typeof selected !== "string") {
      return; // user cancelled
    }

    const config = await setWorkspace(selected);
    appState.update((s) => ({
      ...s,
      provider: config.provider,
      model: config.model,
      sessionId: config.session_id,
      reasoningEffort: config.reasoning_effort,
      recursive: config.recursive,
      workspace: config.workspace,
      maxDepth: config.max_depth,
      maxStepsPerCall: config.max_steps_per_call,
      maxExaAgentCalls: config.max_exa_agent_calls,
      maxOutputTokens: config.max_output_tokens,
      exaAgentTimeoutSec: config.exa_agent_timeout_sec,
      messages: [],
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheCreationTokens: 0,
      currentStep: 0,
      currentDepth: 0,
      inputQueue: [],
    }));

    // Clear chat DOM and refresh sessions, credentials, and the wiki graph
    window.dispatchEvent(new CustomEvent("session-changed", { detail: { isNew: true } }));
    await loadSessions(sessionList);
    await loadCredentials(credsDisplay);
    window.dispatchEvent(new CustomEvent("curator-done"));
  } catch (e) {
    console.error("Failed to switch workspace:", e);
  }
}

async function loadCredentials(container: HTMLElement): Promise<void> {
  try {
    const status = await getCredentialsStatus();
    renderCredentials(container, status);
  } catch (e) {
    console.error("Failed to load credentials:", e);
  }
}

function renderCredentials(container: HTMLElement, status: Record<string, boolean>): void {
  container.innerHTML = "";
  for (const p of CRED_PROVIDERS) {
    const row = document.createElement("div");
    row.className = "cred-row";

    const hasKey = status[p] ?? false;
    const statusEl = document.createElement("span");
    statusEl.className = hasKey ? "cred-ok" : "cred-missing";

    const dot = document.createElement("span");
    dot.className = `cred-status-dot ${hasKey ? "ok" : "missing"}`;
    const name = document.createElement("span");
    name.className = "cred-name";
    name.textContent = p;
    statusEl.append(dot, name);
    row.appendChild(statusEl);

    if (p === "ollama") {
      // Ollama never needs a key \u2014 no Set control.
      const note = document.createElement("span");
      note.className = "cred-local-note";
      note.textContent = "local, no key";
      row.appendChild(note);
      container.appendChild(row);
      continue;
    }

    const setBtn = document.createElement("button");
    setBtn.className = "cred-set-btn btn";
    setBtn.textContent = "Set\u2026";

    const form = document.createElement("div");
    form.className = "cred-set-form";
    form.style.display = "none";

    const input = document.createElement("input");
    input.type = "password";
    input.className = "cred-set-input";
    input.placeholder = `${p} API key`;
    input.autocomplete = "off";

    const saveBtn = document.createElement("button");
    saveBtn.className = "cred-save-btn btn";
    saveBtn.textContent = "Save";

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "cred-cancel-btn btn";
    cancelBtn.textContent = "Cancel";

    form.append(input, saveBtn, cancelBtn);
    row.append(setBtn, form);

    function openForm(): void {
      form.style.display = "flex";
      input.focus();
    }

    function closeForm(): void {
      form.style.display = "none";
      input.value = "";
    }

    setBtn.addEventListener("click", () => {
      const opening = form.style.display === "none";
      if (opening) openForm();
      else closeForm();
    });

    cancelBtn.addEventListener("click", () => closeForm());

    async function doSave(): Promise<void> {
      const value = input.value.trim();
      if (!value) return;
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving\u2026";
      try {
        const newStatus = await setCredential(p, value);
        // Never leave the entered value in the DOM.
        input.value = "";
        form.style.display = "none";
        renderCredentials(container, newStatus);
      } catch (e) {
        console.error(`Failed to save credential for ${p}:`, e);
        saveBtn.disabled = false;
        saveBtn.textContent = "Save";
      }
    }

    saveBtn.addEventListener("click", () => void doSave());
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") void doSave();
      if (e.key === "Escape") closeForm();
    });

    container.appendChild(row);
  }
}
