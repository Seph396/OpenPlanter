// @vitest-environment happy-dom
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { __setHandler, __clearHandlers } from "../__mocks__/tauri";

vi.mock("@tauri-apps/api/core", async () => {
  const mock = await import("../__mocks__/tauri");
  return { invoke: mock.invoke };
});

// Mock sub-components that have heavy dependencies (markdown-it, three.js)
vi.mock("./StatusBar", () => ({
  createStatusBar: () => document.createElement("div"),
}));
vi.mock("./ChatPane", () => ({
  createChatPane: () => document.createElement("div"),
  KEY_ARGS: {},
}));
vi.mock("./GraphPane", () => ({
  createGraphPane: () => document.createElement("div"),
}));

import { appState } from "../state/store";
import { createApp } from "./App";

// Deterministic UUIDs
let uuidCounter = 0;
vi.stubGlobal("crypto", { randomUUID: () => `uuid-${++uuidCounter}` });

const SESSION_A = {
  id: "20260227-100000-aaaa1111",
  created_at: "2026-02-27T10:00:00Z",
  turn_count: 2,
  last_objective: "Test objective A",
};
const SESSION_B = {
  id: "20260227-110000-bbbb2222",
  created_at: "2026-02-27T11:00:00Z",
  turn_count: 0,
  last_objective: null,
};

describe("createApp", () => {
  const originalState = appState.get();

  beforeEach(() => {
    uuidCounter = 0;
    appState.set({ ...originalState, messages: [], sessionId: null });
    __setHandler("list_sessions", () => [SESSION_B, SESSION_A]);
    __setHandler("get_credentials_status", () => ({
      openai: true, anthropic: true, openrouter: false,
      cerebras: false, ollama: true, exa: false,
    }));
    __setHandler("open_session", () => ({
      id: "20260227-120000-cccc3333",
      created_at: "2026-02-27T12:00:00Z",
      turn_count: 0,
      last_objective: null,
    }));
    __setHandler("delete_session", () => {});
    __setHandler("get_session_history", () => []);
    __setHandler("list_models", (args: { provider: string }) =>
      args?.provider === "anthropic" || args?.provider === "all"
        ? [{ id: "claude-opus-4-6", name: "Claude Opus 4.6", provider: "anthropic" }]
        : []
    );
  });

  afterEach(() => {
    __clearHandlers();
    appState.set(originalState);
    document.body.innerHTML = "";
  });

  it("renders sidebar with session list", async () => {
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    // Wait for async loadSessions
    await vi.waitFor(() => {
      const items = root.querySelectorAll(".session-list .session-item");
      expect(items.length).toBe(2);
    });
  });

  it("renders settings display", async () => {
    appState.update((s) => ({ ...s, provider: "anthropic", model: "claude-opus-4-6" }));
    const root = document.createElement("div");
    createApp(root);
    const settings = root.querySelector(".settings-display");
    expect(settings).not.toBeNull();
    expect(settings!.textContent).toContain("anthropic");
    await vi.waitFor(() => {
      expect(settings!.textContent).toContain("claude-opus-4-6");
    });
  });

  it("renders credential status", async () => {
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      const creds = root.querySelector(".cred-status");
      expect(creds!.children.length).toBe(6);
      expect(creds!.querySelector(".cred-ok")!.textContent).toContain("openai");
      expect(creds!.querySelector(".cred-missing")!.textContent).toContain("openrouter");
    });
  });

  it("new session button creates session and clears state", async () => {
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".session-list .session-item").length).toBe(2);
    });

    const newBtn = root.querySelector(".sidebar > .session-item") as HTMLElement;
    expect(newBtn.textContent).toBe("+ New Session");
    newBtn.click();

    await vi.waitFor(() => {
      expect(appState.get().sessionId).toBe("20260227-120000-cccc3333");
    });
  });

  it("shows 'No sessions yet' when list is empty", async () => {
    __setHandler("list_sessions", () => []);
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      const items = root.querySelectorAll(".session-list .session-item");
      expect(items.length).toBe(1);
      expect(items[0].textContent).toBe("No sessions yet");
    });
  });

  it("editing the recursive checkbox calls update_config and updates state", async () => {
    let received: any = null;
    __setHandler("update_config", (partial: any) => {
      received = partial;
      return {
        provider: "anthropic",
        model: "claude-opus-4-6",
        reasoning_effort: null,
        workspace: "/tmp/ws",
        session_id: null,
        recursive: false,
        max_depth: 4,
        max_steps_per_call: 100,
        demo: false,
      };
    });

    appState.update((s) => ({ ...s, recursive: true }));
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    const checkbox = root.querySelector(".settings-recursive-checkbox") as HTMLInputElement;
    expect(checkbox).not.toBeNull();
    checkbox.checked = false;
    checkbox.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(received.partial).toEqual({ recursive: false });
      expect(appState.get().recursive).toBe(false);
    });
  });

  it("editing the max-depth input calls update_config with the new value", async () => {
    let received: any = null;
    __setHandler("update_config", (partial: any) => {
      received = partial;
      return {
        provider: "anthropic",
        model: "claude-opus-4-6",
        reasoning_effort: null,
        workspace: "/tmp/ws",
        session_id: null,
        recursive: true,
        max_depth: 8,
        max_steps_per_call: 100,
        demo: false,
      };
    });

    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    const input = root.querySelector(".settings-maxdepth-input") as HTMLInputElement;
    expect(input).not.toBeNull();
    input.value = "8";
    input.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(received.partial).toEqual({ max_depth: 8 });
      expect(appState.get().maxDepth).toBe(8);
    });
  });

  it("editing the exa call cap input calls update_config with the new value", async () => {
    let received: any = null;
    __setHandler("update_config", (partial: any) => {
      received = partial;
      return {
        provider: "anthropic",
        model: "claude-opus-4-6",
        reasoning_effort: null,
        workspace: "/tmp/ws",
        session_id: null,
        recursive: true,
        max_depth: 4,
        max_steps_per_call: 100,
        demo: false,
        max_exa_agent_calls: 5,
      };
    });

    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    const input = root.querySelector(".settings-max-exa-agent-calls-input") as HTMLInputElement;
    expect(input).not.toBeNull();
    input.value = "5";
    input.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(received.partial).toEqual({ max_exa_agent_calls: 5 });
      expect(appState.get().maxExaAgentCalls).toBe(5);
    });
  });

  it("changing the provider select calls update_config with the new provider", async () => {
    let received: any = null;
    __setHandler("update_config", (partial: any) => {
      received = partial;
      return {
        provider: "cerebras",
        model: "qwen-3-235b-a22b-instruct-2507",
        reasoning_effort: null,
        workspace: "/tmp/ws",
        session_id: null,
        recursive: true,
        max_depth: 4,
        max_steps_per_call: 100,
        demo: false,
      };
    });

    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    const select = root.querySelector(".settings-provider-select") as HTMLSelectElement;
    expect(select).not.toBeNull();
    select.value = "cerebras";
    select.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(received.partial).toEqual({ provider: "cerebras" });
      expect(appState.get().provider).toBe("cerebras");
    });
  });

  it("renders the sub-agent model and leaf model selects with an inherit option", async () => {
    appState.update((s) => ({ ...s, provider: "anthropic" }));
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      const subtaskSelect = root.querySelector(".settings-subtask-model-select") as HTMLSelectElement;
      const executeSelect = root.querySelector(".settings-execute-model-select") as HTMLSelectElement;
      expect(subtaskSelect).not.toBeNull();
      expect(executeSelect).not.toBeNull();
      expect(subtaskSelect.options[0].value).toBe("");
      expect(subtaskSelect.options[0].textContent).toBe("inherit");
      expect(executeSelect.options[0].value).toBe("");
      expect(executeSelect.options[0].textContent).toBe("inherit");
    });
  });

  it("changing the sub-agent model select calls update_config with subtask_model", async () => {
    let received: any = null;
    __setHandler("update_config", (partial: any) => {
      received = partial;
      return {
        provider: "anthropic",
        model: "claude-opus-4-6",
        reasoning_effort: null,
        workspace: "/tmp/ws",
        session_id: null,
        recursive: true,
        max_depth: 4,
        max_steps_per_call: 100,
        demo: false,
        subtask_model: "claude-sonnet-5",
        execute_model: null,
      };
    });

    appState.update((s) => ({ ...s, provider: "anthropic" }));
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    const select = root.querySelector(".settings-subtask-model-select") as HTMLSelectElement;
    expect(select).not.toBeNull();
    await vi.waitFor(() => {
      expect(select.options.length).toBeGreaterThan(1);
    });
    select.value = "claude-opus-4-6";
    select.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(received.partial).toEqual({ subtask_model: "claude-opus-4-6" });
      expect(appState.get().subtaskModel).toBe("claude-sonnet-5");
    });
  });

  it("changing the leaf model select calls update_config with execute_model", async () => {
    let received: any = null;
    __setHandler("update_config", (partial: any) => {
      received = partial;
      return {
        provider: "anthropic",
        model: "claude-opus-4-6",
        reasoning_effort: null,
        workspace: "/tmp/ws",
        session_id: null,
        recursive: true,
        max_depth: 4,
        max_steps_per_call: 100,
        demo: false,
        subtask_model: null,
        execute_model: "claude-haiku-4-5",
      };
    });

    appState.update((s) => ({ ...s, provider: "anthropic" }));
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    const select = root.querySelector(".settings-execute-model-select") as HTMLSelectElement;
    expect(select).not.toBeNull();
    await vi.waitFor(() => {
      expect(select.options.length).toBeGreaterThan(1);
    });
    select.value = "claude-opus-4-6";
    select.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(received.partial).toEqual({ execute_model: "claude-opus-4-6" });
      expect(appState.get().executeModel).toBe("claude-haiku-4-5");
    });
  });

  it("credential input value is cleared and never left in the DOM after save", async () => {
    let savedProvider = "";
    let savedValue = "";
    __setHandler("set_credential", ({ provider, value }: { provider: string; value: string }) => {
      savedProvider = provider;
      savedValue = value;
      return { openai: true, anthropic: true, openrouter: false, cerebras: false, ollama: true, exa: false };
    });

    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".cred-row").length).toBe(6);
    });

    const rows = Array.from(root.querySelectorAll(".cred-row"));
    const openaiRow = rows.find((r) => r.querySelector(".cred-ok, .cred-missing")!.textContent!.includes("openai"))!;
    const setBtn = openaiRow.querySelector(".cred-set-btn") as HTMLElement;
    setBtn.click();

    const input = openaiRow.querySelector(".cred-set-input") as HTMLInputElement;
    expect(input).not.toBeNull();
    expect(input.type).toBe("password");
    input.value = "sk-super-secret-value";
    const saveBtn = openaiRow.querySelector(".cred-save-btn") as HTMLElement;
    saveBtn.click();

    await vi.waitFor(() => {
      expect(savedProvider).toBe("openai");
      expect(savedValue).toBe("sk-super-secret-value");
    });

    // The value must never remain in the DOM (input cleared, container re-rendered).
    // credsDisplay is re-rendered asynchronously after the save resolves, so
    // wait for that continuation rather than asserting immediately.
    await vi.waitFor(() => {
      expect(root.innerHTML).not.toContain("sk-super-secret-value");
      const newInput = root.querySelector(".cred-set-input") as HTMLInputElement | null;
      if (newInput) expect(newInput.value).toBe("");
    });
  });

  it("credential form closes on Escape", async () => {
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".cred-row").length).toBe(6);
    });

    const rows = Array.from(root.querySelectorAll(".cred-row"));
    const openaiRow = rows.find((r) => r.querySelector(".cred-ok, .cred-missing")!.textContent!.includes("openai"))!;
    const setBtn = openaiRow.querySelector(".cred-set-btn") as HTMLElement;
    setBtn.click();

    const form = openaiRow.querySelector(".cred-set-form") as HTMLElement;
    const input = openaiRow.querySelector(".cred-set-input") as HTMLInputElement;
    expect(form.style.display).toBe("flex");

    input.value = "sk-typed-but-not-saved";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));

    expect(form.style.display).toBe("none");
    expect(input.value).toBe("");
  });

  it("hides the 'model id' row unless a custom model is selected", async () => {
    appState.update((s) => ({ ...s, provider: "anthropic", model: "claude-opus-4-6" }));
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    const modelSelect = root.querySelector(".settings-model-select") as HTMLSelectElement;
    await vi.waitFor(() => {
      expect(modelSelect.options.length).toBeGreaterThan(1);
    });

    const modelIdInput = root.querySelector(".settings-model-custom-input") as HTMLInputElement;
    const modelIdRow = modelIdInput.closest(".form-row") as HTMLElement;

    // Known model selected: row stays hidden.
    expect(modelIdRow.classList.contains("hidden")).toBe(true);

    // Switch to "Custom…": row becomes visible.
    modelSelect.value = "__custom__";
    modelSelect.dispatchEvent(new Event("change"));
    expect(modelIdRow.classList.contains("hidden")).toBe(false);
  });
});

describe("session delete confirmation flow", () => {
  const originalState = appState.get();
  let deletedIds: string[] = [];

  beforeEach(() => {
    uuidCounter = 0;
    deletedIds = [];
    appState.set({ ...originalState, messages: [], sessionId: null });
    __setHandler("list_sessions", () => [SESSION_A]);
    __setHandler("get_credentials_status", () => ({}));
    __setHandler("open_session", () => ({
      id: "new-session",
      created_at: "2026-02-27T12:00:00Z",
      turn_count: 0,
      last_objective: null,
    }));
    __setHandler("delete_session", ({ id }: { id: string }) => {
      deletedIds.push(id);
      // After delete, list_sessions returns empty
      __setHandler("list_sessions", () => []);
    });
    __setHandler("get_session_history", () => []);
  });

  afterEach(() => {
    __clearHandlers();
    appState.set(originalState);
    document.body.innerHTML = "";
  });

  it("first click shows 'Delete?' confirmation", async () => {
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".session-list .session-item").length).toBe(1);
    });

    const deleteBtn = root.querySelector(".session-delete") as HTMLElement;
    expect(deleteBtn.textContent).toBe("\u00d7");

    // First click: enters confirmation state
    deleteBtn.click();
    expect(deleteBtn.textContent).toBe("Delete?");
    expect(deleteBtn.style.color).toBe("var(--error)");
    expect(deleteBtn.style.fontWeight).toBe("600");
    expect(deleteBtn.style.display).toBe("inline");

    // Session should NOT be deleted yet
    expect(deletedIds).toEqual([]);
  });

  it("second click actually deletes", async () => {
    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".session-list .session-item").length).toBe(1);
    });

    const deleteBtn = root.querySelector(".session-delete") as HTMLElement;

    // First click: confirm
    deleteBtn.click();
    expect(deleteBtn.textContent).toBe("Delete?");

    // Second click: delete
    deleteBtn.click();

    await vi.waitFor(() => {
      expect(deletedIds).toEqual([SESSION_A.id]);
    });
  });

  it("confirmation resets after timeout", async () => {
    vi.useFakeTimers();

    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    // Wait for async session loading
    await vi.waitFor(() => {
      expect(root.querySelectorAll(".session-list .session-item").length).toBe(1);
    });

    const deleteBtn = root.querySelector(".session-delete") as HTMLElement;

    // First click: confirm
    deleteBtn.click();
    expect(deleteBtn.textContent).toBe("Delete?");

    // Advance past 3s timeout
    vi.advanceTimersByTime(3100);

    // Should be reset
    expect(deleteBtn.textContent).toBe("\u00d7");
    expect(deleteBtn.style.color).toBe("");
    expect(deleteBtn.style.fontWeight).toBe("");
    expect(deleteBtn.style.display).toBe("");
    expect(deletedIds).toEqual([]);

    vi.useRealTimers();
  });

  it("shows error on delete failure", async () => {
    __setHandler("delete_session", () => {
      throw new Error("Permission denied");
    });

    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".session-list .session-item").length).toBe(1);
    });

    const deleteBtn = root.querySelector(".session-delete") as HTMLElement;

    // First click: confirm
    deleteBtn.click();
    // Second click: delete (will fail)
    deleteBtn.click();

    await vi.waitFor(() => {
      expect(deleteBtn.textContent).toBe("Error!");
    });
  });

  it("clicking session label switches session", async () => {
    __setHandler("open_session", ({ id, resume }: any) => {
      expect(id).toBe(SESSION_A.id);
      expect(resume).toBe(true);
      return SESSION_A;
    });

    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".session-list .session-item").length).toBe(1);
    });

    const label = root.querySelector(".session-list .session-item span") as HTMLElement;
    label.click();

    await vi.waitFor(() => {
      expect(appState.get().sessionId).toBe(SESSION_A.id);
    });
  });

  it("deleting active session switches to new one", async () => {
    appState.update((s) => ({ ...s, sessionId: SESSION_A.id }));

    const root = document.createElement("div");
    document.body.appendChild(root);
    createApp(root);

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".session-list .session-item").length).toBe(1);
    });

    const deleteBtn = root.querySelector(".session-delete") as HTMLElement;
    deleteBtn.click(); // confirm
    deleteBtn.click(); // delete

    await vi.waitFor(() => {
      expect(deletedIds).toEqual([SESSION_A.id]);
      // Should have switched to new session
      expect(appState.get().sessionId).toBe("new-session");
    });
  });
});
