// @vitest-environment happy-dom
import { vi, describe, it, expect, beforeAll, beforeEach, afterEach } from "vitest";
import { __setHandler, __clearHandlers } from "../__mocks__/tauri";

vi.mock("@tauri-apps/api/core", async () => {
  const mock = await import("../__mocks__/tauri");
  return { invoke: mock.invoke };
});

vi.mock("../graph/cytoGraph", () => ({
  initGraph: vi.fn(),
  updateGraph: vi.fn(),
  destroyGraph: vi.fn(),
  fitView: vi.fn(),
  focusNode: vi.fn(),
  setLayout: vi.fn(),
  getCurrentLayout: () => "fcose",
  filterByCategory: vi.fn(),
  filterByTier: vi.fn(),
  filterBySearch: () => [],
  filterBySession: () => 0,
  fitSearchMatches: vi.fn(),
  getCategories: () => [],
  getNodeIds: () => new Set<string>(),
}));

vi.mock("../graph/interaction", () => ({
  bindInteractions: vi.fn(),
}));

import { createGraphPane } from "./GraphPane";

function dispatchDelta(kind: string, text: string): void {
  window.dispatchEvent(new CustomEvent("agent-delta", { detail: { kind, text } }));
}

describe("GraphPane: live refresh on wiki writes", () => {
  let getGraphDataCalls = 0;

  // GraphPane attaches its listeners to `window` with no unmount/cleanup API,
  // so we build exactly one instance for this whole suite (matching the real
  // app, which never re-mounts it) instead of one per test — otherwise each
  // test would stack another live listener on the shared `window` and inflate
  // every subsequent test's call count.
  beforeAll(async () => {
    vi.useFakeTimers();
    __setHandler("get_graph_data", () => {
      getGraphDataCalls++;
      return { nodes: [], edges: [] };
    });
    createGraphPane();
    await vi.runOnlyPendingTimersAsync(); // flush the mount-time getGraphData call
  });

  beforeEach(() => {
    getGraphDataCalls = 0;
  });

  afterEach(() => {
    __clearHandlers();
    __setHandler("get_graph_data", () => {
      getGraphDataCalls++;
      return { nodes: [], edges: [] };
    });
  });

  it("does not refresh for a write_file call outside .openplanter/wiki/", async () => {
    dispatchDelta("tool_call_start", "write_file");
    dispatchDelta("tool_call_args", '{"path": "src/notes.md"}');
    await vi.advanceTimersByTimeAsync(2500);

    expect(getGraphDataCalls).toBe(0);
  });

  it("refreshes ~2s (debounced) after a write_file call touching .openplanter/wiki/", async () => {
    dispatchDelta("tool_call_start", "write_file");
    dispatchDelta("tool_call_args", '{"path": "');
    dispatchDelta("tool_call_args", '.openplanter/wiki/fec.md"}');

    // Not yet — still debouncing.
    await vi.advanceTimersByTimeAsync(1000);
    expect(getGraphDataCalls).toBe(0);

    await vi.advanceTimersByTimeAsync(1100);
    expect(getGraphDataCalls).toBe(1);
  });

  it("collapses a burst of edits to the same file into a single refresh", async () => {
    for (let i = 0; i < 5; i++) {
      dispatchDelta("tool_call_start", "edit_file");
      dispatchDelta("tool_call_args", '{"path": ".openplanter/wiki/fec.md"}');
      await vi.advanceTimersByTimeAsync(200);
    }
    await vi.advanceTimersByTimeAsync(2100);

    expect(getGraphDataCalls).toBe(1);
  });

  it("ignores non-path tools (e.g. run_shell)", async () => {
    dispatchDelta("tool_call_start", "run_shell");
    dispatchDelta("tool_call_args", '{"command": "cat .openplanter/wiki/fec.md"}');
    await vi.advanceTimersByTimeAsync(2500);

    expect(getGraphDataCalls).toBe(0);
  });
});
