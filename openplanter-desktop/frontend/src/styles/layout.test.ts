// @vitest-environment happy-dom
//
// Regression test for the three-pane layout collapse bug: during long runs,
// an unbroken long token (path, hash, base64 blob, etc.) in the chat pane
// grew that grid track and pushed the sidebar/graph pane off-screen.
//
// happy-dom does not run a real layout/paint engine, so we can't assert
// pixel widths here. Instead we assert the two halves of the actual fix:
//  1. the CSS rules that prevent grid-track blowout are present in the
//     shipped stylesheet (min-width: 0 on every grid-column pane, plus
//     overflow-wrap: anywhere on .message so a single long token still
//     wraps instead of forcing intrinsic width), and
//  2. rendering a message with a 5,000-char unbroken string doesn't get
//     truncated or throw — the DOM half of the same code path.
import { readFileSync } from "fs";
import { resolve } from "path";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";

vi.mock("@tauri-apps/api/core", async () => {
  const mock = await import("../__mocks__/tauri");
  return { invoke: mock.invoke };
});

vi.mock("./InputBar", () => ({
  createInputBar: () => document.createElement("div"),
}));

// Import from the actual component path used elsewhere in the suite.
import { appState, type ChatMessage } from "../state/store";
import { createChatPane } from "../components/ChatPane";

const CSS_PATH = resolve(__dirname, "main.css");

function extractRuleBlock(css: string, selector: string): string {
  const idx = css.indexOf(`${selector} {`);
  expect(idx, `selector "${selector}" not found in main.css`).toBeGreaterThan(-1);
  const close = css.indexOf("}", idx);
  return css.slice(idx, close);
}

describe("three-pane layout: grid blowout fix (CSS)", () => {
  const css = readFileSync(CSS_PATH, "utf-8");

  it(".sidebar has min-width: 0 so it can shrink below content size", () => {
    expect(extractRuleBlock(css, ".sidebar")).toMatch(/min-width:\s*0/);
  });

  it(".chat-pane has min-width: 0 so long tool output can't push other panes off-screen", () => {
    expect(extractRuleBlock(css, ".chat-pane")).toMatch(/min-width:\s*0/);
  });

  it(".graph-pane has min-width: 0 so it can shrink below content size", () => {
    expect(extractRuleBlock(css, ".graph-pane")).toMatch(/min-width:\s*0/);
  });

  it(".message wraps a single very long unbroken token via overflow-wrap: anywhere", () => {
    expect(extractRuleBlock(css, ".message")).toMatch(/overflow-wrap:\s*anywhere/);
  });
});

describe("three-pane layout: 5,000-char unbroken string renders without truncation", () => {
  const originalState = appState.get();

  beforeEach(() => {
    appState.set({ ...originalState, messages: [] });
  });

  afterEach(() => {
    appState.set(originalState);
  });

  it("renders a message containing a 5000-char unbroken token intact", () => {
    const longToken = "a".repeat(5000);
    const pane = createChatPane();
    const msg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: longToken,
      timestamp: Date.now(),
    };
    appState.update((s) => ({ ...s, messages: [msg] }));

    const el = pane.querySelector(".message.assistant");
    expect(el).not.toBeNull();
    expect(el!.textContent!.length).toBe(5000);
    expect(el!.textContent).toBe(longToken);
  });
});
