// @vitest-environment happy-dom
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  clamp,
  clampLeftWidth,
  clampRightWidth,
  resizeLeftWidth,
  resizeRightWidth,
  loadLayoutState,
  saveLayoutState,
  DEFAULT_LAYOUT,
  STORAGE_KEY,
  LEFT_MIN,
  LEFT_MAX,
  RIGHT_MIN,
  RIGHT_MAX,
} from "./paneLayout";

describe("clamp", () => {
  it("passes values inside the range through unchanged", () => {
    expect(clamp(50, 0, 100)).toBe(50);
  });
  it("clamps below min", () => {
    expect(clamp(-10, 0, 100)).toBe(0);
  });
  it("clamps above max", () => {
    expect(clamp(150, 0, 100)).toBe(100);
  });
});

describe("clampLeftWidth / clampRightWidth", () => {
  it("clamps left width to [180, 480]", () => {
    expect(clampLeftWidth(100)).toBe(LEFT_MIN);
    expect(clampLeftWidth(300)).toBe(300);
    expect(clampLeftWidth(1000)).toBe(LEFT_MAX);
  });
  it("clamps right width to [240, 900]", () => {
    expect(clampRightWidth(100)).toBe(RIGHT_MIN);
    expect(clampRightWidth(500)).toBe(500);
    expect(clampRightWidth(1500)).toBe(RIGHT_MAX);
  });
});

describe("resizeLeftWidth", () => {
  it("grows when dragging right (positive deltaX)", () => {
    expect(resizeLeftWidth(220, 50)).toBe(270);
  });
  it("shrinks when dragging left (negative deltaX), clamped at the minimum", () => {
    // 220 - 50 = 170, below LEFT_MIN (180)
    expect(resizeLeftWidth(220, -50)).toBe(LEFT_MIN);
  });
  it("clamps at the minimum", () => {
    expect(resizeLeftWidth(200, -1000)).toBe(LEFT_MIN);
  });
  it("clamps at the maximum", () => {
    expect(resizeLeftWidth(400, 1000)).toBe(LEFT_MAX);
  });
});

describe("resizeRightWidth", () => {
  it("grows when dragging left (negative deltaX)", () => {
    expect(resizeRightWidth(340, -50)).toBe(390);
  });
  it("shrinks when dragging right (positive deltaX)", () => {
    expect(resizeRightWidth(340, 50)).toBe(290);
  });
  it("clamps at the minimum", () => {
    expect(resizeRightWidth(300, 1000)).toBe(RIGHT_MIN);
  });
  it("clamps at the maximum", () => {
    expect(resizeRightWidth(800, -1000)).toBe(RIGHT_MAX);
  });
});

// happy-dom under this vitest/node setup does not provide a working global
// `localStorage` (Node's own experimental one is disabled without a flag),
// so tests need their own in-memory Storage stand-in.
function createFakeStorage(): Storage {
  const store = new Map<string, string>();
  return {
    getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
    setItem: (k: string, v: string) => {
      store.set(k, v);
    },
    removeItem: (k: string) => {
      store.delete(k);
    },
    clear: () => {
      store.clear();
    },
    key: (i: number) => Array.from(store.keys())[i] ?? null,
    get length() {
      return store.size;
    },
  } as Storage;
}

describe("loadLayoutState / saveLayoutState", () => {
  beforeEach(() => {
    vi.stubGlobal("localStorage", createFakeStorage());
  });

  it("returns defaults when nothing is persisted", () => {
    expect(loadLayoutState()).toEqual(DEFAULT_LAYOUT);
  });

  it("round-trips a saved state", () => {
    const state = { leftWidth: 260, rightWidth: 400, leftCollapsed: true, rightCollapsed: false };
    saveLayoutState(state);
    expect(loadLayoutState()).toEqual(state);
  });

  it("falls back to defaults on corrupt JSON", () => {
    localStorage.setItem(STORAGE_KEY, "{not json");
    expect(loadLayoutState()).toEqual(DEFAULT_LAYOUT);
  });

  it("falls back to defaults on a shape that doesn't match", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ foo: "bar" }));
    expect(loadLayoutState()).toEqual(DEFAULT_LAYOUT);
  });

  it("clamps persisted widths that are out of range", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ leftWidth: 5, rightWidth: 5000, leftCollapsed: false, rightCollapsed: false })
    );
    const loaded = loadLayoutState();
    expect(loaded.leftWidth).toBe(LEFT_MIN);
    expect(loaded.rightWidth).toBe(RIGHT_MAX);
  });

  it("saveLayoutState does not throw when localStorage.setItem throws", () => {
    vi.stubGlobal("localStorage", {
      ...createFakeStorage(),
      setItem: () => {
        throw new Error("quota exceeded");
      },
    });
    expect(() => saveLayoutState(DEFAULT_LAYOUT)).not.toThrow();
  });
});
