/**
 * Pure state/math helpers for the resizable/collapsible three-pane layout.
 * No DOM access here (except localStorage) so drag math and persistence are
 * unit-testable without mounting components.
 */

export const STORAGE_KEY = "op.layout.v1";

export const LEFT_MIN = 180;
export const LEFT_MAX = 480;
export const LEFT_DEFAULT = 220;

export const RIGHT_MIN = 240;
export const RIGHT_MAX = 900;
export const RIGHT_DEFAULT = 340;

/** Width of a collapsed pane's rail (chevron-only). */
export const COLLAPSED_WIDTH = 28;

export interface PaneLayoutState {
  leftWidth: number;
  rightWidth: number;
  leftCollapsed: boolean;
  rightCollapsed: boolean;
}

export const DEFAULT_LAYOUT: PaneLayoutState = {
  leftWidth: LEFT_DEFAULT,
  rightWidth: RIGHT_DEFAULT,
  leftCollapsed: false,
  rightCollapsed: false,
};

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function clampLeftWidth(width: number): number {
  return clamp(width, LEFT_MIN, LEFT_MAX);
}

export function clampRightWidth(width: number): number {
  return clamp(width, RIGHT_MIN, RIGHT_MAX);
}

/** Left pane's gutter sits on its right (inner) edge — dragging right grows it. */
export function resizeLeftWidth(startWidth: number, deltaX: number): number {
  return clampLeftWidth(startWidth + deltaX);
}

/** Right pane's gutter sits on its left (inner) edge — dragging left grows it. */
export function resizeRightWidth(startWidth: number, deltaX: number): number {
  return clampRightWidth(startWidth - deltaX);
}

function isValidLayout(value: unknown): value is PaneLayoutState {
  if (!value || typeof value !== "object") return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.leftWidth === "number" &&
    typeof o.rightWidth === "number" &&
    typeof o.leftCollapsed === "boolean" &&
    typeof o.rightCollapsed === "boolean"
  );
}

/** Load persisted layout state, guarded with try/catch. Falls back to defaults on any failure. */
export function loadLayoutState(): PaneLayoutState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_LAYOUT };
    const parsed: unknown = JSON.parse(raw);
    if (!isValidLayout(parsed)) return { ...DEFAULT_LAYOUT };
    return {
      leftWidth: clampLeftWidth(parsed.leftWidth),
      rightWidth: clampRightWidth(parsed.rightWidth),
      leftCollapsed: parsed.leftCollapsed,
      rightCollapsed: parsed.rightCollapsed,
    };
  } catch {
    return { ...DEFAULT_LAYOUT };
  }
}

/** Persist layout state, guarded with try/catch (quota/availability errors are non-fatal). */
export function saveLayoutState(state: PaneLayoutState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // best-effort only
  }
}
