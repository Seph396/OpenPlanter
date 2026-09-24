/** TypeScript interfaces matching Rust event types. */

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

export interface TraceEvent {
  message: string;
}

export interface StepEvent {
  depth: number;
  step: number;
  tool_name: string | null;
  tokens: TokenUsage;
  elapsed_ms: number;
  is_final: boolean;
}

export type DeltaKind = "text" | "thinking" | "tool_call_start" | "tool_call_args";

export interface DeltaEvent {
  kind: DeltaKind;
  text: string;
}

export interface CompleteEvent {
  result: string;
}

export interface ErrorEvent {
  message: string;
}

export interface CuratorUpdateEvent {
  summary: string;
  files_changed: number;
}

export type NodeType = "source" | "section" | "fact";

export interface GraphNode {
  id: string;
  label: string;
  category: string;
  path: string;
  node_type?: NodeType;
  parent_id?: string;
  content?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string | null;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ConfigView {
  provider: string;
  model: string;
  reasoning_effort: string | null;
  workspace: string;
  session_id: string | null;
  recursive: boolean;
  max_depth: number;
  max_steps_per_call: number;
  demo: boolean;
  subtask_model: string | null;
  execute_model: string | null;
  max_exa_agent_calls: number;
}

export interface PartialConfig {
  provider?: string;
  model?: string;
  reasoning_effort?: string;
  recursive?: boolean;
  max_depth?: number;
  /** Empty string clears the override back to "inherit". */
  subtask_model?: string;
  /** Empty string clears the override back to "inherit". */
  execute_model?: string;
  max_exa_agent_calls?: number;
}

export interface ModelInfo {
  id: string;
  name: string | null;
  provider: string;
}

export interface SessionInfo {
  id: string;
  created_at: string;
  turn_count: number;
  last_objective: string | null;
}

export interface PersistentSettings {
  default_model?: string | null;
  default_reasoning_effort?: string | null;
  default_model_openai?: string | null;
  default_model_anthropic?: string | null;
  default_model_openrouter?: string | null;
  default_model_cerebras?: string | null;
  default_model_ollama?: string | null;
}

export interface SlashResult {
  output: string;
  success: boolean;
}

export interface StepToolCallEntry {
  name: string;
  key_arg: string;
  elapsed: number;
}

export interface ReplayEntry {
  seq: number;
  timestamp: string;
  role: string;
  content: string;
  tool_name?: string;
  is_rendered?: boolean;
  step_number?: number;
  step_tokens_in?: number;
  step_tokens_out?: number;
  step_elapsed?: number;
  step_model_preview?: string;
  step_tool_calls?: StepToolCallEntry[];
}

export type AgentEvent =
  | { type: "trace"; message: string }
  | { type: "step"; depth: number; step: number; tool_name: string | null; tokens: TokenUsage; elapsed_ms: number; is_final: boolean; }
  | { type: "delta"; kind: DeltaKind; text: string }
  | { type: "complete"; result: string }
  | { type: "error"; message: string }
  | { type: "wiki_updated"; nodes: GraphNode[]; edges: GraphEdge[] };
