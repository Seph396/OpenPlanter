use std::collections::HashMap;
use std::path::PathBuf;
use serde::Deserialize;
use tauri::State;
use crate::state::{load_config_for, save_credential, save_ui_prefs, AppState, UiPrefs};
use op_core::events::{ConfigView, ModelInfo};
use op_core::settings::{PersistentSettings, SettingsStore};
use op_core::credentials::credentials_from_env;
use op_core::workspace::persist_last_workspace;

/// Partial configuration update from the frontend's editable sidebar.
///
/// Op-tauri-local (not `op_core::events::PartialConfig`) so the sidebar's
/// recursive/max_depth controls don't require changing the shared op-core
/// event schema owned by the backend agent's worktree.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub recursive: Option<bool>,
    pub max_depth: Option<i64>,
    /// `Some("")` clears the override back to "inherit"; `None` leaves it unchanged.
    pub subtask_model: Option<String>,
    /// `Some("")` clears the override back to "inherit"; `None` leaves it unchanged.
    pub execute_model: Option<String>,
    pub max_exa_agent_calls: Option<u32>,
    pub max_output_tokens: Option<u64>,
    pub exa_agent_timeout_sec: Option<u64>,
}

/// Get the current configuration.
#[tauri::command]
pub async fn get_config(
    state: State<'_, AppState>,
) -> Result<ConfigView, String> {
    let cfg = state.config.lock().await;
    let session_id = state.session_id.lock().await;
    Ok(ConfigView {
        provider: cfg.provider.clone(),
        model: cfg.model.clone(),
        reasoning_effort: cfg.reasoning_effort.clone(),
        workspace: cfg.workspace.display().to_string(),
        session_id: session_id.clone(),
        recursive: cfg.recursive,
        max_depth: cfg.max_depth,
        max_steps_per_call: cfg.max_steps_per_call,
        demo: cfg.demo,
        subtask_model: cfg.subtask_model.clone(),
        execute_model: cfg.execute_model.clone(),
        max_exa_agent_calls: cfg.max_exa_agent_calls,
        max_output_tokens: cfg.max_output_tokens,
        exa_agent_timeout_sec: cfg.exa_agent_timeout_sec,
    })
}

/// Apply a `PartialConfig` update onto `cfg` in place. Pure/testable — no I/O.
pub fn apply_partial_config(cfg: &mut op_core::config::AgentConfig, partial: PartialConfig) {
    if let Some(provider) = partial.provider {
        cfg.provider = provider;
    }
    if let Some(model) = partial.model {
        cfg.model = model;
    }
    if let Some(effort) = partial.reasoning_effort {
        cfg.reasoning_effort = if effort.is_empty() {
            None
        } else {
            Some(effort)
        };
    }
    if let Some(recursive) = partial.recursive {
        cfg.recursive = recursive;
    }
    if let Some(max_depth) = partial.max_depth {
        cfg.max_depth = max_depth;
    }
    if let Some(subtask_model) = partial.subtask_model {
        cfg.subtask_model = if subtask_model.is_empty() { None } else { Some(subtask_model) };
    }
    if let Some(execute_model) = partial.execute_model {
        cfg.execute_model = if execute_model.is_empty() { None } else { Some(execute_model) };
    }
    if let Some(max_exa_agent_calls) = partial.max_exa_agent_calls {
        cfg.max_exa_agent_calls = max_exa_agent_calls;
    }
    if let Some(max_output_tokens) = partial.max_output_tokens {
        cfg.max_output_tokens = max_output_tokens;
    }
    if let Some(exa_agent_timeout_sec) = partial.exa_agent_timeout_sec {
        cfg.exa_agent_timeout_sec = exa_agent_timeout_sec;
    }
}

/// Update configuration fields.
#[tauri::command]
pub async fn update_config(
    partial: PartialConfig,
    state: State<'_, AppState>,
) -> Result<ConfigView, String> {
    let mut cfg = state.config.lock().await;
    apply_partial_config(&mut cfg, partial);

    // Persist the sidebar's selections so they survive relaunch.
    let prefs = UiPrefs {
        provider: Some(cfg.provider.clone()),
        model: Some(cfg.model.clone()),
        reasoning_effort: cfg.reasoning_effort.clone(),
        recursive: Some(cfg.recursive),
        max_depth: Some(cfg.max_depth),
        subtask_model: cfg.subtask_model.clone(),
        execute_model: cfg.execute_model.clone(),
        max_exa_agent_calls: Some(cfg.max_exa_agent_calls),
        max_output_tokens: Some(cfg.max_output_tokens),
        exa_agent_timeout_sec: Some(cfg.exa_agent_timeout_sec),
    };
    if let Err(e) = save_ui_prefs(&cfg.workspace, &prefs) {
        eprintln!("[config] failed to persist UI prefs: {e}");
    }

    let session_id = state.session_id.lock().await;
    Ok(ConfigView {
        provider: cfg.provider.clone(),
        model: cfg.model.clone(),
        reasoning_effort: cfg.reasoning_effort.clone(),
        workspace: cfg.workspace.display().to_string(),
        session_id: session_id.clone(),
        recursive: cfg.recursive,
        max_depth: cfg.max_depth,
        max_steps_per_call: cfg.max_steps_per_call,
        demo: cfg.demo,
        subtask_model: cfg.subtask_model.clone(),
        execute_model: cfg.execute_model.clone(),
        max_exa_agent_calls: cfg.max_exa_agent_calls,
        max_output_tokens: cfg.max_output_tokens,
        exa_agent_timeout_sec: cfg.exa_agent_timeout_sec,
    })
}

/// Save an API key for `provider`: writes to the user credential store
/// (`~/.openplanter/credentials.json`) and, on macOS, the Keychain. Never
/// logs `value`. Reloads config for the active workspace so the new
/// credential is merged in immediately, and returns the fresh status map.
#[tauri::command]
pub async fn set_credential(
    provider: String,
    value: String,
    state: State<'_, AppState>,
) -> Result<HashMap<String, bool>, String> {
    let provider = provider.trim().to_lowercase();
    let value = value.trim().to_string();
    if value.is_empty() {
        return Err("Credential value must not be empty".to_string());
    }

    save_credential(&provider, &value)?;

    let mut cfg = state.config.lock().await;
    let workspace = cfg.workspace.clone();
    *cfg = load_config_for(&workspace);

    Ok(build_credential_status(&cfg))
}

/// Known models per provider for listing.
fn known_models_for_provider(provider: &str) -> Vec<ModelInfo> {
    let models: Vec<(&str, &str)> = match provider {
        "openai" => vec![
            ("gpt-5.2", "GPT-5.2"),
            ("gpt-4o", "GPT-4o"),
            ("gpt-4o-mini", "GPT-4o Mini"),
            ("o1", "o1"),
            ("o3", "o3"),
            ("o4-mini", "o4-mini"),
        ],
        "anthropic" => vec![
            ("claude-opus-4-6", "Claude Opus 4.6"),
            ("claude-sonnet-4-5", "Claude Sonnet 4.5"),
            ("claude-haiku-4-5", "Claude Haiku 4.5"),
            ("claude-opus-5", "Claude Opus 5"),
            ("claude-sonnet-5", "Claude Sonnet 5"),
            ("claude-fable-5-1", "Claude Fable 5.1"),
        ],
        "openrouter" => vec![
            ("anthropic/claude-sonnet-4-5", "Claude Sonnet 4.5 (OR)"),
            ("anthropic/claude-opus-4-6", "Claude Opus 4.6 (OR)"),
            ("openai/gpt-5.2", "GPT-5.2 (OR)"),
        ],
        "cerebras" => vec![
            ("qwen-3-235b-a22b-instruct-2507", "Qwen-3 235B"),
            ("llama-4-scout-17b-16e-instruct", "Llama-4 Scout"),
        ],
        "ollama" => vec![
            ("llama3.2", "Llama 3.2"),
            ("mistral", "Mistral"),
            ("gemma", "Gemma"),
            ("phi", "Phi"),
            ("deepseek", "DeepSeek"),
            ("qwen2", "Qwen 2"),
        ],
        _ => vec![],
    };

    models
        .into_iter()
        .map(|(id, name)| ModelInfo {
            id: id.to_string(),
            name: Some(name.to_string()),
            provider: provider.to_string(),
        })
        .collect()
}

/// List available models for a provider.
#[tauri::command]
pub async fn list_models(
    provider: String,
    _state: State<'_, AppState>,
) -> Result<Vec<ModelInfo>, String> {
    if provider == "all" {
        let mut all = Vec::new();
        for p in &["openai", "anthropic", "openrouter", "cerebras", "ollama"] {
            all.extend(known_models_for_provider(p));
        }
        Ok(all)
    } else {
        Ok(known_models_for_provider(&provider))
    }
}

/// Validate that `path` refers to an existing directory, returning it as a `PathBuf`.
pub fn validate_workspace_dir(path: &str) -> Result<PathBuf, String> {
    let workspace = PathBuf::from(path);
    if !workspace.is_dir() {
        return Err(format!("Workspace directory does not exist: {}", path));
    }
    Ok(workspace)
}

/// Switch the active workspace: validates the directory exists, rebuilds the
/// `AgentConfig` for it (same credential merge as startup), swaps it into
/// state, clears the active session, and persists it as the last-used workspace.
#[tauri::command]
pub async fn set_workspace(
    path: String,
    state: State<'_, AppState>,
) -> Result<ConfigView, String> {
    let workspace = validate_workspace_dir(&path)?;

    let new_cfg = load_config_for(&workspace);
    persist_last_workspace(&new_cfg.workspace);

    let mut cfg = state.config.lock().await;
    *cfg = new_cfg;

    let mut session_id = state.session_id.lock().await;
    *session_id = None;

    Ok(ConfigView {
        provider: cfg.provider.clone(),
        model: cfg.model.clone(),
        reasoning_effort: cfg.reasoning_effort.clone(),
        workspace: cfg.workspace.display().to_string(),
        session_id: session_id.clone(),
        recursive: cfg.recursive,
        max_depth: cfg.max_depth,
        max_steps_per_call: cfg.max_steps_per_call,
        demo: cfg.demo,
        subtask_model: cfg.subtask_model.clone(),
        execute_model: cfg.execute_model.clone(),
        max_exa_agent_calls: cfg.max_exa_agent_calls,
        max_output_tokens: cfg.max_output_tokens,
        exa_agent_timeout_sec: cfg.exa_agent_timeout_sec,
    })
}

/// Save persistent settings to disk.
#[tauri::command]
pub async fn save_settings(
    settings: PersistentSettings,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let cfg = state.config.lock().await;
    let store = SettingsStore::new(&cfg.workspace, &cfg.session_root_dir);
    store.save(&settings).map_err(|e| e.to_string())
}

/// Build credential status from config: which providers/services have API keys configured.
pub fn build_credential_status(cfg: &op_core::config::AgentConfig) -> HashMap<String, bool> {
    let mut status = HashMap::new();
    status.insert("openai".to_string(), cfg.openai_api_key.is_some());
    status.insert("anthropic".to_string(), cfg.anthropic_api_key.is_some());
    status.insert("openrouter".to_string(), cfg.openrouter_api_key.is_some());
    status.insert("cerebras".to_string(), cfg.cerebras_api_key.is_some());
    status.insert("ollama".to_string(), true); // Ollama never needs a key
    status.insert("exa".to_string(), cfg.exa_api_key.is_some());
    status
}

/// Get credential status: which providers/services have API keys configured.
#[tauri::command]
pub async fn get_credentials_status(
    state: State<'_, AppState>,
) -> Result<HashMap<String, bool>, String> {
    let cfg = state.config.lock().await;
    let env_creds = credentials_from_env();

    let mut status = HashMap::new();
    status.insert(
        "openai".to_string(),
        cfg.openai_api_key.is_some() || env_creds.openai_api_key.is_some(),
    );
    status.insert(
        "anthropic".to_string(),
        cfg.anthropic_api_key.is_some() || env_creds.anthropic_api_key.is_some(),
    );
    status.insert(
        "openrouter".to_string(),
        cfg.openrouter_api_key.is_some() || env_creds.openrouter_api_key.is_some(),
    );
    status.insert(
        "cerebras".to_string(),
        cfg.cerebras_api_key.is_some() || env_creds.cerebras_api_key.is_some(),
    );
    status.insert("ollama".to_string(), true); // Ollama never needs a key
    status.insert(
        "exa".to_string(),
        cfg.exa_api_key.is_some() || env_creds.exa_api_key.is_some(),
    );
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ── validate_workspace_dir ──

    #[test]
    fn test_validate_workspace_dir_missing_rejected() {
        let result = validate_workspace_dir("/definitely/does/not/exist/openplanter-xyz");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_workspace_dir_existing_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let result = validate_workspace_dir(dir.path().to_str().unwrap());
        assert_eq!(result.unwrap(), dir.path());
    }

    #[test]
    fn test_validate_workspace_dir_file_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("not-a-dir.txt");
        std::fs::write(&file_path, "x").unwrap();
        let result = validate_workspace_dir(file_path.to_str().unwrap());
        assert!(result.is_err());
    }

    // ── known_models_for_provider ──

    #[test]
    fn test_openai_models_nonempty() {
        let models = known_models_for_provider("openai");
        assert!(!models.is_empty(), "openai should have known models");
    }

    #[test]
    fn test_anthropic_models_nonempty() {
        let models = known_models_for_provider("anthropic");
        assert!(!models.is_empty(), "anthropic should have known models");
    }

    #[test]
    fn test_openrouter_models_nonempty() {
        let models = known_models_for_provider("openrouter");
        assert!(!models.is_empty(), "openrouter should have known models");
    }

    #[test]
    fn test_cerebras_models_nonempty() {
        let models = known_models_for_provider("cerebras");
        assert!(!models.is_empty(), "cerebras should have known models");
    }

    #[test]
    fn test_ollama_models_nonempty() {
        let models = known_models_for_provider("ollama");
        assert!(!models.is_empty(), "ollama should have known models");
    }

    #[test]
    fn test_unknown_provider_empty() {
        let models = known_models_for_provider("foo");
        assert!(models.is_empty(), "unknown provider should return empty vec");
    }

    #[test]
    fn test_all_providers_model_ids_unique() {
        let mut all_ids = HashSet::new();
        for p in &["openai", "anthropic", "openrouter", "cerebras", "ollama"] {
            for m in known_models_for_provider(p) {
                assert!(
                    all_ids.insert(m.id.clone()),
                    "duplicate model ID: {}",
                    m.id
                );
            }
        }
    }

    #[test]
    fn test_model_info_fields() {
        for provider in &["openai", "anthropic", "openrouter", "cerebras", "ollama"] {
            for m in known_models_for_provider(provider) {
                assert!(!m.id.is_empty(), "model id should not be empty");
                assert!(m.name.is_some(), "model name should be Some for {}", m.id);
                assert_eq!(m.provider, *provider, "provider mismatch for {}", m.id);
            }
        }
    }

    // ── build_credential_status ──

    #[test]
    fn test_cred_status_all_none() {
        let cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        // Force all keys to None
        let mut cfg = cfg;
        cfg.openai_api_key = None;
        cfg.anthropic_api_key = None;
        cfg.openrouter_api_key = None;
        cfg.cerebras_api_key = None;
        let status = build_credential_status(&cfg);
        assert_eq!(status["openai"], false);
        assert_eq!(status["anthropic"], false);
        assert_eq!(status["openrouter"], false);
        assert_eq!(status["cerebras"], false);
        assert_eq!(status["ollama"], true, "ollama always true");
    }

    #[test]
    fn test_cred_status_openai_set() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.openai_api_key = Some("sk-test".to_string());
        cfg.anthropic_api_key = None;
        cfg.openrouter_api_key = None;
        cfg.cerebras_api_key = None;
        let status = build_credential_status(&cfg);
        assert_eq!(status["openai"], true);
        assert_eq!(status["anthropic"], false);
    }

    #[test]
    fn test_cred_status_anthropic_set() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.openai_api_key = None;
        cfg.anthropic_api_key = Some("sk-ant-test".to_string());
        cfg.openrouter_api_key = None;
        cfg.cerebras_api_key = None;
        let status = build_credential_status(&cfg);
        assert_eq!(status["anthropic"], true);
        assert_eq!(status["openai"], false);
    }

    #[test]
    fn test_cred_status_ollama_always_true() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.openai_api_key = None;
        cfg.anthropic_api_key = None;
        cfg.openrouter_api_key = None;
        cfg.cerebras_api_key = None;
        let status = build_credential_status(&cfg);
        assert_eq!(status["ollama"], true);
    }

    #[test]
    fn test_cred_status_all_set() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.openai_api_key = Some("k1".to_string());
        cfg.anthropic_api_key = Some("k2".to_string());
        cfg.openrouter_api_key = Some("k3".to_string());
        cfg.cerebras_api_key = Some("k4".to_string());
        cfg.exa_api_key = Some("k5".to_string());
        let status = build_credential_status(&cfg);
        for (provider, has_key) in &status {
            assert!(has_key, "{} should be true when key is set", provider);
        }
    }

    #[test]
    fn test_cred_status_has_six_entries() {
        let cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        let status = build_credential_status(&cfg);
        assert_eq!(status.len(), 6, "should have 6 entries (5 providers + exa)");
    }

    // ── apply_partial_config (recursive / max_depth) ──

    #[test]
    fn test_apply_partial_config_recursive_and_max_depth() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.recursive = true;
        cfg.max_depth = 4;

        apply_partial_config(
            &mut cfg,
            PartialConfig {
                recursive: Some(false),
                max_depth: Some(8),
                ..Default::default()
            },
        );

        assert!(!cfg.recursive);
        assert_eq!(cfg.max_depth, 8);
    }

    #[test]
    fn test_apply_partial_config_max_exa_agent_calls() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.max_exa_agent_calls = 12;

        apply_partial_config(
            &mut cfg,
            PartialConfig {
                max_exa_agent_calls: Some(5),
                ..Default::default()
            },
        );
        assert_eq!(cfg.max_exa_agent_calls, 5);

        // None leaves it unchanged.
        apply_partial_config(&mut cfg, PartialConfig::default());
        assert_eq!(cfg.max_exa_agent_calls, 5);
    }

    #[test]
    fn test_apply_partial_config_max_output_tokens() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.max_output_tokens = 32768;

        apply_partial_config(
            &mut cfg,
            PartialConfig {
                max_output_tokens: Some(8192),
                ..Default::default()
            },
        );
        assert_eq!(cfg.max_output_tokens, 8192);

        // None leaves it unchanged.
        apply_partial_config(&mut cfg, PartialConfig::default());
        assert_eq!(cfg.max_output_tokens, 8192);
    }

    #[test]
    fn test_apply_partial_config_none_fields_preserve_existing() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.recursive = true;
        cfg.max_depth = 4;
        cfg.provider = "anthropic".to_string();

        apply_partial_config(&mut cfg, PartialConfig::default());

        assert!(cfg.recursive);
        assert_eq!(cfg.max_depth, 4);
        assert_eq!(cfg.provider, "anthropic");
    }

    #[test]
    fn test_apply_partial_config_all_fields() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");

        apply_partial_config(
            &mut cfg,
            PartialConfig {
                provider: Some("openrouter".to_string()),
                model: Some("anthropic/claude-sonnet-4-5".to_string()),
                reasoning_effort: Some("medium".to_string()),
                recursive: Some(false),
                max_depth: Some(2),
                subtask_model: Some("claude-sonnet-5".to_string()),
                execute_model: Some("claude-haiku-4-5".to_string()),
                max_exa_agent_calls: Some(20),
                max_output_tokens: Some(65536),
                exa_agent_timeout_sec: Some(180),
            },
        );

        assert_eq!(cfg.provider, "openrouter");
        assert_eq!(cfg.model, "anthropic/claude-sonnet-4-5");
        assert_eq!(cfg.reasoning_effort, Some("medium".to_string()));
        assert!(!cfg.recursive);
        assert_eq!(cfg.max_depth, 2);
        assert_eq!(cfg.max_output_tokens, 65536);
        assert_eq!(cfg.subtask_model, Some("claude-sonnet-5".to_string()));
        assert_eq!(cfg.execute_model, Some("claude-haiku-4-5".to_string()));
        assert_eq!(cfg.max_exa_agent_calls, 20);
        assert_eq!(cfg.exa_agent_timeout_sec, 180);
    }

    #[test]
    fn test_apply_partial_config_exa_agent_timeout_sec() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.exa_agent_timeout_sec = 300;

        apply_partial_config(
            &mut cfg,
            PartialConfig {
                exa_agent_timeout_sec: Some(120),
                ..Default::default()
            },
        );
        assert_eq!(cfg.exa_agent_timeout_sec, 120);

        // None leaves it unchanged.
        apply_partial_config(&mut cfg, PartialConfig::default());
        assert_eq!(cfg.exa_agent_timeout_sec, 120);
    }

    #[test]
    fn test_apply_partial_config_subtask_execute_model_clear_to_inherit() {
        let mut cfg = op_core::config::AgentConfig::from_env("/nonexistent");
        cfg.subtask_model = Some("claude-sonnet-5".into());
        cfg.execute_model = Some("claude-haiku-4-5".into());

        apply_partial_config(
            &mut cfg,
            PartialConfig {
                subtask_model: Some(String::new()),
                execute_model: Some(String::new()),
                ..Default::default()
            },
        );

        assert_eq!(cfg.subtask_model, None);
        assert_eq!(cfg.execute_model, None);
    }

    #[test]
    fn test_partial_config_deserializes_from_json() {
        let json = r#"{"recursive": false, "max_depth": 7}"#;
        let partial: PartialConfig = serde_json::from_str(json).unwrap();
        assert_eq!(partial.recursive, Some(false));
        assert_eq!(partial.max_depth, Some(7));
        assert!(partial.provider.is_none());
    }
}
