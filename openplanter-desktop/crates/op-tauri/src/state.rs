use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use op_core::config::AgentConfig;
use op_core::credentials::{credentials_from_env, discover_env_candidates, parse_env_file, CredentialBundle, UserCredentialStore};
use op_core::workspace::{resolve_workspace, UserSettingsStore};

/// Merge credentials into an AgentConfig.
/// Priority: existing config value > env_creds > file_creds.
pub fn merge_credentials_into_config(
    cfg: &mut AgentConfig,
    env_creds: &CredentialBundle,
    file_creds: &CredentialBundle,
) {
    macro_rules! merge {
        ($field:ident) => {
            if cfg.$field.is_none() {
                cfg.$field = env_creds.$field.clone()
                    .or_else(|| file_creds.$field.clone());
            }
        };
    }
    merge!(openai_api_key);
    merge!(anthropic_api_key);
    merge!(openrouter_api_key);
    merge!(cerebras_api_key);
    merge!(exa_api_key);
    merge!(voyage_api_key);
}

/// Build an `AgentConfig` for `workspace`: env vars, then `.env` file credentials,
/// then (macOS only) Keychain fallback for any keys still missing.
///
/// Shared by startup (`AppState::new`) and the `set_workspace` command so both
/// paths apply the exact same credential merge logic.
pub fn load_config_for(workspace: &Path) -> AgentConfig {
    let mut cfg = AgentConfig::from_env(workspace);

    // Load .env files and merge credentials into config
    let env_creds = credentials_from_env();
    let candidates = discover_env_candidates(&cfg.workspace);
    for candidate in &candidates {
        let file_creds = parse_env_file(candidate);
        merge_credentials_into_config(&mut cfg, &env_creds, &file_creds);
    }

    // If no .env candidates found, still merge from process env
    if candidates.is_empty() {
        let empty = CredentialBundle::default();
        merge_credentials_into_config(&mut cfg, &env_creds, &empty);
    }

    // Merge in the user-level credential store (~/.openplanter/credentials.json),
    // written by the in-app "Set…" credential UI (see `save_credential`).
    let user_creds = UserCredentialStore::new().load();
    merge_credentials_into_config(&mut cfg, &user_creds, &CredentialBundle::default());

    apply_keychain_fallback(&mut cfg);

    apply_ui_prefs(&mut cfg);

    cfg
}

/// Op-tauri-local UI preferences that survive relaunch: provider/model/reasoning/
/// recursive/max_depth chosen via the editable sidebar. Stored separately from
/// op-core's `PersistentSettings` (which only covers model/reasoning defaults)
/// at `{workspace}/.openplanter/ui_prefs.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UiPrefs {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub recursive: Option<bool>,
    pub max_depth: Option<i64>,
    /// Sub-agent model override for `subtask` children. `Some(None)`-shaped
    /// persistence isn't representable here (same limitation as
    /// `reasoning_effort` above) — an explicit "cleared back to inherit"
    /// looks identical to "never set" on reload.
    #[serde(default)]
    pub subtask_model: Option<String>,
    #[serde(default)]
    pub execute_model: Option<String>,
    #[serde(default)]
    pub max_exa_agent_calls: Option<u32>,
}

fn ui_prefs_path(workspace: &Path) -> PathBuf {
    workspace.join(".openplanter").join("ui_prefs.json")
}

/// Load persisted UI prefs for `workspace`, if any. Never fails — returns
/// `UiPrefs::default()` on any read/parse error.
pub fn load_ui_prefs(workspace: &Path) -> UiPrefs {
    let path = ui_prefs_path(workspace);
    match fs::read_to_string(&path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => UiPrefs::default(),
    }
}

/// Persist `prefs` for `workspace`. Best-effort: logs nothing, returns the
/// io::Result so callers can surface a warning if desired.
pub fn save_ui_prefs(workspace: &Path, prefs: &UiPrefs) -> std::io::Result<()> {
    let path = ui_prefs_path(workspace);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(prefs)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    fs::write(&path, json)
}

/// Apply any persisted UI prefs onto `cfg` (called after every load so relaunch
/// restores the last sidebar selection).
fn apply_ui_prefs(cfg: &mut AgentConfig) {
    let prefs = load_ui_prefs(&cfg.workspace);
    if let Some(provider) = prefs.provider {
        cfg.provider = provider;
    }
    if let Some(model) = prefs.model {
        cfg.model = model;
    }
    if prefs.reasoning_effort.is_some() {
        cfg.reasoning_effort = prefs.reasoning_effort;
    }
    if let Some(recursive) = prefs.recursive {
        cfg.recursive = recursive;
    }
    if let Some(max_depth) = prefs.max_depth {
        cfg.max_depth = max_depth;
    }
    if prefs.subtask_model.is_some() {
        cfg.subtask_model = prefs.subtask_model;
    }
    if prefs.execute_model.is_some() {
        cfg.execute_model = prefs.execute_model;
    }
    if let Some(max_exa_agent_calls) = prefs.max_exa_agent_calls {
        cfg.max_exa_agent_calls = max_exa_agent_calls;
    }
}

/// Save `provider`'s credential: to the user credential store, and (macOS only)
/// to the Keychain under `openplanter-<provider>` (or the operator's configured
/// override — see `keychain_service_name`). Never logs `value`.
pub fn save_credential(provider: &str, value: &str) -> Result<(), String> {
    let store = UserCredentialStore::new();
    let mut bundle = store.load();
    match provider {
        "openai" => bundle.openai_api_key = Some(value.to_string()),
        "anthropic" => bundle.anthropic_api_key = Some(value.to_string()),
        "openrouter" => bundle.openrouter_api_key = Some(value.to_string()),
        "cerebras" => bundle.cerebras_api_key = Some(value.to_string()),
        "exa" => bundle.exa_api_key = Some(value.to_string()),
        "voyage" => bundle.voyage_api_key = Some(value.to_string()),
        other => return Err(format!("Unknown provider: {other}")),
    }
    store.save(&bundle).map_err(|e| e.to_string())?;

    #[cfg(target_os = "macos")]
    {
        set_keychain_credential(provider, value);
    }

    Ok(())
}

/// Build the argument list for `security add-generic-password` without executing it.
/// Kept separate from `set_keychain_credential` so it can be unit tested on any platform.
fn build_keychain_add_args(provider: &str, value: &str) -> Vec<String> {
    vec![
        "add-generic-password".to_string(),
        "-U".to_string(),
        "-a".to_string(),
        "openplanter".to_string(),
        "-s".to_string(),
        keychain_service_name(provider),
        "-w".to_string(),
        value.to_string(),
    ]
}

/// Resolve the Keychain service name for `provider`.
///
/// Precedence: `OPENPLANTER_KEYCHAIN_<PROVIDER>` env var > `keychain_services.<provider>`
/// in `~/.openplanter/settings.json` > built-in default `openplanter-<provider>`.
/// The settings.json override lets an operator keep pre-existing Keychain items
/// (e.g. from an earlier fork) without touching the repo.
fn keychain_service_name(provider: &str) -> String {
    let env_key = format!("OPENPLANTER_KEYCHAIN_{}", provider.to_uppercase());
    if let Ok(v) = env::var(&env_key) {
        let trimmed = v.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    let settings = UserSettingsStore::new().load();
    if let Some(services) = settings.keychain_services {
        if let Some(v) = services.get(provider) {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }

    format!("openplanter-{provider}")
}

/// Write `value` into the macOS Keychain for `provider`. Best-effort; never logs
/// or prints `value`, and any failure (missing `security` binary, denied prompt) is
/// silently ignored — the user store save above is the source of truth.
#[cfg(target_os = "macos")]
fn set_keychain_credential(provider: &str, value: &str) {
    use std::process::Command;
    let args = build_keychain_add_args(provider, value);
    let _ = Command::new("security").args(&args).output();
}

/// Fill any still-missing API keys from the macOS Keychain. No-op on other platforms.
/// Never logs or prints a retrieved value. Any lookup failure is silently treated as None.
/// Service name resolution (env override > settings.json override > default) is
/// shared with `save_credential` via `keychain_service_name`.
#[cfg(target_os = "macos")]
fn apply_keychain_fallback(cfg: &mut AgentConfig) {
    if cfg.openai_api_key.is_none() {
        cfg.openai_api_key = keychain_lookup(&keychain_service_name("openai"));
    }
    if cfg.anthropic_api_key.is_none() {
        cfg.anthropic_api_key = keychain_lookup(&keychain_service_name("anthropic"));
    }
    if cfg.openrouter_api_key.is_none() {
        cfg.openrouter_api_key = keychain_lookup(&keychain_service_name("openrouter"));
    }
    if cfg.cerebras_api_key.is_none() {
        cfg.cerebras_api_key = keychain_lookup(&keychain_service_name("cerebras"));
    }
    if cfg.exa_api_key.is_none() {
        cfg.exa_api_key = keychain_lookup(&keychain_service_name("exa"));
    }
}

#[cfg(not(target_os = "macos"))]
fn apply_keychain_fallback(_cfg: &mut AgentConfig) {}

/// Look up a generic password in the macOS Keychain via `security find-generic-password`.
/// Returns `None` on any failure (not found, non-zero exit, empty output, spawn error).
#[cfg(target_os = "macos")]
fn keychain_lookup(service: &str) -> Option<String> {
    use std::process::Command;
    let output = Command::new("security")
        .args(["find-generic-password", "-s", service, "-w"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Application state shared across Tauri commands.
pub struct AppState {
    pub config: Arc<Mutex<AgentConfig>>,
    pub session_id: Arc<Mutex<Option<String>>>,
    pub cancel_token: Arc<Mutex<CancellationToken>>,
}

impl AppState {
    pub fn new() -> Self {
        let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let workspace = resolve_workspace(&cwd);
        let cfg = load_config_for(&workspace);
        op_core::workspace::persist_last_workspace(&cfg.workspace);

        Self {
            config: Arc::new(Mutex::new(cfg)),
            session_id: Arc::new(Mutex::new(None)),
            cancel_token: Arc::new(Mutex::new(CancellationToken::new())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_cfg() -> AgentConfig {
        let mut cfg = AgentConfig::from_env("/nonexistent");
        cfg.openai_api_key = None;
        cfg.anthropic_api_key = None;
        cfg.openrouter_api_key = None;
        cfg.cerebras_api_key = None;
        cfg.exa_api_key = None;
        cfg.voyage_api_key = None;
        cfg
    }

    #[test]
    fn test_merge_fills_missing() {
        let mut cfg = empty_cfg();
        let env_creds = CredentialBundle {
            openai_api_key: Some("env-key".to_string()),
            ..Default::default()
        };
        let file_creds = CredentialBundle::default();
        merge_credentials_into_config(&mut cfg, &env_creds, &file_creds);
        assert_eq!(cfg.openai_api_key, Some("env-key".to_string()));
    }

    #[test]
    fn test_merge_preserves_existing() {
        let mut cfg = empty_cfg();
        cfg.openai_api_key = Some("existing".to_string());
        let env_creds = CredentialBundle {
            openai_api_key: Some("env-key".to_string()),
            ..Default::default()
        };
        let file_creds = CredentialBundle::default();
        merge_credentials_into_config(&mut cfg, &env_creds, &file_creds);
        assert_eq!(cfg.openai_api_key, Some("existing".to_string()));
    }

    #[test]
    fn test_merge_env_over_file() {
        let mut cfg = empty_cfg();
        let env_creds = CredentialBundle {
            anthropic_api_key: Some("env-ant".to_string()),
            ..Default::default()
        };
        let file_creds = CredentialBundle {
            anthropic_api_key: Some("file-ant".to_string()),
            ..Default::default()
        };
        merge_credentials_into_config(&mut cfg, &env_creds, &file_creds);
        assert_eq!(cfg.anthropic_api_key, Some("env-ant".to_string()));
    }

    #[test]
    fn test_merge_file_fills_when_env_missing() {
        let mut cfg = empty_cfg();
        let env_creds = CredentialBundle::default();
        let file_creds = CredentialBundle {
            cerebras_api_key: Some("file-cer".to_string()),
            ..Default::default()
        };
        merge_credentials_into_config(&mut cfg, &env_creds, &file_creds);
        assert_eq!(cfg.cerebras_api_key, Some("file-cer".to_string()));
    }

    // ── UiPrefs ──

    #[test]
    fn test_ui_prefs_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let prefs = UiPrefs {
            provider: Some("anthropic".into()),
            model: Some("claude-sonnet-5".into()),
            reasoning_effort: Some("high".into()),
            recursive: Some(false),
            max_depth: Some(6),
            subtask_model: Some("claude-sonnet-5".into()),
            execute_model: Some("claude-haiku-4-5".into()),
            max_exa_agent_calls: Some(20),
        };
        save_ui_prefs(dir.path(), &prefs).unwrap();
        let loaded = load_ui_prefs(dir.path());
        assert_eq!(loaded.provider, Some("anthropic".into()));
        assert_eq!(loaded.model, Some("claude-sonnet-5".into()));
        assert_eq!(loaded.reasoning_effort, Some("high".into()));
        assert_eq!(loaded.recursive, Some(false));
        assert_eq!(loaded.max_depth, Some(6));
        assert_eq!(loaded.subtask_model, Some("claude-sonnet-5".into()));
        assert_eq!(loaded.execute_model, Some("claude-haiku-4-5".into()));
        assert_eq!(loaded.max_exa_agent_calls, Some(20));
    }

    #[test]
    fn test_ui_prefs_load_missing_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = load_ui_prefs(dir.path());
        assert!(loaded.provider.is_none());
        assert!(loaded.recursive.is_none());
        assert!(loaded.max_depth.is_none());
    }

    #[test]
    fn test_apply_ui_prefs_overrides_config() {
        let dir = tempfile::tempdir().unwrap();
        let prefs = UiPrefs {
            provider: Some("cerebras".into()),
            model: None,
            reasoning_effort: None,
            recursive: Some(false),
            max_depth: Some(9),
            subtask_model: None,
            execute_model: None,
            max_exa_agent_calls: Some(7),
        };
        save_ui_prefs(dir.path(), &prefs).unwrap();

        let mut cfg = AgentConfig::from_env(dir.path().to_str().unwrap());
        cfg.workspace = dir.path().to_path_buf();
        apply_ui_prefs(&mut cfg);
        assert_eq!(cfg.provider, "cerebras");
        assert!(!cfg.recursive);
        assert_eq!(cfg.max_depth, 9);
        assert_eq!(cfg.max_exa_agent_calls, 7);
    }

    // ── save_credential / Keychain arg builder ──
    //
    // `keychain_service_name` reads `~/.openplanter/settings.json` (via
    // `UserSettingsStore`, which uses `$HOME`), so these tests must isolate
    // `HOME` to a scratch dir rather than reading the real operator machine's
    // settings file (same pattern as `op_core::workspace`'s `with_home`).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_isolated_home<F: FnOnce()>(f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let saved_home = env::var("HOME").ok();
        unsafe {
            env::set_var("HOME", dir.path());
        }
        f();
        unsafe {
            match saved_home {
                Some(v) => env::set_var("HOME", v),
                None => env::remove_var("HOME"),
            }
        }
    }

    #[test]
    fn test_build_keychain_add_args_shape() {
        with_isolated_home(|| {
            let args = build_keychain_add_args("anthropic", "sk-secret-value");
            assert_eq!(
                args,
                vec![
                    "add-generic-password",
                    "-U",
                    "-a",
                    "openplanter",
                    "-s",
                    "openplanter-anthropic",
                    "-w",
                    "sk-secret-value",
                ]
            );
        });
    }

    #[test]
    fn test_keychain_service_name_format() {
        with_isolated_home(|| {
            assert_eq!(keychain_service_name("openai"), "openplanter-openai");
            assert_eq!(keychain_service_name("exa"), "openplanter-exa");
        });
    }

    #[test]
    fn test_keychain_service_name_env_override_wins() {
        with_isolated_home(|| {
            unsafe {
                env::set_var("OPENPLANTER_KEYCHAIN_OPENAI", "env-override-service");
            }
            let result = keychain_service_name("openai");
            unsafe {
                env::remove_var("OPENPLANTER_KEYCHAIN_OPENAI");
            }
            assert_eq!(result, "env-override-service");
        });
    }

    #[test]
    fn test_keychain_service_name_settings_override_wins_over_default() {
        with_isolated_home(|| {
            let mut services = std::collections::HashMap::new();
            services.insert("openai".to_string(), "legacy-openai-service".to_string());
            let store = UserSettingsStore::new();
            let mut settings = store.load();
            settings.keychain_services = Some(services);
            store.save(&settings).unwrap();

            assert_eq!(keychain_service_name("openai"), "legacy-openai-service");
            // Untouched provider still falls through to the default.
            assert_eq!(keychain_service_name("exa"), "openplanter-exa");
        });
    }

    #[test]
    fn test_save_credential_unknown_provider_rejected() {
        let result = save_credential("not-a-provider", "value");
        assert!(result.is_err());
    }
}
