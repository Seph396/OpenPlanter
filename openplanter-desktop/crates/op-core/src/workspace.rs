use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::credentials::home_dir;

/// User-level settings, distinct from the per-workspace `PersistentSettings`.
/// Stored at `~/.openplanter/settings.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UserSettings {
    pub last_workspace: Option<String>,
    /// Per-provider macOS Keychain service name overrides (e.g. `{"anthropic": "my-old-service"}`).
    /// Lets an operator keep pre-existing Keychain items without touching the repo.
    /// Precedence: `OPENPLANTER_KEYCHAIN_<PROVIDER>` env var > this map > built-in default.
    #[serde(default)]
    pub keychain_services: Option<std::collections::HashMap<String, String>>,
}

/// User-level settings store at `~/.openplanter/settings.json`.
pub struct UserSettingsStore {
    pub settings_path: PathBuf,
}

impl UserSettingsStore {
    pub fn new() -> Self {
        Self {
            settings_path: home_dir().join(".openplanter").join("settings.json"),
        }
    }

    pub fn load(&self) -> UserSettings {
        let content = match fs::read_to_string(&self.settings_path) {
            Ok(c) => c,
            Err(_) => return UserSettings::default(),
        };
        serde_json::from_str(&content).unwrap_or_default()
    }

    pub fn save(&self, settings: &UserSettings) -> std::io::Result<()> {
        if let Some(parent) = self.settings_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(settings)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(&self.settings_path, json)
    }
}

impl Default for UserSettingsStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve the workspace directory to use.
///
/// Priority order:
/// 1. `OPENPLANTER_WORKSPACE` env var, if set to a non-empty value.
/// 2. `last_workspace` from `~/.openplanter/settings.json`, if that directory still exists.
/// 3. If `cwd` is `/` or the user's home directory, fall back to `~/.openplanter/workspace`
///    (created if missing).
/// 4. Otherwise, `cwd` itself.
pub fn resolve_workspace(cwd: &Path) -> PathBuf {
    if let Ok(env_ws) = env::var("OPENPLANTER_WORKSPACE") {
        let trimmed = env_ws.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    let store = UserSettingsStore::new();
    let settings = store.load();
    if let Some(last) = settings.last_workspace {
        let p = PathBuf::from(&last);
        if p.is_dir() {
            return p;
        }
    }

    let home = home_dir();
    if cwd == Path::new("/") || cwd == home.as_path() {
        let fallback = home.join(".openplanter").join("workspace");
        let _ = fs::create_dir_all(&fallback);
        return fallback;
    }

    cwd.to_path_buf()
}

/// Persist `workspace` as the user's last-used workspace.
pub fn persist_last_workspace(workspace: &Path) {
    let store = UserSettingsStore::new();
    // Preserve any existing keychain_services overrides rather than clobbering them.
    let mut settings = store.load();
    settings.last_workspace = Some(workspace.display().to_string());
    let _ = store.save(&settings);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize env-mutating tests; HOME/OPENPLANTER_WORKSPACE are process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_home<F: FnOnce(&Path)>(f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let saved_home = env::var("HOME").ok();
        let saved_ws = env::var("OPENPLANTER_WORKSPACE").ok();
        unsafe {
            env::set_var("HOME", dir.path());
            env::remove_var("OPENPLANTER_WORKSPACE");
        }
        f(dir.path());
        unsafe {
            match saved_home {
                Some(v) => env::set_var("HOME", v),
                None => env::remove_var("HOME"),
            }
            match saved_ws {
                Some(v) => env::set_var("OPENPLANTER_WORKSPACE", v),
                None => env::remove_var("OPENPLANTER_WORKSPACE"),
            }
        }
    }

    #[test]
    fn test_env_var_takes_priority() {
        with_home(|home| {
            let other = tempfile::tempdir().unwrap();
            unsafe {
                env::set_var("OPENPLANTER_WORKSPACE", other.path());
            }
            let resolved = resolve_workspace(home);
            assert_eq!(resolved, other.path());
            unsafe {
                env::remove_var("OPENPLANTER_WORKSPACE");
            }
        });
    }

    #[test]
    fn test_last_workspace_used_when_dir_exists() {
        with_home(|home| {
            let ws = tempfile::tempdir().unwrap();
            persist_last_workspace(ws.path());
            let resolved = resolve_workspace(home);
            assert_eq!(resolved, ws.path());
        });
    }

    #[test]
    fn test_last_workspace_ignored_when_dir_missing() {
        with_home(|home| {
            let store = UserSettingsStore::new();
            store
                .save(&UserSettings {
                    last_workspace: Some("/definitely/does/not/exist/xyz".to_string()),
                    keychain_services: None,
                })
                .unwrap();
            // cwd is some arbitrary dir that is not home or "/"
            let cwd = tempfile::tempdir().unwrap();
            let resolved = resolve_workspace(cwd.path());
            assert_eq!(resolved, cwd.path());
        });
    }

    #[test]
    fn test_fallback_when_cwd_is_root() {
        with_home(|home| {
            let resolved = resolve_workspace(Path::new("/"));
            assert_eq!(resolved, home.join(".openplanter").join("workspace"));
            assert!(resolved.is_dir());
        });
    }

    #[test]
    fn test_fallback_when_cwd_is_home() {
        with_home(|home| {
            let resolved = resolve_workspace(home);
            assert_eq!(resolved, home.join(".openplanter").join("workspace"));
        });
    }

    #[test]
    fn test_cwd_used_when_not_root_or_home() {
        with_home(|_home| {
            let cwd = tempfile::tempdir().unwrap();
            let resolved = resolve_workspace(cwd.path());
            assert_eq!(resolved, cwd.path());
        });
    }

    #[test]
    fn test_persist_and_load_round_trip() {
        with_home(|_home| {
            let ws = tempfile::tempdir().unwrap();
            persist_last_workspace(ws.path());
            let store = UserSettingsStore::new();
            let loaded = store.load();
            assert_eq!(loaded.last_workspace, Some(ws.path().display().to_string()));
        });
    }
}
