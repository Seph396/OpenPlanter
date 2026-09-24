/// Engine construction and provider inference.
///
/// Mirrors `agent/builder.py` — regex-based provider detection,
/// model validation, and engine factory.
use std::collections::HashMap;

use regex::Regex;
use std::sync::LazyLock;

use crate::config::{AgentConfig, PROVIDER_DEFAULT_MODELS};
use crate::model::BaseModel;
use crate::model::openai::OpenAIModel;
use crate::model::anthropic::AnthropicModel;

/// Error type for model/builder operations.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("{0}")]
    Message(String),
}

// Provider inference regexes — order matters (Cerebras `qwen-3` before Ollama `qwen`).
static ANTHROPIC_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^claude").unwrap());

static OPENAI_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(gpt|o[1-4]-|o[1-4]$|chatgpt|dall-e|tts-|whisper)").unwrap());

static CEREBRAS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(llama.*cerebras|qwen-3|gpt-oss|zai-glm)").unwrap());

// Ollama regex: `qwen` without lookahead — Cerebras check runs first, so
// `qwen-3*` is already caught before we reach this regex.
static OLLAMA_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(llama|mistral|gemma|phi|codellama|deepseek|vicuna|tinyllama|neural-chat|dolphin|wizardlm|orca|nous-hermes|command-r|qwen)",
    )
    .unwrap()
});

/// Infer the likely provider for a model name, or `None` if ambiguous.
pub fn infer_provider_for_model(model: &str) -> Option<&'static str> {
    if model.contains('/') {
        return Some("openrouter");
    }
    if ANTHROPIC_RE.is_match(model) {
        return Some("anthropic");
    }
    if CEREBRAS_RE.is_match(model) {
        return Some("cerebras");
    }
    if OPENAI_RE.is_match(model) {
        return Some("openai");
    }
    if OLLAMA_RE.is_match(model) {
        return Some("ollama");
    }
    None
}

/// Validate that a model name is compatible with the given provider.
pub fn validate_model_provider(model_name: &str, provider: &str) -> Result<(), ModelError> {
    if provider == "openrouter" {
        return Ok(());
    }
    let inferred = infer_provider_for_model(model_name);
    match inferred {
        None | Some("openrouter") => Ok(()),
        Some(p) if p == provider => Ok(()),
        Some(p) => Err(ModelError::Message(format!(
            "Model '{}' belongs to provider '{}', not '{}'. \
             Use --provider {} or pick a model that matches the current provider.",
            model_name, p, provider, p
        ))),
    }
}

/// Resolve the model name from config, handling the "newest" keyword.
pub fn resolve_model_name(cfg: &AgentConfig) -> Result<String, ModelError> {
    let selected = cfg.model.trim();
    if !selected.is_empty() && selected.to_lowercase() != "newest" {
        return Ok(selected.to_string());
    }
    if selected.to_lowercase() == "newest" {
        // In the full implementation this would call list_models for the provider.
        // For now, fall through to defaults.
        return Ok(PROVIDER_DEFAULT_MODELS
            .get(cfg.provider.as_str())
            .unwrap_or(&"claude-opus-4-6")
            .to_string());
    }
    Ok(PROVIDER_DEFAULT_MODELS
        .get(cfg.provider.as_str())
        .unwrap_or(&"claude-opus-4-6")
        .to_string())
}

/// Resolve the provider, handling "auto" by inferring from model name
/// or falling back to the first provider with an available API key.
pub fn resolve_provider(cfg: &AgentConfig) -> Result<String, ModelError> {
    let provider = cfg.provider.trim().to_lowercase();
    if !provider.is_empty() && provider != "auto" {
        return Ok(provider);
    }

    // Try to infer from model name
    let model = cfg.model.trim();
    if !model.is_empty() {
        if let Some(inferred) = infer_provider_for_model(model) {
            return Ok(inferred.to_string());
        }
    }

    // Fallback: first provider with an available key
    let candidates: &[(&str, &Option<String>)] = &[
        ("anthropic", &cfg.anthropic_api_key),
        ("openai", &cfg.openai_api_key),
        ("openrouter", &cfg.openrouter_api_key),
        ("cerebras", &cfg.cerebras_api_key),
        ("ollama", &None), // ollama is always last — no key needed
    ];

    for (name, key) in candidates {
        if key.is_some() {
            return Ok(name.to_string());
        }
    }

    // Default to ollama (no key needed)
    Ok("ollama".to_string())
}

/// Resolve the base URL and API key for the given provider.
pub fn resolve_endpoint(
    cfg: &AgentConfig,
    provider: &str,
) -> Result<(String, String), ModelError> {
    match provider {
        "anthropic" => {
            let key = cfg
                .anthropic_api_key
                .as_deref()
                .or(cfg.api_key.as_deref())
                .filter(|k| !k.is_empty())
                .ok_or_else(|| {
                    ModelError::Message(
                        "No Anthropic API key. Set ANTHROPIC_API_KEY or OPENPLANTER_ANTHROPIC_API_KEY.".into(),
                    )
                })?;
            // Anthropic base URL does NOT include /v1 suffix for /messages endpoint —
            // the model adapter appends /messages itself. The config stores it with /v1.
            Ok((cfg.anthropic_base_url.clone(), key.to_string()))
        }
        "openai" => {
            let key = cfg
                .openai_api_key
                .as_deref()
                .or(cfg.api_key.as_deref())
                .filter(|k| !k.is_empty())
                .ok_or_else(|| {
                    ModelError::Message(
                        "No OpenAI API key. Set OPENAI_API_KEY or OPENPLANTER_OPENAI_API_KEY.".into(),
                    )
                })?;
            Ok((cfg.openai_base_url.clone(), key.to_string()))
        }
        "openrouter" => {
            let key = cfg
                .openrouter_api_key
                .as_deref()
                .or(cfg.api_key.as_deref())
                .filter(|k| !k.is_empty())
                .ok_or_else(|| {
                    ModelError::Message(
                        "No OpenRouter API key. Set OPENROUTER_API_KEY or OPENPLANTER_OPENROUTER_API_KEY.".into(),
                    )
                })?;
            Ok((cfg.openrouter_base_url.clone(), key.to_string()))
        }
        "cerebras" => {
            let key = cfg
                .cerebras_api_key
                .as_deref()
                .or(cfg.api_key.as_deref())
                .filter(|k| !k.is_empty())
                .ok_or_else(|| {
                    ModelError::Message(
                        "No Cerebras API key. Set CEREBRAS_API_KEY or OPENPLANTER_CEREBRAS_API_KEY.".into(),
                    )
                })?;
            Ok((cfg.cerebras_base_url.clone(), key.to_string()))
        }
        "ollama" => {
            // Ollama doesn't need a real key — use a dummy
            Ok((cfg.ollama_base_url.clone(), "ollama".to_string()))
        }
        _ => Err(ModelError::Message(format!("Unknown provider: {provider}"))),
    }
}

// ---------------------------------------------------------------------
// Sub-agent model tier selection (per-tier models for `subtask`/`execute`).
//
// Tier order (cheapest last): opus(1) > sonnet(2) > haiku(3), by substring
// match on the model name. Unknown model names have no known tier and are
// always treated as equal to the parent's tier (allowed — no clamp).
// ---------------------------------------------------------------------

/// Capability tier for a model name, lower = more expensive/capable.
/// `None` if the name doesn't match a known Anthropic tier keyword.
fn known_tier(model_name: &str) -> Option<u8> {
    let lower = model_name.to_lowercase();
    if lower.contains("opus") {
        Some(1)
    } else if lower.contains("sonnet") {
        Some(2)
    } else if lower.contains("haiku") {
        Some(3)
    } else {
        None
    }
}

/// Clamp `desired_model` to `parent_model`'s tier if it would be a more
/// expensive (lower-numbered) tier than the parent. A child may never use a
/// more capable/expensive tier than its parent — only equal or cheaper.
///
/// If either name has an unknown tier, no clamp is applied (treated as
/// equal/allowed), per the brief's explicit "unknown => treat as parent
/// tier" rule.
pub fn enforce_downward_tier(parent_model: &str, desired_model: &str) -> String {
    match (known_tier(parent_model), known_tier(desired_model)) {
        (Some(parent_tier), Some(desired_tier)) if desired_tier < parent_tier => {
            parent_model.to_string()
        }
        _ => desired_model.to_string(),
    }
}

/// Resolve the model name a `subtask`/`execute` delegation should use, given
/// the static per-tier config fields and the calling level's current model
/// name. Returns `None` when there is no override configured (inherit the
/// parent's model unchanged — no new model needs to be built).
///
/// Precedence: `execute` children use `execute_model`, else `subtask_model`,
/// else inherit. `subtask` children use `subtask_model`, else inherit.
pub fn select_child_model_name(
    cfg: &AgentConfig,
    is_execute: bool,
    parent_model_name: &str,
) -> Option<String> {
    let desired = if is_execute {
        cfg.execute_model.clone().or_else(|| cfg.subtask_model.clone())
    } else {
        cfg.subtask_model.clone()
    }?;
    Some(enforce_downward_tier(parent_model_name, &desired))
}

/// Build a model instance from the agent configuration.
pub fn build_model(cfg: &AgentConfig) -> Result<Box<dyn BaseModel>, ModelError> {
    let provider = resolve_provider(cfg)?;
    let model_name = resolve_model_name(cfg)?;
    validate_model_provider(&model_name, &provider)?;
    let (base_url, api_key) = resolve_endpoint(cfg, &provider)?;

    match provider.as_str() {
        "anthropic" => Ok(Box::new(AnthropicModel::new(
            model_name,
            base_url,
            api_key,
            cfg.reasoning_effort.clone(),
            cfg.max_output_tokens,
        ))),
        _ => {
            // OpenAI-compatible: openai, openrouter, cerebras, ollama
            let mut extra_headers = HashMap::new();
            if provider == "openrouter" {
                extra_headers.insert(
                    "HTTP-Referer".to_string(),
                    "https://github.com/openplanter".to_string(),
                );
                extra_headers.insert("X-Title".to_string(), "OpenPlanter".to_string());
            }
            Ok(Box::new(OpenAIModel::with_max_output_tokens(
                model_name,
                provider,
                base_url,
                api_key,
                cfg.reasoning_effort.clone(),
                extra_headers,
                cfg.max_output_tokens,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_anthropic() {
        assert_eq!(
            infer_provider_for_model("claude-opus-4-6"),
            Some("anthropic")
        );
        assert_eq!(
            infer_provider_for_model("claude-sonnet-4-5"),
            Some("anthropic")
        );
        assert_eq!(
            infer_provider_for_model("claude-haiku-4-5"),
            Some("anthropic")
        );
    }

    #[test]
    fn test_infer_openai() {
        assert_eq!(infer_provider_for_model("gpt-5.2"), Some("openai"));
        assert_eq!(infer_provider_for_model("o1-preview"), Some("openai"));
        assert_eq!(infer_provider_for_model("o3"), Some("openai"));
        assert_eq!(infer_provider_for_model("chatgpt-4o"), Some("openai"));
    }

    #[test]
    fn test_infer_openrouter() {
        assert_eq!(
            infer_provider_for_model("anthropic/claude-sonnet-4-5"),
            Some("openrouter")
        );
        assert_eq!(
            infer_provider_for_model("openai/gpt-5.2"),
            Some("openrouter")
        );
    }

    #[test]
    fn test_infer_cerebras() {
        assert_eq!(
            infer_provider_for_model("qwen-3-235b-a22b-instruct-2507"),
            Some("cerebras")
        );
    }

    #[test]
    fn test_infer_ollama() {
        assert_eq!(infer_provider_for_model("llama3.2"), Some("ollama"));
        assert_eq!(infer_provider_for_model("mistral"), Some("ollama"));
        assert_eq!(infer_provider_for_model("phi"), Some("ollama"));
        assert_eq!(infer_provider_for_model("deepseek"), Some("ollama"));
        // qwen without -3 should be ollama
        assert_eq!(infer_provider_for_model("qwen2"), Some("ollama"));
    }

    #[test]
    fn test_cerebras_before_ollama_qwen() {
        // qwen-3 → cerebras, qwen (no -3) → ollama
        assert_eq!(infer_provider_for_model("qwen-3"), Some("cerebras"));
        assert_eq!(infer_provider_for_model("qwen2"), Some("ollama"));
    }

    #[test]
    fn test_infer_unknown() {
        assert_eq!(infer_provider_for_model("some-random-model"), None);
    }

    #[test]
    fn test_validate_model_provider_ok() {
        assert!(validate_model_provider("gpt-5.2", "openai").is_ok());
        assert!(validate_model_provider("claude-opus-4-6", "anthropic").is_ok());
        // OpenRouter accepts anything
        assert!(validate_model_provider("gpt-5.2", "openrouter").is_ok());
        // Unknown model is accepted
        assert!(validate_model_provider("some-model", "openai").is_ok());
    }

    #[test]
    fn test_validate_model_provider_mismatch() {
        let result = validate_model_provider("gpt-5.2", "anthropic");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("openai"));
        assert!(err.contains("anthropic"));
    }

    #[test]
    fn test_resolve_model_name_explicit() {
        let cfg = AgentConfig {
            model: "gpt-5.2".into(),
            provider: "openai".into(),
            ..Default::default()
        };
        assert_eq!(resolve_model_name(&cfg).unwrap(), "gpt-5.2");
    }

    #[test]
    fn test_resolve_model_name_default() {
        let cfg = AgentConfig {
            model: "".into(),
            provider: "openai".into(),
            ..Default::default()
        };
        assert_eq!(resolve_model_name(&cfg).unwrap(), "gpt-5.2");
    }

    // ── resolve_provider ──

    #[test]
    fn test_resolve_provider_explicit() {
        let cfg = AgentConfig {
            provider: "openai".into(),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&cfg).unwrap(), "openai");
    }

    #[test]
    fn test_resolve_provider_auto_infers_from_model() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "claude-opus-4-6".into(),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&cfg).unwrap(), "anthropic");
    }

    #[test]
    fn test_resolve_provider_auto_falls_back_to_key() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "some-unknown-model".into(),
            openai_api_key: Some("sk-test".into()),
            ..Default::default()
        };
        // anthropic checked first but no key, openai has key
        assert_eq!(resolve_provider(&cfg).unwrap(), "openai");
    }

    #[test]
    fn test_resolve_provider_auto_no_keys_defaults_ollama() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "some-unknown-model".into(),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&cfg).unwrap(), "ollama");
    }

    #[test]
    fn test_resolve_provider_anthropic_key_preferred_first() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "some-unknown-model".into(),
            anthropic_api_key: Some("sk-ant-test".into()),
            openai_api_key: Some("sk-test".into()),
            ..Default::default()
        };
        assert_eq!(resolve_provider(&cfg).unwrap(), "anthropic");
    }

    // ── resolve_endpoint ──

    #[test]
    fn test_resolve_endpoint_anthropic() {
        let cfg = AgentConfig {
            anthropic_api_key: Some("sk-ant-key".into()),
            ..Default::default()
        };
        let (url, key) = resolve_endpoint(&cfg, "anthropic").unwrap();
        assert_eq!(url, "https://api.anthropic.com/v1");
        assert_eq!(key, "sk-ant-key");
    }

    #[test]
    fn test_resolve_endpoint_anthropic_fallback_to_api_key() {
        let cfg = AgentConfig {
            api_key: Some("fallback-key".into()),
            ..Default::default()
        };
        let (_, key) = resolve_endpoint(&cfg, "anthropic").unwrap();
        assert_eq!(key, "fallback-key");
    }

    #[test]
    fn test_resolve_endpoint_anthropic_missing_key() {
        let cfg = AgentConfig::default();
        let result = resolve_endpoint(&cfg, "anthropic");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Anthropic API key"));
    }

    #[test]
    fn test_resolve_endpoint_openai() {
        let cfg = AgentConfig {
            openai_api_key: Some("sk-openai".into()),
            ..Default::default()
        };
        let (url, key) = resolve_endpoint(&cfg, "openai").unwrap();
        assert_eq!(url, "https://api.openai.com/v1");
        assert_eq!(key, "sk-openai");
    }

    #[test]
    fn test_resolve_endpoint_ollama_dummy_key() {
        let cfg = AgentConfig::default();
        let (url, key) = resolve_endpoint(&cfg, "ollama").unwrap();
        assert!(url.contains("11434"));
        assert_eq!(key, "ollama");
    }

    #[test]
    fn test_resolve_endpoint_unknown_provider() {
        let cfg = AgentConfig::default();
        let result = resolve_endpoint(&cfg, "unknown");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown provider"));
    }

    // ── build_model ──

    #[test]
    fn test_build_model_anthropic() {
        let cfg = AgentConfig {
            provider: "anthropic".into(),
            model: "claude-opus-4-6".into(),
            anthropic_api_key: Some("sk-ant-key".into()),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.model_name(), "claude-opus-4-6");
        assert_eq!(model.provider_name(), "anthropic");
    }

    #[test]
    fn test_build_model_openai() {
        let cfg = AgentConfig {
            provider: "openai".into(),
            model: "gpt-4o".into(),
            openai_api_key: Some("sk-key".into()),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.model_name(), "gpt-4o");
        assert_eq!(model.provider_name(), "openai");
    }

    #[test]
    fn test_build_model_ollama_no_key_needed() {
        let cfg = AgentConfig {
            provider: "ollama".into(),
            model: "llama3.2".into(),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.model_name(), "llama3.2");
        assert_eq!(model.provider_name(), "ollama");
    }

    #[test]
    fn test_build_model_auto_anthropic() {
        let cfg = AgentConfig {
            provider: "auto".into(),
            model: "claude-sonnet-4-5".into(),
            anthropic_api_key: Some("sk-ant-key".into()),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.provider_name(), "anthropic");
        assert_eq!(model.model_name(), "claude-sonnet-4-5");
    }

    #[test]
    fn test_build_model_missing_key_errors() {
        let cfg = AgentConfig {
            provider: "openai".into(),
            model: "gpt-4o".into(),
            // No key set
            ..Default::default()
        };
        let result = build_model(&cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_model_mismatch_errors() {
        let cfg = AgentConfig {
            provider: "anthropic".into(),
            model: "gpt-4o".into(),
            anthropic_api_key: Some("key".into()),
            ..Default::default()
        };
        let result = build_model(&cfg);
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        };
        assert!(err_msg.contains("openai"), "error should mention openai: {err_msg}");
    }

    // ── tier enforcement / model selection ──

    #[test]
    fn test_enforce_downward_tier_blocks_upgrade() {
        // Parent sonnet, child requests opus (more expensive) => clamped to parent's model.
        let result = enforce_downward_tier("claude-sonnet-4-5", "claude-opus-4-6");
        assert_eq!(result, "claude-sonnet-4-5");
    }

    #[test]
    fn test_enforce_downward_tier_allows_downgrade() {
        // Parent sonnet, child requests haiku (cheaper) => allowed as requested.
        let result = enforce_downward_tier("claude-sonnet-4-5", "claude-haiku-4-5");
        assert_eq!(result, "claude-haiku-4-5");
    }

    #[test]
    fn test_enforce_downward_tier_allows_equal() {
        let result = enforce_downward_tier("claude-sonnet-4-5", "claude-sonnet-5");
        assert_eq!(result, "claude-sonnet-5");
    }

    #[test]
    fn test_enforce_downward_tier_unknown_parent_allowed() {
        // Parent tier unknown (custom model) => no clamp, desired passes through.
        let result = enforce_downward_tier("some-custom-model", "claude-opus-4-6");
        assert_eq!(result, "claude-opus-4-6");
    }

    #[test]
    fn test_enforce_downward_tier_unknown_desired_allowed() {
        // Desired tier unknown => no clamp, desired passes through.
        let result = enforce_downward_tier("claude-opus-4-6", "some-custom-model");
        assert_eq!(result, "some-custom-model");
    }

    #[test]
    fn test_select_child_model_name_none_when_unconfigured() {
        let cfg = AgentConfig::default();
        assert_eq!(select_child_model_name(&cfg, false, "claude-opus-4-6"), None);
        assert_eq!(select_child_model_name(&cfg, true, "claude-opus-4-6"), None);
    }

    #[test]
    fn test_select_child_model_name_subtask_precedence() {
        let cfg = AgentConfig {
            subtask_model: Some("claude-sonnet-5".into()),
            execute_model: None,
            ..Default::default()
        };
        // subtask call uses subtask_model
        assert_eq!(
            select_child_model_name(&cfg, false, "claude-opus-4-6"),
            Some("claude-sonnet-5".into())
        );
        // execute call falls back to subtask_model when execute_model unset
        assert_eq!(
            select_child_model_name(&cfg, true, "claude-opus-4-6"),
            Some("claude-sonnet-5".into())
        );
    }

    #[test]
    fn test_select_child_model_name_execute_overrides_subtask() {
        let cfg = AgentConfig {
            subtask_model: Some("claude-sonnet-5".into()),
            execute_model: Some("claude-haiku-4-5".into()),
            ..Default::default()
        };
        assert_eq!(
            select_child_model_name(&cfg, false, "claude-opus-4-6"),
            Some("claude-sonnet-5".into())
        );
        assert_eq!(
            select_child_model_name(&cfg, true, "claude-opus-4-6"),
            Some("claude-haiku-4-5".into())
        );
    }

    #[test]
    fn test_select_child_model_name_tier_clamped() {
        // Parent already downgraded to sonnet; execute requests opus => clamp to sonnet.
        let cfg = AgentConfig {
            subtask_model: None,
            execute_model: Some("claude-opus-4-6".into()),
            ..Default::default()
        };
        assert_eq!(
            select_child_model_name(&cfg, true, "claude-sonnet-4-5"),
            Some("claude-sonnet-4-5".into())
        );
    }

    #[test]
    fn test_build_model_openrouter_has_extra_headers() {
        let cfg = AgentConfig {
            provider: "openrouter".into(),
            model: "anthropic/claude-sonnet-4-5".into(),
            openrouter_api_key: Some("sk-or-key".into()),
            ..Default::default()
        };
        let model = build_model(&cfg).unwrap();
        assert_eq!(model.provider_name(), "openrouter");
    }
}
