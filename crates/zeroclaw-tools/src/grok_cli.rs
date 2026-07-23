use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use zeroclaw_api::tool::{Tool, ToolOutput, ToolResult};
use zeroclaw_config::policy::SecurityPolicy;
use zeroclaw_config::policy::ToolOperation;
use zeroclaw_config::schema::GrokCliConfig;

/// Environment variables safe to pass through to the `grok` subprocess.
const SAFE_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "USER", "SHELL", "TMPDIR",
];

/// Delegates coding tasks to the Grok Build CLI (`grok -p` headless).
///
/// This creates a two-tier agent architecture: ZeroClaw orchestrates high-level
/// tasks and delegates complex coding work to Grok Build, which has its own
/// agent loop with file editing and shell tools.
///
/// Authentication uses the `grok` binary's own session by default. No API key
/// is needed unless `env_passthrough` includes `XAI_API_KEY`.
pub struct GrokCliTool {
    security: Arc<SecurityPolicy>,
    config: GrokCliConfig,
}

impl GrokCliTool {
    pub fn new(security: Arc<SecurityPolicy>, config: GrokCliConfig) -> Self {
        Self { security, config }
    }
}

#[async_trait]
impl Tool for GrokCliTool {
    fn name(&self) -> &str {
        "grok_cli"
    }

    fn description(&self) -> &str {
        "Delegate a coding task to Grok Build CLI (grok -p headless). Supports file editing and shell execution. Use for complex coding work that benefits from Grok's full agent loop. Optional session_id resumes a prior Grok session."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The coding task to delegate to Grok Build CLI"
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory within the workspace (must be inside workspace_dir)"
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional Grok session ID to resume with --resume (from a previous --output-format json response)"
                },
                "model": {
                    "type": "string",
                    "description": "Optional model ID override (e.g. grok-4.5). Falls back to [grok_cli].default_model when set."
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        // Rate limiting is applied by the RateLimitedTool wrapper at
        // registration time (see zeroclaw-runtime::tools::mod).

        // Enforce act policy
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "grok_cli")
        {
            return Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(error),
            });
        }

        // Extract prompt (required)
        let prompt = args.get("prompt").and_then(|v| v.as_str()).ok_or_else(|| {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({"param": "prompt"})),
                "grok_cli: missing prompt parameter"
            );
            anyhow::Error::msg("Missing 'prompt' parameter")
        })?;

        // Validate working directory — require both paths to exist (reject
        // non-existent paths instead of falling back to the raw value, which
        // could bypass the workspace containment check via symlinks or
        // specially-crafted path components).
        let work_dir = if let Some(wd) = args.get("working_directory").and_then(|v| v.as_str()) {
            let wd_path = std::path::PathBuf::from(wd);
            // Resolve relative working_directory against workspace_dir, NOT
            // the daemon's current working directory. This prevents the bug
            // where an external coding tool's relative working_directory
            // would silently resolve to a path outside the workspace when
            // the daemon cwd differs from workspace_dir.
            let wd_path = if wd_path.is_relative() {
                self.security.workspace_dir.join(&wd_path)
            } else {
                wd_path
            };
            let workspace = &self.security.workspace_dir;
            let canonical_wd = match wd_path.canonicalize() {
                Ok(p) => p,
                Err(_) => {
                    return Ok(ToolResult {
                        success: false,
                        output: ToolOutput::default(),
                        error: Some(format!(
                            "working_directory '{}' does not exist or is not accessible",
                            wd
                        )),
                    });
                }
            };
            let canonical_ws = match workspace.canonicalize() {
                Ok(p) => p,
                Err(_) => {
                    return Ok(ToolResult {
                        success: false,
                        output: ToolOutput::default(),
                        error: Some(format!(
                            "workspace directory '{}' does not exist or is not accessible",
                            workspace.display()
                        )),
                    });
                }
            };
            if !canonical_wd.starts_with(&canonical_ws) {
                return Ok(ToolResult {
                    success: false,
                    output: ToolOutput::default(),
                    error: Some(format!(
                        "working_directory '{}' is outside the workspace '{}'",
                        wd,
                        workspace.display()
                    )),
                });
            }
            canonical_wd
        } else {
            self.security.workspace_dir.clone()
        };

        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());

        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| {
                self.config
                    .default_model
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
            });

        // Build CLI command: `grok -p <prompt> [flags...]`
        // Headless mode runs the full Grok agent loop then exits.
        // See https://docs.x.ai/build/cli/headless-scripting
        let grok_bin = if cfg!(target_os = "windows") {
            "grok.cmd"
        } else {
            "grok"
        };
        let mut cmd = Command::new(grok_bin);
        cmd.arg("-p").arg(prompt);
        cmd.arg("--output-format").arg("plain");
        cmd.arg("--no-auto-update");

        // Non-interactive tool use needs auto-approval; without these flags
        // headless Grok can block waiting for permission prompts.
        if self.config.auto_approve {
            cmd.arg("--always-approve");
            cmd.arg("--permission-mode").arg("bypassPermissions");
        }

        if let Some(model) = model {
            cmd.arg("-m").arg(model);
        }

        if let Some(session_id) = session_id {
            cmd.arg("--resume").arg(session_id);
        }

        // Append operator-configured extra arguments (e.g. --max-turns, --rules)
        for arg in &self.config.extra_args {
            let trimmed = arg.trim();
            if !trimmed.is_empty() {
                cmd.arg(trimmed);
            }
        }

        // Environment: clear everything, pass only safe vars + configured passthrough.
        cmd.env_clear();
        for var in SAFE_ENV_VARS {
            if let Ok(val) = std::env::var(var) {
                cmd.env(var, val);
            }
        }
        for var in &self.config.env_passthrough {
            let trimmed = var.trim();
            if !trimmed.is_empty()
                && let Ok(val) = std::env::var(trimmed)
            {
                cmd.env(trimmed, val);
            }
        }

        cmd.current_dir(&work_dir);
        // Execute with timeout — use kill_on_drop(true) so the child process
        // is automatically killed when the future is dropped on timeout,
        // preventing zombie processes.
        let timeout = Duration::from_secs(self.config.timeout_secs);
        cmd.kill_on_drop(true);

        let result = tokio::time::timeout(timeout, cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                // Truncate to max_output_bytes with char-boundary safety
                if stdout.len() > self.config.max_output_bytes {
                    let mut b = self.config.max_output_bytes.min(stdout.len());
                    while b > 0 && !stdout.is_char_boundary(b) {
                        b -= 1;
                    }
                    stdout.truncate(b);
                    stdout.push_str("\n... [output truncated]");
                }

                Ok(ToolResult {
                    success: output.status.success(),
                    output: stdout.into(),
                    error: if stderr.is_empty() {
                        None
                    } else {
                        Some(stderr)
                    },
                })
            }
            Ok(Err(e)) => {
                let err_msg = e.to_string();
                let msg = if err_msg.contains("No such file or directory")
                    || err_msg.contains("not found")
                    || err_msg.contains("cannot find")
                {
                    "Grok Build CLI ('grok') not found in PATH. Install with: curl -fsSL https://x.ai/cli/install.sh | bash (then ensure ~/.grok/bin is on PATH)".into()
                } else {
                    format!("Failed to execute grok: {e}")
                };
                Ok(ToolResult {
                    success: false,
                    output: ToolOutput::default(),
                    error: Some(msg),
                })
            }
            Err(_) => {
                // Timeout — kill_on_drop(true) ensures the child is killed
                // when the future is dropped.
                Ok(ToolResult {
                    success: false,
                    output: ToolOutput::default(),
                    error: Some(format!(
                        "Grok Build CLI timed out after {}s and was killed",
                        self.config.timeout_secs
                    )),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroclaw_config::autonomy::AutonomyLevel;
    use zeroclaw_config::policy::SecurityPolicy;
    use zeroclaw_config::schema::GrokCliConfig;

    fn test_config() -> GrokCliConfig {
        GrokCliConfig::default()
    }

    fn test_security(autonomy: AutonomyLevel) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        })
    }

    fn test_security_with_workspace(
        autonomy: AutonomyLevel,
        workspace_dir: std::path::PathBuf,
    ) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy,
            workspace_dir,
            ..SecurityPolicy::default()
        })
    }

    #[test]
    fn grok_cli_tool_name() {
        let tool = GrokCliTool::new(test_security(AutonomyLevel::Supervised), test_config());
        assert_eq!(tool.name(), "grok_cli");
    }

    #[test]
    fn grok_cli_tool_schema_has_prompt() {
        let tool = GrokCliTool::new(test_security(AutonomyLevel::Supervised), test_config());
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["prompt"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .expect("schema required should be an array")
                .contains(&json!("prompt"))
        );
        assert!(schema["properties"]["working_directory"].is_object());
        assert!(schema["properties"]["session_id"].is_object());
        assert!(schema["properties"]["model"].is_object());
    }

    #[tokio::test]
    async fn grok_cli_blocks_rate_limited() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            max_actions_per_hour: 0,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        });
        let tool = GrokCliTool::new(security, test_config());
        let result = tool
            .execute(json!({"prompt": "hello"}))
            .await
            .expect("rate-limited should return a result");
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("Rate limit"));
    }

    #[tokio::test]
    async fn grok_cli_blocks_readonly() {
        let tool = GrokCliTool::new(test_security(AutonomyLevel::ReadOnly), test_config());
        let result = tool
            .execute(json!({"prompt": "hello"}))
            .await
            .expect("readonly should return a result");
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("read-only mode")
        );
    }

    #[tokio::test]
    async fn grok_cli_missing_prompt_param() {
        let tool = GrokCliTool::new(test_security(AutonomyLevel::Supervised), test_config());
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("prompt"));
    }

    #[tokio::test]
    async fn grok_cli_rejects_path_outside_workspace() {
        let tool = GrokCliTool::new(test_security(AutonomyLevel::Full), test_config());
        let result = tool
            .execute(json!({
                "prompt": "hello",
                "working_directory": "/etc"
            }))
            .await
            .expect("should return a result for path validation");
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("outside the workspace")
        );
    }

    #[tokio::test]
    async fn grok_cli_resolves_relative_working_directory_under_workspace() {
        let workspace = tempfile::TempDir::new().expect("temp workspace");
        let empty_path = tempfile::TempDir::new().expect("empty PATH dir");
        let relative_working_directory = "relative-workdir";
        std::fs::create_dir(workspace.path().join(relative_working_directory))
            .expect("relative working directory");

        let previous_path = std::env::var_os("PATH");
        // SAFETY: this test is intended to run with `--test-threads=1` and
        // restores PATH before returning.
        unsafe { std::env::set_var("PATH", empty_path.path()) };
        let _path_guard = scopeguard::guard(previous_path, |previous_path| match previous_path {
            Some(previous_path) => {
                // SAFETY: restoring the process PATH captured before this test.
                unsafe { std::env::set_var("PATH", previous_path) }
            }
            None => {
                // SAFETY: restoring the process PATH captured before this test.
                unsafe { std::env::remove_var("PATH") }
            }
        });

        let tool = GrokCliTool::new(
            test_security_with_workspace(AutonomyLevel::Full, workspace.path().to_path_buf()),
            test_config(),
        );
        let result = tool
            .execute(json!({
                "prompt": "hello",
                "working_directory": relative_working_directory
            }))
            .await
            .expect("should return a result after path validation");
        let error = result.error.as_deref().unwrap_or("");

        assert!(!result.success);
        assert!(
            !error.contains("outside the workspace"),
            "relative working_directory should resolve inside workspace; got {error:?}"
        );
        assert!(
            error.contains("Grok Build CLI ('grok') not found in PATH"),
            "expected missing Grok CLI after path validation; got {error:?}"
        );
    }

    #[test]
    fn grok_cli_env_passthrough_defaults() {
        let config = GrokCliConfig::default();
        assert!(
            config.env_passthrough.is_empty(),
            "env_passthrough should default to empty"
        );
    }

    #[test]
    fn grok_cli_extra_args_defaults() {
        let config = GrokCliConfig::default();
        assert!(
            config.extra_args.is_empty(),
            "extra_args should default to empty"
        );
    }

    #[test]
    fn grok_cli_default_config_values() {
        let config = GrokCliConfig::default();
        assert!(!config.enabled);
        assert!(config.auto_approve);
        assert_eq!(config.timeout_secs, 600);
        assert_eq!(config.max_output_bytes, 2_097_152);
        assert!(config.default_model.is_none());
    }
}
