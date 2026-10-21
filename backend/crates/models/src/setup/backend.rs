use std::{path::PathBuf, process::Stdio, time::Duration};
use tokio::{
    io::AsyncReadExt,
    process::{Child, Command},
};

const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

pub(super) async fn resolve_backend(requested: &str) -> &str {
    let macos = cfg!(target_os = "macos");
    if requested != "auto" || macos {
        return select_backend(requested, macos, false);
    }
    for program in nvidia_programs() {
        let mut command = Command::new(program);
        command.args(["--query-gpu=name", "--format=csv,noheader"]);
        if probe_command(command, PROBE_TIMEOUT).await {
            return select_backend(requested, false, true);
        }
    }
    select_backend(requested, false, false)
}

fn select_backend(requested: &str, macos: bool, nvidia: bool) -> &str {
    match requested {
        "auto" if !macos && nvidia => "cuda",
        "auto" => "cpu",
        explicit => explicit,
    }
}

fn nvidia_programs() -> Vec<PathBuf> {
    let mut programs = vec![PathBuf::from("nvidia-smi")];
    if cfg!(windows) {
        for variable in ["ProgramW6432", "ProgramFiles"] {
            if let Some(directory) = std::env::var_os(variable) {
                let program =
                    PathBuf::from(directory).join("NVIDIA Corporation/NVSMI/nvidia-smi.exe");
                if !programs.contains(&program) {
                    programs.push(program);
                }
            }
        }
        if let Some(directory) = std::env::var_os("SystemRoot") {
            programs.push(PathBuf::from(directory).join("System32/nvidia-smi.exe"));
        }
    }
    programs
}

async fn probe_command(mut command: Command, timeout: Duration) -> bool {
    let Ok(mut child) = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
    else {
        return false;
    };
    match tokio::time::timeout(timeout, probe_output(&mut child)).await {
        Ok(Ok(available)) => available,
        _ => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            false
        }
    }
}

async fn probe_output(child: &mut Child) -> std::io::Result<bool> {
    if !child.wait().await?.success() {
        return Ok(false);
    }
    let Some(stdout) = child.stdout.take() else {
        return Ok(false);
    };
    let mut bytes = Vec::new();
    stdout.take(64 * 1024).read_to_end(&mut bytes).await?;
    Ok(bytes.iter().any(|byte| !byte.is_ascii_whitespace()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(windows: &str, unix: &str) -> Command {
        if cfg!(windows) {
            let mut command = Command::new("cmd.exe");
            command.args(["/D", "/C", windows]);
            command
        } else {
            let mut command = Command::new("sh");
            command.args(["-c", unix]);
            command
        }
    }

    #[test]
    fn automatic_selection_requires_a_working_nvidia_device_and_keeps_explicit_choices() {
        assert_eq!(select_backend("auto", false, false), "cpu");
        assert_eq!(select_backend("auto", false, true), "cuda");
        assert_eq!(select_backend("auto", true, true), "cpu");
        assert_eq!(select_backend("cpu", false, true), "cpu");
        assert_eq!(select_backend("cuda", false, false), "cuda");
    }

    #[tokio::test]
    async fn gpu_probe_requires_success_and_nonempty_output() {
        assert!(
            probe_command(
                command("echo NVIDIA GPU", "printf 'NVIDIA GPU\\n'"),
                PROBE_TIMEOUT
            )
            .await
        );
        assert!(!probe_command(command("exit /b 0", "exit 0"), PROBE_TIMEOUT).await);
        assert!(
            !probe_command(
                command("echo unavailable & exit /b 1", "printf unavailable; exit 1"),
                PROBE_TIMEOUT
            )
            .await
        );
        assert!(!probe_command(Command::new("desynth-missing-nvidia-smi"), PROBE_TIMEOUT).await);
    }

    #[tokio::test]
    async fn stalled_gpu_probe_is_stopped_before_falling_back_to_cpu() {
        let sleeper = if cfg!(windows) {
            let mut command = Command::new("powershell.exe");
            command.args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Start-Sleep -Seconds 20",
            ]);
            command
        } else {
            command("", "exec sleep 20")
        };
        let started = std::time::Instant::now();
        assert!(!probe_command(sleeper, Duration::from_millis(100)).await);
        assert!(started.elapsed() < Duration::from_secs(3));
    }
}
