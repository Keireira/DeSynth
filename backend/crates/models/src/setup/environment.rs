use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use tokio::process::Command;

pub fn environment_python(directory: &Path) -> PathBuf {
    directory.join(if cfg!(windows) {
        "Scripts/python.exe"
    } else {
        "bin/python"
    })
}

pub(super) async fn require_environment(directory: &Path) -> Result<PathBuf> {
    let python = environment_python(directory);
    ensure!(
        directory.join("pyvenv.cfg").is_file() && python.is_file(),
        "No ML environment at {}. Run make install first.",
        directory.display()
    );
    let output = Command::new(&python)
        .arg("--version")
        .kill_on_drop(true)
        .output()
        .await
        .context("Cannot run the ML environment's Python")?;
    let version = String::from_utf8_lossy(&output.stdout);
    ensure!(
        output.status.success() && supported_python_version(&version),
        "The ML dependencies require Python 3.11–3.13; found {}.",
        version.trim()
    );
    Ok(python)
}

async fn bootstrap_python() -> Result<(PathBuf, Vec<String>)> {
    let mut candidates = Vec::new();
    if let Some(python) = std::env::var_os("DESYNTH_BOOTSTRAP_PYTHON") {
        candidates.push((PathBuf::from(python), Vec::new()));
    } else {
        if cfg!(windows) {
            for version in ["3.11", "3.12", "3.13"] {
                candidates.push((PathBuf::from("py"), vec![format!("-{version}")]));
            }
        }
        for program in [
            "python3.11",
            "python3.12",
            "python3.13",
            "python3",
            "python",
        ] {
            candidates.push((PathBuf::from(program), Vec::new()));
        }
    }
    for (program, arguments) in candidates {
        let output = Command::new(&program)
            .args(&arguments)
            .arg("--version")
            .kill_on_drop(true)
            .output()
            .await;
        if let Ok(output) = output {
            let version = String::from_utf8_lossy(&output.stdout);
            if output.status.success() && supported_python_version(&version) {
                return Ok((program, arguments));
            }
        }
    }
    bail!("Install Python 3.11–3.13, or set DESYNTH_BOOTSTRAP_PYTHON to its executable path.")
}

fn supported_python_version(version: &str) -> bool {
    let Some(version) = version.trim().strip_prefix("Python ") else {
        return false;
    };
    let mut parts = version.split('.');
    parts.next() == Some("3") && matches!(parts.next(), Some("11" | "12" | "13"))
}

async fn checked_command(root: &Path, program: &Path, arguments: &[String]) -> Result<()> {
    println!("> {} {}", program.display(), arguments.join(" "));
    let mut child = Command::new(program)
        .args(arguments)
        .current_dir(root)
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("Cannot start {}", program.display()))?;
    let status = tokio::select! {
        status = child.wait() => status?,
        signal = tokio::signal::ctrl_c() => {
            signal.context("Cannot listen for Ctrl+C")?;
            let _ = child.kill().await;
            let _ = child.wait().await;
            bail!("Setup cancelled");
        }
    };
    ensure!(
        status.success(),
        "{} failed with {status}",
        program.display()
    );
    Ok(())
}

pub(super) async fn create_environment(root: &Path, directory: &Path) -> Result<()> {
    if directory.exists() {
        require_environment(directory).await?;
        println!("Using ML environment: {}", directory.display());
        return Ok(());
    }
    let (program, mut arguments) = bootstrap_python().await?;
    arguments.extend([
        "-m".into(),
        "venv".into(),
        directory.to_string_lossy().into_owned(),
    ]);
    checked_command(root, &program, &arguments).await?;
    require_environment(directory).await?;
    Ok(())
}

pub(super) async fn install(root: &Path, directory: &Path, backend: &str) -> Result<()> {
    create_environment(root, directory).await?;
    let python = require_environment(directory).await?;
    let requirements = root.join(if backend == "cuda" {
        "inference/requirements/cuda.txt"
    } else {
        "inference/requirements/cpu.txt"
    });
    for arguments in [
        vec![
            "-m".into(),
            "pip".into(),
            "install".into(),
            "--upgrade".into(),
            "pip".into(),
        ],
        vec![
            "-m".into(),
            "pip".into(),
            "install".into(),
            "-r".into(),
            requirements.to_string_lossy().into_owned(),
        ],
        vec!["-m".into(), "pip".into(), "check".into()],
    ] {
        checked_command(root, &python, &arguments).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpreter_version_checks_keep_the_supported_minor_versions() {
        for version in ["Python 3.11.9", "Python 3.12.0rc1", "Python 3.13.7\n"] {
            assert!(supported_python_version(version));
        }
        for version in [
            "Python 2.7.18",
            "Python 3.10.9",
            "Python 3.14.0",
            "not installed",
            "",
        ] {
            assert!(!supported_python_version(version));
        }
    }
}
