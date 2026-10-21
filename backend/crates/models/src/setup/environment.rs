use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use tokio::process::Command;

fn python_command(program: &Path) -> Command {
    let mut command = Command::new(dunce::simplified(program));
    command.env_remove("PYTHONHOME").env_remove("PYTHONPATH");
    command
}

pub fn environment_python(directory: &Path) -> PathBuf {
    dunce::simplified(directory).join(if cfg!(windows) {
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
    let output = python_command(&python)
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

async fn bootstrap_python(minor: Option<&str>) -> Result<(PathBuf, Vec<String>)> {
    let mut candidates = Vec::new();
    let versions: Vec<_> = minor.map_or_else(|| vec!["3.11", "3.12", "3.13"], |minor| vec![minor]);
    if let Some(python) = std::env::var_os("DESYNTH_BOOTSTRAP_PYTHON") {
        candidates.push((PathBuf::from(python), Vec::new()));
    } else {
        if cfg!(windows) {
            for version in &versions {
                candidates.push((PathBuf::from("py"), vec![format!("-{version}")]));
            }
        }
        for version in versions {
            candidates.push((PathBuf::from(format!("python{version}")), Vec::new()));
        }
        for program in ["python3", "python"] {
            candidates.push((PathBuf::from(program), Vec::new()));
        }
    }
    for (program, arguments) in candidates {
        let output = python_command(&program)
            .args(&arguments)
            .arg("--version")
            .kill_on_drop(true)
            .output()
            .await;
        if let Ok(output) = output {
            let version = String::from_utf8_lossy(&output.stdout);
            if output.status.success()
                && supported_python_version(&version)
                && minor.is_none_or(|minor| version.starts_with(&format!("Python {minor}.")))
            {
                return Ok((program, arguments));
            }
        }
    }
    if let Some(minor) = minor {
        bail!(
            "Install Python {minor} to repair this ML environment, or choose a new VENV directory. Set DESYNTH_BOOTSTRAP_PYTHON if Python is installed outside PATH."
        );
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
    let mut child = python_command(program)
        .args(arguments)
        .current_dir(dunce::simplified(root))
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

#[derive(Debug, PartialEq, Eq)]
enum EnvironmentState {
    Ready,
    Create(Option<String>),
}

fn activation_script(directory: &Path) -> PathBuf {
    directory.join(if cfg!(windows) {
        "Scripts/activate"
    } else {
        "bin/activate"
    })
}

fn environment_state(directory: &Path) -> Result<EnvironmentState> {
    if !directory.exists() {
        return Ok(EnvironmentState::Create(None));
    }
    ensure!(
        directory.is_dir(),
        "{} is not a directory.",
        directory.display()
    );
    let config = directory.join("pyvenv.cfg");
    if !config.is_file() {
        ensure!(
            std::fs::read_dir(directory)?.next().transpose()?.is_none(),
            "{} is not a recognized ML environment or an empty directory. Choose a different VENV directory.",
            directory.display()
        );
        return Ok(EnvironmentState::Create(None));
    }
    let config =
        std::fs::read_to_string(config).context("Cannot read ML environment configuration")?;
    let version = config.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key.trim() == "version").then(|| value.trim())
    });
    if let Some(version) = version {
        ensure!(
            supported_python_version(&format!("Python {version}")),
            "The existing ML environment uses Python {version}; dependencies require Python 3.11–3.13. Choose a new VENV directory."
        );
    }
    if environment_python(directory).is_file() && activation_script(directory).is_file() {
        return Ok(EnvironmentState::Ready);
    }
    // Match its Python minor version so existing native packages remain compatible.
    let minor = version.map(|version| version.split('.').take(2).collect::<Vec<_>>().join("."));
    Ok(EnvironmentState::Create(minor))
}

pub(super) async fn create_environment(root: &Path, directory: &Path) -> Result<()> {
    let root = dunce::simplified(root);
    let directory = dunce::simplified(directory);
    let minor = match environment_state(directory)? {
        EnvironmentState::Ready => {
            let python = require_environment(directory).await?;
            println!("Using ML environment: {}", directory.display());
            return ensure_pip(root, &python).await;
        }
        EnvironmentState::Create(minor) => minor,
    };
    if directory.exists() {
        println!("Repairing ML environment: {}", directory.display());
    }
    if environment_python(directory).is_file() {
        require_environment(directory).await?;
    }
    let (program, mut arguments) = bootstrap_python(minor.as_deref()).await?;
    arguments.extend([
        "-m".into(),
        "venv".into(),
        "--without-pip".into(),
        directory.to_string_lossy().into_owned(),
    ]);
    checked_command(root, &program, &arguments).await?;
    let python = require_environment(directory).await?;
    ensure_pip(root, &python).await
}

async fn ensure_pip(root: &Path, python: &Path) -> Result<()> {
    let output = python_command(python)
        .args(["-m", "pip", "--version"])
        .kill_on_drop(true)
        .output()
        .await
        .context("Cannot check pip in the ML environment")?;
    if !output.status.success() {
        checked_command(
            root,
            python,
            &[
                "-m".into(),
                "ensurepip".into(),
                "--upgrade".into(),
                "--default-pip".into(),
            ],
        )
        .await?;
    }
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
            "-r".into(),
            requirements.to_string_lossy().into_owned(),
            "-r".into(),
            root.join("inference/requirements/dev.txt")
                .to_string_lossy()
                .into_owned(),
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

    #[test]
    fn environment_recovery_only_accepts_missing_empty_or_recognized_directories() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("environment");
        assert_eq!(
            environment_state(&directory).unwrap(),
            EnvironmentState::Create(None)
        );
        std::fs::create_dir(&directory).unwrap();
        assert_eq!(
            environment_state(&directory).unwrap(),
            EnvironmentState::Create(None)
        );
        let unrelated = directory.join("important.txt");
        std::fs::write(&unrelated, "Keep this file").unwrap();
        assert!(environment_state(&directory).is_err());
        assert_eq!(
            std::fs::read_to_string(&unrelated).unwrap(),
            "Keep this file"
        );
        std::fs::write(directory.join("pyvenv.cfg"), "version = 3.12.7\n").unwrap();
        assert_eq!(
            environment_state(&directory).unwrap(),
            EnvironmentState::Create(Some("3.12".into()))
        );
        assert_eq!(
            std::fs::read_to_string(&unrelated).unwrap(),
            "Keep this file"
        );
    }

    #[test]
    fn existing_interpreters_are_checked_instead_of_overwritten() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("pyvenv.cfg"), "version = 3.11.9\n").unwrap();
        let python = environment_python(root.path());
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, "Existing interpreter").unwrap();
        assert_eq!(
            environment_state(root.path()).unwrap(),
            EnvironmentState::Create(Some("3.11".into()))
        );
        std::fs::write(activation_script(root.path()), "Existing activation script").unwrap();
        assert_eq!(
            environment_state(root.path()).unwrap(),
            EnvironmentState::Ready
        );
        std::fs::write(root.path().join("pyvenv.cfg"), "version = 3.14.0\n").unwrap();
        let error = environment_state(root.path()).unwrap_err();
        assert!(error.to_string().contains("Python 3.14.0"));
        assert_eq!(
            std::fs::read_to_string(python).unwrap(),
            "Existing interpreter"
        );
    }

    #[tokio::test]
    async fn canonical_paths_create_and_recover_the_complete_environment_offline() {
        let Ok((program, mut arguments)) = bootstrap_python(None).await else {
            eprintln!("Skipping offline venv recovery: no Python 3.11–3.13 is available.");
            return;
        };
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project with spaces");
        std::fs::create_dir(&project).unwrap();
        let project = project.canonicalize().unwrap();
        let fresh = project.join("new environment");
        std::fs::create_dir(&fresh).unwrap();
        let fresh = fresh.canonicalize().unwrap();
        crate::setup::run(&project, "venv", &fresh, "cpu", "auto", "auto")
            .await
            .unwrap();
        let fresh_python = require_environment(&fresh).await.unwrap();
        assert!(activation_script(&fresh).is_file());
        assert!(
            python_command(&fresh_python)
                .args(["-m", "pip", "--version"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );
        if cfg!(windows) {
            let config = std::fs::read_to_string(fresh.join("pyvenv.cfg")).unwrap();
            assert!(!config.contains(r"\\?\"));
        }

        let directory = project.join("interrupted environment");
        arguments.extend([
            "-m".into(),
            "venv".into(),
            "--without-pip".into(),
            directory.to_string_lossy().into_owned(),
        ]);
        checked_command(&project, &program, &arguments)
            .await
            .unwrap();
        let directory = directory.canonicalize().unwrap();
        let python = require_environment(&directory).await.unwrap();
        let pip = python_command(&python)
            .args(["-m", "pip", "--version"])
            .output()
            .await
            .unwrap();
        assert!(!pip.status.success());

        let preserved = directory.join("preserved.txt");
        std::fs::write(&preserved, "Keep existing environment files").unwrap();
        let package = directory.join(if cfg!(windows) {
            "Lib/site-packages/preserved_package.py"
        } else {
            "lib/preserved_package.py"
        });
        std::fs::write(&package, "preserved = True\n").unwrap();
        std::fs::remove_file(activation_script(&directory)).unwrap();
        crate::setup::run(&project, "venv", &directory, "cpu", "auto", "auto")
            .await
            .unwrap();
        assert!(activation_script(&directory).is_file());
        assert!(
            python_command(&python)
                .args(["-m", "pip", "--version"])
                .output()
                .await
                .unwrap()
                .status
                .success()
        );

        std::fs::remove_file(&python).unwrap();
        crate::setup::run(&project, "venv", &directory, "cpu", "auto", "auto")
            .await
            .unwrap();
        crate::setup::run(&project, "venv", &directory, "cpu", "auto", "auto")
            .await
            .unwrap();
        require_environment(&directory).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(preserved).unwrap(),
            "Keep existing environment files"
        );
        assert_eq!(
            std::fs::read_to_string(package).unwrap(),
            "preserved = True\n"
        );
    }
}
