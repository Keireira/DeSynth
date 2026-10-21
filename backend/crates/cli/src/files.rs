use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

use anyhow::{Context, Result};

pub(super) fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    publish(path, |file| {
        file.write_all(bytes)?;
        Ok(())
    })
}

pub(super) fn copy(source: &Path, destination: &Path) -> Result<()> {
    let mut source = File::open(source)?;
    publish(destination, |file| {
        std::io::copy(&mut source, file)?;
        Ok(())
    })
}

fn publish(path: &Path, write: impl FnOnce(&mut File) -> Result<()>) -> Result<()> {
    let parent = path.parent().context("Output path has no parent")?;
    std::fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".desynth-{}.tmp", uuid::Uuid::new_v4().simple()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        write(&mut file)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_publish_preserves_the_previous_file_and_removes_partial_data() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let output = directory.path().join("result.png");
        std::fs::write(&output, b"previous")?;
        let result = publish(&output, |file| {
            file.write_all(b"partial")?;
            anyhow::bail!("simulated write failure")
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(&output)?, b"previous");
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        write(&output, b"replacement")?;
        assert_eq!(std::fs::read(output)?, b"replacement");
        Ok(())
    }

    #[test]
    fn publishing_replaces_an_existing_hardlink_without_modifying_its_other_name() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("original.png");
        let output = directory.path().join("result.png");
        std::fs::write(&source, b"source")?;
        std::fs::hard_link(&source, &output)?;
        write(&output, b"result")?;
        assert_eq!(std::fs::read(source)?, b"source");
        assert_eq!(std::fs::read(output)?, b"result");
        Ok(())
    }
}
