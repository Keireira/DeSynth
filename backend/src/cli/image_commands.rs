use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use image::RgbaImage;

use crate::images::{self, DiffChannel};

use super::{CompareArgs, DiffArgs, rooted, write_image};

pub(super) fn compare(root: &Path, args: CompareArgs) -> Result<()> {
    let source_path = rooted(root, &args.original);
    let result_path = rooted(root, &args.result);
    let source = load(&source_path)?;
    let mut result = load(&result_path)?;
    if source.dimensions() != result.dimensions() {
        println!(
            "Resizing comparison from {}x{} to {}x{}",
            result.width(),
            result.height(),
            source.width(),
            source.height()
        );
        result = images::resize_for_comparison(&result, source.width(), source.height());
    }
    let metrics = images::extended_metrics(&source, &result)?;
    for (label, key) in [
        ("PSNR (dB)", "psnr_db"),
        ("SSIM", "ssim"),
        ("SSIM low-pass", "ssim_low_pass"),
        ("SSIM high-pass", "ssim_high_pass"),
        ("MAE", "mae"),
        ("MSE", "mse"),
        ("Histogram correlation", "mean_hist_corr"),
        ("Changed (%)", "changed_pct"),
        ("Mean alpha difference", "mean_alpha"),
    ] {
        match metrics[key].as_f64() {
            Some(value) => println!("  {label:<24} {value:.6}"),
            None => println!("  {label:<24} -"),
        }
    }
    if args.visual || args.visual_out.is_some() {
        let default_name = format!(
            "comparison_{}.png",
            result_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        );
        let path = args
            .visual_out
            .map(|path| rooted(root, &path))
            .unwrap_or_else(|| root.join("out").join(default_name));
        check_output(&path, &[&source_path, &result_path])?;
        write_image(
            &path,
            &images::comparison_visual(&source, &result, &metrics)?,
            None,
        )?;
    }
    Ok(())
}

pub(super) fn diff(root: &Path, args: DiffArgs) -> Result<()> {
    let source_path = rooted(root, &args.original);
    let result_path = rooted(root, &args.result);
    let output = rooted(root, &args.out);
    let raw = args.raw_out.map(|path| rooted(root, &path));
    let preview = args.preview_out.map(|path| rooted(root, &path));
    let paths: Vec<&Path> = [&source_path, &result_path, &output]
        .into_iter()
        .map(PathBuf::as_path)
        .chain(raw.as_deref())
        .chain(preview.as_deref())
        .collect();
    for index in 2..paths.len() {
        check_output(paths[index], &paths[..index])?;
    }
    let source = load(&source_path)?;
    let result = load(&result_path)?;
    let visual = images::difference(&source, &result, args.gain, DiffChannel::Rgba)?;
    write_image(&output, &visual, None)?;
    if let Some(raw) = raw {
        write_image(
            &raw,
            &images::difference(&source, &result, args.gain, DiffChannel::Raw)?,
            None,
        )?;
    }
    if let Some(preview) = preview {
        let mut image = RgbaImage::new(source.width() * 2, source.height());
        image::imageops::replace(
            &mut image,
            &images::difference(&source, &result, args.gain, DiffChannel::Rgb)?,
            0,
            0,
        );
        image::imageops::replace(
            &mut image,
            &images::difference(&source, &result, args.gain, DiffChannel::Alpha)?,
            i64::from(source.width()),
            0,
        );
        write_image(&preview, &image, None)?;
    }
    let mut changed = 0_u64;
    let mut rgba_delta = 0_u64;
    let mut alpha_delta = 0_u64;
    for (left, right) in source.pixels().zip(result.pixels()) {
        changed += u64::from(left != right);
        for channel in 0..4 {
            rgba_delta += u64::from(left[channel].abs_diff(right[channel]));
        }
        alpha_delta += u64::from(left[3].abs_diff(right[3]));
    }
    let pixels = u64::from(source.width()) * u64::from(source.height());
    println!(
        "Visible RGBA diff: {}x{}; gain {}; changed {:.2}%; mean RGBA difference {:.3}; mean alpha difference {:.3}",
        source.width(),
        source.height(),
        args.gain,
        changed as f64 / pixels as f64 * 100.0,
        rgba_delta as f64 / (pixels * 4) as f64,
        alpha_delta as f64 / pixels as f64,
    );
    Ok(())
}

fn load(path: &Path) -> Result<RgbaImage> {
    images::open(&std::fs::read(path).with_context(|| format!("Cannot read {}", path.display()))?)
        .with_context(|| format!("Cannot read {}", path.display()))
}

fn output_identity(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.canonicalize()?);
    }
    let absolute = std::path::absolute(path)?;
    let parent = absolute.parent().context("Output path has no parent")?;
    let name = absolute
        .file_name()
        .context("Output path has no filename")?;
    Ok(output_identity(parent)?.join(name))
}

pub(super) fn check_output(output: &Path, sources: &[&Path]) -> Result<()> {
    let resolved_output = output_identity(output)?;
    for source in sources {
        let same_path = resolved_output == output_identity(source)?;
        let same_file =
            output.exists() && source.exists() && same_file::is_same_file(output, source)?;
        ensure!(
            !same_path && !same_file,
            "Output must not overwrite another image: {}",
            source.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outputs_cannot_overwrite_input_hardlinks_or_duplicate_destinations() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("source.png");
        std::fs::write(&source, [1, 2, 3])?;
        let link = directory.path().join("link.png");
        std::fs::hard_link(&source, &link)?;
        assert!(check_output(&link, &[&source]).is_err());
        let future = directory.path().join("new/nested.png");
        assert!(check_output(&future, &[&future]).is_err());
        check_output(&future, &[&source])?;
        Ok(())
    }
}
