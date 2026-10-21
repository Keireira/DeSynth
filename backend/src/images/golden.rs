//! Golden inputs were produced by the pre-migration Pillow/OpenCV implementation.
//! A one-level channel difference covers floating-point summation and uint8 truncation;
//! this must not conceal altered detail bands, alpha changes or orientation mistakes.

use std::path::PathBuf;

use anyhow::Result;
use image::RgbaImage;

use super::{RestorationParams, finish, metrics, open, prepare};

fn fixture(name: &str) -> Result<RgbaImage> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/images")
        .join(name);
    open(&std::fs::read(path)?)
}

fn assert_channels_close(actual: &RgbaImage, expected: &RgbaImage, max_error: u8, max_mean: f64) {
    assert_eq!(actual.dimensions(), expected.dimensions());
    let mut error = 0_u64;
    for (actual, expected) in actual.pixels().zip(expected.pixels()) {
        assert_eq!(actual[3], expected[3], "source alpha changed");
        for channel in 0..3 {
            let delta = actual[channel].abs_diff(expected[channel]);
            assert!(
                delta <= max_error,
                "channel error {delta} exceeds {max_error}: {actual:?} != {expected:?}"
            );
            error += u64::from(delta);
        }
    }
    let mean = error as f64 / f64::from(actual.width() * actual.height() * 3);
    assert!(mean <= max_mean, "mean RGB error {mean} exceeds {max_mean}");
}

#[test]
fn restoration_matches_pre_migration_gaussian_edge_and_unsharp() -> Result<()> {
    let source = fixture("source.png")?;
    let clean = fixture("clean.png")?;
    for method in ["gaussian", "edge"] {
        for unsharp in [0.0, 0.2] {
            let expected = fixture(&format!("{method}-{unsharp}.png"))?;
            let actual = finish(
                &clean,
                &source,
                &RestorationParams {
                    restore_mode: method.into(),
                    unsharp,
                    ..Default::default()
                },
            )?;
            assert_channels_close(&actual, &expected, 1, 0.02);
        }
    }
    Ok(())
}

#[test]
fn resized_pipeline_output_retains_lanczos_four_detail() -> Result<()> {
    let actual = finish(
        &fixture("small-clean.png")?,
        &fixture("source.png")?,
        &RestorationParams {
            restore_mode: "edge".into(),
            ..Default::default()
        },
    )?;
    assert_channels_close(&actual, &fixture("resized-edge.png")?, 2, 0.2);
    Ok(())
}

#[test]
fn every_exif_orientation_matches_pillow_display_pixels() -> Result<()> {
    for orientation in 1..=8 {
        assert_eq!(
            fixture(&format!("exif-{orientation}.png"))?,
            fixture(&format!("exif-{orientation}-expected.png"))?,
            "EXIF orientation {orientation}"
        );
    }
    Ok(())
}

#[test]
fn icc_conversion_matches_linear_rgb_reference_without_losing_alpha() -> Result<()> {
    assert_channels_close(
        &fixture("icc-linear.png")?,
        &fixture("icc-expected.png")?,
        0,
        0.0,
    );
    Ok(())
}

#[test]
fn lab_tiff_profiles_are_applied_before_rgb_conversion_for_raw_and_compressed_strips() -> Result<()>
{
    let expected = fixture("lab-expected.png")?;
    for name in [
        "lab-raw.tiff",
        "lab-tiff_lzw.tiff",
        "lab-tiff_adobe_deflate.tiff",
    ] {
        assert_channels_close(&fixture(name)?, &expected, 0, 0.0);
    }
    Ok(())
}

#[test]
fn cmyk_tiff_and_jpeg_profiles_are_applied_to_ink_samples() -> Result<()> {
    for extension in ["tiff", "jpg"] {
        assert_channels_close(
            &fixture(&format!("cmyk.{extension}"))?,
            &fixture(&format!("cmyk-{extension}-expected.png"))?,
            0,
            0.0,
        );
    }
    Ok(())
}

#[test]
fn lab_and_cmyk_keep_exact_alpha_hidden_colors_and_exif_orientation() -> Result<()> {
    for kind in ["lab", "cmyk"] {
        assert_channels_close(
            &fixture(&format!("{kind}-alpha-exif.tiff"))?,
            &fixture(&format!("{kind}-alpha-exif-expected.png"))?,
            0,
            0.0,
        );
    }
    Ok(())
}

#[test]
fn ycck_jpeg_is_converted_through_its_original_ink_profile() -> Result<()> {
    assert_channels_close(
        &fixture("ycck.jpg")?,
        &fixture("ycck-expected.png")?,
        1,
        0.25,
    );
    Ok(())
}

#[test]
fn malformed_lab_and_cmyk_profiles_return_actionable_errors() -> Result<()> {
    for name in ["lab-raw.tiff", "cmyk.tiff", "cmyk.jpg"] {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/images")
            .join(name);
        let mut bytes = std::fs::read(path)?;
        let signature = bytes
            .windows(4)
            .position(|value| value == b"acsp")
            .expect("ICC test fixture has a profile header");
        bytes[signature..signature + 4].copy_from_slice(b"bad!");
        assert!(
            open(&bytes)
                .unwrap_err()
                .to_string()
                .contains("ICC color profile")
        );
    }
    Ok(())
}

#[test]
fn metrics_match_numpy_and_skimage_at_full_resolution() -> Result<()> {
    let measured = metrics(&fixture("source.png")?, &fixture("clean.png")?)?;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/images/metrics.json");
    let expected: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    for key in ["mse", "mae", "psnr_db"] {
        assert!(
            (measured[key].as_f64().unwrap() - expected[key].as_f64().unwrap()).abs() < 1e-10,
            "{key}: {} != {}",
            measured[key],
            expected[key]
        );
    }
    assert!((measured["ssim"].as_f64().unwrap() - expected["ssim"].as_f64().unwrap()).abs() < 1e-5);
    Ok(())
}

#[test]
fn model_input_and_nonrestored_output_match_pillow_lanczos_exactly() -> Result<()> {
    let source = fixture("source.png")?;
    assert_eq!(prepare(&source, false), fixture("prepared.png")?);
    let rgb = image::DynamicImage::ImageRgba8(source).into_rgb8();
    for (width, height, name) in [(13, 9, "downscaled.png"), (113, 79, "upscaled.png")] {
        let actual = super::resize::lanczos_three(&rgb, width, height);
        let expected = image::DynamicImage::ImageRgba8(fixture(name)?).into_rgb8();
        assert_eq!(actual, expected, "Pillow Lanczos-3 {width}x{height}");
    }
    Ok(())
}
