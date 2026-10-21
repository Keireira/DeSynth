use anyhow::{Result, ensure};
use font8x8::UnicodeFonts;
use image::{DynamicImage, Rgba, RgbaImage};
use serde_json::Value;

use super::resize::lanczos_three;

/// A full-image comparison is an output artifact, independent from the browser canvas.
pub fn comparison_visual(
    source: &RgbaImage,
    result: &RgbaImage,
    metrics: &Value,
) -> Result<RgbaImage> {
    ensure!(
        source.width() > 0 && source.height() > 0 && result.width() > 0 && result.height() > 0,
        "Image dimensions must be positive"
    );
    let target_height = 720;
    let max_width = 1280;
    let fitted = |image: &RgbaImage| {
        let rgb = DynamicImage::ImageRgba8(image.clone()).into_rgb8();
        // A very wide, short input must not expand into a multi-gigabyte canvas.
        let width_at_target = u64::from(image.width()) * target_height / u64::from(image.height());
        let (width, height) = if width_at_target > max_width {
            (
                max_width as u32,
                (u64::from(image.height()) * max_width / u64::from(image.width())).max(1) as u32,
            )
        } else {
            (width_at_target.max(1) as u32, target_height as u32)
        };
        DynamicImage::ImageRgb8(lanczos_three(&rgb, width, height)).into_rgba8()
    };
    let left = fitted(source);
    let right = fitted(result);
    let mut canvas = RgbaImage::from_pixel(
        left.width() + right.width() + 64,
        888,
        Rgba([24, 24, 24, 255]),
    );
    image::imageops::replace(&mut canvas, &left, 24, 60);
    image::imageops::replace(&mut canvas, &right, i64::from(left.width() + 40), 60);
    text(&mut canvas, 24, 28, "Original", [220, 220, 220, 255], 2);
    text(
        &mut canvas,
        left.width() + 40,
        28,
        "Result",
        [220, 220, 220, 255],
        2,
    );
    let number = |key: &str, precision: usize| match metrics[key].as_f64() {
        Some(value) => format!("{value:.precision$}"),
        None => "-".into(),
    };
    let line = format!(
        "PSNR {} dB   SSIM {}   MAE {}",
        number("psnr_db", 2),
        number("ssim", 4),
        number("mae", 2)
    );
    text(&mut canvas, 24, 800, &line, [245, 245, 245, 255], 2);
    let line = format!(
        "SSIM low {}   SSIM high {}   Histogram {}",
        number("ssim_low_pass", 4),
        number("ssim_high_pass", 4),
        number("mean_hist_corr", 4)
    );
    text(&mut canvas, 24, 834, &line, [190, 190, 190, 255], 2);
    Ok(canvas)
}

fn text(image: &mut RgbaImage, mut x: u32, y: u32, content: &str, color: [u8; 4], scale: u32) {
    for character in content.chars() {
        if let Some(glyph) = font8x8::BASIC_FONTS.get(character) {
            for (row, bits) in glyph.into_iter().enumerate() {
                for column in 0..8 {
                    if bits & (1 << column) == 0 {
                        continue;
                    }
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let (px, py) = (x + column * scale + dx, y + row as u32 * scale + dy);
                            if px < image.width() && py < image.height() {
                                image.put_pixel(px, py, Rgba(color));
                            }
                        }
                    }
                }
            }
        }
        x += 9 * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_canvas_is_bounded_for_a_very_wide_input() -> Result<()> {
        let source = RgbaImage::from_pixel(12_000, 1, Rgba([80, 120, 200, 255]));
        let visual = comparison_visual(&source, &source, &serde_json::json!({}))?;
        assert_eq!(visual.dimensions(), (2624, 888));
        assert_eq!(visual.get_pixel(24, 60), &Rgba([80, 120, 200, 255]));
        Ok(())
    }

    #[test]
    fn visual_is_labelled_and_opaque_even_for_transparent_inputs() -> Result<()> {
        let source = RgbaImage::from_pixel(8, 8, Rgba([120, 60, 20, 0]));
        let metrics = serde_json::json!({"ssim":1.0,"mae":0.0});
        let visual = comparison_visual(&source, &source, &metrics)?;
        assert_eq!(visual.dimensions(), (1504, 888));
        assert_eq!(visual.get_pixel(30, 70), &Rgba([120, 60, 20, 255]));
        assert!(visual.pixels().all(|pixel| pixel[3] == 255));
        assert!((24..44).any(|y| (24..180).any(|x| visual.get_pixel(x, y)[0] > 100)));
        Ok(())
    }
}
