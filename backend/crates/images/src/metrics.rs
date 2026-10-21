use anyhow::{Result, ensure};
use image::RgbaImage;
use serde_json::{Value, json};

use super::restore::gaussian_plane;

/// RGB similarity retains the UI contract; changed pixels and mean alpha use all four channels.
pub fn metrics(source: &RgbaImage, result: &RgbaImage) -> Result<Value> {
    ensure!(
        source.dimensions() == result.dimensions(),
        "Image dimensions differ. Comparison requires images of the same size."
    );
    ensure!(
        source.width() > 0 && source.height() > 0,
        "Image dimensions must be positive"
    );
    let mut squared = 0.0_f64;
    let mut absolute = 0.0_f64;
    let mut alpha = 0.0_f64;
    let mut changed = 0_u64;
    for (left, right) in source.pixels().zip(result.pixels()) {
        for channel in 0..3 {
            let delta = f64::from(left[channel]) - f64::from(right[channel]);
            squared += delta * delta;
            absolute += delta.abs();
        }
        alpha += f64::from(left[3].abs_diff(right[3]));
        changed += u64::from(left != right);
    }
    let pixels = u64::from(source.width()) * u64::from(source.height());
    let mse = squared / (pixels * 3) as f64;
    let mut scores = Vec::new();
    for channel in 0..3 {
        let a: Vec<f32> = source
            .pixels()
            .map(|pixel| f32::from(pixel[channel]))
            .collect();
        let b: Vec<f32> = result
            .pixels()
            .map(|pixel| f32::from(pixel[channel]))
            .collect();
        if let Some(score) = ssim(&a, &b, source.width() as usize, source.height() as usize) {
            scores.push(score);
        }
    }
    Ok(json!({
        "shape": [source.height(), source.width(), 3],
        "mse": mse,
        "mae": absolute / (pixels * 3) as f64,
        "psnr_db": if mse > 0.0 { Some(10.0 * (255.0_f64.powi(2) / mse).log10()) } else { None },
        "ssim": mean(&scores),
        "changed_pct": changed as f64 / pixels as f64 * 100.0,
        "identical": changed == 0,
        "mean_alpha": alpha / pixels as f64,
    }))
}

pub fn extended_metrics(source: &RgbaImage, result: &RgbaImage) -> Result<Value> {
    let mut result_metrics = metrics(source, result)?;
    let mut correlations = Vec::new();
    let mut low = Vec::new();
    let mut high = Vec::new();
    let (width, height) = (source.width() as usize, source.height() as usize);
    for channel in 0..3 {
        let a: Vec<f32> = source
            .pixels()
            .map(|pixel| f32::from(pixel[channel]))
            .collect();
        let b: Vec<f32> = result
            .pixels()
            .map(|pixel| f32::from(pixel[channel]))
            .collect();
        correlations.push(histogram_correlation(&a, &b));
        let low_a = gaussian_plane(&a, width, height, 1.95);
        let low_b = gaussian_plane(&b, width, height, 1.95);
        let quantize = |value: f32| f32::from(value.clamp(0.0, 255.0) as u8);
        let high_a: Vec<f32> = a
            .iter()
            .zip(&low_a)
            .map(|(a, low)| quantize(a - low + 128.0))
            .collect();
        let high_b: Vec<f32> = b
            .iter()
            .zip(&low_b)
            .map(|(b, low)| quantize(b - low + 128.0))
            .collect();
        let low_a: Vec<f32> = low_a.into_iter().map(quantize).collect();
        let low_b: Vec<f32> = low_b.into_iter().map(quantize).collect();
        if let Some(value) = ssim(&low_a, &low_b, width, height) {
            low.push(value);
        }
        if let Some(value) = ssim(&high_a, &high_b, width, height) {
            high.push(value);
        }
    }
    result_metrics["ssim_low_pass"] = json!(mean(&low));
    result_metrics["ssim_high_pass"] = json!(mean(&high));
    result_metrics["mean_hist_corr"] = json!(mean(&correlations));
    result_metrics["hist_corr_rgb"] = json!(correlations);
    Ok(result_metrics)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn histogram_correlation(a: &[f32], b: &[f32]) -> f64 {
    let mut left = [0.0_f64; 256];
    let mut right = [0.0_f64; 256];
    for value in a {
        left[*value as usize] += 1.0;
    }
    for value in b {
        right[*value as usize] += 1.0;
    }
    let avg = a.len() as f64 / 256.0;
    let mut covariance = 0.0;
    let mut variance_a = 0.0;
    let mut variance_b = 0.0;
    for index in 0..256 {
        let da = left[index] - avg;
        let db = right[index] - avg;
        covariance += da * db;
        variance_a += da * da;
        variance_b += db * db;
    }
    let denominator = (variance_a * variance_b).sqrt();
    if denominator <= f64::EPSILON {
        1.0
    } else {
        covariance / denominator
    }
}

/// The 7x7 sample-covariance SSIM window and three-pixel crop match skimage defaults.
/// Six-row halos are implicit in these horizontal sums, avoiding full-sized f64 integral images.
fn ssim(a: &[f32], b: &[f32], width: usize, height: usize) -> Option<f64> {
    if width < 7 || height < 7 {
        return None;
    }
    let columns = width - 6;
    let mut rows = vec![vec![[0.0_f64; 5]; columns]; 7];
    let mut totals = vec![[0.0_f64; 5]; columns];
    let mut score = 0.0;
    for y in 0..height {
        let ring = &mut rows[y % 7];
        let mut window = [0.0_f64; 5];
        for x in 0..width {
            let add = |index: usize| {
                let aa = f64::from(a[index]);
                let bb = f64::from(b[index]);
                [aa, bb, aa * aa, bb * bb, aa * bb]
            };
            let incoming = add(y * width + x);
            for index in 0..5 {
                window[index] += incoming[index];
            }
            if x >= 7 {
                let outgoing = add(y * width + x - 7);
                for index in 0..5 {
                    window[index] -= outgoing[index];
                }
            }
            if x < 6 {
                continue;
            }
            let column = x - 6;
            for index in 0..5 {
                totals[column][index] += window[index] - ring[column][index];
            }
            ring[column] = window;
            if y < 6 {
                continue;
            }
            let [ma, mb, aa, bb, ab] = totals[column].map(|value| value / 49.0);
            let covariance_norm = 49.0 / 48.0;
            let va = covariance_norm * (aa - ma * ma);
            let vb = covariance_norm * (bb - mb * mb);
            let covariance = covariance_norm * (ab - ma * mb);
            score += ((2.0 * ma * mb + 6.5025) * (2.0 * covariance + 58.5225))
                / ((ma * ma + mb * mb + 6.5025) * (va + vb + 58.5225));
        }
    }
    Some(score / (columns * (height - 6)) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn identity_and_hidden_rgba_differences_are_reported() -> Result<()> {
        let source = RgbaImage::from_pixel(9, 9, Rgba([50, 100, 200, 0]));
        let identity = metrics(&source, &source)?;
        assert_eq!(identity["ssim"], 1.0);
        assert_eq!(identity["psnr_db"], Value::Null);
        assert_eq!(identity["identical"], true);
        let mut result = source.clone();
        result.get_pixel_mut(4, 4)[0] += 5;
        result.get_pixel_mut(0, 0)[3] = 81;
        let measured = metrics(&source, &result)?;
        assert_eq!(measured["mean_alpha"], 1.0);
        assert_eq!(measured["changed_pct"], 200.0 / 81.0);
        assert_eq!(measured["mae"], 5.0 / 243.0);
        assert!(measured["ssim"].as_f64().unwrap() < 1.0);
        Ok(())
    }

    #[test]
    fn small_images_skip_unavailable_ssim() -> Result<()> {
        let source = RgbaImage::new(1, 1);
        assert_eq!(metrics(&source, &source)?["ssim"], Value::Null);
        assert!(metrics(&source, &RgbaImage::new(2, 1)).is_err());
        Ok(())
    }
}
