use anyhow::{Result, ensure};
use image::{Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RestorationParams {
    pub restore: bool,
    pub restore_sigma: f64,
    pub restore_mode: String,
    pub unsharp: f64,
}

impl Default for RestorationParams {
    fn default() -> Self {
        Self {
            restore: true,
            restore_sigma: 1.95,
            restore_mode: "gaussian".into(),
            unsharp: 0.0,
        }
    }
}

impl RestorationParams {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.restore_sigma.is_finite() && (0.1..=10.0).contains(&self.restore_sigma),
            "restore_sigma must be a finite number between 0.1 and 10"
        );
        ensure!(
            self.unsharp.is_finite() && (0.0..=3.0).contains(&self.unsharp),
            "unsharp must be a finite number between 0 and 3"
        );
        ensure!(
            matches!(self.restore_mode.as_str(), "gaussian" | "edge"),
            "restore_mode must be gaussian or edge"
        );
        Ok(())
    }
}

pub fn finish(
    clean: &RgbaImage,
    source: &RgbaImage,
    params: &RestorationParams,
) -> Result<RgbaImage> {
    params.validate()?;
    ensure!(
        source.width() > 0 && source.height() > 0 && clean.width() > 0 && clean.height() > 0,
        "Image dimensions must be positive"
    );
    let (width, height) = (source.width() as usize, source.height() as usize);
    let mut output = RgbaImage::new(source.width(), source.height());
    if !params.restore {
        let clean_rgb = image::DynamicImage::ImageRgba8(clean.clone()).into_rgb8();
        let clean = super::resize::lanczos_three(&clean_rgb, source.width(), source.height());
        for ((clean, original), output) in
            clean.pixels().zip(source.pixels()).zip(output.pixels_mut())
        {
            *output = Rgba([clean[0], clean[1], clean[2], original[3]]);
        }
        return Ok(output);
    }
    let clean = resize_clean(clean, source.width(), source.height());
    let edge_mask = if params.restore_mode == "edge" {
        Some(contour_mask(source))
    } else {
        None
    };
    let sigma_edge = (params.restore_sigma * 2.0)
        .min(params.restore_sigma + 2.0)
        .min(10.0)
        .max(params.restore_sigma);
    for channel in 0..3 {
        let original: Vec<f32> = source
            .pixels()
            .map(|pixel| f32::from(pixel[channel]))
            .collect();
        let clean: Vec<f32> = clean
            .pixels()
            .map(|pixel| f32::from(pixel[channel]))
            .collect();
        let mut combined = restore_plane(&clean, &original, width, height, params.restore_sigma);
        if let Some(mask) = &edge_mask {
            let sharp = restore_plane(&clean, &original, width, height, sigma_edge);
            for ((value, sharp), mask) in combined.iter_mut().zip(sharp).zip(mask) {
                *value = (mask * sharp + (1.0 - mask) * *value).clamp(0.0, 255.0);
            }
        }
        if params.unsharp > 0.0 {
            let blurred = gaussian_plane(&combined, width, height, 1.0);
            for (value, blurred) in combined.iter_mut().zip(blurred) {
                *value = (*value + params.unsharp as f32 * (*value - blurred)).clamp(0.0, 255.0);
            }
        }
        for (pixel, value) in output.pixels_mut().zip(combined) {
            pixel[channel] = value as u8;
        }
    }
    for (pixel, source) in output.pixels_mut().zip(source.pixels()) {
        pixel[3] = source[3];
    }
    Ok(output)
}

fn restore_plane(
    clean: &[f32],
    original: &[f32],
    width: usize,
    height: usize,
    sigma: f64,
) -> Vec<f32> {
    let low_clean = gaussian_plane(clean, width, height, sigma);
    let low_original = gaussian_plane(original, width, height, sigma);
    low_clean
        .into_iter()
        .zip(original)
        .zip(low_original)
        .map(|((low, original), original_low)| (low + (*original - original_low)).clamp(0.0, 255.0))
        .collect()
}

fn contour_mask(source: &RgbaImage) -> Vec<f32> {
    let (width, height) = (source.width() as usize, source.height() as usize);
    // OpenCV's 8-bit RGB to gray conversion rounds before Sobel, rather than using float luma.
    let gray: Vec<f32> = source
        .pixels()
        .map(|pixel| {
            ((u32::from(pixel[0]) * 9798
                + u32::from(pixel[1]) * 19235
                + u32::from(pixel[2]) * 3735
                + 16384)
                >> 15) as f32
        })
        .collect();
    let mut magnitude = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let left = reflect(x as isize - 1, width);
            let right = reflect(x as isize + 1, width);
            let above = reflect(y as isize - 1, height);
            let below = reflect(y as isize + 1, height);
            let gx = gray[above * width + right] - gray[above * width + left]
                + 2.0 * (gray[y * width + right] - gray[y * width + left])
                + gray[below * width + right]
                - gray[below * width + left];
            let gy = gray[below * width + left]
                + 2.0 * gray[below * width + x]
                + gray[below * width + right]
                - gray[above * width + left]
                - 2.0 * gray[above * width + x]
                - gray[above * width + right];
            magnitude[y * width + x] = (gx * gx + gy * gy).sqrt();
        }
    }
    let mut mask = gaussian_plane(&magnitude, width, height, 2.0);
    let rank = (mask.len() - 1) as f64 * 0.99;
    let index = rank.floor() as usize;
    let mut sorted = mask.clone();
    sorted.select_nth_unstable_by(index, f32::total_cmp);
    let lower = sorted[index];
    let upper = if index + 1 < sorted.len() {
        sorted[index + 1..]
            .iter()
            .copied()
            .min_by(f32::total_cmp)
            .unwrap_or(lower)
    } else {
        lower
    };
    let percentile = f64::from(lower) + (f64::from(upper) - f64::from(lower)) * rank.fract();
    let divisor = (percentile + 1e-6) as f32;
    for value in &mut mask {
        *value = (*value / divisor).clamp(0.0, 1.0);
    }
    mask
}

/// Gaussian kernel size and reflect-101 borders match cv2.GaussianBlur on float32 images.
pub(super) fn gaussian_plane(source: &[f32], width: usize, height: usize, sigma: f64) -> Vec<f32> {
    let size = ((sigma * 8.0 + 1.0).round_ties_even() as usize) | 1;
    let radius = (size / 2) as isize;
    let mut kernel: Vec<f64> = (-radius..=radius)
        .map(|offset| (-(offset * offset) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f64 = kernel.iter().sum();
    for value in &mut kernel {
        *value /= sum;
    }
    let kernel: Vec<f32> = kernel.into_iter().map(|value| value as f32).collect();
    let mut horizontal = vec![0.0; source.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0_f32;
            for (index, weight) in kernel.iter().enumerate() {
                value += source[y * width + reflect(x as isize + index as isize - radius, width)]
                    * weight;
            }
            horizontal[y * width + x] = value;
        }
    }
    let mut output = vec![0.0; source.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0_f32;
            for (index, weight) in kernel.iter().enumerate() {
                value += horizontal
                    [reflect(y as isize + index as isize - radius, height) * width + x]
                    * weight;
            }
            output[y * width + x] = value;
        }
    }
    output
}

fn reflect(mut index: isize, length: usize) -> usize {
    if length == 1 {
        return 0;
    }
    while index < 0 || index >= length as isize {
        index = if index < 0 {
            -index
        } else {
            2 * length as isize - index - 2
        };
    }
    index as usize
}

/// Lanczos-4 sampling is used for restoring pipeline output to the source dimensions.
/// Alpha is deliberately ignored; the source alpha is copied after restoration.
fn resize_clean(clean: &RgbaImage, width: u32, height: u32) -> RgbaImage {
    if clean.dimensions() == (width, height) {
        return clean.clone();
    }
    let coefficients = |input: u32, output: u32| -> Vec<Vec<(u32, f32)>> {
        (0..output)
            .map(|position| {
                let center =
                    (f64::from(position) + 0.5) * f64::from(input) / f64::from(output) - 0.5;
                let first = center.floor() as i64 - 3;
                let mut weights = Vec::with_capacity(8);
                let mut sum = 0.0;
                for index in first..first + 8 {
                    let distance = center - index as f64;
                    let weight = if distance.abs() < 1e-9 {
                        1.0
                    } else if distance.abs() >= 4.0 {
                        0.0
                    } else {
                        let x = std::f64::consts::PI * distance;
                        x.sin() * (x / 4.0).sin() * 4.0 / (x * x)
                    };
                    weights.push((index.clamp(0, i64::from(input) - 1) as u32, weight as f32));
                    sum += weight;
                }
                for (_, weight) in &mut weights {
                    *weight /= sum as f32;
                }
                weights
            })
            .collect()
    };
    let x_weights = coefficients(clean.width(), width);
    let y_weights = coefficients(clean.height(), height);
    let mut horizontal = vec![[0.0_f32; 3]; width as usize * clean.height() as usize];
    for y in 0..clean.height() {
        for (x, weights) in x_weights.iter().enumerate() {
            let output = &mut horizontal[y as usize * width as usize + x];
            for (source_x, weight) in weights {
                let pixel = clean.get_pixel(*source_x, y);
                for channel in 0..3 {
                    output[channel] += f32::from(pixel[channel]) * weight;
                }
            }
        }
    }
    let mut result = RgbaImage::new(width, height);
    for (y, weights) in y_weights.iter().enumerate() {
        for x in 0..width as usize {
            let mut color = [0.0_f32; 3];
            for (source_y, weight) in weights {
                let pixel = horizontal[*source_y as usize * width as usize + x];
                for channel in 0..3 {
                    color[channel] += pixel[channel] * weight;
                }
            }
            result.put_pixel(
                x as u32,
                y as u32,
                Rgba([
                    color[0].round_ties_even().clamp(0.0, 255.0) as u8,
                    color[1].round_ties_even().clamp(0.0, 255.0) as u8,
                    color[2].round_ties_even().clamp(0.0, 255.0) as u8,
                    255,
                ]),
            );
        }
    }
    result
}

pub fn resize_for_comparison(image: &RgbaImage, width: u32, height: u32) -> RgbaImage {
    resize_clean(image, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_methods_preserve_every_source_alpha_when_pipeline_size_differs() -> Result<()> {
        let source = RgbaImage::from_fn(37, 23, |x, y| {
            Rgba([(x * 7) as u8, (y * 11) as u8, 128, (x + y * 37) as u8])
        });
        let clean = RgbaImage::from_pixel(32, 16, Rgba([70, 80, 90, 255]));
        for method in ["gaussian", "edge"] {
            let result = finish(
                &clean,
                &source,
                &RestorationParams {
                    restore_mode: method.into(),
                    ..Default::default()
                },
            )?;
            assert_eq!(result.dimensions(), source.dimensions());
            assert!(
                result
                    .pixels()
                    .zip(source.pixels())
                    .all(|(a, b)| a[3] == b[3])
            );
        }
        Ok(())
    }

    #[test]
    fn disabled_restoration_preserves_clean_color_and_source_alpha() -> Result<()> {
        let source = RgbaImage::from_pixel(1, 1, Rgba([10, 20, 30, 0]));
        let clean = RgbaImage::from_pixel(1, 1, Rgba([120, 70, 90, 255]));
        let result = finish(
            &clean,
            &source,
            &RestorationParams {
                restore: false,
                ..Default::default()
            },
        )?;
        assert_eq!(result.get_pixel(0, 0), &Rgba([120, 70, 90, 0]));
        assert!(
            finish(
                &clean,
                &source,
                &RestorationParams {
                    restore_sigma: f64::NAN,
                    ..Default::default()
                }
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn edge_restores_a_broader_band_of_source_contours() -> Result<()> {
        let source = RgbaImage::from_fn(192, 128, |x, _| {
            let value = if (x / 24) % 2 == 0 { 40 } else { 195 };
            Rgba([value, value, value, 255])
        });
        let plane: Vec<f32> = source.pixels().map(|pixel| f32::from(pixel[0])).collect();
        let blurred = gaussian_plane(&plane, 192, 128, 3.0);
        let clean = RgbaImage::from_fn(192, 128, |x, y| {
            let value = blurred[(y * 192 + x) as usize].round() as u8;
            Rgba([value, value, value, 255])
        });
        let gaussian = finish(&clean, &source, &RestorationParams::default())?;
        let edge = finish(
            &clean,
            &source,
            &RestorationParams {
                restore_mode: "edge".into(),
                ..Default::default()
            },
        )?;
        let error = |image: &RgbaImage| {
            image
                .pixels()
                .zip(source.pixels())
                .map(|(a, b)| f64::from(a[0].abs_diff(b[0])))
                .sum::<f64>()
        };
        assert!(error(&edge) < error(&gaussian) * 0.8);
        Ok(())
    }
}
