use image::{Rgb, RgbImage};

/// Pillow-compatible Lanczos-3: antialias when shrinking, and round each pass to
/// eight bits using 22-bit filter coefficients. Rounding only after both passes
/// changes model input pixels, even when both resizers use the same Lanczos kernel.
pub(super) fn lanczos_three(source: &RgbImage, width: u32, height: u32) -> RgbImage {
    if source.dimensions() == (width, height) {
        return source.clone();
    }
    let horizontal = if source.width() == width {
        source.clone()
    } else {
        let filters = coefficients(source.width(), width);
        RgbImage::from_fn(width, source.height(), |x, y| {
            let filter = &filters[x as usize];
            let mut sum = [1_i64 << 21; 3];
            for (index, weight) in filter.weights.iter().enumerate() {
                let pixel = source.get_pixel(filter.start + index as u32, y);
                for channel in 0..3 {
                    sum[channel] += i64::from(pixel[channel]) * i64::from(*weight);
                }
            }
            Rgb(sum.map(|value| (value >> 22).clamp(0, 255) as u8))
        })
    };
    if source.height() == height {
        return horizontal;
    }
    let filters = coefficients(source.height(), height);
    RgbImage::from_fn(width, height, |x, y| {
        let filter = &filters[y as usize];
        let mut sum = [1_i64 << 21; 3];
        for (index, weight) in filter.weights.iter().enumerate() {
            let pixel = horizontal.get_pixel(x, filter.start + index as u32);
            for channel in 0..3 {
                sum[channel] += i64::from(pixel[channel]) * i64::from(*weight);
            }
        }
        Rgb(sum.map(|value| (value >> 22).clamp(0, 255) as u8))
    })
}

struct Filter {
    start: u32,
    weights: Vec<i32>,
}

fn coefficients(input: u32, output: u32) -> Vec<Filter> {
    let scale = f64::from(input) / f64::from(output);
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    (0..output)
        .map(|position| {
            let center = (f64::from(position) + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as u32;
            let end = ((center + support + 0.5) as u32).min(input);
            let weights: Vec<f64> = (start..end)
                .map(|index| {
                    let distance = (f64::from(index) - center + 0.5) / filter_scale;
                    if !(-3.0..3.0).contains(&distance) {
                        return 0.0;
                    }
                    if distance == 0.0 {
                        return 1.0;
                    }
                    let x = distance * std::f64::consts::PI;
                    (x.sin() / x) * ((x / 3.0).sin() / (x / 3.0))
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let weights = weights
                .into_iter()
                .map(|weight| {
                    let weight = weight / total * f64::from(1 << 22);
                    if weight < 0.0 {
                        (weight - 0.5) as i32
                    } else {
                        (weight + 0.5) as i32
                    }
                })
                .collect();
            Filter { start, weights }
        })
        .collect()
}
