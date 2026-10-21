use anyhow::{Result, ensure};
use image::{Rgba, RgbaImage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffChannel {
    Rgba,
    Rgb,
    Alpha,
    Raw,
}

/// Opaque previews include alpha differences without hiding RGB beneath unchanged alpha.
/// Raw mode is the literal four-channel delta and can intentionally be transparent.
pub fn difference(
    source: &RgbaImage,
    result: &RgbaImage,
    gain: f64,
    channel: DiffChannel,
) -> Result<RgbaImage> {
    ensure!(
        source.dimensions() == result.dimensions(),
        "Image dimensions differ. Comparison requires images of the same size."
    );
    ensure!(
        gain.is_finite() && gain > 0.0,
        "gain must be a finite positive number"
    );
    let gain = gain.min(255.0);
    let mut output = RgbaImage::new(source.width(), source.height());
    for ((left, right), output) in source
        .pixels()
        .zip(result.pixels())
        .zip(output.pixels_mut())
    {
        let mut delta = [0_u8; 4];
        for channel in 0..4 {
            delta[channel] =
                (f64::from(left[channel].abs_diff(right[channel])) * gain).min(255.0) as u8;
        }
        *output = match channel {
            DiffChannel::Rgba => Rgba([
                delta[0].max(delta[3]),
                delta[1].max(delta[3]),
                delta[2].max(delta[3]),
                255,
            ]),
            DiffChannel::Rgb => Rgba([delta[0], delta[1], delta[2], 255]),
            DiffChannel::Alpha => Rgba([delta[3], delta[3], delta[3], 255]),
            DiffChannel::Raw => Rgba(delta),
        };
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_diff_includes_hidden_colors_and_alpha_with_fractional_gain() -> Result<()> {
        let left = RgbaImage::from_vec(2, 1, vec![10, 20, 30, 0, 1, 2, 3, 4]).unwrap();
        let right = RgbaImage::from_vec(2, 1, vec![14, 10, 29, 0, 1, 2, 3, 20]).unwrap();
        assert_eq!(
            difference(&left, &right, 1.5, DiffChannel::Rgba)?.into_raw(),
            vec![6, 15, 1, 255, 24, 24, 24, 255]
        );
        assert_eq!(
            difference(&left, &right, 1.0, DiffChannel::Raw)?.into_raw(),
            vec![4, 10, 1, 0, 0, 0, 0, 16]
        );
        assert!(difference(&left, &right, f64::NAN, DiffChannel::Rgba).is_err());
        assert!(difference(&left, &RgbaImage::new(1, 1), 1.0, DiffChannel::Rgba).is_err());
        Ok(())
    }
}
