//! Image decoding, color normalization and output assembly. No model runtime is needed here.

mod color;
mod diff;
mod formats;
#[cfg(test)]
mod golden;
mod metrics;
mod resize;
mod restore;
mod visual;

pub use diff::{DiffChannel, difference};
pub use metrics::{extended_metrics, metrics};
pub use restore::{RestorationParams, finish, resize_for_comparison};
pub use visual::comparison_visual;

use std::io::Cursor;

use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbaImage};
use lcms2::{ColorSpaceSignature, PixelFormat, Profile, ThreadContext};
use serde_json::Value;

pub const MAX_PIXELS: u64 = 36_000_000;

/// Decode before exposing an asset, applying EXIF orientation and its embedded ICC profile once.
/// Pixel alpha and RGB values hidden beneath zero alpha are never premultiplied.
pub fn open(bytes: &[u8]) -> Result<RgbaImage> {
    if let Some(image) = formats::decode_special_color(bytes)? {
        return Ok(image);
    }
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("Cannot identify image format")?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(512 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().context("Cannot read image")?;
    let (width, height) = decoder.dimensions();
    ensure!(width > 0 && height > 0, "Image dimensions must be positive");
    ensure!(
        u64::from(width) * u64::from(height) <= MAX_PIXELS,
        "Image exceeds 36 million pixels"
    );
    let orientation = decoder
        .orientation()
        .context("Cannot read image orientation")?;
    let profile = decoder
        .icc_profile()
        .context("Cannot read embedded ICC color profile")?;
    let mut decoded = DynamicImage::from_decoder(decoder).context("Cannot decode image")?;
    decoded.apply_orientation(orientation);
    let rgba = decoded.to_rgba8();
    let Some(profile) = profile else {
        return Ok(rgba);
    };
    normalize_color(&decoded, rgba, &profile)
}

fn normalize_color(decoded: &DynamicImage, mut rgba: RgbaImage, icc: &[u8]) -> Result<RgbaImage> {
    let context = ThreadContext::new();
    let source = Profile::new_icc_context(&context, icc)
        .context("Cannot convert embedded ICC color profile to sRGB")?;
    let (format, input) = match source.color_space() {
        ColorSpaceSignature::RgbData => (PixelFormat::RGB_8, decoded.to_rgb8().into_raw()),
        ColorSpaceSignature::GrayData
            if matches!(
                decoded.color(),
                image::ColorType::L8
                    | image::ColorType::La8
                    | image::ColorType::L16
                    | image::ColorType::La16
            ) =>
        {
            (PixelFormat::GRAY_8, decoded.to_luma8().into_raw())
        }
        _ => bail!(
            "Cannot convert embedded ICC color profile to sRGB: profile and decoded image color space do not match"
        ),
    };
    let alpha: Vec<u8> = rgba.pixels().map(|pixel| pixel[3]).collect();
    rgba = color::transform(&input, rgba.dimensions(), &context, &source, format, None)?;
    for (pixel, alpha) in rgba.pixels_mut().zip(alpha) {
        pixel[3] = alpha;
    }
    Ok(rgba)
}

/// All API and saved images use the same normalized, lossless RGBA representation.
pub fn encode_png(image: &RgbaImage, metadata: Option<&Value>) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut output, image.width(), image.height());
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        if let Some(metadata) = metadata {
            encoder.add_itxt_chunk("desynth".into(), serde_json::to_string(metadata)?)?;
        }
        let mut writer = encoder.write_header()?;
        writer.write_image_data(image.as_raw())?;
    }
    Ok(output)
}

/// Recover generation provenance when registering an existing PNG result.
pub fn png_metadata(bytes: &[u8]) -> Option<Value> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let reader = decoder.read_info().ok()?;
    let info = reader.info();
    for entry in &info.uncompressed_latin1_text {
        if entry.keyword == "desynth" {
            return serde_json::from_str(&entry.text).ok();
        }
    }
    for entry in &info.utf8_text {
        if entry.keyword == "desynth" {
            return serde_json::from_str(&entry.get_text().ok()?).ok();
        }
    }
    for entry in &info.compressed_latin1_text {
        if entry.keyword == "desynth" {
            return serde_json::from_str(&entry.get_text().ok()?).ok();
        }
    }
    None
}

/// Give Diffusers explicit multiples of 16 while preserving the previous preview limit.
pub fn prepare(source: &RgbaImage, preview: bool) -> RgbaImage {
    let ratio = if preview {
        (768.0 / f64::from(source.width().max(source.height()))).min(1.0)
    } else {
        1.0
    };
    let dimension =
        |value: u32| (((f64::from(value) * ratio / 16.0 + 1e-9).floor() as u32) * 16).max(16);
    let (width, height) = (dimension(source.width()), dimension(source.height()));
    // Converting to RGB before resize prevents alpha from suppressing hidden source colors.
    let rgb = DynamicImage::ImageRgba8(source.clone()).to_rgb8();
    let resized = resize::lanczos_three(&rgb, width, height);
    DynamicImage::ImageRgb8(resized).into_rgba8()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn png_round_trip_keeps_hidden_colors_and_alpha() -> Result<()> {
        let source =
            RgbaImage::from_vec(3, 1, vec![91, 12, 244, 0, 6, 72, 8, 17, 255, 25, 10, 255])
                .unwrap();
        let metadata = serde_json::json!({"seed":42,"source_name":"Исходник 🎨.png"});
        let encoded = encode_png(&source, Some(&metadata))?;
        assert_eq!(png_metadata(&encoded), Some(metadata.clone()));
        assert_eq!(open(&encoded)?, source);
        let decoder = png::Decoder::new(Cursor::new(encoded)).read_info()?;
        let chunk = &decoder.info().utf8_text[0];
        assert_eq!(chunk.keyword, "desynth");
        assert_eq!(chunk.get_text()?, serde_json::to_string(&metadata)?);
        Ok(())
    }

    #[test]
    fn preparation_uses_hidden_colors_and_multiple_of_sixteen() {
        let source = RgbaImage::from_pixel(31, 27, Rgba([89, 134, 244, 0]));
        let prepared = prepare(&source, false);
        assert_eq!(prepared.dimensions(), (16, 16));
        assert!(
            prepared
                .pixels()
                .all(|pixel| *pixel == Rgba([89, 134, 244, 255]))
        );
        assert_eq!(
            prepare(&RgbaImage::new(1536, 1024), true).dimensions(),
            (768, 512)
        );
    }

    #[test]
    fn invalid_profile_is_rejected_instead_of_silently_recolored() -> Result<()> {
        let mut info = png::Info::with_size(1, 1);
        info.color_type = png::ColorType::Rgba;
        info.icc_profile = Some(std::borrow::Cow::Borrowed(b"broken"));
        let mut bytes = Vec::new();
        {
            let mut writer = png::Encoder::with_info(&mut bytes, info)?.write_header()?;
            writer.write_image_data(&[1, 2, 3, 4])?;
        }
        assert!(
            open(&bytes)
                .unwrap_err()
                .to_string()
                .contains("ICC color profile")
        );
        Ok(())
    }
}
