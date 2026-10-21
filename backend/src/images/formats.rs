//! Decode profile-bearing Lab/CMYK samples before a format decoder converts them to RGB.

use std::io::Cursor;

use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, RgbaImage, metadata::Orientation};
use lcms2::{ColorSpaceSignature, PixelFormat, Profile, ThreadContext};
use tiff::{
    decoder::{Decoder, DecodingResult},
    tags::Tag,
};
use zune_core::{bytestream::ZCursor, colorspace::ColorSpace};

use super::MAX_PIXELS;

pub(super) fn decode_special_color(bytes: &[u8]) -> Result<Option<RgbaImage>> {
    if bytes.starts_with(b"II*\0")
        || bytes.starts_with(b"MM\0*")
        || bytes.starts_with(b"II+\0")
        || bytes.starts_with(b"MM\0+")
    {
        return tiff_color(bytes);
    }
    if bytes.starts_with(&[0xff, 0xd8]) {
        return jpeg_color(bytes);
    }
    Ok(None)
}

fn tiff_color(bytes: &[u8]) -> Result<Option<RgbaImage>> {
    let mut decoder = Decoder::new(Cursor::new(bytes)).context("Cannot read TIFF image")?;
    let interpretation = decoder
        .get_tag_u32(Tag::PhotometricInterpretation)
        .context("TIFF color interpretation is missing")?;
    if ![5, 8, 9].contains(&interpretation) {
        return Ok(None);
    }
    let profile = decoder
        .find_tag(Tag::IccProfile)?
        .map(|value| value.into_u8_vec())
        .transpose()?;
    let Some(profile) = profile else {
        ensure!(
            interpretation == 5,
            "Cannot convert Lab TIFF to sRGB without its embedded ICC color profile"
        );
        return Ok(None);
    };
    let context = ThreadContext::new();
    let profile = Profile::new_icc_context(&context, &profile)
        .context("Cannot convert embedded ICC color profile to sRGB")?;
    ensure!(
        (interpretation == 5 && profile.color_space() == ColorSpaceSignature::CmykData)
            || (interpretation != 5 && profile.color_space() == ColorSpaceSignature::LabData),
        "Cannot convert embedded ICC color profile to sRGB: TIFF samples and ICC color space do not match"
    );
    let (width, height) = decoder.dimensions()?;
    check_size(width, height)?;
    let samples = decoder.get_tag_u32(Tag::SamplesPerPixel)? as usize;
    let color_samples = if interpretation == 5 { 4 } else { 3 };
    ensure!(
        (color_samples..=color_samples + 1).contains(&samples),
        "Unsupported Lab/CMYK TIFF channel count: {samples}"
    );
    let extra_samples = decoder
        .find_tag(Tag::ExtraSamples)?
        .map(|value| value.into_u16_vec())
        .transpose()?
        .unwrap_or_default();
    ensure!(
        extra_samples.len() <= samples - color_samples,
        "TIFF extra samples do not match its color channels"
    );
    let alpha_kind = extra_samples.first().copied().unwrap_or(0);
    let alpha = matches!(alpha_kind, 1 | 2).then_some(color_samples);
    let orientation = decoder.get_tag_u32(Tag::Orientation).unwrap_or(1) as u8;
    let planar = decoder.get_tag_u32(Tag::PlanarConfiguration).unwrap_or(1) == 2;
    let patched;
    let input = if interpretation == 8 || interpretation == 9 {
        // The TIFF decoder cannot expose Lab buffers. Treat their channels as raw
        // RGB bytes in an in-memory directory copy, then apply the original Lab ICC.
        patched = unsigned_lab_tiff(bytes)?;
        patched.as_slice()
    } else {
        bytes
    };
    let mut limits = tiff::decoder::Limits::default();
    limits.decoding_buffer_size = 512 * 1024 * 1024;
    let mut decoder = Decoder::new(Cursor::new(input))?.with_limits(limits);
    let mut decoded = DecodingResult::U8(Vec::new());
    let layout = decoder
        .read_image_to_buffer(&mut decoded)
        .context("Cannot decode TIFF color samples")?;
    let mut pixels = match decoded {
        DecodingResult::U8(pixels) => pixels,
        DecodingResult::U16(pixels) => pixels.into_iter().map(|value| (value >> 8) as u8).collect(),
        _ => bail!("Only 8-bit and 16-bit Lab/CMYK TIFF samples are supported"),
    };
    let count = width as usize * height as usize;
    let decoded_samples = pixels.len() / count;
    ensure!(
        pixels.len().is_multiple_of(count)
            && (decoded_samples == samples
                || (alpha.is_none() && decoded_samples == color_samples))
            && layout.complete_len <= 512 * 1024 * 1024,
        "TIFF color samples are incomplete"
    );
    // The TIFF decoder may omit an explicitly unspecified extra sample.
    let samples = decoded_samples;
    if planar {
        let planes = pixels;
        pixels = vec![0; count * samples];
        for pixel in 0..count {
            for channel in 0..samples {
                pixels[pixel * samples + channel] = planes[channel * count + pixel];
            }
        }
    }
    if alpha_kind == 1 {
        unassociate(&mut pixels, samples, color_samples, interpretation == 8);
    }
    if interpretation == 8 {
        // CIE Lab stores signed a/b. ICC Lab uses their unsigned midpoint at 128.
        for pixel in pixels.chunks_exact_mut(samples) {
            pixel[1] ^= 128;
            pixel[2] ^= 128;
        }
    }
    let layout = match (interpretation == 5, samples) {
        (true, 4) => PixelFormat::CMYK_8,
        (true, 5) => PixelFormat::CMYKA_8,
        (false, 3) => PixelFormat::Lab_8,
        (false, 4) => PixelFormat(PixelFormat::Lab_8.0 | (1 << 7)),
        _ => bail!("Unsupported Lab/CMYK TIFF channel count: {samples}"),
    };
    let result =
        super::color::transform(&pixels, (width, height), &context, &profile, layout, alpha)?;
    Ok(Some(orient(result, orientation)))
}

fn unassociate(pixels: &mut [u8], samples: usize, color_samples: usize, signed_lab: bool) {
    for pixel in pixels.chunks_exact_mut(samples) {
        let alpha = i32::from(pixel[color_samples]);
        if alpha == 0 {
            continue;
        }
        for (channel, value) in pixel[..color_samples].iter_mut().enumerate() {
            if signed_lab && matches!(channel, 1 | 2) {
                let straight = i32::from(*value as i8) * 255 / alpha;
                *value = straight.clamp(-128, 127) as i8 as u8;
            } else {
                *value = (i32::from(*value) * 255 / alpha).clamp(0, 255) as u8;
            }
        }
    }
}

fn jpeg_color(bytes: &[u8]) -> Result<Option<RgbaImage>> {
    let mut decoder = zune_jpeg::JpegDecoder::new(ZCursor::new(bytes));
    decoder
        .decode_headers()
        .map_err(|error| anyhow::anyhow!("Cannot read JPEG image: {error}"))?;
    let Some(icc) = decoder.icc_profile() else {
        return Ok(None);
    };
    let context = ThreadContext::new();
    let profile = Profile::new_icc_context(&context, &icc)
        .context("Cannot convert embedded ICC color profile to sRGB")?;
    if profile.color_space() != ColorSpaceSignature::CmykData {
        return Ok(None);
    }
    let source_space = decoder
        .input_colorspace()
        .context("JPEG color space is missing")?;
    ensure!(
        matches!(source_space, ColorSpace::CMYK | ColorSpace::YCCK),
        "Cannot convert embedded ICC color profile to sRGB: JPEG samples and ICC color space do not match"
    );
    let info = decoder.info().context("JPEG dimensions are missing")?;
    let (width, height) = (u32::from(info.width), u32::from(info.height));
    check_size(width, height)?;
    let orientation = decoder
        .exif()
        .and_then(|bytes| Orientation::from_exif_chunk(bytes).map(|value| value.to_exif()))
        .unwrap_or(1);
    decoder.set_options(decoder.options().jpeg_set_out_colorspace(source_space));
    let mut pixels = decoder
        .decode()
        .map_err(|error| anyhow::anyhow!("Cannot decode JPEG color samples: {error}"))?;
    ensure!(
        pixels.len() == width as usize * height as usize * 4,
        "JPEG color samples are incomplete"
    );
    for pixel in pixels.chunks_exact_mut(4) {
        if source_space == ColorSpace::YCCK {
            let y = f64::from(pixel[0]);
            let cb = f64::from(pixel[1]) - 128.0;
            let cr = f64::from(pixel[2]) - 128.0;
            // YCCK encodes the inverted first three ink channels as YCbCr.
            pixel[0] = (y + 1.402 * cr).round().clamp(0.0, 255.0) as u8;
            pixel[1] = (y - 0.344136 * cb - 0.714136 * cr)
                .round()
                .clamp(0.0, 255.0) as u8;
            pixel[2] = (y + 1.772 * cb).round().clamp(0.0, 255.0) as u8;
            pixel[3] = 255 - pixel[3];
        } else {
            for value in pixel {
                *value = 255 - *value;
            }
        }
    }
    let result = super::color::transform(
        &pixels,
        (width, height),
        &context,
        &profile,
        PixelFormat::CMYK_8,
        None,
    )?;
    Ok(Some(orient(result, orientation)))
}

fn check_size(width: u32, height: u32) -> Result<()> {
    ensure!(
        width > 0 && height > 0 && u64::from(width) * u64::from(height) <= MAX_PIXELS,
        "Image exceeds 36 million pixels or has empty dimensions"
    );
    Ok(())
}

fn orient(image: RgbaImage, value: u8) -> RgbaImage {
    let mut image = DynamicImage::ImageRgba8(image);
    image.apply_orientation(Orientation::from_exif(value).unwrap_or(Orientation::NoTransforms));
    image.into_rgba8()
}

/// Adjust the first image's SHORT photometric tag in an in-memory copy only.
/// Compressed strips, ICC data, EXIF and all offsets retain their original bytes.
fn unsigned_lab_tiff(bytes: &[u8]) -> Result<Vec<u8>> {
    let little = bytes.starts_with(b"II");
    let read = |offset: usize, length: usize| -> Result<u64> {
        let data = bytes
            .get(offset..offset.checked_add(length).context("TIFF offset overflow")?)
            .context("TIFF directory is truncated")?;
        let mut value = 0_u64;
        for (index, byte) in data.iter().enumerate() {
            let shift = if little {
                index * 8
            } else {
                (length - index - 1) * 8
            };
            value |= u64::from(*byte) << shift;
        }
        Ok(value)
    };
    let big = read(2, 2)? == 43;
    let directory = read(if big { 8 } else { 4 }, if big { 8 } else { 4 })?;
    let directory = usize::try_from(directory).context("TIFF directory offset is too large")?;
    let count_size = if big { 8 } else { 2 };
    let entry_size = if big { 20 } else { 12 };
    let entries =
        usize::try_from(read(directory, count_size)?).context("TIFF directory is too large")?;
    ensure!(
        entries <= bytes.len() / entry_size,
        "TIFF directory is too large"
    );
    for index in 0..entries {
        let entry = directory
            .checked_add(count_size)
            .and_then(|offset| offset.checked_add(index * entry_size))
            .context("TIFF directory overflow")?;
        if read(entry, 2)? != 262 {
            continue;
        }
        ensure!(
            read(entry + 2, 2)? == 3 && read(entry + 4, if big { 8 } else { 4 })? == 1,
            "Invalid TIFF photometric tag"
        );
        let offset = entry + if big { 12 } else { 8 };
        let mut result = bytes.to_vec();
        let replacement = if little {
            2_u16.to_le_bytes()
        } else {
            2_u16.to_be_bytes()
        };
        result
            .get_mut(offset..offset + 2)
            .context("TIFF photometric tag is truncated")?
            .copy_from_slice(&replacement);
        return Ok(result);
    }
    bail!("TIFF photometric tag is missing")
}
