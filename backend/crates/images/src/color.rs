use anyhow::{Context, Result, ensure};
use image::RgbaImage;
use lcms2::{Intent, PixelFormat, Profile, ThreadContext, Transform};

/// Keep each operation's profiles and transform in a private LCMS context.
/// LittleCMS is statically linked and matches the previous Pillow color engine.
pub(super) fn transform(
    pixels: &[u8],
    dimensions: (u32, u32),
    context: &ThreadContext,
    profile: &Profile<ThreadContext>,
    format: PixelFormat,
    alpha: Option<usize>,
) -> Result<RgbaImage> {
    let (width, height) = dimensions;
    let count = width as usize * height as usize;
    ensure!(
        count > 0 && pixels.len().is_multiple_of(count),
        "Image color samples are incomplete"
    );
    let destination = Profile::new_srgb_context(context);
    let transform: Transform<u8, u8, ThreadContext> = Transform::new_context(
        context,
        profile,
        format,
        &destination,
        PixelFormat::RGBA_8,
        Intent::Perceptual,
    )
    .context("Cannot convert embedded ICC color profile to sRGB")?;
    let mut result = RgbaImage::new(width, height);
    transform.transform_pixels(pixels, result.as_mut());
    let samples = pixels.len() / count;
    for (pixel, source) in result.pixels_mut().zip(pixels.chunks_exact(samples)) {
        pixel[3] = alpha.map_or(255, |index| source[index]);
    }
    Ok(result)
}
