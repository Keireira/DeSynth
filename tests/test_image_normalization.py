"""Shared image color/orientation handling without model loading or system profiles."""
from __future__ import annotations

import base64
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms

import compare
import diff
import gui
from processing import finish_image, load_rgba, normalize_rgba, srgb_profile_bytes


def linear_rgb_profile():
    """Use sRGB primaries with a standard ICC type-0 gamma=1 transfer curve."""
    profile = bytearray(srgb_profile_bytes())
    for index in range(struct.unpack_from(">I", profile, 128)[0]):
        signature, offset, size = struct.unpack_from(">4sII", profile, 132 + 12 * index)
        if signature in (b"rTRC", b"gTRC", b"bTRC"):
            curve = b"para" + bytes(8) + struct.pack(">i", 65536)
            profile[offset:offset + size] = curve + bytes(size - len(curve))
    return bytes(profile)


class ImageNormalizationTests(unittest.TestCase):
    def tagged_source(self):
        image = Image.new("RGBA", (3, 2))
        image.putdata([(100, 150, 200, 0), (20, 60, 90, 17), (130, 10, 70, 255),
                       (200, 170, 80, 254), (50, 200, 30, 1), (0, 255, 0, 127)])
        image.info["icc_profile"] = linear_rgb_profile()
        return image

    def test_non_srgb_color_is_converted_once_with_exact_alpha(self):
        source = self.tagged_source()
        expected = ImageCms.profileToProfile(source.convert("RGB"),
                                            ImageCms.ImageCmsProfile(io.BytesIO(source.info["icc_profile"])),
                                            ImageCms.ImageCmsProfile(io.BytesIO(srgb_profile_bytes())),
                                            outputMode="RGB")
        actual = normalize_rgba(source)
        self.assertNotEqual(actual.convert("RGB").tobytes(), source.convert("RGB").tobytes())
        self.assertEqual(actual.convert("RGB").tobytes(), expected.tobytes())
        self.assertEqual(actual.getchannel("A").tobytes(), source.getchannel("A").tobytes())
        self.assertEqual(actual.info["icc_profile"], srgb_profile_bytes())
        with patch.object(ImageCms, "profileToProfile", side_effect=AssertionError("already sRGB")):
            self.assertEqual(normalize_rgba(actual).tobytes(), actual.tobytes())

    def test_untagged_rgba_preserves_hidden_color_and_metadata(self):
        source = self.tagged_source()
        source.info.pop("icc_profile")
        source.info["desynth"] = '{"sampling":"test"}'
        with patch.object(ImageCms, "profileToProfile", side_effect=AssertionError("untagged sRGB")):
            actual = normalize_rgba(source)
        self.assertEqual(actual.tobytes(), source.tobytes())
        self.assertEqual(actual.info["desynth"], source.info["desynth"])

    def test_lab_profile_is_converted_before_pillow_rgb_conversion(self):
        source = Image.new("LAB", (2, 2), (128, 150, 180))
        source.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
        actual = normalize_rgba(source)
        self.assertEqual(actual.mode, "RGBA")
        self.assertEqual(actual.getchannel("A").getextrema(), (255, 255))

    def test_rgb_transparency_key_survives_profile_conversion(self):
        source = self.tagged_source().convert("RGB")
        source.info["transparency"] = (100, 150, 200)
        expected_alpha = source.convert("RGBA").getchannel("A").tobytes()
        actual = normalize_rgba(source)
        self.assertEqual(actual.getchannel("A").tobytes(), expected_alpha)
        self.assertEqual(actual.getpixel((0, 0))[3], 0)
        self.assertNotIn("transparency", actual.info)
        self.assertEqual(normalize_rgba(actual).tobytes(), actual.tobytes())

    def test_broken_and_mismatched_profiles_are_rejected(self):
        for profile in (b"broken", b"", ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()):
            with self.subTest(profile=profile[:10]):
                source = Image.new("RGB", (2, 2))
                source.info["icc_profile"] = profile
                with self.assertRaisesRegex(ValueError, "ICC color profile"):
                    normalize_rgba(source)

    def test_cli_diff_compare_and_processing_share_color_and_orientation(self):
        source = self.tagged_source()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            for orientation in (1, 3, 5, 6, 8):
                exif = Image.Exif()
                exif[274] = orientation
                source.save(path, exif=exif)
                actual = load_rgba(path)
                np.testing.assert_array_equal(diff.load_rgba(path), np.asarray(actual))
                np.testing.assert_array_equal(compare.load(path), np.asarray(actual.convert("RGB")))
                self.assertIn(actual.getexif().get(274), (None, 1))

    def test_finished_result_retains_srgb_tag_and_alpha(self):
        source = normalize_rgba(self.tagged_source())
        clean = source.convert("RGB")
        for restore in (False, True):
            with self.subTest(restore=restore):
                result = finish_image(clean, source,
                                      dict(restore=restore, restore_sigma=1.95, restore_mode="gaussian", unsharp=0),
                                      lambda clean, original, **kwargs: clean.copy())
                data = io.BytesIO()
                result.save(data, format="PNG")
                data.seek(0)
                with Image.open(data) as saved:
                    self.assertEqual(saved.info["icc_profile"], srgb_profile_bytes())
                    self.assertEqual(saved.tobytes(), source.tobytes())

    def test_gui_upload_and_registered_original_use_same_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "out").mkdir()
            source = self.tagged_source()
            exif = Image.Exif()
            exif[274] = 6
            path = root / "original.png"
            source.save(path, exif=exif)
            with patch.object(gui, "ROOT", root), patch.object(gui, "STORE", root / "out" / "gui"), \
                    patch.object(gui.Studio, "start_runtime_probe"):
                studio = gui.Studio()
                try:
                    registered = studio.assets[studio.default_source]
                    self.assertEqual((registered["width"], registered["height"]), (2, 3))
                    expected = load_rgba(path)
                    self.assertEqual(studio.image(registered["id"]).tobytes(), expected.tobytes())
                    with Image.open(io.BytesIO(studio.image_bytes(registered["id"]))) as displayed:
                        self.assertEqual(displayed.size, expected.size)
                        self.assertEqual(displayed.tobytes(), expected.tobytes())
                        self.assertEqual(displayed.info["icc_profile"], srgb_profile_bytes())
                    uploaded = studio.upload(dict(data=base64.b64encode(path.read_bytes()).decode(), name="source.png"))
                    self.assertEqual(studio.image(uploaded["id"]).tobytes(), expected.tobytes())
                    with Image.open(io.BytesIO(studio.image_bytes(uploaded["id"]))) as saved:
                        self.assertEqual(saved.info["icc_profile"], srgb_profile_bytes())
                finally:
                    studio.close()

    def test_gui_invalid_profile_reports_actionable_client_error(self):
        source = Image.new("RGB", (2, 2))
        data = io.BytesIO()
        source.save(data, format="PNG", icc_profile=b"broken")
        with self.assertRaisesRegex(gui.APIError, "ICC color profile"):
            gui.Studio.upload(None, dict(data=base64.b64encode(data.getvalue()).decode()))


if __name__ == "__main__":
    unittest.main()
