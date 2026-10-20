"""Retain the generation settings when a GUI result is restored and exported."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import gui


class DetailMetadataTests(unittest.TestCase):
    def test_png_metadata_survives_live_restoration(self):
        with tempfile.TemporaryDirectory(prefix="desynth-detail-metadata-") as directory:
            root = Path(directory)
            with (patch.object(gui, "ROOT", root), patch.object(gui, "STORE", root / "out" / "gui"),
                  patch.object(gui.Studio, "start_runtime_probe")):
                studio = gui.Studio()
                try:
                    source = studio.save(Image.new("RGBA", (16, 16), (50, 60, 70, 17)), "source", "source.png")
                    metadata = dict(sampling="linear-sigma-v1", starting_sigma=.25,
                                    params=dict(denoise=.25, steps=32, seed=42))
                    clean = studio.save(Image.new("RGBA", (16, 16), (51, 61, 71, 255)), "clean", "clean.png",
                                        temporary=True, metadata=metadata)
                    studio.clean_sources[clean["id"]] = source["id"]
                    response = studio.quick_restore(dict(source_id=source["id"], clean_id=clean["id"],
                                                         params=dict(restore=False, restore_sigma=2.5)))
                    result = studio.image(response["result"]["id"])
                    recorded = json.loads(result.info["desynth"])
                    self.assertEqual(recorded["starting_sigma"], .25)
                    self.assertEqual(recorded["sampling"], "linear-sigma-v1")
                    self.assertEqual(recorded["params"]["seed"], 42)
                    self.assertEqual(recorded["params"]["steps"], 32)
                    self.assertEqual(recorded["params"]["restore_sigma"], 2.5)
                    self.assertFalse(recorded["params"]["restore"])
                    self.assertEqual(result.getchannel("A").getextrema(), (17, 17))
                finally:
                    studio.close()


if __name__ == "__main__":
    unittest.main()
