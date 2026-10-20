"""All entry points must discover the same preferred local weights."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import model_catalog as catalog
from scripts import project


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="desynth-catalog-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "ai").mkdir()

    def asset(self, name, size=1, legacy=False):
        path = (self.root if legacy else self.root / "ai") / name
        path.write_bytes(b"x" * size)
        return path

    def test_all_entry_points_choose_the_same_model_and_adapter(self):
        import desynth
        import gui
        self.asset("Qwen-Image-2512-Q2_K_M.gguf", 7)
        self.asset("Qwen-Image-2512-Q4_0.gguf", 12)
        preferred = self.asset("Qwen-Image-2512-Q4_K_M.gguf", 13)
        self.asset("Qwen-Image-2512-Q8_0.gguf", 21)
        self.asset("Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors")
        adapter = self.asset("Qwen-Image-2512-Lightning-8steps-V1.1-fp32.safetensors")
        with (patch.object(desynth, "ROOT", self.root),
              patch.object(desynth, "AI_DIR", self.root / "ai"),
              patch.object(project, "ROOT", self.root),
              patch.object(project, "MODEL_DIR", self.root / "ai")):
            models, adapters = gui.model_inventory(self.root)
            self.assertEqual(desynth.discover_gguf_models()[0], preferred)
            self.assertEqual(project.selected_gguf("auto"), preferred)
            self.assertEqual(models[0]["name"], preferred.name)
            self.assertEqual(desynth.discover_lightning_loras()[0], adapter)
            self.assertEqual(project.selected_lightning("auto"), adapter)
            self.assertEqual(adapters[0]["name"], adapter.name)

    def test_empty_duplicate_does_not_hide_valid_legacy_file(self):
        legacy = self.asset("Qwen-Image-2512-Q4_K_M.gguf", legacy=True)
        self.asset(legacy.name, 0)
        self.assertEqual(catalog.discover_models(self.root), [legacy])
        self.assertEqual(catalog.resolve_model_path(legacy.name, self.root), legacy)
        current = self.asset(legacy.name)
        self.assertEqual(catalog.discover_models(self.root), [current])

    def test_fallback_only_includes_existing_smaller_models(self):
        q4 = self.asset("Qwen-Image-2512-Q4_K_M.gguf", 13)
        q2 = self.asset("Qwen-Image-2512-Q2_K_M.gguf", 7)
        self.asset("Qwen-Image-2512-Q8_0.gguf", 21)
        models = [{"path": p.relative_to(self.root).as_posix()}
                  for p in catalog.discover_models(self.root)]
        models.append({"path": "ai/removed.gguf"})
        self.assertEqual(catalog.automatic_model_paths(models, self.root),
                         [p.relative_to(self.root).as_posix() for p in (q4, q2)])

    def test_case_insensitive_extensions_and_empty_embeddings(self):
        model = self.asset("Qwen-Image-2512-Q4_K_M.GGUF")
        self.assertEqual(catalog.discover_models(self.root), [model])
        legacy = self.asset("embeds_cache.pt", legacy=True)
        self.asset("embeds_cache.pt", 0)
        self.assertEqual(catalog.embeddings_path(self.root), legacy)


if __name__ == "__main__":
    unittest.main()
