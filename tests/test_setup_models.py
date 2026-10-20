"""Download selection tests using local placeholders and a mocked Hugging Face API."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from scripts import project


Q4 = "Qwen-Image-2512-Q4_K_M.gguf"
Q8 = "Qwen-Image-2512-Q8_0.gguf"
LORA = "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors"


class SetupModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="desynth-setup-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ai = self.root / "ai"
        self.ai.mkdir()
        (self.root / "embeds_cache.pt").write_bytes(b"embeddings")
        (self.ai / LORA).write_bytes(b"adapter")
        for item in (patch.object(project, "ROOT", self.root), patch.object(project, "MODEL_DIR", self.ai)):
            item.start()
            self.addCleanup(item.stop)
        self.download = Mock()
        self.api = Mock()
        self.api.return_value.list_repo_files.return_value = [Q4, Q8]
        hub = SimpleNamespace(HfApi=self.api, hf_hub_download=self.download)
        item = patch.dict("sys.modules", {"huggingface_hub": hub})
        item.start()
        self.addCleanup(item.stop)

    def prepare(self, model="auto", answers=(), interactive=False):
        terminal = SimpleNamespace(isatty=lambda: interactive)
        with (patch.object(project.sys, "stdin", terminal),
              patch("builtins.input", side_effect=answers) as prompt,
              redirect_stdout(io.StringIO())):
            project.download_models(model=model)
        return prompt

    def assert_downloaded(self, filename):
        self.api.return_value.list_repo_files.assert_called_once_with(repo_id="Frederic75/Qwen-Image-2512-GGUF")
        self.download.assert_called_once_with(
            repo_id="Frederic75/Qwen-Image-2512-GGUF", filename=filename, local_dir=str(self.ai),
        )

    def test_fresh_interactive_install_offers_q4_as_enter_default(self):
        prompt = self.prepare(answers=[""], interactive=True)
        prompt.assert_called_once_with("Choice [1]: ")
        self.assert_downloaded(Q4)

    def test_interactive_selection_can_choose_q8_and_retry_invalid_input(self):
        prompt = self.prepare(answers=["bad", "2"], interactive=True)
        self.assertEqual(prompt.call_count, 2)
        self.assert_downloaded(Q8)

    def test_noninteractive_install_uses_q4_without_reading_stdin(self):
        prompt = self.prepare()
        prompt.assert_not_called()
        self.assert_downloaded(Q4)

    def test_interactive_eof_falls_back_to_q4(self):
        self.prepare(answers=[EOFError()], interactive=True)
        self.assert_downloaded(Q4)

    def test_auto_reuses_installed_q8_without_prompt_or_network(self):
        (self.ai / Q8).write_bytes(b"model")
        prompt = self.prepare(interactive=True)
        prompt.assert_not_called()
        self.api.assert_not_called()
        self.download.assert_not_called()

    def test_explicit_q8_downloads_even_when_q4_is_installed(self):
        (self.ai / Q4).write_bytes(b"model")
        prompt = self.prepare(model="q8", interactive=True)
        prompt.assert_not_called()
        self.assert_downloaded(Q8)

    def test_explicit_q4_downloads_even_when_q8_is_installed(self):
        (self.ai / Q8).write_bytes(b"model")
        self.prepare(model="q4", interactive=True)
        self.assert_downloaded(Q4)

    def test_explicit_selection_reuses_legacy_root_file(self):
        (self.root / Q8).write_bytes(b"model")
        self.prepare(model="q8", interactive=True)
        self.api.assert_not_called()
        self.download.assert_not_called()

    def test_explicit_selection_does_not_accept_another_quant_or_empty_file(self):
        (self.ai / Q4).write_bytes(b"model")
        (self.ai / Q8).write_bytes(b"")
        self.assertIsNone(project.selected_gguf("q8"))
        self.assertEqual(project.selected_gguf("q4"), self.ai / Q4)

    def test_worker_forwards_selection_when_entering_virtual_environment(self):
        directory = self.root / ".venv"
        python = directory / "python"
        with (patch.object(project.sys, "prefix", str(self.root)),
              patch.object(project, "require_env", return_value=python),
              patch.object(project, "run") as run):
            project.worker("models", directory, "cuda", "4", "q8")
        run.assert_called_once_with(
            str(python), str(Path(project.__file__).resolve()), "models",
            "--venv", str(directory), "--backend", "cuda", "--lightning", "4", "--model", "q8",
        )

    def test_setup_passes_requested_model_to_download_and_doctor(self):
        with (patch.object(project.sys, "argv", ["project.py", "setup", "--model", "q8", "--lightning", "4"]),
              patch.object(project, "install") as install,
              patch.object(project, "worker") as worker):
            project.main()
        directory = (self.root / ".venv").resolve()
        backend = "cpu" if project.sys.platform == "darwin" else "cuda"
        install.assert_called_once_with(directory, backend)
        self.assertEqual(worker.call_args_list, [
            call(command, directory, backend, "4", "q8") for command in ("models", "prepare", "doctor")
        ])


if __name__ == "__main__":
    unittest.main()
