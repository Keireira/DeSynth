# DeSynth

Local image processing with Qwen-Image GGUF, Lightning and source detail restoration.
Includes a browser GUI and CLI for comparing results and studying invisible image watermarks.

## Results

`original.png`, 1460 × 1078. Q4_K_M, 8-step FP32 Lightning, Steps 32,
one pass, Seed 42, restoration at Sigma 1.95, no sharpening.

Left to right: original, **Gaussian**, **Edge**, at Denoise **0.25**.
Matching crops, enlarged without adding detail.

![Original vs Gaussian and Edge: face, hair, sign and shirt print](docs/images/detail-comparison.jpg)

| Metric | Gaussian 0.25 | Edge 0.25 |
|---|---:|---:|
| PSNR ↑ | 32.72 dB | 33.77 dB |
| SSIM ↑ | 0.9533 | 0.9608 |
| MAE ↓ | 3.85 | 3.57 |
| Sampling steps | 8 | 8 |

Edge keeps contours closer to the source,
especially lettering and print. Similarity scores do not establish watermark removal.

## How it works

Qwen-Image + Lightning runs img2img. Frequency restoration combines its
low-frequency content with fine detail from the source. The RGBA output retains
the original dimensions and alpha channel.

## Usage

### Setup

Use **Python 3.11**. From the repository folder in Windows PowerShell:

```powershell
py -3.11 scripts/project.py setup
.\.venv\Scripts\python.exe gui.py
```

Setup installs dependencies and downloads models into `ai/`. The default is
**Q4_K_M**, the smaller model. Keep the included `embeds_cache.pt`.

On Linux/macOS, use `python3.11` for setup and `.venv/bin/python` to run scripts.
For CPU-only installation, add `--backend cpu` to setup.

With GNU Make: `make setup`, then `make gui`. Use `make doctor` to check setup.

### Models

One GGUF model and one Lightning adapter are required:

| File in `ai/` | Size | Use |
|---|---:|---|
| `Qwen-Image-2512-Q4_K_M.gguf` | ~13 GB | Default model |
| `Qwen-Image-2512-Q8_0.gguf` | ~21.8 GB | Optional, more memory required |
| `Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors` | ~1.7 GB | Default adapter |

Sources: [GGUF models](https://huggingface.co/Frederic75/Qwen-Image-2512-GGUF)
and [Lightning adapters](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning).
The 4-step adapter is also supported. Other quantizations of the same
Qwen-Image-2512 model can be selected in Settings or with `--transformer`.

To download Q8 after setup:

```powershell
py -3.11 scripts/project.py models --model q8
```

### GUI

Open **http://127.0.0.1:8758**. Stop the server with **Ctrl+C** in its terminal.

![DeSynth GUI: draggable comparison, processing controls and export](docs/images/workspace.jpg)

Open an image, adjust the sliders and click **Process image**. Use the bottom
toolbar to compare images, drag the divider or view Diff. Restoration sliders
update the result without another generation. **Export PNG** saves the result;
in Original or Diff, it saves that image.

**Settings** contains models, device options and presets. **Refresh** finds newly
added models; **Save settings** applies choices. Sequential offload uses RAM to
reduce VRAM use. Leave Precision on Auto.

### CLI

Commands below use the Windows virtual environment. On Linux/macOS, replace
`.\.venv\Scripts\python.exe` with `.venv/bin/python`.

```powershell
# Process original.png with automatic model selection
.\.venv\Scripts\python.exe desynth.py --seed 42

# Repeat with Edge restoration
.\.venv\Scripts\python.exe desynth.py original.png --denoise 0.25 --steps 32 --seed 42 --restore-mode edge

# Select Q8 explicitly
.\.venv\Scripts\python.exe desynth.py original.png --transformer ai/Qwen-Image-2512-Q8_0.gguf --seed 42
```

For `original.png`, final images and `comparison_...` PNGs go to `out/` with
short names such as `Original_desynth_s32_d0.250_p1_r1.95_edge.png`.
Repeated settings update these reference images. Full runs, settings and
intermediates stay in the local `out/runs/` archive.

### Flags

Defaults below assume the 8-step adapter.

| Flag | Default | When to use |
|---|---|---|
| `--seed N` | Random | Repeat a run with the same settings |
| `--denoise X [X ...]` | `0.25` | Control starting noise; lower values preserve more detail |
| `--steps N` | `32` | Full schedule; at Denoise 0.25, 32 gives 8 sampling steps |
| `--passes N` | `1` | Process the previous result again |
| `--restore-sigma X` | `1.95` | Higher values restore larger structures from the source |
| `--restore-mode M` | `gaussian` | Use `edge` for more source detail around contours |
| `--unsharp X` | `0` | Sharpen after restoration |
| `--no-restore` | Off | Skip source detail restoration |
| `--keep-intermediate` | Off | Also save the image before restoration |
| `--transformer PATH` | Automatic, prefers Q4 | Select a GGUF model |
| `--lightning PATH` | Automatic, prefers 8-step FP32 | Select a Lightning adapter |

The 4-step adapter keeps its CLI default of 8 total steps. Set `--steps 16`
at Denoise 0.25 for four sampling steps. Use `--help` for all options.

### Quality check

```powershell
$result = "out/Original_desynth_s32_d0.250_p1_r1.95.png"
.\.venv\Scripts\python.exe compare.py original.png $result
.\.venv\Scripts\python.exe diff.py original.png $result --out out/diff.png --gain 8
```

`compare.py` prints similarity metrics. `diff.py` saves a visible RGBA difference
map; Gain amplifies small changes. Black means identical pixels.

## Hardware

Tested on an RTX 5070 Ti with 16 GB VRAM and 64 GB RAM. CUDA uses sequential
CPU offload by default. If memory runs out, use Q4 or a smaller input image.

## Limitations

- Higher Denoise and repeated passes can change faces, text and shapes.
- Detail restoration cannot recover every structure altered by generation.

## Credits

Original project: [0xROOTPLS/DeSynth](https://github.com/0xROOTPLS/DeSynth).
Baseline workflow: [00quebec/Synthid-Bypass](https://github.com/00quebec/Synthid-Bypass).
Models: [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image-2512)
and [LightX2V Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning).
