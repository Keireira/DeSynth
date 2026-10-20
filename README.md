# DeSynth

DeSynth processes images locally with Qwen-Image and a Lightning adapter, then
restores fine detail from the source. The project explores how this affects
invisible image watermarks. You can process an image, compare the result and
inspect RGBA differences in the browser or from the command line.

![DeSynth: draggable comparison and processing controls](docs/images/workspace.jpg)

## Get started

Use **Python 3.11**. Run these commands from the repository folder in Windows PowerShell:

```powershell
py -3.11 scripts/project.py setup
.\.venv\Scripts\python.exe gui.py
```

Setup creates `.venv`, installs dependencies, downloads missing weights into
`ai/`, caches the VAE/configs and checks that everything is ready. Keep the
included `embeds_cache.pt`; it is required by the pipeline.

On a fresh install, setup asks which GGUF model to download. Press Enter for
**Q4_K_M**, the smaller option (~13 GB), or choose **Q8_0** (~21.8 GB).
Existing weights are reused. The default Lightning download is **8-step FP32**
(~1.7 GB). Allow extra disk space for dependencies and the Hugging Face cache.

To request Q8 explicitly, including when Q4 is already installed:

```powershell
py -3.11 scripts/project.py models --model q8
```

Without an interactive terminal, setup uses Q4 for a fresh install. Use
`--model q4` or `--model q8` to choose explicitly, and `--lightning 4` or
`--lightning 8` to prepare a particular adapter.

If you use GNU Make, the same workflow is:

```powershell
make setup
make gui
# Optional: download the larger model
make models MODEL=q8
```

On Linux/macOS, use `python3.11` for setup and `.venv/bin/python` to run the
scripts. For a machine without NVIDIA CUDA, add `--backend cpu` to setup
(`BACKEND=cpu` with Make). macOS selects the MPS/CPU dependency set automatically.

The GUI opens at **http://127.0.0.1:8758**. Keep its terminal open and press
**Ctrl+C** there to stop it. Use `gui.py --port 7864 --no-browser` for another
port without opening a browser. Images stay on your machine; preparation needs
Internet access to download missing assets.

## Use the interface

1. Click **Open image** or drop a file into the viewer.
2. Set **Denoise**, **Steps**, **Passes** and **Seed**, then click **Process image**.
   Moving these controls does not start a processing run.
3. Compare with **Original**, **Slider**, **Side by side** or **Result** in the
   bottom toolbar. Drag the divider in Slider mode; scroll to zoom and drag to pan.
4. Adjust **Detail restoration** after processing to change Sigma, Method or
   Sharpen without running diffusion again.
5. Click **Export PNG**. Original exports the source, Diff exports the visible
   difference map, and the other modes export the result.

### Settings and presets

Open **Settings** in the bottom toolbar. Model, Lightning adapter, Device,
Offload and Precision are on the right; presets are on the left.

![Settings: presets on the left, model and runtime options on the right](docs/images/settings.jpg)

- **Refresh** rescans local weights and available hardware without restarting or
  losing the current images. Use it after adding or moving model files.
- **Save settings** applies and remembers your choices. **Cancel**, Escape and
  the close button discard unsaved changes.
- To save a preset, enter a name and click **Save preset**. It stores generation,
  restoration, model and runtime settings. Saving the same name updates it.
- To use a preset, select it and click **Save settings**. Settings and presets are
  stored in this browser.

**Offload** uses system RAM to reduce GPU memory use. Sequential transfers
individual layers and uses the least VRAM; Model transfers whole components;
Disabled keeps the pipeline on the selected device. Offloading is slower than
keeping everything on the GPU.

Leave **Precision** on Auto unless you need to change it. BF16 is used where
supported; CPU uses FP32. FP16 is disabled because it can produce black images
with this pipeline. FP32 in an adapter filename describes the saved weights;
it does not force the entire pipeline to run in FP32.

### Difference map

Choose **Diff** to see pixel changes. Black means identical; brighter pixels
mean a larger difference. **Gain** makes small changes easier to see. Choose
RGBA for all channels, RGB for color only or Alpha for transparency only.

![RGBA difference map with gain and channel controls](docs/images/diff.jpg)

## Models and steps

You need **one GGUF model and one Lightning adapter**. Put them directly in
`ai/`. Older installations with weights in the repository root also work.

| File in `ai/` | Size | Use |
|---|---:|---|
| `Qwen-Image-2512-Q4_K_M.gguf` | ~13 GB | Default, lower memory use |
| `Qwen-Image-2512-Q8_0.gguf` | ~21.8 GB | Optional, larger model |
| `Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors` | ~1.7 GB | Default adapter |
| `Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors` | ~1.7 GB | Fewer sampling steps |

Download sources: [GGUF models](https://huggingface.co/Frederic75/Qwen-Image-2512-GGUF)
and [Lightning adapters](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning).
After downloading, select the model in Settings or with `--transformer`.
Choose Q4 or Q8 independently of the adapter. Other GGUF quantizations of this
same Qwen-Image-2512 model can be selected when supported by the installed loader;
an unrelated model architecture is not interchangeable.

**Steps is the full schedule. Denoise determines how much of it runs.** At
Denoise `0.25`, use:

| Adapter | Steps | Actual sampling steps per pass |
|---|---:|---:|
| 8-step Lightning | 32 | 8 |
| 4-step Lightning | 16 | 4 |

Changing the adapter in Settings does not change your Steps value. The CLI
default is 32 for the 8-step adapter. The 4-step adapter keeps its older default
of 8, which runs only 2 actual steps at Denoise 0.25; set 16 explicitly for four.

The screenshots use Q4_K_M with the 8-step FP32 adapter, Denoise 0.25, Steps 32,
Seed 42 and Gaussian restoration at Sigma 1.95.

## Command line

Commands below use Windows PowerShell after setup. Replace
`.\.venv\Scripts\python.exe` with `.venv/bin/python` on Linux/macOS.
For multiline commands, also replace the trailing PowerShell backticks with `\`.

Automatic selection prefers installed Q4 and 8-step Lightning:

```powershell
.\.venv\Scripts\python.exe desynth.py original.png --seed 42
```

Choose Q4 and the 8-step FP32 adapter explicitly:

```powershell
.\.venv\Scripts\python.exe desynth.py original.png `
  --transformer ai/Qwen-Image-2512-Q4_K_M.gguf `
  --lightning ai/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors `
  --steps 32 --denoise 0.25 --seed 42
```

For Q8, keep the same adapter and sampling settings:

```powershell
.\.venv\Scripts\python.exe desynth.py original.png `
  --transformer ai/Qwen-Image-2512-Q8_0.gguf `
  --lightning ai/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors `
  --steps 32 --denoise 0.25 --seed 42
```

Both explicit examples write `out/original_desynth_s32_d0.250_p1_r1.95.png`.
Copy the result before trying another model or seed: these choices are not
included in the filename. Device, precision and offload are automatic in the
CLI; use Settings in the GUI to select them manually.

### Options

| Option | Default | What it changes |
|---|---|---|
| `--seed N` | Random | Fix the seed to repeat a run with the same settings. |
| `--denoise X [X ...]` | `0.25` | Strength of changes. Multiple values save separate results. |
| `--steps N` | `32` for 8-step; `8` for 4-step | Full schedule per pass; actual count depends on denoise. |
| `--passes N` | `1` | Process the previous result again, N times in total. |
| `--transformer PATH` | Automatic | Select a GGUF model. |
| `--lightning PATH` | Installed 8-step, then 4-step | Select the Lightning adapter. |
| `--lightning-scale X` | `0.8` | Strength of the Lightning adapter. |
| `--restore-sigma X` | `1.95` | Scale of source detail brought back after generation. |
| `--restore-mode gaussian\|edge` | `gaussian` | Uniform detail restoration or more source detail around edges. |
| `--unsharp X` | `0` | Sharpen after restoration. |
| `--no-restore` | Off | Save generation output without restoring source detail. |
| `--keep-intermediate` | Off | Also save the image before restoration. |

### Compare and export a diff

Print similarity metrics, or save a side-by-side comparison:

```powershell
.\.venv\Scripts\python.exe compare.py original.png out/original_desynth_s32_d0.250_p1_r1.95.png
.\.venv\Scripts\python.exe compare.py original.png out/original_desynth_s32_d0.250_p1_r1.95.png --visual --visual-out out/comparison.png
```

`compare.py` measures RGB similarity and resizes the second image if needed.
SSIM and PSNR describe similarity to the source; they do not test for a watermark.

Save a visible difference map including alpha:

```powershell
.\.venv\Scripts\python.exe diff.py original.png out/original_desynth_s32_d0.250_p1_r1.95.png --out out/diff.png --gain 8
```

`diff.py` requires matching image dimensions. Its main PNG stays opaque so that
color and alpha differences are visible. Optional `--preview-out out/preview.png`
separates color and alpha changes. `--raw-out out/raw.png --gain 1` saves exact
RGBA deltas; that raw file can look empty when unchanged alpha becomes zero.
Use the main diff PNG for viewing.

### Useful Make commands

| Command | Action |
|---|---|
| `make install` | Install dependencies without downloading weights. |
| `make models MODEL=q4` | Prepare Q4; `MODEL=q8` selects Q8. |
| `make models LIGHTNING=8` | Prepare the 8-step adapter; use `4` for the 4-step version. |
| `make prepare` | Cache VAE and pipeline configs. |
| `make doctor` | Check dependencies, hardware and model files. |
| `make run INPUT=original.png ARGS="--seed 42"` | Process an image. |
| `make compare` / `make visual` / `make diff` | Compare the default source and s32 output. |

Comparison targets accept `INPUT` and `OUTPUT`. Run `make` to list commands.
Without Make, use `py -3.11 scripts/project.py doctor` for a readiness check.

## If something goes wrong

- **Out of memory:** choose Q4, Sequential offload or a smaller image. Q8 needs
  more RAM/VRAM; the file size is not the total runtime memory requirement.
- **A model is missing from Settings:** put it directly in `ai/`, click Refresh
  and select it again. Refresh updates the list; it does not download weights.
- **CUDA is unavailable:** check the NVIDIA driver and run `make doctor`. For
  CPU-only use, rerun setup with `--backend cpu`.
- **Make will not start:** use the Python commands above; Make is optional.

## Credits

Based on the workflow explored in
[00quebec/Synthid-Bypass](https://github.com/00quebec/Synthid-Bypass), using
[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image-2512) and
[LightX2V Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning).
