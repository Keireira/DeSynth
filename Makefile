.DEFAULT_GOAL := help

# Environment creation and model downloads must not race under make -j.
.NOTPARALLEL:

include make/config.mk
include make/backend.mk
include make/inference.mk
include make/run.mk
include make/web.mk

.PHONY: help
help:
	@echo Desynth commands
	@echo make setup   - create environment, install dependencies, prepare models and check readiness
	@echo make venv    - create or reuse the Python virtual environment
	@echo make install - install Python dependencies only
	@echo make models  - download missing GGUF and LoRA files
	@echo make models MODEL=q8 - download Q8_0 instead of the lower-memory Q4_K_M default
	@echo make models LIGHTNING=8 - prepare the 8-step Lightning adapter in inference/models/
	@echo make prepare - cache the VAE and pipeline configs
	@echo make doctor  - check dependencies, device and model assets
	@echo make gui    - open the local image editor and comparison workspace
	@echo make web-build - install frontend dependencies and build the editor
	@echo make web-dev - start the React dev server, with the backend running on PORT
	@echo make web-check - typecheck, lint, format check and frontend tests
	@echo make web-format - format frontend and shared JavaScript configs
	@echo make backend-check - Rust formatting, lint and backend tests
	@echo make inference-check - PyTorch and Diffusers worker tests
	@echo make run INPUT=original.png
	@echo make compare INPUT=original.png OUTPUT=out/your_result.png
	@echo make visual INPUT=original.png OUTPUT=out/your_result.png
	@echo make diff INPUT=original.png OUTPUT=out/your_result.png
	@echo Variables: VENV BACKEND MODEL LIGHTNING LIGHTNING_LORA LIGHTNING_SCALE PORT GUI_ARGS INPUT OUTPUT ARGS VISUAL_OUT DIFF_OUT PREVIEW_OUT RAW_OUT GAIN
