.DEFAULT_GOAL := help

# Environment creation and model downloads must not race under make -j.
.NOTPARALLEL:

include make/config.mk
include make/setup.mk
include make/run.mk

.PHONY: help
help:
	@echo Desynth commands
	@echo make setup   - create environment, install dependencies, prepare models and check readiness
	@echo make venv    - create or reuse the Python virtual environment
	@echo make install - install Python dependencies only
	@echo make models  - download missing GGUF and LoRA files
	@echo make models MODEL=q8 - download Q8_0 instead of the lower-memory Q4_K_M default
	@echo make models LIGHTNING=8 - prepare the 8-step Lightning adapter in ai/
	@echo make prepare - cache the VAE and pipeline configs
	@echo make doctor  - check dependencies, device and model assets
	@echo make gui    - open the local image editor and comparison workspace
	@echo make run INPUT=original.png
	@echo make compare INPUT=original.png OUTPUT=out/your_result.png
	@echo make visual INPUT=original.png OUTPUT=out/your_result.png
	@echo make diff INPUT=original.png OUTPUT=out/your_result.png
	@echo Variables: PYTHON VENV BACKEND MODEL LIGHTNING LIGHTNING_LORA LIGHTNING_SCALE PORT GUI_ARGS INPUT OUTPUT ARGS VISUAL_OUT DIFF_OUT PREVIEW_OUT RAW_OUT GAIN
