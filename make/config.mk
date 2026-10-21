# Use forward slashes so the same recipes work in cmd.exe and POSIX shells.
VENV ?= .venv
BACKEND ?= auto
MODEL ?= auto
LIGHTNING ?= auto
LIGHTNING_LORA ?=
LIGHTNING_SCALE ?= 0.8

ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
VENV_PYTHON = $(VENV)/Scripts/python.exe
DESYNTH = backend/target/release/desynth.exe
else
VENV_PYTHON = $(VENV)/bin/python
DESYNTH = backend/target/release/desynth
endif

INPUT ?= original.png
OUTPUT ?=
ARGS ?=
VISUAL_OUT ?= out/comparison.png
DIFF_OUT ?= out/diff.png
PREVIEW_OUT ?=
RAW_OUT ?=
GAIN ?= 8
PORT ?= 8758
GUI_ARGS ?=
PNPM ?= pnpm
NODE ?= node
CARGO ?= cargo
WEB_PORT ?= 3000

ENVIRONMENT_ARGS = --venv "$(VENV)" --backend "$(BACKEND)" --model "$(MODEL)" --lightning "$(LIGHTNING)"
