# Use forward slashes so the same recipes work in cmd.exe and POSIX shells.
VENV ?= .venv
BACKEND ?= auto
LIGHTNING ?= auto
LIGHTNING_LORA ?=
LIGHTNING_SCALE ?= 0.8

ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
PYTHON ?= py -3.11
VENV_PYTHON = $(VENV)/Scripts/python.exe
else
PYTHON ?= python3.11
VENV_PYTHON = $(VENV)/bin/python
endif

INPUT ?= original.png
OUTPUT ?= out/original_desynth_s32_d0.250_p1_r1.95.png
ARGS ?=
VISUAL_OUT ?= out/comparison.png
DIFF_OUT ?= out/diff.png
PREVIEW_OUT ?=
RAW_OUT ?=
GAIN ?= 8
PORT ?= 7860
GUI_ARGS ?=
