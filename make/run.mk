.PHONY: gui run compare visual diff

gui: | venv
	"$(VENV_PYTHON)" gui.py --port "$(PORT)" $(GUI_ARGS)

run: | venv
	"$(VENV_PYTHON)" desynth.py "$(INPUT)" $(if $(LIGHTNING_LORA),--lightning "$(LIGHTNING_LORA)") --lightning-scale "$(LIGHTNING_SCALE)" $(ARGS)

compare: | venv
	$(if $(strip $(OUTPUT)),,$(error Set OUTPUT to the result PNG printed by desynth.py))
	"$(VENV_PYTHON)" compare.py "$(INPUT)" "$(OUTPUT)"

visual: | venv
	$(if $(strip $(OUTPUT)),,$(error Set OUTPUT to the result PNG printed by desynth.py))
	"$(VENV_PYTHON)" compare.py "$(INPUT)" "$(OUTPUT)" --visual --visual-out "$(VISUAL_OUT)"

diff: | venv
	$(if $(strip $(OUTPUT)),,$(error Set OUTPUT to the result PNG printed by desynth.py))
	"$(VENV_PYTHON)" diff.py "$(INPUT)" "$(OUTPUT)" --out "$(DIFF_OUT)" --gain "$(GAIN)" $(if $(PREVIEW_OUT),--preview-out "$(PREVIEW_OUT)") $(if $(RAW_OUT),--raw-out "$(RAW_OUT)")
