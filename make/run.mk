.PHONY: gui run compare visual diff

gui: web-build backend-build
	"$(DESYNTH)" --python "$(VENV_PYTHON)" gui --port "$(PORT)" $(GUI_ARGS)

run: backend-build
	"$(DESYNTH)" --python "$(VENV_PYTHON)" run "$(INPUT)" $(if $(LIGHTNING_LORA),--lightning "$(LIGHTNING_LORA)") --lightning-scale "$(LIGHTNING_SCALE)" $(ARGS)

compare: backend-build
	$(if $(strip $(OUTPUT)),,$(error Set OUTPUT to the result PNG printed by make run))
	"$(DESYNTH)" compare "$(INPUT)" "$(OUTPUT)"

visual: backend-build
	$(if $(strip $(OUTPUT)),,$(error Set OUTPUT to the result PNG printed by make run))
	"$(DESYNTH)" compare "$(INPUT)" "$(OUTPUT)" --visual --visual-out "$(VISUAL_OUT)"

diff: backend-build
	$(if $(strip $(OUTPUT)),,$(error Set OUTPUT to the result PNG printed by make run))
	"$(DESYNTH)" diff "$(INPUT)" "$(OUTPUT)" --out "$(DIFF_OUT)" --gain "$(GAIN)" $(if $(PREVIEW_OUT),--preview-out "$(PREVIEW_OUT)") $(if $(RAW_OUT),--raw-out "$(RAW_OUT)")
