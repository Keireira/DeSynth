.PHONY: setup setup-project tools require-environment

# Bootstrap before the recursive make reads the generated local environment.
setup: tools
	$(MAKE) setup-project

tools:
ifeq ($(OS),Windows_NT)
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -Venv "$(VENV)"
else
	sh scripts/bootstrap.sh --venv "$(VENV)"
endif

setup-project: web-build backend-build
	"$(DESYNTH)" setup $(ENVIRONMENT_ARGS)

require-environment:
ifeq ($(wildcard .tools/environment.mk),)
	$(error Run make setup first to prepare the local project environment)
endif
