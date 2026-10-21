.PHONY: setup venv install models prepare doctor inference-check

inference-check: | venv
	"$(VENV_PYTHON)" -m unittest discover -s inference/tests -v

# Rust orchestrates the isolated Python inference environment.
setup venv install models prepare doctor: backend-build
	"$(DESYNTH)" $@ --venv "$(VENV)" --backend "$(BACKEND)" --model "$(MODEL)" --lightning "$(LIGHTNING)"
