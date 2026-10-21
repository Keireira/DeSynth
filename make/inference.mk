.PHONY: venv install models prepare doctor inference-lint inference-check

inference-lint: | venv
	"$(VENV_PYTHON)" -m pylint --rcfile=inference/.pylintrc inference

inference-check: inference-lint
	"$(VENV_PYTHON)" -m unittest discover -s inference/tests -v

# Rust orchestrates the isolated Python inference environment.
venv install models prepare doctor: backend-build
	"$(DESYNTH)" $@ $(ENVIRONMENT_ARGS)
