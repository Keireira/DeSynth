.PHONY: setup venv install models prepare doctor

# The Python helper keeps bootstrap steps portable and setup sequential.
setup venv install models prepare doctor:
	$(PYTHON) scripts/project.py $@ --venv "$(VENV)" --backend "$(BACKEND)" --model "$(MODEL)" --lightning "$(LIGHTNING)"
