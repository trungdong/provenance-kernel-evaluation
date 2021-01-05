SHELL := bash
PROVCONVERT=utility/provconvert/bin/provconvert
PYTHON=venv/bin/python3
MAKEFILE_SINGLE=$(abspath dataset.makefile)

DATASETS = MIMIC-PXC7 CM-Buildings CM-Routes CM-RouteSets PG-T PG-D
OUTPUT_DIRS = $(DATASETS:%=outputs/%)

help:
	@echo "Provenance Kernels Evaluation Pipeline Makefile"
	@echo "Current supported targets: venv, data, kernels, experiments, plots"

# Setting up the Python virtual environment
venv: venv/bin/activate

venv/bin/activate: scripts/requirements.txt
	@echo "> Setting up the Python virtual environment"
	@test -d venv || python3 -m venv ./venv
	@. venv/bin/activate; pip install -U pip; pip install -Ur scripts/requirements.txt
	@touch venv/bin/activate

# Setting up the datasets
datasets/MIMIC-PXC7:
	@echo "> Generating MIMIC provenance graphs..."
	@venv/bin/python3 scripts/mimic/generate_provenance.py

datasets/CM-Buildings:
	@echo "> Unpacking CM-Buildings graphs..."
	@tar -xzf datasets/CM-Buildings.tar.gz --directory datasets

datasets/CM-Routes:
	@echo "> Unpacking CM-Routes graphs..."
	@tar -xzf datasets/CM-Routes.tar.gz --directory datasets

datasets/CM-RouteSets:
	@echo "> Unpacking CM-RouteSets graphs..."
	@tar -xzf datasets/CM-RouteSets.tar.gz --directory datasets

datasets/PG-T:
	@echo "> Unpacking PG-T graphs..."
	@tar -xzf datasets/PG-T.tar.gz --directory datasets

datasets/PG-D:
	@echo "> Unpacking PG-D graphs..."
	@tar -xzf datasets/PG-D.tar.gz --directory datasets

data: datasets/CM-Buildings datasets/CM-Routes datasets/CM-RouteSets datasets/PG-T datasets/PG-D


# The following goals will be called using the dataset Makefile on the each dataset
kernels types experiments plots clean-app-data clean-kernels clean-pickled-kernels clean-experiments: $(DATASETS)

$(DATASETS): venv data
	@echo "--------- Execute [$(MAKECMDGOALS)] on $@ dataset ---------"
	@$(MAKE) --file $(MAKEFILE_SINGLE) $(MAKECMDGOALS) DATASET=$@


# Other maintenance goals
clean:
	rm -rf venv
	find . -name "*.pyc" -delete
	rm -rf outputs
	rm -rf plots

.PHONY: help data clean kernels types experiments plots clean-app-data clean-kernels clean-pickled-kernels clean-experiments
.PHONY: $(DATASETS)