SHELL := bash
PROVCONVERT=utility/provconvert/bin/provconvert
PYTHON=venv/bin/python3

DATASET_FOLDER=datasets/$(DATASET)
OUTPUT=outputs/$(DATASET)
TYPES_OUTPUT=outputs/types/$(DATASET)
PLOTDIR=plots/$(DATASET)

$(OUTPUT):
	@mkdir -p $@

# Collect all the graphs and calculate their network metrics
$(OUTPUT)/graphs.pickled: $(DATASET_FOLDER)/graphs.csv 		$(OUTPUT)
	@echo "> Generating PNA metrics and application data for $${DATASET}..."
	@[ -s $@ ] || $(PYTHON) -m scripts.data.$(DATASET)

clean-app-data:
	rm $(OUTPUT)/graphs.pickled

# Generating provenance kernels
PYTHON_F_KERNELS = $(shell echo $(OUTPUT)/kernels/F{A,G}_{0..5}.pickled)

$(PYTHON_F_KERNELS): $(OUTPUT)/graphs.pickled
	@$(PYTHON) scripts/gen-flattypes-kernel-tables.py $(DATASET_FOLDER) $(OUTPUT)

clean-kernels:
	rm -rf $(OUTPUT)/kernels/*.pickled

# Counting the provenance types produced by each kernels
$(OUTPUT)/type_counts.csv: $(PYTHON_F_KERNELS)
	@$(PYTHON) scripts/count-kernel-types.py $@ $(PYTHON_F_KERNELS)

kernels: $(PYTHON_F_KERNELS) $(OUTPUT)/type_counts.csv

# Running all the experiments on this dataset
$(OUTPUT)/scoring.pickled: $(OUTPUT)/graphs.pickled $(PYTHON_F_KERNELS)
	$(PYTHON) -m scripts.experiments.$(DATASET)

experiments: $(OUTPUT)/scoring.pickled

clean-experiments:
	rm -f $(OUTPUT)/scoring.pickled
	rm -rf $(PLOTDIR)

# Generating the plots for results of this dataset
$(PLOTDIR)/type_counts.pdf: $(OUTPUT)/type_counts.csv
	@[ -d $(@D) ] || mkdir -p $(@D)
	@$(PYTHON) scripts/plotting/type_counts.py $(OUTPUT)/type_counts.csv $@

plots: $(OUTPUT)/scoring.pickled $(PLOTDIR)/type_counts.pdf
	$(PYTHON) scripts/plotting/$(DATASET).py


.PHONY: kernels experiments plots clean-app-data clean-kernels clean-experiments
