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
PYTHON_KERNELS = $(shell echo $(OUTPUT)/kernels/{F,D}{A,G}_{0..5}.pickled)
SCALA_KERNELS = $(shell echo $(OUTPUT)/kernels/{H,T}{A,G}_{-5..5}.pickled)

$(PYTHON_KERNELS): $(OUTPUT)/graphs.pickled
	@echo "> Producing linear kernels for $${DATASET}..."
	@mkdir -p $(OUTPUT)/kernels
	@$(PYTHON) scripts/gen-flattypes-kernel-tables.py $(DATASET_FOLDER) $(OUTPUT)

$(SCALA_KERNELS):
	@echo "> Producing hierarchical kernels for $${DATASET}..."
	@mkdir -p $(OUTPUT)/kernels
	@$(PYTHON) scripts/gen-hierarchical-kernel-tables.py $(DATASET_FOLDER) $(OUTPUT)

clean-kernels:
	rm -rf $(OUTPUT)/kernels
	rm -rf $(OUTPUT)/type_counts.csv

# Counting the provenance types produced by each kernels
$(OUTPUT)/type_counts.csv: $(PYTHON_KERNELS) $(SCALA_KERNELS)
	@$(PYTHON) scripts/count-kernel-types.py $@ $(PYTHON_KERNELS) $(SCALA_KERNELS)

kernels: $(PYTHON_KERNELS) $(SCALA_KERNELS) $(OUTPUT)/type_counts.csv

# Running all the experiments on this dataset
$(OUTPUT)/scoring.pickled: $(OUTPUT)/graphs.pickled $(PYTHON_KERNELS) $(SCALA_KERNELS)
	@$(PYTHON) -m scripts.experiments.$(DATASET)

experiments: $(OUTPUT)/scoring.pickled

clean-cached-experiments:
	rm -f $(OUTPUT)/selected.csv
	rm -f $(OUTPUT)/cv_sets.pickled
	rm -f $(OUTPUT)/scorings/*

clean-experiments:
	rm -f $(OUTPUT)/scoring.pickled
	rm -rf $(PLOTDIR)

# Generating the plots for results of this dataset
$(PLOTDIR)/type_counts.pdf: $(OUTPUT)/type_counts.csv
	@[ -d $(@D) ] || mkdir -p $(@D)
	@$(PYTHON) scripts/plotting/type_counts.py $(OUTPUT)/type_counts.csv $@

plots: $(OUTPUT)/scoring.pickled $(PLOTDIR)/type_counts.pdf
	$(PYTHON) scripts/plotting/$(DATASET).py
	$(PYTHON) -m scripts.plotting.type_stats $(OUTPUT) $(PLOTDIR) FA
	$(PYTHON) -m scripts.plotting.type_stats $(OUTPUT) $(PLOTDIR) FG


.PHONY: kernels experiments plots clean-app-data clean-kernels clean-experiments
