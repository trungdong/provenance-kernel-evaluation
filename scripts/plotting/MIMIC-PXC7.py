from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk")


def plot_bars(data, measure="accuracy", ylim=(0.5, 0.9)):
    plot = sns.barplot(x="method", y=measure, data=data, errwidth=1, capsize=0.02)
    plt.xticks(rotation=90)
    plot.set_xlabel("Method")
    plot.set_ylabel(measure.capitalize())
    plot.set_ylim(ylim)
    plot.figure.set_size_inches(18, 9)
    plt.tight_layout()
    return plot


ROOT_DIR = Path(__file__).parents[2]
dataset_id = "MIMIC-PXC7"
results_folder = ROOT_DIR / "outputs" / dataset_id
plots_folder = ROOT_DIR / "plots" / dataset_id
plots_folder.mkdir(parents=True, exist_ok=True)


# Plotting the results for predicting a patient will die in an admission
results = pd.read_pickle(results_folder / "scoring.pickled")

plot = plot_bars(results)
plot.figure.savefig(plots_folder / f"{dataset_id}-accuracy.pdf")
plt.close()
plot = plot_bars(results, "f1")
plot.figure.savefig(plots_folder / f"{dataset_id}-f1.pdf")
plt.close()
plot = plot_bars(results, "recall")
plot.figure.savefig(plots_folder / f"{dataset_id}-recall.pdf")
plt.close()
