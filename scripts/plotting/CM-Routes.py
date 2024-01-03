from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk")


def plot_bars(data, measure="accuracy", ylim=(0.5, 1.0)):
    plot = sns.barplot(x="method", y=measure, data=data, err_kws={"linewidth": 1}, capsize=0.02)
    plt.xticks(rotation=90)
    plot.set_xlabel("Method")
    plot.set_ylabel(measure.capitalize())
    plot.set_ylim(ylim)
    plot.figure.set_size_inches(18, 9)
    plt.tight_layout()
    return plot


ROOT_DIR = Path(__file__).parents[2]
dataset_id = "CM-Routes"
results_folder = ROOT_DIR / "outputs" / dataset_id
plots_folder = ROOT_DIR / "plots" / dataset_id
plots_folder.mkdir(parents=True, exist_ok=True)


# Plotting the results for predicting a patient will die in an admission
results = pd.read_pickle(results_folder / "scoring.pickled")

plot = plot_bars(results)
plot.figure.savefig(plots_folder / f"{dataset_id}-accuracy.pdf")
plt.close()
