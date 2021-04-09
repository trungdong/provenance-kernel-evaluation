import logging
import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.experiments.common import load_kernel_ml_data, SVM_C_PARAMS
from scripts.utils import load_graph_index, Timer


def train_provenance_kernel_pipeline(
    graphs: pd.DataFrame,
    output_path: Path,
    kernel: str,
    level: int,
    y_column: str,
    including_edge_type_counts: bool = False,
) -> Pipeline:
    X, y = load_kernel_ml_data(
        graphs, output_path, kernel, level, y_column, including_edge_type_counts
    )
    clf = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("svm", SVC(kernel="rbf", gamma="scale", class_weight="balanced")),
        ]
    )
    gs = GridSearchCV(
        estimator=clf,
        param_grid={
            "svm__C": SVM_C_PARAMS,
        },
        refit=True,
        n_jobs=-1,
    )
    with Timer():
        gs.fit(X, y)

    clf = gs.best_estimator_
    print(" - Best params:", gs.best_params_)
    print(" - Best score:", gs.best_score_)
    print(" - Accuracy:", clf.score(X, y))

    return clf


@click.command()
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("kernel")
@click.argument("y_column")
@click.argument("to_level", default=5)
def main(output_folder: str, kernel: str, y_column: str, to_level: int):
    output_path = Path(output_folder)
    dataset_id = output_path.name
    graphs_index = load_graph_index(dataset_id)
    selected_samples_filepath = output_path / "selected.csv"
    selected_graphfiles = pd.read_csv(selected_samples_filepath, index_col=0)
    selected_graphs = graphs_index.iloc[selected_graphfiles.index].copy()

    print(f"Training {kernel}-{to_level} on {len(selected_graphs)} {dataset_id} graphs...")
    clf = train_provenance_kernel_pipeline(
        selected_graphs, output_path, kernel, to_level, y_column
    )
    models_folder = output_path / "models"
    models_folder.mkdir(parents=True, exist_ok=True)
    model_filepath = models_folder / f"model_{kernel}_{to_level}.pickled"
    with model_filepath.open("wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
