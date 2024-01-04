"""Common experiment code."""
import itertools
import logging
from operator import itemgetter
from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from grakel.kernels import (
    GraphletSampling,
    RandomWalk,
    RandomWalkLabeled,
    ShortestPath,
    ShortestPathAttr,
    WeisfeilerLehman,
    WeisfeilerLehmanOptimalAssignment,
    NeighborhoodHash,
    PyramidMatch,
    SubgraphMatching,
    NeighborhoodSubgraphPairwiseDistance,
    LovaszTheta,
    SvmTheta,
    OddSth,
    Propagation,
    PropagationAttr,
    HadamardCode,
    MultiscaleLaplacian,
    VertexHistogram,
    EdgeHistogram,
    GraphHopper,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scipy.sparse import coo_matrix
from torch_geometric.loader import DataLoader

from scripts.data.common import PROV_RELATION_NAMES
from scripts.experiments.gnn import GATv2Wrapper
from scripts.experiments.prov2pyg import convert_PROV_graphs_to_PyG_data
from scripts.utils import Timer, TimeoutException

logger = logging.getLogger(__name__)

# no parallel for graph kernels to measure time costs
N_JOBS = None  # type: Optional[int]
NORMALIZING_GRAPH_KERNELS = True
TIMEOUT = 60 * 60  # one hour in seconds
SVM_C_PARAMS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

ML_CLASSIFIERS = {
    "DTree": DecisionTreeClassifier(max_depth=5),
    "RF": RandomForestClassifier(max_depth=5, n_estimators=10),
    "K-NB": KNeighborsClassifier(8),
    "NBayes": GaussianNB(),
    "NN": MLPClassifier(alpha=1, max_iter=1000),
    "SVM": Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", SVC(gamma="scale", class_weight="balanced")),
        ]
    ),
}

scoring = ["accuracy", "f1_micro", "precision_micro", "recall_micro"]
score_fields = list(map("test_{}".format, scoring))

# defining short names for methods to save space in plots
method_short_names = {
    "FG-0": "G0",
    "FG-1": "G1",
    "FG-2": "G2",
    "FG-3": "G3",
    "FG-4": "G4",
    "FG-5": "G5",
    "FA-0": "A0",
    "FA-1": "A1",
    "FA-2": "A2",
    "FA-3": "A3",
    "FA-4": "A4",
    "FA-5": "A5",
    "DG-0": "GD0",
    "DG-1": "GD1",
    "DG-2": "GD2",
    "DG-3": "GD3",
    "DG-4": "GD4",
    "DG-5": "GD5",
    "DA-0": "AD0",
    "DA-1": "AD1",
    "DA-2": "AD2",
    "DA-3": "AD3",
    "DA-4": "AD4",
    "DA-5": "AD5",
    "PNA-DTree": "DT",
    "PNA-RF": "RF",
    "PNA-K-NB": "K-NB",
    "PNA-NBayes": "NB",
    "PNA-NN": "NN",
    "PNA-SVM": "SVM",
    "GK-SPath": "SP",
    "GK-EHist": "EH",
    "GK-VHist": "VH",
    "GK-GSamp": "GS",
    "GK-WL-1": "WL1",
    "GK-WL-2": "WL2",
    "GK-WL-3": "WL3",
    "GK-WL-4": "WL4",
    "GK-WL-5": "WL5",
    "GK-NH": "NH",
    "GK-HC-1": "HC1",
    "GK-HC-2": "HC2",
    "GK-HC-3": "HC3",
    "GK-HC-4": "HC4",
    "GK-HC-5": "HC5",
    "GK-NSPD": "NSPD",
    "GK-WL-OA-1": "WLO1",
    "GK-WL-OA-2": "WLO2",
    "GK-WL-OA-3": "WLO3",
    "GK-WL-OA-4": "WLO4",
    "GK-WL-OA-5": "WLO5",
    "GK-OddSth": "ODD",
    "HG-0": "GH0",
    "HG-1": "GH+1",
    "HG-2": "GH+2",
    "HG-3": "GH+3",
    "HG-4": "GH+4",
    "HG-5": "GH+5",
    "HG--1": "GH-1",
    "HG--2": "GH-2",
    "HG--3": "GH-3",
    "HG--4": "GH-4",
    "HG--5": "GH-5",
    "HA-0": "AH0",
    "HA-1": "AH+1",
    "HA-2": "AH+2",
    "HA-3": "AH+3",
    "HA-4": "AH+4",
    "HA-5": "AH+5",
    "HA--1": "AH-1",
    "HA--2": "AH-2",
    "HA--3": "AH-3",
    "HA--4": "AH-4",
    "HA--5": "AH-5",
    "TG-0": "GT0",
    "TG-1": "GT+1",
    "TG-2": "GT+2",
    "TG-3": "GT+3",
    "TG-4": "GT+4",
    "TG-5": "GT+5",
    "TG--1": "GT-1",
    "TG--2": "GT-2",
    "TG--3": "GT-3",
    "TG--4": "GT-4",
    "TG--5": "GT-5",
    "TA-0": "AT0",
    "TA-1": "AT+1",
    "TA-2": "AT+2",
    "TA-3": "AT+3",
    "TA-4": "AT+4",
    "TA-5": "AT+5",
    "TA--1": "AT-1",
    "TA--2": "AT-2",
    "TA--3": "AT-3",
    "TA--4": "AT-4",
    "TA--5": "AT-5",
    "GAT2": "GATv2",
}


def save_experiment_scorings(output_path: Path, method_id: str, scorings: pd.DataFrame):
    scorings_path = output_path / "scorings"
    # making sure that the folder is there
    scorings_path.mkdir(parents=True, exist_ok=True)
    scorings_filepath = scorings_path / (method_id + ".pickled")
    scorings.to_pickle(scorings_filepath)


def load_experiment_scorings(output_path: Path, method_id: str):
    scorings_filepath = output_path / "scorings" / (method_id + ".pickled")
    if not scorings_filepath.is_file():
        return None
    return pd.read_pickle(scorings_filepath)


def get_fixed_CV_sets(
    X,
    y,
    n_splits: int = 10,
    n_repeats: int = 10,
    random_state=None,
    output_path: Path = None,
):
    if output_path is not None:
        # try loading the cached CV_sets if it exists
        # this is to make sure performance comparison is fair across runs
        cv_sets_filepath = output_path / "cv_sets.pickled"
        if cv_sets_filepath.is_file():
            print("> Loaded CV sets from: ", cv_sets_filepath)
            with cv_sets_filepath.open("rb") as f:
                return pickle.load(f)

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    cv_sets = list(rskf.split(X, y))

    if output_path is not None:
        # remember these sets for later runs, if any
        cv_sets_filepath = output_path / "cv_sets.pickled"
        with cv_sets_filepath.open("wb") as f:
            pickle.dump(cv_sets, f)

    return cv_sets


def pd_df_to_coo(df: pd.DataFrame):
    # Converting a Pandas DataFrame to sparse.coo_matrix
    cols, rows, datas = [], [], []
    for col, name in enumerate(df):
        s = df[name]
        row = s.array.sp_index.to_int_index().indices
        cols.append(np.repeat(col, len(row)))
        rows.append(row)
        # SVM works best with float64 dtype
        datas.append(s.array.sp_values.astype(np.float64, copy=False))

    cols = np.concatenate(cols)
    rows = np.concatenate(rows)
    datas = np.concatenate(datas)
    return coo_matrix((datas, (rows, cols)), shape=df.shape)


def merge_timings_to_graph_index(graphs: pd.DataFrame, timings: pd.DataFrame):
    timings = timings.rename(
        lambda col_name: "timings_" + col_name, axis="columns", copy=False
    )
    return graphs.join(timings, on="graph_file")


def read_provenance_kernel_dataframes(
    output_path: Path, kernel_set: str, from_level: int = 0, to_level: int = None
) -> pd.DataFrame:
    kernel_df_filepath = output_path / "kernels" / f"{kernel_set}_{from_level}.pickled"
    logger.debug("Reading pickled dataframe: %s", kernel_df_filepath)
    kernels_df = pd.read_pickle(kernel_df_filepath)

    # going up for forward kernels, down for backward kernels
    increment = 1 if to_level > from_level else -1
    if to_level is not None:
        level = from_level
        while level != to_level:
            level += increment  # reading the next level
            kernel_df_filepath = (
                output_path / "kernels" / f"{kernel_set}_{level}.pickled"
            )
            logger.debug("Reading pickled dataframe: %s", kernel_df_filepath)
            df = pd.read_pickle(kernel_df_filepath)
            kernels_df = kernels_df.join(df)

    return kernels_df


def load_provenance_kernel_ml_data(
    graphs: pd.DataFrame,                      # loading the rows for only those graphs
    output_path: Path,                         # the path where the pre-computed `kernels` folder will be found
    kernel_set: str,                           # which type of provenance kernel (FG or FA)
    level: int = 0,                            # specifying the level of the kernel (0, 1, 2, 3, 4, 5, ...)
    y_column: str = "label",                   # the property of a graph to be used as the classification label (y)
    including_edge_type_counts: bool = False,  # joining the numbers of provenance relations (as ML features)
    sparse: bool = True,                       # converting the data (X) into a sparse matrix to save memory
):
    print("  - Reading kernels...")
    kernels_df = read_provenance_kernel_dataframes(output_path, kernel_set, to_level=level)
    # filtering the kernels to only selected graphs
    selected_kernels = kernels_df.loc[graphs.graph_file]

    X = selected_kernels
    if including_edge_type_counts:
        # putting the counts of PROV relation types (edge types) together with the kernels
        X = X.join(graphs[PROV_RELATION_NAMES])

    if sparse:
        # Converting the selected kernels into a sparse matrix
        X = pd_df_to_coo(X)
        # SVM works best with sparse CSR format and float64 dtype
        X = X.tocsr()

    y = graphs[y_column]
    return X, y


def score_accuracy_provenance_kernels(
    graphs: pd.DataFrame,
    output_path: Path,
    kernel_set: str = "summary",
    level: int = 0,
    y_column: str = "label",
    cv: int = 10,
    including_edge_type_counts: bool = False,
):
    print(f"> Testing {kernel_set} up to level-{level}:")
    X, y = load_provenance_kernel_ml_data(
        graphs, output_path, kernel_set, level, y_column, including_edge_type_counts
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
    )
    scores = cross_validate(gs, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    print(
        "  - Accuracy: %0.2f (+/- %0.2f)"
        % (scores["test_accuracy"].mean(), scores["test_accuracy"].std() * 2)
    )
    return scores


def test_prediction_on_ml_classifiers(
    X: pd.DataFrame, output_path: Path, y: pd.Series, cv_sets=10, test_prefix=None
):
    if cv_sets is None:
        cv = 10
        print("> Using 10-fold cross validation...")
    else:
        cv = cv_sets
        print(f"> Using {len(cv_sets)}x preselected train/test sets...")

    results = []
    for clf_name, clf in ML_CLASSIFIERS.items():
        method_id = clf_name if test_prefix is None else test_prefix + clf_name
        # load existing scorings
        scorings = load_experiment_scorings(output_path, method_id)

        if scorings is None:
            print("> Testing ML method:", method_id)
            timer = Timer()
            with timer:
                scores = cross_validate(clf, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            print(
                "  - Accuracy: %0.2f (+/- %0.2f) <-- %s"
                % (
                    scores["test_accuracy"].mean(),
                    scores["test_accuracy"].std() * 2,
                    clf_name,
                )
            )
            data = {
                score_type: scores[score_field]
                for score_type, score_field in zip(scoring, score_fields)
            }
            data["method"] = method_id
            data["time"] = timer.interval
            scorings = pd.DataFrame(data)
            save_experiment_scorings(output_path, method_id, scorings)

        results.append(scorings)

    return pd.concat(results, ignore_index=True)


def test_prediction_with_provenance_kernels(
    graphs: pd.DataFrame, output_path: Path, y_column: str, cv_sets=10
):
    if cv_sets is None:
        cv = 10
        print("> Using 10-fold cross validation...")
    else:
        cv = cv_sets
        print(f"> Using {len(cv_sets)}x preselected train/test sets...")

    results = []
    # Enumerating the provenance kernels to be tested
    kernels_levels = itertools.product(["FG", "FA"], range(6))
    for kernel_set, level in kernels_levels:
        method_id = f"{kernel_set}-{level}"

        # load existing scorings
        scorings = load_experiment_scorings(output_path, method_id)

        if scorings is None:
            # run the experiment
            scores = score_accuracy_provenance_kernels(
                graphs,
                output_path,
                kernel_set,
                level,
                y_column=y_column,
                cv=cv,
            )
            data = {
                score_type: scores[score_field]
                for score_type, score_field in zip(scoring, score_fields)
            }
            data["method"] = method_id
            timings_column_name = f"timings_{kernel_set}_{level}"
            try:
                data["time"] = graphs[timings_column_name].sum()
            except KeyError:
                data["time"] = 0.0
            scorings = pd.DataFrame(data)
            save_experiment_scorings(output_path, method_id, scorings)

        results.append(scorings)

    return pd.concat(results, ignore_index=True)


GRAKEL_KERNELS = {
    "GK-SPath": lambda: ShortestPath(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-EHist": lambda: EdgeHistogram(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-VHist": lambda: VertexHistogram(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-GSamp": lambda: GraphletSampling(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-WL-1": lambda: WeisfeilerLehman(
        n_iter=1, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-WL-2": lambda: WeisfeilerLehman(
        n_iter=2, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-WL-3": lambda: WeisfeilerLehman(
        n_iter=3, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-WL-4": lambda: WeisfeilerLehman(
        n_iter=4, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-WL-5": lambda: WeisfeilerLehman(
        n_iter=5, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-NH": lambda: NeighborhoodHash(
        n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-HC-1": lambda: HadamardCode(
        n_iter=1, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-HC-2": lambda: HadamardCode(
        n_iter=2, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-HC-3": lambda: HadamardCode(
        n_iter=3, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-HC-4": lambda: HadamardCode(
        n_iter=4, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-HC-5": lambda: HadamardCode(
        n_iter=5, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-NSPD": lambda: NeighborhoodSubgraphPairwiseDistance(
        normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-WL-OA-1": lambda: WeisfeilerLehmanOptimalAssignment(
        n_iter=1, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-WL-OA-2": lambda: WeisfeilerLehmanOptimalAssignment(
        n_iter=2, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-WL-OA-3": lambda: WeisfeilerLehmanOptimalAssignment(
        n_iter=3, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-WL-OA-4": lambda: WeisfeilerLehmanOptimalAssignment(
        n_iter=4, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-WL-OA-5": lambda: WeisfeilerLehmanOptimalAssignment(
        n_iter=5, n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),
    "GK-OddSth": lambda: OddSth(normalize=NORMALIZING_GRAPH_KERNELS),
}

NOT_TESTED = {
    "GK-ShortestPathA": lambda: ShortestPathAttr(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-RandomWalk": lambda: RandomWalk(
        n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),  # taking too long
    "GK-RandomWalkLabeled": lambda: RandomWalkLabeled(
        n_jobs=N_JOBS, normalize=NORMALIZING_GRAPH_KERNELS
    ),  # taking too long
    "GK-GraphHopper": lambda: GraphHopper(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-PyramidMatch": lambda: PyramidMatch(
        normalize=NORMALIZING_GRAPH_KERNELS
    ),  # Error with PG
    "GK-LovaszTheta": lambda: LovaszTheta(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-SvmTheta": lambda: SvmTheta(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-Propagation": lambda: Propagation(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-PropagationA": lambda: PropagationAttr(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-MScLaplacian": lambda: MultiscaleLaplacian(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-SubgraphMatching": lambda: SubgraphMatching(
        normalize=NORMALIZING_GRAPH_KERNELS
    ),  # taking too long
}


def test_prediction_with_generic_graph_kernels(
    graphs: pd.DataFrame,
    output_path: Path,
    y_column: str,
    cv_sets=None,
    ignore_kernels=None,
):
    if cv_sets is None:
        cv = 10
        print("> Using 10-fold cross validation...")
    else:
        cv = cv_sets
        print(f"> Using {len(cv_sets)}x preselected train/test sets...")
    if ignore_kernels is None:
        ignore_kernels = set()
    results = []
    for method_id, gk_class in GRAKEL_KERNELS.items():
        if method_id in ignore_kernels:
            logger.info("Skipping testing kernel: %s", method_id)
            continue

        # load existing scorings
        scorings = load_experiment_scorings(output_path, method_id)

        if scorings is None:
            # run the experiment
            logger.info("Testing graph kernel: %s", method_id)
            print("> Testing GraKeL kernel:", method_id)
            gk = gk_class()
            failed = False
            try:
                timer = Timer(timeout=TIMEOUT)
                with timer:
                    # TODO: break if timed out
                    # only time the kerneling cost
                    X = gk.fit_transform(graphs.grakel_graphs)
            except TimeoutException:
                failed = True
                print("*** TIMED OUT - %s ***" % method_id)
            except Exception as e:
                failed = True
                print(f"*** EXCEPTION - {method_id} ***\n{e}")

            if failed:
                # skip this, go to the next experiment
                continue

            clf = SVC(kernel="precomputed", gamma="scale", class_weight="balanced")
            gs = GridSearchCV(
                estimator=clf,
                param_grid={
                    "C": SVM_C_PARAMS,
                },
            )
            scores = cross_validate(
                gs, X, graphs[y_column], scoring=scoring, cv=cv, n_jobs=-1
            )
            print(
                "  - Accuracy: %0.2f (+/- %0.2f) <-- %s"
                % (
                    scores["test_accuracy"].mean(),
                    scores["test_accuracy"].std() * 2,
                    method_id,
                )
            )
            data = {
                score_type: scores[score_field]
                for score_type, score_field in zip(scoring, score_fields)
            }
            data["method"] = method_id
            data["time"] = timer.interval
            scorings = pd.DataFrame(data)
            save_experiment_scorings(output_path, method_id, scorings)

        results.append(scorings)
    return pd.concat(results, ignore_index=True)


def test_prediction_with_gnn(
    graphs: pd.DataFrame,
    y_column: str,
    dataset_folder: Path,
    output_path: Path,
    cv_sets,  # this is mandatory
):
    method_id = "GAT2"
    # load existing scorings
    scorings = load_experiment_scorings(output_path, method_id)

    if scorings is None:
        # run the experiment
        print("Testing GNN: %s" % method_id)

        print(f"> Converting {len(graphs)} PROV graphs to PyTorch Geometric data...")
        data_list, num_classes = convert_PROV_graphs_to_PyG_data(graphs, y_column, dataset_folder)
        acc_scores = []
        timings = []
        n_runs = len(cv_sets)
        n_epochs = 50
        print(f"> Using {n_runs}x preselected train/test sets...")
        print(f"> Number of training epochs: {n_epochs} (per run)")
        for train_idx, test_idx in cv_sets:
            train_list = itemgetter(*train_idx)(data_list)
            test_list = itemgetter(*test_idx)(data_list)
            train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_list, batch_size=32)
            # gnn_wrapper = GCNWrapper(num_classes, hidden_channels=64)
            gnn_wrapper = GATv2Wrapper(num_classes, edge_dim=data_list[0].edge_attr.shape[1], hidden_channels=64)
            timer = Timer(verbose=False)
            with timer:
                gnn_wrapper.train(train_loader, epochs=50)
            accuracy = gnn_wrapper.test(test_loader)
            acc_scores.append(accuracy)
            timings.append(timer.interval)
            print(f"  - Training time ({len(acc_scores):3d}/{n_runs}): {timer.interval:.2f}s – Accuracy: {accuracy:.2f}")

        scorings = pd.DataFrame({
            "accuracy": acc_scores,
            "method": method_id,
            "time": timings,
        })
        print(f"> Mean accuracy: {scorings.accuracy.mean():.2f}")
        print(f"> Mean duration: {scorings.time.mean():.2f}s")
        save_experiment_scorings(output_path, method_id, scorings)

    return scorings
