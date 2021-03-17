"""Common experiment code."""
import itertools
import logging
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    make_scorer,
)

from scipy.sparse import coo_matrix, hstack

from scripts.data.common import PROV_RELATION_NAMES
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

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score, average="micro"),
    "recall": make_scorer(recall_score, average="micro"),
    "precision": make_scorer(precision_score, average="micro"),
    # "roc_auc": make_scorer(roc_auc_score)
}
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


def read_kernel_dataframes(
    output_path: Path, kernel_set: str, from_level: int = 0, to_level: int = None
) -> pd.DataFrame:
    kernel_df_filepath = output_path / "kernels" / f"{kernel_set}_{from_level}.pickled"
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


def score_accuracy_kernels(
    graphs: pd.DataFrame,
    output_path: Path,
    kernel_set: str = "summary",
    level: int = 0,
    y_column: str = "label",
    cv: int = 10,
    including_edge_type_counts: bool = False,
):
    print(f"> Testing {kernel_set} up to level-{level}:")
    print("  - Reading kernels...")
    kernels_df = read_kernel_dataframes(output_path, kernel_set, to_level=level)
    # filtering the kernels to only selected graphs
    selected_kernels = kernels_df.loc[graphs.graph_file]

    # Converting the selected kernels into a sparse matrix
    X = pd_df_to_coo(selected_kernels)
    if including_edge_type_counts:
        # putting the counts of PROV relation types (edge types) together with the kernels
        edge_type_features = coo_matrix(graphs[PROV_RELATION_NAMES])
        X = hstack([edge_type_features, X])
    # SVM works best with sparse CSR format and float64 dtype
    X = X.tocsr()
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
    scores = cross_validate(gs, X, graphs[y_column], scoring=scoring, cv=cv, n_jobs=-1)
    print(
        "  - Accuracy: %0.2f (+/- %0.2f)"
        % (scores["test_accuracy"].mean(), scores["test_accuracy"].std() * 2)
    )
    return scores


def test_prediction_on_classifiers(
    X: pd.DataFrame, output_path: Path, y: pd.Series, cv_sets=10, test_prefix=None
):
    if cv_sets is None:
        cv = 10
        print("> Using 10-fold cross validation...")
    else:
        cv = cv_sets
        print(f"> Using {len(cv_sets)}x preselected train/test sets...")

    results = pd.DataFrame()
    for clf_name, clf in ML_CLASSIFIERS.items():
        method_id = clf_name if test_prefix is None else test_prefix + clf_name
        # load existing scorings
        scorings = load_experiment_scorings(output_path, method_id)

        if scorings is None:
            timer = Timer()
            with timer:
                scores = cross_validate(clf, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            print(
                "Accuracy: %0.2f (+/- %0.2f) <-- %s"
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

        results = results.append(scorings, ignore_index=True)

    return results


def test_prediction_on_kernels(
    graphs: pd.DataFrame, output_path: Path, y_column: str, cv_sets=10
):
    if cv_sets is None:
        cv = 10
        print("> Using 10-fold cross validation...")
    else:
        cv = cv_sets
        print(f"> Using {len(cv_sets)}x preselected train/test sets...")

    results = pd.DataFrame()
    # Enumerating the provenance kernels to be tested
    kernels_levels = itertools.chain(
        # Forward propagation kernels
        itertools.product(["FG", "DG", "TG", "FA", "DA", "TA"], range(6)),
        # Backward propagation kernels
        itertools.product(["TG", "TA"], range(-1, -6, -1)),
    )
    for kernel_set, level in kernels_levels:
        method_id = f"{kernel_set}-{level}"

        # load existing scorings
        scorings = load_experiment_scorings(output_path, method_id)

        if scorings is None:
            # run the experiment
            scores = score_accuracy_kernels(
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

        results = results.append(scorings, ignore_index=True)

    return results


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


def test_prediction_on_Grakel_kernels(
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
    results = pd.DataFrame()
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
            has_timed_out = False
            try:
                timer = Timer(timeout=TIMEOUT)
                with timer:
                    # TODO: break if timed out
                    # only time the kerneling cost
                    X = gk.fit_transform(graphs.grakel_graphs)
            except TimeoutException:
                has_timed_out = True
                print("*** TIMED OUT - %s ***" % method_id)

            if has_timed_out:
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

        results = results.append(scorings, ignore_index=True)
    return results
