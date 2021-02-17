"""Common experiment code."""
import logging
from pathlib import Path
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
from sklearn.model_selection import cross_validate, GridSearchCV, RepeatedStratifiedKFold
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
TIMEOUT = 15 * 60  # 15 mins in seconds
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


def get_fixed_CV_sets(X, y, n_splits=10, n_repeats=10, random_state=None):
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    return list(rskf.split(X, y))


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
    timings = timings.rename(lambda col_name: "timings_" + col_name, axis="columns", copy=False)
    return graphs.join(timings, on="graph_file")


def read_kernel_dataframes(
    kernels_folder: Path, kernel_set: str, from_level: int = 0, to_level: int = None
) -> pd.DataFrame:
    kernel_df_filepath = (
        kernels_folder / "kernels" / f"{kernel_set}_{from_level}.pickled"
    )
    kernels_df = pd.read_pickle(kernel_df_filepath)

    if to_level is not None:
        level = from_level
        while level < to_level:
            level += 1  # reading the next level
            kernel_df_filepath = (
                kernels_folder / "kernels" / f"{kernel_set}_{level}.pickled"
            )
            logger.debug("Reading pickled dataframe: %s", kernel_df_filepath)
            df = pd.read_pickle(kernel_df_filepath)
            kernels_df = kernels_df.join(df)

    return kernels_df


def score_accuracy_kernels(
    graphs: pd.DataFrame,
    kernels_folder: Path,
    kernel_set: str = "summary",
    level: int = 0,
    y_column: str = "label",
    cv: int = 10,
    including_edge_type_counts: bool = False,
):
    print(f"> Testing {kernel_set} up to level-{level}:")
    print("  - Reading kernels...")
    kernels_df = read_kernel_dataframes(kernels_folder, kernel_set, to_level=level)
    # filtering the kernels to only selected graphs
    selected_kernels = kernels_df.loc[graphs.graph_file]

    print(f"  - Calculating accuracy scores with SVM...")
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
        "    Accuracy: %0.2f (+/- %0.2f)"
        % (scores["test_accuracy"].mean(), scores["test_accuracy"].std() * 2)
    )
    return scores


def test_prediction_on_classifiers(
    X: pd.DataFrame, y: pd.Series, cv_sets=10, test_prefix=None
):
    if cv_sets is None:
        cv = 10
        print("Using 10-fold cross validation...")
    else:
        cv = cv_sets
        print(f"Using {len(cv_sets)}x preselected train/test sets...")

    results = pd.DataFrame()
    for clf_name, clf in ML_CLASSIFIERS.items():
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
        data["method"] = clf_name if test_prefix is None else test_prefix + clf_name
        data["time"] = timer.interval
        results = results.append(pd.DataFrame(data), ignore_index=True)
    return results


def test_prediction_on_kernels(
    graphs: pd.DataFrame, kernels_folder: Path, y_column: str, cv_sets=10
):
    if cv_sets is None:
        cv = 10
        print("Using 10-fold cross validation...")
    else:
        cv = cv_sets
        print(f"Using {len(cv_sets)}x preselected train/test sets...")

    results = pd.DataFrame()
    # Testing the provenance kernels
    for level in range(6):
        for kernel_set in ["FG", "DG", "FA", "DA"]:
            method_id = f"{kernel_set}-{level}"
            scores = score_accuracy_kernels(
                graphs, kernels_folder, kernel_set, level, y_column=y_column, cv=cv,
            )
            data = {
                score_type: scores[score_field]
                for score_type, score_field in zip(scoring, score_fields)
            }
            data["method"] = method_id
            data["time"] = graphs[f"timings_{kernel_set}_{level}"].sum()
            results = results.append(pd.DataFrame(data), ignore_index=True)

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
    "GK-OddSth": lambda: OddSth(normalize=NORMALIZING_GRAPH_KERNELS),
    "GK-SubgraphMatching": lambda: SubgraphMatching(
        normalize=NORMALIZING_GRAPH_KERNELS
    ),  # taking too long
}


def test_prediction_on_Grakel_kernels(
    graphs: pd.DataFrame, y_column: str, cv_sets=None, ignore_kernels=None
):
    if cv_sets is None:
        cv = 10
        print("Using 10-fold cross validation...")
    else:
        cv = cv_sets
        print(f"Using {len(cv_sets)}x preselected train/test sets...")
    if ignore_kernels is None:
        ignore_kernels = set()
    results = pd.DataFrame()
    for method_id, gk_class in GRAKEL_KERNELS.items():
        if method_id in ignore_kernels:
            logger.info("Skipping testing kernel: %s", method_id)
            continue

        logger.info("Testing graph kernel: %s", method_id)
        print("Testing GraKeL kernel:", method_id)
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

        if not has_timed_out:
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
                "Accuracy: %0.2f (+/- %0.2f) <-- %s"
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
            results = results.append(pd.DataFrame(data), ignore_index=True)
    return results
