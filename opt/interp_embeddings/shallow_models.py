import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sklearn.base import ClassifierMixin
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.exceptions import NotFittedError


Array = np.ndarray


@dataclass
class ModelEvaluation:
    model_name: str  # "svm" or "random_forest"
    original_label_values: Tuple[Any, Any]
    best_params: Dict[str, Any]
    cv_scores: Dict[str, np.ndarray]
    test_classification_report: Dict[str, Any]
    test_confusion_matrix: Array
    test_roc_auc: float
    test_average_precision: float
    y_test: Array = field(repr=False)
    y_pred: Array = field(repr=False)
    y_proba: Array = field(repr=False)


def encode_binary_label(
    y: Union[Sequence, pd.Series],
    positive_label: Optional[Any] = None,
) -> Tuple[np.ndarray, Tuple[Any, Any]]:
    """
    Encode a binary label sequence into 0/1 array.
    Returns (y_binary, (negative_label, positive_label)).
    If `positive_label` is provided, that value is mapped to 1; the other unique value is 0.
    If not provided, the two unique values are sorted and the second is taken as positive.
    """
    arr = pd.Series(y).reset_index(drop=True)
    uniques = arr.dropna().unique()
    if len(uniques) != 2:
        raise ValueError(f"Expected exactly 2 distinct label values, got {uniques}")

    if positive_label is not None:
        if positive_label not in uniques:
            raise ValueError(
                f"Provided positive_label {positive_label} not in label values {uniques}"
            )
        neg, _ = (u for u in uniques if u != positive_label), positive_label
        neg_label = next(neg)
        pos_label = positive_label
    else:
        sorted_vals = sorted(uniques, key=lambda x: (str(type(x)), str(x)))
        neg_label, pos_label = sorted_vals[0], sorted_vals[1]

    y_binary = arr.apply(lambda v: 1 if v == pos_label else 0).to_numpy()
    return y_binary, (neg_label, pos_label)


def build_base_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Create base (untuned) pipelines for SVM and Random Forest.
    """
    svm_pipeline = make_pipeline(
        StandardScaler(),
        SVC(probability=True, class_weight="balanced", random_state=random_state),
    )
    rf_pipeline = make_pipeline(
        RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=-1
        )
    )
    return {"svm": svm_pipeline, "random_forest": rf_pipeline}


def get_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    return {
        "svm": {
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", "auto"],
        },
        "random_forest": {
            "randomforestclassifier__n_estimators": [100, 300],
            "randomforestclassifier__max_depth": [None, 10, 30],
            "randomforestclassifier__min_samples_leaf": [1, 2],
            "randomforestclassifier__max_features": ["sqrt", "log2"],
        },
    }


def tune_model(
    X: Array,
    y: Array,
    pipeline: Pipeline,
    param_grid: Dict[str, List[Any]],
    cv: int = 5,
    scoring: str = "f1",
    n_jobs: int = -1,
    verbose: int = 0,
) -> GridSearchCV:
    """
    Grid-search tune the given pipeline. Returns the fitted GridSearchCV.
    """
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0),
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=False,
    )
    grid.fit(X, y)
    return grid


def cross_validated_scores(
    estimator: ClassifierMixin,
    X: Array,
    y: Array,
    cv: int = 5,
    scoring: Sequence[str] = ("f1", "accuracy", "precision", "recall"),
) -> Dict[str, np.ndarray]:
    """
    Compute cross-validated test scores on training data for given estimator.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    res = cross_validate(
        estimator,
        X,
        y,
        cv=cv_splitter,
        scoring=list(scoring),
        return_train_score=False,
        n_jobs=-1,
    )
    return {metric: res[metric] for metric in res if metric.startswith("test_")}


def evaluate_on_test(
    estimator: ClassifierMixin,
    X_test: Array,
    y_test: Array,
    label_names: Tuple[str, str],
) -> Tuple[Array, Array, Dict[str, Any], Array, float, float]:
    """
    Evaluate a fitted estimator on test split. Returns y_pred, y_proba, classification report dict,
    confusion matrix, ROC AUC, average precision.
    """
    try:
        y_pred = estimator.predict(X_test)
    except NotFittedError:
        raise

    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_test)[:, 1]
    else:
        if hasattr(estimator, "decision_function"):
            df = estimator.decision_function(X_test)
            y_proba = (df - df.min()) / (df.max() - df.min())
        else:
            y_proba = np.zeros_like(y_test, dtype=float)

    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = (
        roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else float("nan")
    )
    avg_prec = (
        average_precision_score(y_test, y_proba)
        if len(np.unique(y_test)) == 2
        else float("nan")
    )
    return y_pred, y_proba, report, cm, roc_auc, avg_prec


def plot_decision_boundary_2d(
    embedding_2d: Array,
    binary_labels: Array,
    title: str,
    label_names: Tuple[str, str],
    resolution: int = 300,
) -> None:
    """
    Fit an RBF SVM on 2D embedding and plot decision boundary.
    """
    vis_clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", probability=True, gamma="scale", random_state=0),
    )
    vis_clf.fit(embedding_2d, binary_labels)

    x_min, x_max = embedding_2d[:, 0].min() - 1, embedding_2d[:, 0].max() + 1
    y_min, y_max = embedding_2d[:, 1].min() - 1, embedding_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = vis_clf.predict(grid_points).reshape(xx.shape)

    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.2, levels=[-0.5, 0.5, 1.5], cmap="bwr")
    sc = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=binary_labels,
        edgecolor="k",
        cmap="bwr",
        s=35,
        alpha=0.85,
    )
    plt.legend(handles=sc.legend_elements()[0], labels=list(label_names))
    plt.title(title)
    plt.xlabel("Embedding dim 1")
    plt.ylabel("Embedding dim 2")
    plt.tight_layout()
    plt.show()


def train_compare_binary(
    X: Array,
    y_raw: Union[Sequence, pd.Series],
    positive_label: Optional[Any] = None,
    embedding_2d: Optional[Array] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 5,
    scoring: str = "f1",
    visualize_embedding: bool = False,
) -> Dict[str, ModelEvaluation]:
    """
    Train, tune, and evaluate SVM and Random Forest on one binary label.
    Returns dict with keys "svm" and "random_forest" mapping to their ModelEvaluation.
    """
    y_binary, (neg_label, pos_label) = encode_binary_label(
        y_raw, positive_label=positive_label
    )
    label_names = (str(neg_label), str(pos_label))

    # train/test split preserving balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, stratify=y_binary, random_state=random_state
    )

    pipelines = build_base_pipelines(random_state=random_state)
    param_grids = get_param_grids()

    evaluations: Dict[str, ModelEvaluation] = {}

    for model_key in ("svm", "random_forest"):
        grid = tune_model(
            X_train,
            y_train,
            pipeline=pipelines[model_key],
            param_grid=param_grids[model_key],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )
        best = grid.best_estimator_

        cv_scores = cross_validated_scores(best, X_train, y_train, cv=cv)

        y_pred, y_proba, report, cm, roc_auc, avg_prec = evaluate_on_test(
            best, X_test, y_test, label_names=label_names
        )

        eval_obj = ModelEvaluation(
            model_name=model_key,
            original_label_values=(neg_label, pos_label),
            best_params=grid.best_params_,
            cv_scores=cv_scores,
            test_classification_report=report,
            test_confusion_matrix=cm,
            test_roc_auc=roc_auc,
            test_average_precision=avg_prec,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
        )
        evaluations[model_key] = eval_obj

        if visualize_embedding and embedding_2d is not None:
            plot_decision_boundary_2d(
                embedding_2d,
                y_binary,
                title=f"{model_key.upper()} decision boundary",
                label_names=label_names,
            )

    return evaluations


def annotate_with_predictions(
    df: pd.DataFrame,
    X: Array,
    evals: Dict[str, ModelEvaluation],
    base_label_name: str,
) -> pd.DataFrame:
    """
    Retrain best estimators on full data and annotate predictions/confidences.
    """
    y_raw = df[base_label_name]
    y_binary, (neg_label, pos_label) = encode_binary_label(y_raw)

    pipelines = build_base_pipelines()
    annotated = df.copy()

    for model_key, eval_obj in evals.items():
        pipeline = pipelines[model_key]
        pipeline.set_params(**eval_obj.best_params)
        pipeline.fit(X, y_binary)
        pred = pipeline.predict(X)
        proba = (
            pipeline.predict_proba(X)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )

        suffix = f"{base_label_name}_{model_key}"
        annotated[f"{suffix}_pred"] = np.where(pred == 1, pos_label, neg_label)
        if proba is not None:
            annotated[f"{suffix}_confidence_positive"] = proba

    # agreement between the two if both present
    if "svm" in evals and "random_forest" in evals:
        key1 = f"{base_label_name}_svm_pred"
        key2 = f"{base_label_name}_random_forest_pred"
        if key1 in annotated.columns and key2 in annotated.columns:
            annotated[f"{base_label_name}_agreement"] = (
                annotated[key1] == annotated[key2]
            )

    return annotated


def linear_separability_diagnostics(
    X: np.ndarray, y: np.ndarray, tol: float = 1e-6, margin: int = 1e6
):
    clf = SVC(kernel="linear", C=margin)  # large C ~ hard margin
    clf.fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_[0]

    # functional margins
    functional_margins = y * (X @ w + b)  # not normalized
    min_func_margin = functional_margins.min()
    # geometric margin
    norm_w = np.linalg.norm(w)
    geometric_margins = functional_margins / norm_w
    min_geo_margin = geometric_margins.min()

    # hinge losses
    hinge_losses = np.maximum(0, 1 - functional_margins)
    avg_hinge = hinge_losses.mean()

    # violations (strictly less than 1 - tol)
    violations = np.sum(functional_margins < 1 - tol)

    # training accuracy
    train_pred = clf.predict(X)
    accuracy = np.mean(train_pred == y)

    return {
        "training_accuracy": accuracy,
        "min_functional_margin": min_func_margin,
        "min_geometric_margin": min_geo_margin,
        "average_hinge_loss": avg_hinge,
        "violations": int(violations),
        "n_support_vectors": clf.n_support_.sum(),
    }


def prepare_aromaticity_paper_example():
    embedding_files = glob.glob("compas_2/*.csv")
    filepath = embedding_files[3]
    compas = pd.read_csv(filepath)
    antiaromatic = compas.cyclobutadiene.values > 0
    hidden_size = 768
    X = compas[[str(i) for i in range(hidden_size)]].values
    return X, antiaromatic


def prepare_condensation_paper_example():
    embedding_files = glob.glob("compas/mean_compas_*.csv")
    filepath = embedding_files[-1]
    compas = pd.read_csv(filepath)
    hidden_size = 768
    X = compas[[str(i) for i in range(hidden_size)]].values
    return X, compas["dataset"].values == 1


if __name__ == "__main__":
    # X, y = prepare_condensation_paper_example()
    X, y = prepare_aromaticity_paper_example()
    X_embedded = None

    # linear_sep = linear_separability_diagnostics(X, y)
    # pprint.pprint(linear_sep)

    evaluations = train_compare_binary(
        X=X,
        y_raw=y,
        positive_label=None,
        embedding_2d=X_embedded,
        test_size=0.2,
        random_state=42,
        cv=5,
        scoring="f1",
        visualize_embedding=False,
    )

    for model, eval in evaluations.items():
        print(evaluations.keys())
        print(
            f"{model} \n {eval.test_confusion_matrix} \n AUROC: {eval.test_average_precision}"
        )
