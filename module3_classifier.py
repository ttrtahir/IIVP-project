"""
Module 3 — Classification, Evaluation & Submission
KEN3238 IIVP2026 Kaggle Challenge
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import joblib

SEED = 42
OUTPUTS_DIR = "outputs"
CACHE_DIR = "cache"

# Class names: 0-9 Latin, 10-19 Hindi
LABEL_NAMES = (
    [f"Latin-{i}" for i in range(10)] +
    [f"Hindi-{i}" for i in range(10)]
)


# ---------------------------------------------------------------------------
# SVM training with grid search
# ---------------------------------------------------------------------------

SEARCH_SUBSET = 3000   # samples used for grid search CV (stratified)
SEARCH_CV_FOLDS = 3   # folds during search; final model trains on full data


def train_svm(X_train, y_train):
    """
    Hyperparameter search on a stratified subsample, then retrain the winner
    on the full training set.

    Linear kernel: uses LinearSVC (orders of magnitude faster than SVC).
    RBF kernel:    uses SVC with gamma='scale' only (rarely worse than 'auto').

    Grid:  linear  × C in {0.1, 1, 10, 100}     →  4 candidates
           rbf     × C in {0.1, 1, 10, 100}     →  4 candidates
    Total: 8 candidates × 3 folds = 24 fits on SEARCH_SUBSET samples.
    """
    # ---- stratified subsample for search ----
    n = min(SEARCH_SUBSET, len(X_train))
    rng = np.random.default_rng(SEED)
    # one pass of stratified sampling
    classes, counts = np.unique(y_train, return_counts=True)
    per_class = max(1, n // len(classes))
    idxs = []
    for cls in classes:
        cls_idxs = np.where(y_train == cls)[0]
        chosen = rng.choice(cls_idxs, min(per_class, len(cls_idxs)), replace=False)
        idxs.append(chosen)
    idxs = np.concatenate(idxs)
    X_sub, y_sub = X_train[idxs], y_train[idxs]
    print(f"\nGrid search on {len(X_sub)} samples ({SEARCH_CV_FOLDS}-fold CV)...")

    cv = StratifiedKFold(n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=SEED)
    C_values = [0.1, 1, 10, 100]
    best_score, best_params = -1, None

    # Linear candidates (LinearSVC — much faster than SVC kernel='linear')
    for C in C_values:
        lsvc = LinearSVC(C=C, max_iter=2000, random_state=SEED)
        fold_scores = []
        for train_idx, val_idx in cv.split(X_sub, y_sub):
            lsvc.fit(X_sub[train_idx], y_sub[train_idx])
            fold_scores.append(accuracy_score(y_sub[val_idx], lsvc.predict(X_sub[val_idx])))
        score = float(np.mean(fold_scores))
        print(f"  linear  C={C:<5} CV acc={score:.4f}")
        if score > best_score:
            best_score, best_params = score, {"kernel": "linear", "C": C}

    # RBF candidates
    for C in C_values:
        svc = SVC(kernel="rbf", C=C, gamma="scale", random_state=SEED)
        fold_scores = []
        for train_idx, val_idx in cv.split(X_sub, y_sub):
            svc.fit(X_sub[train_idx], y_sub[train_idx])
            fold_scores.append(accuracy_score(y_sub[val_idx], svc.predict(X_sub[val_idx])))
        score = float(np.mean(fold_scores))
        print(f"  rbf     C={C:<5} CV acc={score:.4f}")
        if score > best_score:
            best_score, best_params = score, {"kernel": "rbf", "C": C, "gamma": "scale"}

    print(f"Best params : {best_params}")
    print(f"Best CV acc : {best_score:.4f}")

    # ---- retrain winner on full training set ----
    print("Retraining on full training set...")
    if best_params["kernel"] == "linear":
        # Wrap LinearSVC in CalibratedClassifierCV to get a predict() interface
        # identical to SVC (needed for consistent evaluation calls)
        base = LinearSVC(C=best_params["C"], max_iter=2000, random_state=SEED)
        model = CalibratedClassifierCV(base, cv=3)
    else:
        model = SVC(kernel="rbf", C=best_params["C"], gamma="scale",
                    random_state=SEED, decision_function_shape="ovr")
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, "svm_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    return model


def load_svm():
    """Load a previously saved SVM model."""
    model_path = os.path.join(CACHE_DIR, "svm_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model at {model_path}. Run train_and_evaluate() first.")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# KNN baseline
# ---------------------------------------------------------------------------

def train_knn(X_train, y_train, k_values=(3, 5, 7)):
    """
    Train KNN classifiers for several k values; return the best one on training
    cross-val accuracy (quick CV, 3-fold) plus a dict of all k→accuracy.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_k, best_score, best_model = None, -1, None
    scores = {}

    print("\nTraining KNN baselines...")
    for k in k_values:
        param_grid = {"n_neighbors": [k]}
        knn = KNeighborsClassifier(n_jobs=-1)
        gs = GridSearchCV(knn, param_grid, cv=cv, scoring="accuracy")
        gs.fit(X_train, y_train)
        scores[k] = gs.best_score_
        print(f"  k={k}: CV accuracy = {gs.best_score_:.4f}")
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_k = k
            best_model = gs.best_estimator_

    print(f"Best KNN: k={best_k} (CV acc={best_score:.4f})")
    return best_model, scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, split_name="Validation"):
    """
    Compute accuracy, precision, recall, F1 (macro).
    Print per-class report + Latin vs Hindi breakdown.
    Returns (y_pred, metrics_dict).
    """
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"\n--- {split_name} Evaluation ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}  (macro)")
    print(f"Recall    : {rec:.4f}  (macro)")
    print(f"F1-score  : {f1:.4f}  (macro)")

    # Per-class report
    present_labels = sorted(np.unique(np.concatenate([y, y_pred])))
    present_names  = [LABEL_NAMES[i] if i < len(LABEL_NAMES) else str(i) for i in present_labels]
    print("\nPer-class report:")
    print(classification_report(y, y_pred, labels=present_labels, target_names=present_names,
                                 zero_division=0))

    # Latin vs Hindi breakdown
    latin_mask = y < 10
    hindi_mask = y >= 10
    if latin_mask.any():
        lat_acc = accuracy_score(y[latin_mask], y_pred[latin_mask])
        print(f"Latin-digit accuracy : {lat_acc:.4f}")
    if hindi_mask.any():
        hin_acc = accuracy_score(y[hindi_mask], y_pred[hindi_mask])
        print(f"Hindi-digit accuracy : {hin_acc:.4f}")

    # Most confused pairs
    cm = confusion_matrix(y, y_pred, labels=present_labels)
    _report_confused_pairs(cm, present_labels, present_names)

    return y_pred, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _report_confused_pairs(cm, labels, names, top_n=5):
    """Print the top-N most confused label pairs."""
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat = cm_no_diag.ravel()
    top_idxs = np.argsort(flat)[::-1][:top_n]
    print(f"\nTop-{top_n} most confused pairs (true → predicted : count):")
    n = len(labels)
    for idx in top_idxs:
        i, j = divmod(idx, n)
        if cm_no_diag[i, j] > 0:
            print(f"  {names[i]} → {names[j]} : {cm_no_diag[i, j]}")


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, save_path="outputs/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    present_names  = [LABEL_NAMES[i] if i < len(LABEL_NAMES) else str(i) for i in present_labels]
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    n = len(present_labels)
    fig_size = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=present_names, yticklabels=present_names,
        linewidths=0.3, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Kaggle submission
# ---------------------------------------------------------------------------

def generate_submission(model, X_test, test_ids, save_path="outputs/submission.csv"):
    """Run inference on X_test and write a Kaggle submission CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if len(X_test) == 0:
        print("No test images found — skipping submission generation.")
        return

    y_pred = model.predict(X_test)
    df = pd.DataFrame({"image_id": test_ids, "label": y_pred})
    df.to_csv(save_path, index=False)
    print(f"Submission saved → {save_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, test_ids=None,
    run_knn_baseline=True,
):
    """
    Full pipeline:
      1. Train SVM (grid search + CV)
      2. Evaluate on validation set
      3. Plot confusion matrix
      4. Optionally compare with KNN baseline
      5. Generate Kaggle submission CSV

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val,   y_val   : validation features and labels
    X_test           : test features (no labels)
    test_ids         : list of test image IDs for submission CSV
    run_knn_baseline : whether to also train/evaluate KNN

    Returns
    -------
    svm_model : fitted SVM
    metrics   : dict with accuracy, precision, recall, f1
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ---- SVM ----
    svm_model = train_svm(X_train, y_train)
    y_pred_val, svm_metrics = evaluate(svm_model, X_val, y_val, split_name="Validation (SVM)")
    plot_confusion_matrix(y_val, y_pred_val)

    # ---- KNN baseline ----
    if run_knn_baseline:
        knn_model, knn_scores = train_knn(X_train, y_train)
        y_pred_knn, knn_metrics = evaluate(knn_model, X_val, y_val, split_name="Validation (KNN best-k)")
        print("\n--- SVM vs KNN Comparison ---")
        print(f"SVM F1 : {svm_metrics['f1']:.4f}")
        print(f"KNN F1 : {knn_metrics['f1']:.4f}")

    # ---- Kaggle submission ----
    if test_ids is None:
        test_ids = [f"{i}.png" for i in range(len(X_test))]
    generate_submission(svm_model, X_test, test_ids)

    return svm_model, svm_metrics


if __name__ == "__main__":
    from module1_preprocessing import load_and_preprocess
    from module2_features import build_features

    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_preprocess()
    X_tr_feat, X_v_feat, X_te_feat, _ = build_features(
        X_train, y_train, X_val, X_test, method="hog"
    )
    train_and_evaluate(X_tr_feat, y_train, X_v_feat, y_val, X_te_feat, test_ids)
