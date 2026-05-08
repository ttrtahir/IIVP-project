"""
Module 2 — Feature Extraction
KEN3238 IIVP2026 Kaggle Challenge
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from sklearn.decomposition import PCA

SEED = 42
CACHE_DIR = "cache"

# HOG defaults — tuned for 32×32 digit images
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

# Histogram defaults
HIST_BINS = 32

PCA_VARIANCE = 0.95  # keep 95% of variance


# ---------------------------------------------------------------------------
# HOG features
# ---------------------------------------------------------------------------

def _extract_hog_single(img, pixels_per_cell, cells_per_block, orientations):
    """Extract HOG feature vector from a single (H, W) float32 image."""
    features, _ = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    return features


def extract_hog(
    images,
    pixels_per_cell=HOG_PIXELS_PER_CELL,
    cells_per_block=HOG_CELLS_PER_BLOCK,
    orientations=HOG_ORIENTATIONS,
):
    """
    Extract HOG descriptors for every image.

    Parameters
    ----------
    images : ndarray [N, H, W]  float32

    Returns
    -------
    features : ndarray [N, D]  float32
    """
    feats = []
    for img in images:
        f = _extract_hog_single(img, pixels_per_cell, cells_per_block, orientations)
        feats.append(f)
    features = np.array(feats, dtype=np.float32)
    print(f"HOG features: {features.shape}  "
          f"(pixels_per_cell={pixels_per_cell}, orientations={orientations})")
    return features


# ---------------------------------------------------------------------------
# Pixel intensity histogram features
# ---------------------------------------------------------------------------

def extract_histogram(images, bins=HIST_BINS):
    """
    Compute a normalized intensity histogram per image as a feature vector.

    Parameters
    ----------
    images : ndarray [N, H, W]  float32

    Returns
    -------
    features : ndarray [N, bins]  float32
    """
    feats = []
    for img in images:
        hist, _ = np.histogram(img.ravel(), bins=bins, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))
    features = np.array(feats, dtype=np.float32)
    print(f"Histogram features: {features.shape}  (bins={bins})")
    return features


# ---------------------------------------------------------------------------
# PCA dimensionality reduction
# ---------------------------------------------------------------------------

def fit_pca(features, variance=PCA_VARIANCE):
    """Fit PCA to training features, retaining `variance` fraction of variance."""
    pca = PCA(n_components=variance, random_state=SEED, svd_solver="full")
    pca.fit(features)
    n_components = pca.n_components_
    print(f"PCA: {features.shape[1]} → {n_components} components "
          f"({variance*100:.0f}% variance retained)")
    return pca


def apply_pca(features, pca):
    """Transform features with a fitted PCA object."""
    return pca.transform(features).astype(np.float32)


# ---------------------------------------------------------------------------
# HOG visualisation
# ---------------------------------------------------------------------------

def visualize_hog(images, labels, n=5, save_path="outputs/hog_visualization.png"):
    """Save side-by-side HOG gradient images for `n` samples."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(images), n, replace=False)

    fig, axes = plt.subplots(n, 2, figsize=(5, 2.5 * n))
    for row, idx in enumerate(idxs):
        img = images[idx]
        _, hog_img = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm="L2-Hys",
            visualize=True,
            feature_vector=True,
        )
        hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Label {labels[idx]}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(hog_img_rescaled, cmap="inferno")
        axes[row, 1].set_title("HOG", fontsize=9)
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"HOG visualization saved → {save_path}")


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def extract_features(images, labels=None, method="hog", pca=None):
    """
    Extract feature vectors from preprocessed images.

    Parameters
    ----------
    images  : ndarray [N, H, W]  float32
    labels  : ndarray [N]  (optional, used only for visualization)
    method  : 'hog' | 'histogram' | 'combined'
    pca     : fitted PCA object (or None to skip PCA)

    Returns
    -------
    features : ndarray [N, D]  float32
    """
    if method == "hog":
        features = extract_hog(images)
    elif method == "histogram":
        features = extract_histogram(images)
    elif method == "combined":
        hog_feats  = extract_hog(images)
        hist_feats = extract_histogram(images)
        features = np.concatenate([hog_feats, hist_feats], axis=1)
        print(f"Combined features: {features.shape}")
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'hog', 'histogram', or 'combined'.")

    if pca is not None:
        features = apply_pca(features, pca)
        print(f"After PCA: {features.shape}")

    return features


def build_features(
    X_train, y_train, X_val, X_test,
    method="hog",
    use_pca=False,
    visualize=True,
):
    """
    Full feature extraction pipeline with caching.

    Returns
    -------
    X_train_feat, X_val_feat, X_test_feat : ndarray [N, D]  float32
    pca_model                              : fitted PCA or None
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    suffix = method + ("_pca" if use_pca else "")
    cache = {
        "X_train": os.path.join(CACHE_DIR, f"feat_train_{suffix}.npy"),
        "X_val":   os.path.join(CACHE_DIR, f"feat_val_{suffix}.npy"),
        "X_test":  os.path.join(CACHE_DIR, f"feat_test_{suffix}.npy"),
        "pca":     os.path.join(CACHE_DIR, f"pca_{suffix}.npy"),
    }

    if all(os.path.exists(cache[k]) for k in ["X_train", "X_val", "X_test"]):
        print(f"Loading cached features ({suffix})...")
        X_tr = np.load(cache["X_train"])
        X_v  = np.load(cache["X_val"])
        X_te = np.load(cache["X_test"])
        pca_model = None
        print(f"Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")
        return X_tr, X_v, X_te, pca_model

    # Visualise HOG on 5 samples from train
    if visualize:
        visualize_hog(X_train, y_train)

    # Extract raw features
    print("\nExtracting train features...")
    X_tr_raw = extract_features(X_train, method=method)
    print("Extracting val features...")
    X_v_raw  = extract_features(X_val,   method=method)
    print("Extracting test features...")
    X_te_raw = extract_features(X_test,  method=method) \
        if len(X_test) > 0 else np.empty((0, X_tr_raw.shape[1]), dtype=np.float32)

    # Optional PCA (fit on train only, transform all)
    pca_model = None
    if use_pca:
        print("\nFitting PCA on training features...")
        pca_model = fit_pca(X_tr_raw)
        X_tr = apply_pca(X_tr_raw, pca_model)
        X_v  = apply_pca(X_v_raw,  pca_model)
        X_te = apply_pca(X_te_raw, pca_model) if len(X_te_raw) > 0 else X_te_raw
    else:
        X_tr, X_v, X_te = X_tr_raw, X_v_raw, X_te_raw

    print(f"\nFinal feature shapes — Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

    # Cache
    np.save(cache["X_train"], X_tr)
    np.save(cache["X_val"],   X_v)
    np.save(cache["X_test"],  X_te)
    print(f"Cached feature arrays in {CACHE_DIR}/")

    return X_tr, X_v, X_te, pca_model


if __name__ == "__main__":
    from module1_preprocessing import load_and_preprocess
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_preprocess()
    build_features(X_train, y_train, X_val, X_test, method="hog", use_pca=False)
