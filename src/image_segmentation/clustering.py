import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from .features import make_features


def cluster_image(image: np.ndarray, method: str, feature_mode: str, clusters: int = 9, eps: float = .05, min_samples: int = 3) -> np.ndarray:
    features = make_features(image, feature_mode)
    if method == "dbscan": labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(features)
    elif method == "kmeans": labels = KMeans(n_clusters=clusters, n_init=5, random_state=42).fit_predict(features)
    elif method == "agglomerative": labels = AgglomerativeClustering(n_clusters=clusters).fit_predict(features)
    else: raise ValueError(f"Unknown clustering method: {method}")
    return labels.reshape(image.shape[:2])


def benchmark(image: np.ndarray, feature_mode: str = "rgb_xy", sample_size: int = 2500, seed: int = 42) -> pd.DataFrame:
    features = make_features(image, feature_mode); rng = np.random.default_rng(seed); sample = features[rng.choice(len(features), min(sample_size, len(features)), replace=False)]; rows = []
    for method, estimator in (("kmeans", KMeans(6, n_init=5, random_state=seed)), ("dbscan", DBSCAN(eps=.1, min_samples=20)), ("agglomerative", AgglomerativeClustering(6))):
        labels = estimator.fit_predict(sample); real_clusters = len(set(labels)) - (-1 in labels); rows.append({"method": method, "clusters": real_clusters, "noise_fraction": float(np.mean(labels == -1)), "silhouette": silhouette_score(sample, labels) if real_clusters > 1 else np.nan})
    return pd.DataFrame(rows)


def tune_dbscan(image: np.ndarray, feature_mode: str, eps_values, min_samples_values, sample_size: int = 3000, seed: int = 42) -> pd.DataFrame:
    features = make_features(image, feature_mode); rng = np.random.default_rng(seed); sample = features[rng.choice(len(features), min(sample_size, len(features)), replace=False)]; rows = []
    for min_samples in min_samples_values:
        for eps in eps_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(sample); count = len(set(labels)) - (-1 in labels); rows.append({"eps": eps, "min_samples": min_samples, "clusters": count, "noise_fraction": float(np.mean(labels == -1)), "silhouette": silhouette_score(sample, labels) if count > 1 else np.nan})
    return pd.DataFrame(rows)
