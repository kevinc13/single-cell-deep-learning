import numpy as np
import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def cluster_gmm(features, n_components):
    """
    Fit Gaussian mixture models with varying covariance types
    and number of components

    Args:
        features: Numpy array of samples (rows) by features (columns)
        n_components: Number (or list) of components to use

    Returns:
        Dictionary containing best GMM model (lowest BIC),
        number of components used, covariance type, and resulting clusters
    """
    lowest_bic = np.infty
    best_gmm = None

    covar_types = ["full", "diag", "tied", "spherical"]
    if not isinstance(n_components, list):
        n_components = [n_components]
    search_space = list(itertools.product(n_components, covar_types))

    for k, covar_type in search_space:
        gmm = GaussianMixture(n_components=k,
                              covariance_type=covar_type)
        gmm.fit(features)
        bic = gmm.bic(features)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    return {
        "model": best_gmm,
        "bic": lowest_bic,
        "n_components": best_gmm.n_components,
        "covariance_type": best_gmm.covariance_type,
        "clusters": best_gmm.predict(features) + 1
    }


def cluster_km(features, n_clusters):
    """
    Run k-means clustering algorithm

    Args:
        features:
        n_clusters:

    Returns:

    """
    if not isinstance(n_clusters, list):
        n_clusters = [n_clusters]

    best_model = None
    best_score = 0
    for k in n_clusters:
        km = KMeans(init="k-means++", n_clusters=k, n_init=10)
        km.fit(features)

        score = silhouette_score(features, km.labels_)
        if score > best_score:
            best_score = score
            best_model = km

    return {
        "model": best_model,
        "silhouette_score": best_score,
        "clusters": best_model.labels_ + 1
    }


def cluster_hc(features, n_clusters):
    if not isinstance(n_clusters, list):
        n_clusters = [n_clusters]

    best_model = None
    best_score = 0
    for k in n_clusters:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        model.fit(features)

        score = silhouette_score(features, model.labels_)
        if score > best_score:
            best_score = score
            best_model = model

    return {
        "model": best_model,
        "silhouette_score": best_score,
        "clusters": best_model.labels_ + 1
    }

