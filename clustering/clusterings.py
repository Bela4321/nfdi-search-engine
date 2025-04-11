from typing import Any, List

import numpy as np
from hdbscan import HDBSCAN
from pyclustering.cluster.xmeans import xmeans
from sklearn.decomposition import PCA


def hdbscan_clustering(data_matrix: np.array) -> tuple[Any, None]:
    #preprpcessing
    reduced_data_matrix = optimal_pca(data_matrix,0.4)
    clusterer = HDBSCAN(max_cluster_size=80, min_cluster_size=5,min_samples = 1)
    labels = clusterer.fit_predict(reduced_data_matrix)
    return labels.tolist(), clusterer


def xmeans_clustering(data_matrix: np.array) -> [List[int],np.array]:
    k_start = min(3, data_matrix.shape[0])
    initial_centers = data_matrix[np.random.choice(data_matrix.shape[0], k_start, replace=False)]
    xmeans_instance = xmeans(data_matrix, initial_centers)
    xmeans_instance.process()

    labels = np.zeros(len(data_matrix), dtype=int)  # Assign cluster labels
    for cluster_idx, cluster in enumerate(xmeans_instance.get_clusters()):
        for idx in cluster:
            labels[idx] = cluster_idx

    centroids = xmeans_instance.get_centers()

    return labels.tolist(),np.array(centroids)


def optimal_pca(embedding, variance_threshold=0.95):
    n_samples, n_features = embedding.shape
    n_components = min(n_samples, n_features)

    pca = PCA(n_components=n_components)
    pca.fit(embedding)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    n_components_to_retain = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    pca_final = PCA(n_components=n_components_to_retain)
    return pca_final.fit_transform(embedding)