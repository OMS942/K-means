"""Miary jakości: WCSS, sylwetka, ARI."""

import numpy as np


def inertia(X, labels, centroids):
    # suma kwadratów odległości od centroidów
    X, centroids = np.asarray(X, dtype=float), np.asarray(centroids, dtype=float)
    return float(((X - centroids[np.asarray(labels)]) ** 2).sum())


def silhouette_samples(X, labels):
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    clusters = np.unique(labels)
    if not 2 <= len(clusters) <= len(X) - 1:
        raise ValueError("Sylwetka wymaga od 2 do n-1 klastrów")
    members = {c: labels == c for c in clusters}
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = members[labels[i]]
        size = own.sum()
        if size == 1:
            # klaster jednoelementowy: s(i) = 0
            continue
        d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        # a(i): własny klaster
        a = d[own].sum() / (size - 1)
        # b(i): najbliższy obcy klaster
        b = min(d[members[c]].mean() for c in clusters if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return s


def silhouette_score(X, labels):
    return float(silhouette_samples(X, labels).mean())


def contingency_matrix(labels_true, labels_pred):
    # n_ij: klasa i, klaster j
    _, t = np.unique(np.asarray(labels_true), return_inverse=True)
    _, p = np.unique(np.asarray(labels_pred), return_inverse=True)
    table = np.zeros((t.max() + 1, p.max() + 1), dtype=np.int64)
    np.add.at(table, (t, p), 1)
    return table


def _pairs(x):
    # x po 2
    x = np.asarray(x, dtype=float)
    return x * (x - 1) / 2


def adjusted_rand_score(labels_true, labels_pred):
    table = contingency_matrix(labels_true, labels_pred)
    index = _pairs(table).sum()
    sum_a = _pairs(table.sum(axis=1)).sum()
    sum_b = _pairs(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / _pairs(table.sum())
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:
        # podziały trywialne
        return 1.0
    return float((index - expected) / (max_index - expected))
