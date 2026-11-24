"""K-Means (algorytm Lloyda) z inicjalizacją Forgy, Random Partition i k-means++."""

import numpy as np


class KMeans:
    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        X = self._validate(X)
        rng = np.random.default_rng(self.random_state)
        runs = self.n_init if isinstance(self.init, str) else 1
        best = None
        for _ in range(runs):
            # zostaje uruchomienie o najmniejszym WCSS
            result = self._lloyd(X, self._initial_centroids(X, rng))
            if best is None or result["inertia"] < best["inertia"]:
                best = result
        self.cluster_centers_ = best["centroids"]
        self.labels_ = best["labels"]
        self.inertia_ = best["inertia"]
        self.n_iter_ = best["n_iter"]
        self.inertia_history_ = best["history"]
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Model nie jest dopasowany - najpierw wywołaj fit()")
        return self._assign(self._validate(X, fitted=True), self.cluster_centers_)[0]

    def _lloyd(self, X, centroids):
        history = []
        converged = False
        n_iter = 0
        while not converged and n_iter < self.max_iter:
            n_iter += 1
            # przypisanie do najbliższego centroidu
            labels, sq_dist = self._assign(X, centroids)
            history.append(float(sq_dist.sum()))
            # aktualizacja: średnie klastrów
            new_centroids = self._update(X, labels, sq_dist, centroids)
            # stop, gdy centroidy się nie zmieniają
            converged = np.array_equal(new_centroids, centroids)
            centroids = new_centroids
        if not converged:
            # po max_iter: przypisanie do ostatnich centroidów
            labels, sq_dist = self._assign(X, centroids)
        return {
            "centroids": centroids,
            "labels": labels,
            "inertia": float(sq_dist.sum()),
            "n_iter": n_iter,
            "history": history,
        }

    @staticmethod
    def _assign(X, centroids):
        # kwadraty odległości punkt-centroid
        sq_dist = np.stack([((X - c) ** 2).sum(axis=1) for c in centroids], axis=1)
        # przy remisie pierwszy
        labels = sq_dist.argmin(axis=1)
        return labels, sq_dist[np.arange(len(X)), labels]

    def _update(self, X, labels, sq_dist, centroids):
        labels = labels.copy()
        empty = [i for i in range(self.n_clusters) if not np.any(labels == i)]
        if empty and sq_dist.max() > 0:
            # pusty klaster dostaje najdalszy punkt (jak w sklearn)
            farthest = np.argsort(sq_dist, kind="stable")[::-1][: len(empty)]
            labels[farthest] = empty
        new_centroids = centroids.copy()
        for i in range(self.n_clusters):
            members = X[labels == i]
            if len(members):
                new_centroids[i] = members.mean(axis=0)
        return new_centroids

    def _initial_centroids(self, X, rng):
        n, k = len(X), self.n_clusters
        if not isinstance(self.init, str):
            return np.asarray(self.init, dtype=float).copy()
        if self.init == "forgy":
            # k losowych punktów
            return X[rng.choice(n, size=k, replace=False)].copy()
        if self.init == "random_partition":
            # losowy podział, potem średnie
            labels = rng.integers(k, size=n)
            return np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else X[rng.integers(n)]
                for i in range(k)
            ])
        return self._kmeans_plus_plus(X, rng)

    def _kmeans_plus_plus(self, X, rng):
        n = len(X)
        centroids = [X[rng.integers(n)]]
        # D(x)^2 do najbliższego wybranego środka
        d2 = ((X - centroids[0]) ** 2).sum(axis=1)
        while len(centroids) < self.n_clusters:
            # losowanie z p ~ D(x)^2
            total = d2.sum()
            idx = rng.choice(n, p=d2 / total) if total > 0 else rng.integers(n)
            centroids.append(X[idx])
            d2 = np.minimum(d2, ((X - X[idx]) ** 2).sum(axis=1))
        return np.array(centroids)

    def _validate(self, X, fitted=False):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.all(np.isfinite(X)):
            raise ValueError("X musi być dwuwymiarową tablicą skończonych liczb")
        if fitted:
            return X
        if not isinstance(self.n_clusters, (int, np.integer)) or not 1 <= self.n_clusters <= len(X):
            raise ValueError("n_clusters musi być liczbą całkowitą od 1 do liczby próbek")
        if self.n_init < 1 or self.max_iter < 1:
            raise ValueError("n_init i max_iter muszą być >= 1")
        if isinstance(self.init, str):
            if self.init not in ("k-means++", "forgy", "random_partition"):
                raise ValueError("init: 'k-means++', 'forgy', 'random_partition' albo tablica centroidów")
        elif np.shape(self.init) != (self.n_clusters, X.shape[1]):
            raise ValueError("Tablica init musi mieć kształt (n_clusters, n_cech)")
        return X
