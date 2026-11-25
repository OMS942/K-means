import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import load_iris

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kmeans import KMeans, adjusted_rand_score, contingency_matrix, silhouette_samples, silhouette_score  # noqa: E402

INITS = ["k-means++", "forgy", "random_partition"]
BEST_WCSS = 78.85144142614601


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize("init", INITS)
def test_znajduje_najlepsze_minimum_i_oddziela_setose(iris, init):
    X, y = iris
    km = KMeans(3, init=init, n_init=10, random_state=0).fit(X)
    assert km.inertia_ == pytest.approx(BEST_WCSS, rel=1e-12)
    table = contingency_matrix(y, km.labels_)
    setosa_cluster = table[0].argmax()
    assert table[0, setosa_cluster] == 50 and table[:, setosa_cluster].sum() == 50


def test_wcss_rowne_sumie_kwadratow(iris):
    X, _ = iris
    km = KMeans(3, random_state=0).fit(X)
    assert km.inertia_ == pytest.approx(((X - km.cluster_centers_[km.labels_]) ** 2).sum())


def test_punkt_staly_centroidy_to_srednie_klastrow(iris):
    X, _ = iris
    km = KMeans(3, random_state=0).fit(X)
    for j in range(3):
        np.testing.assert_array_equal(km.cluster_centers_[j], X[km.labels_ == j].mean(axis=0))


def test_kazdy_punkt_przypisany_do_najblizszego_centroidu(iris):
    X, _ = iris
    km = KMeans(3, random_state=0).fit(X)
    d = ((X[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    assert np.all(d[np.arange(len(X)), km.labels_] == d.min(axis=1))


@pytest.mark.parametrize("init", INITS)
def test_wcss_nie_rosnie_w_kolejnych_iteracjach(iris, init):
    X, _ = iris
    for seed in range(20):
        h = np.array(KMeans(3, init=init, n_init=1, random_state=seed).fit(X).inertia_history_)
        assert np.all(np.diff(h) <= 1e-9 * h[0])


def test_powtarzalnosc_dla_ustalonego_ziarna(iris):
    X, _ = iris
    a = KMeans(3, random_state=42).fit(X)
    b = KMeans(3, random_state=42).fit(X)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)


def test_predict_zgodny_z_labels(iris):
    X, _ = iris
    km = KMeans(3, random_state=0).fit(X)
    np.testing.assert_array_equal(km.predict(X), km.labels_)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_identyczny_wynik_jak_sklearn_przy_tym_samym_starcie(iris, seed):
    X, _ = iris
    c0 = X[np.random.default_rng(seed).choice(len(X), 3, replace=False)]
    own = KMeans(3, init=c0).fit(X)
    sk = SkKMeans(3, init=c0, n_init=1, algorithm="lloyd", tol=0.0).fit(X)
    assert own.inertia_ == pytest.approx(sk.inertia_, rel=1e-12)
    np.testing.assert_allclose(own.cluster_centers_, sk.cluster_centers_, atol=1e-12)
    np.testing.assert_array_equal(own.labels_, sk.labels_)
    assert own.n_iter_ == sk.n_iter_


def test_ten_sam_wynik_koncowy_co_sklearn_dla_100_startow(iris):
    X, _ = iris
    for seed in range(100):
        c0 = X[np.random.default_rng(seed).choice(len(X), 3, replace=False)]
        own = KMeans(3, init=c0).fit(X)
        sk = SkKMeans(3, init=c0, n_init=1, algorithm="lloyd", tol=0.0).fit(X)
        np.testing.assert_array_equal(own.labels_, sk.labels_)
        np.testing.assert_allclose(own.cluster_centers_, sk.cluster_centers_, atol=1e-12)


@pytest.mark.parametrize("init", ["k-means++", "forgy"])
def test_srodki_poczatkowe_to_punkty_danych(iris, init):
    X, _ = iris
    c = KMeans(5, init=init)._initial_centroids(X, np.random.default_rng(0))
    assert all(np.any(np.all(X == r, axis=1)) for r in c)


def test_kmeanspp_nie_wybiera_dwa_razy_tego_samego_punktu(iris):
    X, _ = iris
    for seed in range(20):
        c = KMeans(10)._kmeans_plus_plus(X, np.random.default_rng(seed))
        assert len(np.unique(c, axis=0)) == 10


def test_random_partition_daje_srodki_blisko_sredniej_danych(iris):
    X, _ = iris
    c = KMeans(3, init="random_partition")._initial_centroids(X, np.random.default_rng(0))
    assert np.all(np.abs(c - X.mean(axis=0)) < 0.5 * X.std(axis=0))


def test_pusty_klaster_przejmuje_najdalszy_punkt(iris):
    X, _ = iris
    init = np.vstack([X[0], X[100], np.full(4, 100.0)])
    km = KMeans(3, init=init).fit(X)
    assert len(np.unique(km.labels_)) == 3
    assert np.all(np.isfinite(km.cluster_centers_))


def test_k_rowne_n_daje_zerowe_wcss(iris):
    X = np.unique(iris[0], axis=0)[:10]
    assert KMeans(10, random_state=0).fit(X).inertia_ == pytest.approx(0.0, abs=1e-12)


def test_k_rowne_1_daje_srednia(iris):
    X, _ = iris
    np.testing.assert_allclose(KMeans(1, random_state=0).fit(X).cluster_centers_[0], X.mean(axis=0))


def _with_nan(X):
    X = X.copy()
    X[0, 0] = np.nan
    return X


@pytest.mark.parametrize(
    "kwargs, prepare",
    [
        ({"n_clusters": 200}, lambda X: X),
        ({"n_clusters": 3, "init": "zla"}, lambda X: X),
        ({"n_clusters": 3, "init": np.zeros((2, 4))}, lambda X: X),
        ({"n_clusters": 3}, _with_nan),
        ({"n_clusters": 3}, lambda X: X[:, 0]),
    ],
)
def test_walidacja_wejscia(iris, kwargs, prepare):
    with pytest.raises(ValueError):
        KMeans(**kwargs).fit(prepare(iris[0]))


def test_predict_przed_fit(iris):
    with pytest.raises(RuntimeError):
        KMeans(3).predict(iris[0])


def test_sylwetka_zgodna_ze_sklearn(iris):
    X, _ = iris
    labels = KMeans(3, random_state=0).fit_predict(X)
    np.testing.assert_allclose(silhouette_samples(X, labels), skm.silhouette_samples(X, labels), atol=1e-12)
    assert silhouette_score(X, labels) == pytest.approx(skm.silhouette_score(X, labels), rel=1e-12)


def test_sylwetka_klaster_jednoelementowy_ma_zero(iris):
    X, y = iris
    labels = y.copy()
    labels[0] = 3
    s = silhouette_samples(X, labels)
    assert s[0] == 0.0
    np.testing.assert_allclose(s, skm.silhouette_samples(X, labels), atol=1e-12)


@pytest.mark.parametrize("init", INITS)
def test_ari_zgodny_ze_sklearn(iris, init):
    X, y = iris
    for seed in range(5):
        labels = KMeans(3, init=init, n_init=1, random_state=seed).fit_predict(X)
        assert adjusted_rand_score(y, labels) == pytest.approx(skm.adjusted_rand_score(y, labels), abs=1e-12)


def test_ari_niezalezny_od_numeracji(iris):
    _, y = iris
    assert adjusted_rand_score(y, (y + 1) % 3 + 7) == pytest.approx(1.0)
