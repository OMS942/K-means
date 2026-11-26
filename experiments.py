"""Eksperymenty na zbiorze Iris, wyniki w wyniki/."""

import json
import time
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import load_iris

from kmeans import KMeans, adjusted_rand_score, contingency_matrix, silhouette_score

OUT = Path(__file__).resolve().parent / "wyniki"
OUT.mkdir(exist_ok=True)

SPECIES = ["setosa", "versicolor", "virginica"]
SPECIES_COLORS = ["#2a6fdb", "#e8743b", "#19a979"]
INITS = {"forgy": ("Forgy", "#e8743b"),
         "random_partition": ("Random Partition", "#8e6bc9"),
         "k-means++": ("k-means++", "#2a6fdb")}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

RESULTS = {}


def load_data():
    X, y = load_iris(return_X_y=True)
    return X, y


def sk_kmeans(k, **kwargs):
    # sklearn: Lloyd, tol=0
    return SkKMeans(k, algorithm="lloyd", tol=0.0, **kwargs)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  zapisano {name}")


def pca(X):
    # PCA przez SVD
    mean = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    explained = s ** 2 / (s ** 2).sum()
    return (lambda Z: (Z - mean) @ Vt[:2].T), explained


def species_of_clusters(y, labels):
    # najlepsze dopasowanie klaster -> gatunek
    table = contingency_matrix(y, labels)
    best = max(permutations(range(table.shape[1])), key=lambda p: sum(table[p[j], j] for j in range(len(p))))
    return np.array(best), sum(table[best[j], j] for j in range(len(best))) / len(y)


def summary(X, y, km):
    mapping, accuracy = species_of_clusters(y, km.labels_)
    return {
        "wcss": km.inertia_,
        "iteracje": km.n_iter_,
        "ari_vs_gatunki": adjusted_rand_score(y, km.labels_),
        "dokladnosc_po_dopasowaniu": accuracy,
        "bledne_przypisania": int(round((1 - accuracy) * len(y))),
        "sylwetka": silhouette_score(X, km.labels_),
        # kolumny w kolejności gatunków
        "tablica_kontyngencji": contingency_matrix(y, km.labels_)[:, np.argsort(mapping)].tolist(),
    }


def scatter_clusters(ax, P, y, labels, title):
    mapping, _ = species_of_clusters(y, labels)
    predicted = mapping[labels]
    wrong = predicted != y
    ax.scatter(P[~wrong, 0], P[~wrong, 1], s=20, c=[SPECIES_COLORS[s] for s in predicted[~wrong]], alpha=0.8)
    ax.scatter(P[wrong, 0], P[wrong, 1], s=46, c=[SPECIES_COLORS[s] for s in predicted[wrong]],
               edgecolors="black", linewidths=1.2)
    ax.set_title(title)
    ax.set_xlabel("Składowa główna 1")


def legend_clusters(ax):
    handles = [Line2D([], [], ls="", marker="o", ms=6, color=c, label=f"klaster: {s}")
               for s, c in zip(SPECIES, SPECIES_COLORS)]
    handles.append(Line2D([], [], ls="", marker="o", ms=7, markerfacecolor="white", markeredgecolor="black",
                          markeredgewidth=1.2, label="niezgodny z gatunkiem"))
    ax.legend(handles=handles, loc="lower center", fontsize=8.5)


def exp_sklearn_and_species():
    print("1-2. Porównanie ze scikit-learn i zgodność z gatunkami")
    X, y = load_data()
    project, explained = pca(X)
    P = project(X)

    # ten sam start dla obu
    runs, identical, differences = 100, 0, []
    for seed in range(runs):
        c0 = X[np.random.default_rng(seed).choice(len(X), 3, replace=False)]
        own = KMeans(3, init=c0).fit(X)
        sk = sk_kmeans(3, init=c0, n_init=1).fit(X)
        same = (np.array_equal(own.labels_, sk.labels_) and own.n_iter_ == sk.n_iter_
                and np.allclose(own.cluster_centers_, sk.cluster_centers_, rtol=0, atol=1e-12))
        identical += same
        if not same:
            differences.append({
                "ziarno": seed,
                "iteracje_wlasna": own.n_iter_,
                "iteracje_sklearn": int(sk.n_iter_),
                "ari_koncowych_podzialow": adjusted_rand_score(own.labels_, sk.labels_),
                "max_roznica_centroidow": float(np.abs(own.cluster_centers_ - sk.cluster_centers_).max()),
            })

    own = KMeans(3, n_init=10, random_state=0).fit(X)
    sk = sk_kmeans(3, n_init=10, random_state=0).fit(X)
    RESULTS["porownanie_sklearn"] = {
        "uruchomienia_z_tym_samym_startem": runs,
        "identyczne_we_wszystkim": int(identical),
        "roznice": differences,
        "n_init_10": {
            "wcss_wlasna": own.inertia_,
            "wcss_sklearn": float(sk.inertia_),
            "ari_wlasna_vs_sklearn": adjusted_rand_score(own.labels_, sk.labels_),
        },
    }
    RESULTS["zgodnosc_z_gatunkami"] = summary(X, y, own)
    RESULTS["pca_wyjasniona_wariancja"] = explained[:2].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    for i, name in enumerate(SPECIES):
        axes[0].scatter(P[y == i, 0], P[y == i, 1], s=20, color=SPECIES_COLORS[i], alpha=0.8, label=name)
    axes[0].set_title("Gatunki (etykiety prawdziwe)")
    axes[0].set_xlabel("Składowa główna 1")
    axes[0].legend(loc="lower center")
    scatter_clusters(axes[1], P, y, own.labels_, f"Własna implementacja\nWCSS = {own.inertia_:.2f}")
    scatter_clusters(axes[2], P, y, sk.labels_, f"scikit-learn\nWCSS = {sk.inertia_:.2f}")
    for km, ax in ((own, axes[1]), (sk, axes[2])):
        Pc = project(km.cluster_centers_)
        ax.scatter(Pc[:, 0], Pc[:, 1], marker="X", s=180, c="black", edgecolors="white", linewidths=1.5)
    legend_clusters(axes[1])
    axes[0].set_ylabel("Składowa główna 2")
    save(fig, "01_gatunki_i_klastry.png")


def exp_convergence():
    print("3. Zbieżność")
    X, _ = load_data()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    RESULTS["zbieznosc"] = {}
    for init, (name, color) in INITS.items():
        histories = [KMeans(3, init=init, n_init=1, random_state=s).fit(X).inertia_history_ for s in range(5)]
        for s, h in enumerate(histories):
            ax.plot(range(1, len(h) + 1), h, color=color, alpha=0.8, lw=1.6, marker="o", ms=3,
                    label=name if s == 0 else None)
        RESULTS["zbieznosc"][init] = {
            "wcss_pierwsza_iteracja": [h[0] for h in histories],
            "wcss_koncowe": [h[-1] for h in histories],
            "liczba_iteracji": [len(h) for h in histories],
        }
    ax.set_yscale("log")
    ax.set_xlabel("Iteracja")
    ax.set_ylabel("WCSS (skala log)")
    ax.set_title("WCSS w kolejnych iteracjach – 5 uruchomień każdej inicjalizacji")
    ax.legend(title="Inicjalizacja")
    save(fig, "02_zbieznosc.png")


def exp_init():
    print("4. Porównanie inicjalizacji")
    X, y = load_data()
    runs = 100
    data = {}
    for init in INITS:
        rows = []
        for seed in range(runs):
            km = KMeans(3, init=init, n_init=1, random_state=seed).fit(X)
            rows.append((km.inertia_, km.n_iter_, adjusted_rand_score(y, km.labels_)))
        data[init] = {key: np.array(col) for key, col in zip(("wcss", "iteracje", "ari"), zip(*rows))}

    best = min(d["wcss"].min() for d in data.values())
    categories = [("najlepsze minimum", lambda w: w - best < 1e-6, "#19a979"),
                  ("gorsze o mniej niż 1%", lambda w: (w - best >= 1e-6) & (w < 1.01 * best), "#c2a12b"),
                  ("minimum lokalne (gorsze o ponad 1%)", lambda w: w >= 1.01 * best, "#d64550")]
    RESULTS["inicjalizacja"] = {"liczba_uruchomien": runs, "najlepsze_wcss": float(best),
                                "kategorie": [c[0] for c in categories]}
    for init, d in data.items():
        RESULTS["inicjalizacja"][init] = {
            "uruchomienia_w_kategoriach": [int(f(d["wcss"]).sum()) for _, f, _ in categories],
            "rozne_wyniki_wcss": sorted({round(float(w), 3) for w in d["wcss"]}),
            "wcss_srednie": float(d["wcss"].mean()),
            "iteracje_srednio": float(d["iteracje"].mean()),
            "iteracje_min_max": [int(d["iteracje"].min()), int(d["iteracje"].max())],
            "ari_srednie": float(d["ari"].mean()),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [v[0] for v in INITS.values()]
    bottom = np.zeros(len(INITS))
    for label, f, color in categories:
        counts = np.array([f(data[init]["wcss"]).sum() for init in INITS])
        axes[0].bar(names, counts, bottom=bottom, color=color, label=label, width=0.6)
        bottom += counts
    axes[0].set_ylabel("Liczba uruchomień (ze 100)")
    axes[0].set_title("Wynik pojedynczego uruchomienia (WCSS)")
    axes[0].set_ylim(0, 128)
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].legend(loc="upper center", fontsize=8.5, ncol=2)
    rng = np.random.default_rng(0)
    for i, (init, (_, color)) in enumerate(INITS.items()):
        it = data[init]["iteracje"]
        axes[1].scatter(i + rng.uniform(-0.18, 0.18, it.size), it + rng.uniform(-0.2, 0.2, it.size),
                        s=14, color=color, alpha=0.6, linewidths=0)
        axes[1].hlines(np.median(it), i - 0.28, i + 0.28, color="black", lw=2)
    axes[1].set_xticks(range(len(INITS)), names)
    axes[1].set_ylabel("Liczba iteracji")
    axes[1].set_title("Liczba iteracji do zbieżności (kreska = mediana)")
    save(fig, "03_inicjalizacja.png")


def exp_choose_k():
    print("5. Dobór liczby klastrów")
    X, _ = load_data()
    ks = list(range(1, 11))
    wcss, sils = [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=0).fit(X)
        wcss.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_) if k >= 2 else None)
    best_k = max(ks[1:], key=lambda k: sils[k - 1])
    RESULTS["dobor_k"] = {"k": ks, "wcss": wcss, "sylwetka": sils, "k_max_sylwetki": best_k}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, wcss, marker="o", color="#2a6fdb")
    axes[0].set_title("Metoda łokcia")
    axes[0].set_ylabel("WCSS")
    axes[1].plot(ks[1:], sils[1:], marker="o", color="#19a979")
    axes[1].set_title(f"Współczynnik sylwetki (maksimum dla k = {best_k})")
    axes[1].set_ylabel("Średnia sylwetka")
    for ax in axes:
        ax.axvline(3, color="gray", ls="--", lw=1, label="liczba gatunków (k = 3)")
        ax.set_xticks(ks)
        ax.set_xlabel("Liczba klastrów k")
        ax.legend()
    save(fig, "04_dobor_k.png")


def exp_standardization():
    print("6. Standaryzacja cech")
    X, y = load_data()
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    project, _ = pca(X)
    P = project(X)
    fits = {"surowe": KMeans(3, n_init=10, random_state=0).fit(X),
            "standaryzowane": KMeans(3, n_init=10, random_state=0).fit(Xs)}
    RESULTS["standaryzacja"] = {
        "odchylenie_std_cech_cm": X.std(axis=0).tolist(),
        **{name: summary(data, y, km) for (name, km), data in zip(fits.items(), (X, Xs))},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for ax, (name, km) in zip(axes, fits.items()):
        r = RESULTS["standaryzacja"][name]
        scatter_clusters(ax, P, y, km.labels_,
                         f"Cechy {name}: {r['dokladnosc_po_dopasowaniu']:.1%} zgodności, "
                         f"ARI = {r['ari_vs_gatunki']:.3f}")
    legend_clusters(axes[0])
    axes[0].set_ylabel("Składowa główna 2")
    save(fig, "05_standaryzacja.png")


def main():
    t0 = time.perf_counter()
    for experiment in (exp_sklearn_and_species, exp_convergence, exp_init, exp_choose_k, exp_standardization):
        experiment()
    RESULTS["czas_calkowity_s"] = time.perf_counter() - t0
    with open(OUT / "wyniki.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)
    print(f"Gotowe w {RESULTS['czas_calkowity_s']:.1f} s")


if __name__ == "__main__":
    main()
