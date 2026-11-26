# K-Means od zera w Pythonie

Autorzy: Mateusz Polek, Oliwier Sajdak

Implementacja algorytmu K-Means w NumPy, zgodna z pseudokodem algorytmu Lloyda z Wikipedii
([k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)), z trzema metodami
inicjalizacji: Forgy, Random Partition i k-means++
([Wikipedia](https://en.wikipedia.org/wiki/K-means%2B%2B), Arthur i Vassilvitskii, 2007).
Miary jakości – współczynnik sylwetki
([Wikipedia](https://en.wikipedia.org/wiki/Silhouette_(clustering))) i skorygowany indeks Randa
([Wikipedia](https://en.wikipedia.org/wiki/Rand_index)) – też są napisane od podstaw.

Wszystkie eksperymenty i testy korzystają ze zbioru danych Iris
([UCI](https://archive.ics.uci.edu/ml/datasets/iris)): 150 kwiatów, 4 cechy w cm,
3 gatunki po 50 próbek. Biblioteka scikit-learn służy wyłącznie do wczytania zbioru
(`load_iris`) i jako wzorzec do porównania wyników.

## Struktura

```
kmeans_projekt/
├── kmeans/
│   ├── __init__.py
│   ├── kmeans.py        # klasa KMeans: fit, predict, fit_predict
│   └── metrics.py       # WCSS, sylwetka, ARI, tablica kontyngencji
├── tests/
│   └── test_kmeans.py   # 36 testów pytest na zbiorze Iris
├── experiments.py       # 6 eksperymentów na zbiorze Iris -> wykresy i wyniki.json w wyniki/
├── wyniki/              # wygenerowane wykresy PNG i wyniki.json
├── requirements.txt
└── README.md
```

## Instalacja i uruchomienie

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python experiments.py              # kilka sekund, zapisuje wykresy w wyniki/
python -m pytest -q tests          # testy jednostkowe
```

Jeśli Windows nie zna polecenia `python`, użyj `py`.

## Użycie

```python
from sklearn.datasets import load_iris
from kmeans import KMeans, adjusted_rand_score

X, y = load_iris(return_X_y=True)
km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(X)

km.cluster_centers_   # centroidy (3, 4)
km.labels_            # numer klastra każdego kwiatu
km.inertia_           # WCSS: suma kwadratów odległości od centroidów
km.n_iter_            # liczba iteracji najlepszego uruchomienia
km.inertia_history_   # WCSS po każdym kroku przypisania

adjusted_rand_score(y, km.labels_)   # zgodność klastrów z gatunkami
```

## Parametry `KMeans`

| Parametr | Domyślnie | Znaczenie |
|---|---|---|
| `n_clusters` | 8 | liczba klastrów k |
| `init` | `"k-means++"` | `"k-means++"`, `"forgy"`, `"random_partition"` albo tablica centroidów (k, d) |
| `n_init` | 10 | liczba uruchomień, zostaje to o najmniejszym WCSS |
| `max_iter` | 300 | maksymalna liczba iteracji jednego uruchomienia |
| `random_state` | `None` | ziarno generatora losowego |

Algorytm kończy się, gdy nowe centroidy są równe poprzednim (przypisania się nie zmieniają),
albo po `max_iter` iteracjach.

## Eksperymenty (zbiór Iris)

1. Porównanie ze scikit-learn – te same centroidy startowe, 100 uruchomień
2. Zgodność klastrów z gatunkami – ARI, odsetek zgodnych przypisań, tablica kontyngencji
3. Zbieżność – WCSS w kolejnych iteracjach
4. Porównanie inicjalizacji Forgy, Random Partition i k-means++ (po 100 uruchomień)
5. Dobór liczby klastrów – metoda łokcia i współczynnik sylwetki
6. Standaryzacja cech – cechy surowe a standaryzowane

Wyniki są pokazane w rzucie na dwie składowe główne (PCA liczone przez SVD w NumPy).
