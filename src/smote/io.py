from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from src.utils import distance, divide_dataset


class InnerOuterSMOTE:
    def __init__(self, n: int, eta: float = .15, eta1: float = .1, eta2: float = .2):
        self._fitted = False
        self._n = n
        self._scaler = MinMaxScaler()
        self._knn = KNeighborsClassifier(n_neighbors=n)
        self._eta = eta
        self._eta1 = eta1
        self._eta2 = eta2
        self._x_majority: Optional[np.array] = None
        self._y_majority: Optional[np.array] = None
        self._x_minority: Optional[np.array] = None
        self._y_minority: Optional[np.array] = None
        self._X: Optional[np.array] = None
        self._y: Optional[np.array] = None
        self._inner = []
        self._outer = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> InnerOuterSMOTE:
        self._X = X
        self._y = y
        X_scaled = self._scaler.fit_transform(X)

        (self._x_majority, self._y_majority), (self._x_minority, self._y_minority) = divide_dataset(X_scaled, y)
        self._knn.fit(X_scaled, y)
        self._set_inner_and_outer(y)
        self._fitted = True

        return self

    def _set_inner_and_outer(self, y: np.ndarray) -> None:
        inner = []
        outer = []
        features_count = self._x_minority.shape[1]
        minority_label = self._y_minority[0]
        for x in self._x_minority:
            neighbours_ids = self._knn.kneighbors(
                x.reshape(-1, features_count),
                n_neighbors=self._n,
                return_distance=False
            ).squeeze(axis=0)
            neighbours_count = len(neighbours_ids)
            minority_count = 0
            for n_id in neighbours_ids:
                if y[n_id] == minority_label:
                    minority_count += 1
            if minority_count > neighbours_count / 2:
                inner.append(x)
            else:
                outer.append(x)
        self._inner = inner
        self._outer = outer

    def _find_closest_outer(self, inner: np.ndarray) -> np.ndarray:
        distances = []
        for x in self._outer:
            distances.append(distance(x, inner))

        return self._outer[np.argmin(distances)]

    def resample(self) -> Tuple[np.array, np.array]:
        if not self._fitted:
            raise RuntimeError("You must first call fit()")

        features_count = self._x_minority.shape[1]
        sampled = np.empty((0, features_count))
        labels = np.empty((0,))
        if self._inner and self._outer:
            for inner in self._inner:
                closest_outer = self._find_closest_outer(inner)
                new_sample = self._eta * inner + (1 - self._eta) * closest_outer
                sampled = np.append(
                    sampled,
                    self._scaler.inverse_transform(
                        new_sample.reshape(-1, features_count)
                    ),
                    axis=0
                )
                labels = np.append(labels, self._y_minority[0])
        else:
            indices = np.arange(self._x_minority.shape[0])
            i1, i2, i3 = np.random.choice(indices, size=3, replace=False)
            x1, x2, x3 = self._x_minority[i1], self._x_minority[i2], self._x_minority[i3]
            y = self._eta1 * x2 + (1 - self._eta1) * x3
            new_sample = self._eta2 * x1 + (1 - self._eta2) * y
            sampled = np.append(
                sampled,
                self._scaler.inverse_transform(
                    new_sample.reshape(-1, features_count)
                ),
                axis=0
            )
            labels = np.append(labels, self._y_minority[0])

        X = np.append(sampled, self._X, axis=0)
        y = np.append(labels, self._y, axis=0)

        return X, y

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).resample()
