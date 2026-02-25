from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.utils import distance, divide_dataset


class CenterPointSMOTE:
    def __init__(self, n: int, eta: float = .15, random_state: Optional[int] = None):
        self._fitted = False
        self._n = n
        self._eta = eta
        self._scaler = MinMaxScaler()
        self._kmeans = KMeans(
            n_clusters=n,
            n_init=10,
            max_iter=1000,
            random_state=random_state
        )
        self._x_majority: Optional[np.array] = None
        self._y_majority: Optional[np.array] = None
        self._x_minority: Optional[np.array] = None
        self._y_minority: Optional[np.array] = None
        self._X: Optional[np.array] = None
        self._y: Optional[np.array] = None
        self._center_points = []
        self._d = {}  # {cluster_id: min_dist_to_majority_sample}
        self._dis: List[Tuple[int, float]] = []  # (cluster_id, distance_to_it)

    def fit(self, X: np.ndarray, y: np.ndarray) -> CenterPointSMOTE:
        self._dis = []
        self._d = {}
        self._X = X
        self._y = y

        X_scaled = self._scaler.fit_transform(X)
        (self._x_majority, self._y_majority), (self._x_minority, self._y_minority) = divide_dataset(X_scaled, y)
        self._kmeans.fit(self._x_minority)
        self._center_points = self._kmeans.cluster_centers_
        for i in range(self._n):
            self._d[i] = self._calculate_d(self._center_points[i])
        features_count = self._x_minority.shape[1]
        for x in self._x_minority:
            self._dis.append(self._calculate_dis(x.reshape(-1, features_count)))  # expecting 2D array
        self._fitted = True

        return self

    def _calculate_d(self, center_point: np.ndarray) -> float:
        distances = []
        for x in self._x_majority:
            distances.append(distance(x, center_point))

        return min(distances)

    def _calculate_dis(self, minority_sample: np.ndarray) -> Tuple[int, float]:
        distances = self._kmeans.transform(minority_sample)

        return np.argmin(distances), distances.min()

    def resample(self) -> Tuple[np.array, np.array]:
        if not self._fitted:
            raise RuntimeError("You must first call fit()")

        features_count = self._x_minority.shape[1]
        sampled = np.empty((0, features_count), float)
        labels = np.empty((0,))
        for minority_id, (cluster_id, dis) in enumerate(self._dis):
            if dis < self._d[cluster_id]:
                minority_sample = self._x_minority[minority_id]
                new_sample = self._eta * minority_sample + (1 - self._eta) * self._center_points[cluster_id]
                sampled = np.append(
                    sampled,
                    self._scaler.inverse_transform(
                        new_sample.reshape(-1, features_count)
                    ),
                    axis=0
                )
                labels = np.append(labels, self._y_minority[minority_id])

        X = np.append(sampled, self._X, axis=0)
        y = np.append(labels, self._y, axis=0)

        return X, y

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.array, np.array]:
        return self.fit(X, y).resample()
