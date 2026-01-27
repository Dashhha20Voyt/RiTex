# ritex/core/transformer.py
"""
Main RiTex Transformer class implementing batch effect correction
using Riemannian geometry on SPD matrices.
"""
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from ..riemannian.metrics import RiemannianSPDMetrics
from ..utils.helpers import _stabilize_matrix, _safe_eigh

logger = logging.getLogger(__name__)

class RiTexTransformer(BaseEstimator, TransformerMixin):
    """
    RiTex трансформер с интеграцией римановой геометрии SPD-матриц.
    Использует аффинно-инвариантную метрику для устойчивой оценки ковариаций.
    """
    def __init__(self, n_components=20, k_features=100, lmbda=0.5, alpha=0.1):
        self.n_components = n_components
        self.k_features = k_features
        self.lmbda = lmbda
        self.alpha = alpha
        self.selector = None
        self.selected_indices = None
        self.W = None
        self._is_fitted = False
        self.riemannian_metrics = RiemannianSPDMetrics()

    def fit(self, X: np.ndarray, y: np.ndarray, batch_ids: np.ndarray):
        """Обучение с использованием меток классов и римановой геометрии."""
        if len(X) != len(y) or len(X) != len(batch_ids):
            raise ValueError(f"Mismatch in input dimensions")
        if len(np.unique(batch_ids)) < 2:
            logger.warning("Only one batch detected. Using PCA instead.")
            return self._fit_pca_fallback(X)

        # Выбор признаков
        k = min(self.k_features, X.shape[1])
        if k < 10:
            k = min(10, X.shape[1])
        self.selector = SelectKBest(f_classif, k=k)
        X_sel = self.selector.fit_transform(X, y)
        self.selected_indices = self.selector.get_support(indices=True)
        n_features = X_sel.shape[1]

        # Матрица биологической дисперсии
        S_bio = self._compute_biological_variance(X_sel, y)
        # Матрица глобальной корреляции
        S_global = self._compute_global_correlation(X_sel, y)
        # Комбинированная матрица
        S_bio_weighted = (1 - self.lmbda) * S_bio + self.lmbda * S_global

        # Матрица batch-дисперсии с применением римановой геометрии
        S_batch = self._compute_batch_variance_riemannian(X_sel, batch_ids)

        # Регуляризация
        reg = self.alpha * np.trace(S_batch) / max(n_features, 1)
        S_reg = S_batch + reg * np.eye(n_features)

        # Обобщенная проблема собственных значений
        evals, evecs = _safe_eigh(S_bio_weighted, S_reg)

        # Выбор главных компонент
        n_comp = min(self.n_components, len(evals))
        idx = np.argsort(evals)[::-1][:n_comp]
        self.W = evecs[:, idx]
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Применение трансформации."""
        if not self._is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")
        X_sel = X[:, self.selected_indices]
        return X_sel @ self.W

    def _fit_pca_fallback(self, X: np.ndarray):
        """Fallback на PCA."""
        n_comp = min(self.n_components, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X)
        self.selector = type('DummySelector', (), {
            'get_support': lambda indices=True: np.ones(X.shape[1], dtype=bool) if indices else np.arange(X.shape[1]),
            'transform': lambda x: x
        })()
        self.selected_indices = np.arange(X.shape[1])
        self.W = pca.components_.T
        self._is_fitted = True
        return self

    def _compute_biological_variance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Вычисление биологической дисперсии."""
        n_features = X.shape[1]
        S_bio = np.zeros((n_features, n_features))
        for c in np.unique(y):
            X_c = X[y == c]
            if len(X_c) < 2:
                continue
            mu_c = X_c.mean(axis=0)
            mu_total = X.mean(axis=0)
            diff = (mu_c - mu_total).reshape(-1, 1)
            S_bio += len(X_c) * (diff @ diff.T)
        return _stabilize_matrix(S_bio)

    def _compute_global_correlation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Вычисление глобальной корреляции."""
        n_features = X.shape[1]
        corrs = np.zeros(n_features)
        for i in range(n_features):
            try:
                valid_mask = ~(np.isnan(X[:, i]) | np.isinf(X[:, i]))
                if np.sum(valid_mask) < 3:
                    corrs[i] = 0.0
                    continue
                x_clean = X[valid_mask, i]
                y_clean = y[valid_mask]
                if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
                    corrs[i] = 0.0
                else:
                    corr = np.corrcoef(x_clean, y_clean)[0, 1]
                    corrs[i] = 0.0 if np.isnan(corr) else corr
            except:
                corrs[i] = 0.0
        return np.diag(np.clip(corrs ** 2, 0, 1))

    def _compute_batch_variance_riemannian(self, X: np.ndarray, batch_ids: np.ndarray) -> np.ndarray:
        """
        Вычисление batch-дисперсии с использованием римановой геометрии.
        Использует Frechet mean на многообразии SPD-матриц.
        """
        n_features = X.shape[1]
        total_samples = len(X)
        batch_matrices = []
        batch_weights = []
        for b in np.unique(batch_ids):
            X_b = X[batch_ids == b]
            if len(X_b) < 2:
                continue
            try:
                from sklearn.covariance import LedoitWolf
                cov_estimator = LedoitWolf(assume_centered=True)
                cov_matrix = cov_estimator.fit(X_b).covariance_
            except:
                try:
                    cov_matrix = np.cov(X_b.T, bias=True)
                except:
                    continue
            batch_matrices.append(cov_matrix)
            batch_weights.append(len(X_b) / total_samples)

        if not batch_matrices:
            return np.eye(n_features)

        # Вычисляем Frechet mean на многообразии SPD
        try:
            S_batch = self.riemannian_metrics.frechet_mean(batch_matrices,
                                                          weights=np.array(batch_weights))
        except:
            # Fallback на взвешенное среднее
            S_batch = np.average(batch_matrices, axis=0, weights=batch_weights)

        if np.all(S_batch == 0):
            S_batch = np.eye(n_features)
        return _stabilize_matrix(S_batch)
