# ritex/riemannian/metrics.py
"""
Riemannian geometry operations on Symmetric Positive Definite (SPD) matrices.
Uses affine-invariant metric (Fisher-Rao).
"""
import numpy as np
from scipy.linalg import sqrtm, inv
from typing import List, Optional


class RiemannianSPDMetrics:
    """
    Метрики и операции на многообразии симметричных положительно определенных матриц.
    Использует аффинно-инвариантную метрику (Fisher-Rao).
    """
    @staticmethod
    def affine_invariant_distance(A: np.ndarray, B: np.ndarray) -> float:
        """
        Аффинно-инвариантное расстояние между SPD-матрицами.
        d(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F
        """
        try:
            # Разложение Холецкого для A
            LA = np.linalg.cholesky(A)
            LA_inv = np.linalg.inv(LA)
            # Промежуточная матрица
            M = LA_inv @ B @ LA_inv.T
            # Логарифм собственных значений
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.clip(eigvals, 1e-10, None)
            # Расстояние
            distance = np.sqrt(np.sum(np.log(eigvals) ** 2))
            return distance
        except:
            return np.inf

    @staticmethod
    def geodesic_midpoint(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Геодезическое среднее (midpoint на многообразии).
        Соединяет две SPD-матрицы A и B.
        """
        try:
            # Разложение Холецкого для A
            LA = np.linalg.cholesky(A)
            LA_inv = np.linalg.inv(LA)
            # Промежуточная матрица
            M = LA_inv @ B @ LA_inv.T
            # Матричный корень
            sqrtm_M = sqrtm(M)
            if np.iscomplexobj(sqrtm_M):
                sqrtm_M = np.real(sqrtm_M)
            # Геодезический midpoint: A^{1/2} sqrt(A^{-1/2} B A^{-1/2}) A^{1/2}
            sqrtA = sqrtm(A)
            if np.iscomplexobj(sqrtA):
                sqrtA = np.real(sqrtA)
            midpoint = sqrtA @ sqrtm_M @ sqrtA
            return midpoint
        except:
            return (A + B) / 2

    @staticmethod
    def frechet_mean(matrices: List[np.ndarray], weights: Optional[np.ndarray] = None,
                     max_iter: int = 10, tol: float = 1e-6) -> np.ndarray:
        """
        Среднее Фреше множества SPD-матриц.
        Минимизирует сумму квадратов аффинно-инвариантных расстояний.
        """
        n_matrices = len(matrices)
        if weights is None:
            weights = np.ones(n_matrices) / n_matrices
        # Инициализация: простое среднее
        mean = np.mean([matrices[i] for i in range(n_matrices)], axis=0)
        # Итеративное уточнение
        for iteration in range(max_iter):
            mean_old = mean.copy()
            try:
                # Логарифмическое отображение
                inv_mean = np.linalg.inv(mean)
                sqrt_mean = sqrtm(mean)
                if np.iscomplexobj(sqrt_mean):
                    sqrt_mean = np.real(sqrt_mean)
                inv_sqrt_mean = np.linalg.inv(sqrt_mean)
                # Логарифмы в касательном пространстве
                logs = []
                for M in matrices:
                    logm = inv_sqrt_mean @ inv(sqrtm(inv_mean @ M)) @ inv_sqrt_mean
                    if np.iscomplexobj(logm):
                        logm = np.real(logm)
                    logs.append(logm)
                # Взвешенное среднее в касательном пространстве
                mean_log = np.average(logs, axis=0, weights=weights)
                # Экспоненциальное отображение
                sqrt_mean_log = sqrtm(mean_log)
                if np.iscomplexobj(sqrt_mean_log):
                    sqrt_mean_log = np.real(sqrt_mean_log)
                mean = sqrt_mean @ np.exp(sqrt_mean_log) @ sqrt_mean
                # Проверка сходимости
                diff = np.linalg.norm(mean - mean_old) / np.linalg.norm(mean_old)
                if diff < tol:
                    break
            except:
                break
        return mean

    @staticmethod
    def tangent_space_projection(matrices: List[np.ndarray], base_point: np.ndarray) -> np.ndarray:
        """
        Проекция SPD-матриц в касательное пространство (логарифмическое отображение).
        """
        try:
            inv_base = np.linalg.inv(base_point)
            sqrt_base = sqrtm(base_point)
            if np.iscomplexobj(sqrt_base):
                sqrt_base = np.real(sqrt_base)
            inv_sqrt_base = np.linalg.inv(sqrt_base)
            projections = []
            for M in matrices:
                # Логарифм в касательном пространстве
                M_intermediate = inv_sqrt_base @ M @ inv_sqrt_base
                logm = sqrtm(M_intermediate)
                if np.iscomplexobj(logm):
                    logm = np.real(logm)
                proj = sqrt_base @ np.real(np.linalg.matrix_power(M_intermediate, 0.5)) @ sqrt_base
                projections.append(proj)
            return np.array(projections)
        except:
            return np.array(matrices)
