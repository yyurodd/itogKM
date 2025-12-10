"""
Вспомогательные функции для построения функций распределения
и статистического анализа - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def empirical_distribution(Y_values):
    """
    Построение эмпирической функции распределения F_n*(y)
    
    Parameters:
    -----------
    Y_values : numpy.ndarray
        Выборочный ряд значений случайной величины
    
    Returns:
    --------
    numpy.ndarray : отсортированные значения (вариационный ряд)
    numpy.ndarray : значения функции распределения
    """
    n = len(Y_values)
    sorted_values = np.sort(Y_values)
    empirical_cdf = np.arange(1, n + 1) / n
    return sorted_values, empirical_cdf


def theoretical_distribution(x, sigma, t):
    """
    Теоретическая функция распределения F_σW_t(x)
    для винеровского процесса, умноженного на σ
    
    Важно: σ здесь вычисляется как σ̂ = √(E[Y_t²] / t)
    Поэтому std = σ * √t = √(E[Y_t²])
    
    Parameters:
    -----------
    x : float или numpy.ndarray
        Значение аргумента
    sigma : float
        σ̂ = √(E[Y_t²] / t) из формулы (1.8)
    t : float
        Время
    
    Returns:
    --------
    float или numpy.ndarray : значение функции распределения
    """
    std = sigma * np.sqrt(t)  # = √(E[Y_t²])
    return norm.cdf(x, loc=0, scale=std)


def theoretical_distribution_vectorized(x_values, sigma, t):
    """
    Векторизованная версия теоретической функции распределения
    """
    return theoretical_distribution(x_values, sigma, t)


def compute_theoretical_cdf_for_empirical(empirical_values, sigma, t):
    """
    Вычисление теоретической функции распределения
    для тех же точек, что и эмпирическая
    """
    return theoretical_distribution_vectorized(empirical_values, sigma, t)


def max_deviation(empirical_values, empirical_cdf, theoretical_cdf):
    """
    Вычисление максимального отклонения между эмпирической
    и теоретической функциями распределения
    """
    deviation = np.abs(empirical_cdf - theoretical_cdf)
    return np.max(deviation)


def compute_sigma_hat(Y_values, t):
    """
    Вычисление оценки σ̂ по формуле (1.8):
    σ̂ = √(E[Y_t²] / t)
    
    Parameters:
    -----------
    Y_values : numpy.ndarray
        Массив значений Y_t
    t : float
        Время
    
    Returns:
    --------
    float : оценка σ̂
    """
    if len(Y_values) == 0 or t <= 0:
        return 0.0
    
    E_Y2 = np.mean(Y_values**2)
    sigma_hat = np.sqrt(E_Y2 / t)
    return sigma_hat


def expected_phi_semiemprical(phi_func, sigma, t, method='integral'):
    """
    'Полуэмпирическое' вычисление E[φ(q_t)] по формуле (1.12)
    
    E[φ(q_t)] ≈ E[φ(|W_t * σ|)] = ∫ φ(|x * σ|) * f_W_t(x) dx
    
    Parameters:
    -----------
    phi_func : callable
        Функция φ(x)
    sigma : float
        Оценка σ̂
    t : float
        Время
    method : str
        'integral' - численное интегрирование
        'montecarlo' - метод Монте-Карло
    
    Returns:
    --------
    float : приближенное значение E[φ(q_t)]
    """
    if method == 'integral':
        # Численное интегрирование
        def integrand(x):
            return norm.pdf(x, scale=np.sqrt(t)) * phi_func(np.abs(x * sigma))
        
        result, _ = quad(integrand, -np.inf, np.inf)
        return result
    
    elif method == 'montecarlo':
        # Метод Монте-Карло
        n_samples = 10000
        W_samples = np.random.normal(0, np.sqrt(t), n_samples)
        Y_samples = np.abs(W_samples * sigma)
        phi_values = phi_func(Y_samples)
        return np.mean(phi_values)
    
    else:
        raise ValueError("method должен быть 'integral' или 'montecarlo'")