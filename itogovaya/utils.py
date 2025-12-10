import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from typing import Callable

# ====================
# 1. Параметры модели
# ====================
lambda_ = 1.0   # интенсивность прибытия
mu = 1.0        # интенсивность обслуживания (должно быть lambda_ == mu)
T = 10.0        # время моделирования
L = 100         # дискретизация: шаг = 1/L
n_trajectories = 1000  # количество траекторий для моделирования
np.random.seed(42)

# ====================
# 2. Дискретизация
# ====================
dt = 1 / L
steps = int(T * L) + 1
time_grid = np.linspace(0, T, steps)

# ====================
# 3. Моделирование процессов
# ====================
def simulate_processes(lambda_: float, mu: float, steps: int, dt: float):
    """
    Моделирование дискретных процессов A, B, D, q, U, R, Y
    по формулам (2.1)-(2.6)
    """
    # Инициализация
    A = np.zeros(steps, dtype=int)
    B = np.zeros(steps, dtype=int)
    D = np.zeros(steps, dtype=int)
    q = np.zeros(steps, dtype=int)
    U = np.zeros(steps, dtype=int)
    R = np.zeros(steps, dtype=int)
    Y = np.zeros(steps, dtype=int)
    
    # Случайные величины ξ, η
    xi = np.random.uniform(0, 1, steps)
    eta = np.random.uniform(0, 1, steps)
    
    for k in range(steps - 1):
        # ΔA и ΔB по формулам (2.7)
        delta_A = 1 if (0.5 < xi[k] < 0.5 + lambda_ * dt) else 0
        delta_B = 1 if (0.5 < eta[k] < 0.5 + mu * dt) else 0
        
        A[k+1] = A[k] + delta_A
        B[k+1] = B[k] + delta_B
        
        # D[k+1] по (2.3)
        D[k+1] = D[k] + (delta_B if q[k] > 0 else 0)
        
        # q[k+1] по (1.4)
        q[k+1] = A[k+1] - D[k+1]
        
        # U[k+1] по (2.4)
        if A[k+1] == A[k] or A[k+1] == 0:
            U[k+1] = U[k]
        else:
            # ξ_{k+1} для знака
            sign_xi = np.random.uniform(0, 1)
            if sign_xi < 0.5:
                U[k+1] = U[k] - 1
            else:
                U[k+1] = U[k] + 1
        
        # R[k+1] по (2.5)
        delta_U = U[k+1] - U[k]
        delta_D = D[k+1] - D[k]
        
        if q[k] == 0:
            R[k+1] = R[k] + delta_U
        elif q[k] == 1:
            R[k+1] = R[k] - R[k] * delta_D
        else:
            R[k+1] = R[k]
        
        # Y[k+1] по (1.7) и (2.6)
        Y[k+1] = q[k+1] * R[k+1]
    
    return {
        'A': A, 'B': B, 'D': D, 'q': q, 
        'U': U, 'R': R, 'Y': Y, 'time': time_grid
    }

# ====================
# 4. Моделирование множества траекторий
# ====================
Y_values_at_T = []

for _ in range(n_trajectories):
    result = simulate_processes(lambda_, mu, steps, dt)
    Y_values_at_T.append(result['Y'][-1])  # значение в момент T

Y_values_at_T = np.array(Y_values_at_T)

# ====================
# 5. Оценка σ̂ по формуле (1.8)
# ====================
sigma_hat = np.sqrt(np.mean(Y_values_at_T**2) / T)
print(f"Оценка σ̂ = {sigma_hat:.4f}")

# ====================
# 6. Эмпирическая функция распределения (1.10)
# ====================
def empirical_cdf(sample: np.ndarray, x: float) -> float:
    """Эмпирическая функция распределения F^эмп(x)"""
    return np.mean(sample <= x)

# Сортировка для построения графика
sorted_Y = np.sort(Y_values_at_T)
empirical_cdf_vals = np.array([empirical_cdf(Y_values_at_T, x) for x in sorted_Y])

# ====================
# 7. Теоретическая функция распределения винеровского процесса (1.9)
# ====================
def theoretical_cdf(x: float, sigma: float, t: float) -> float:
    """F_{σW_t}(x) - функция распределения винеровского процесса"""
    return norm.cdf(x, scale=sigma * np.sqrt(t))

theoretical_cdf_vals = theoretical_cdf(sorted_Y, sigma_hat, T)

# ====================
# 8. Максимальное отклонение (1.11)
# ====================
max_deviation = np.max(np.abs(empirical_cdf_vals - theoretical_cdf_vals))
print(f"Максимальное отклонение = {max_deviation:.4f}")

# ====================
# 9. График сравнения функций распределения
# ====================
plt.figure(figsize=(10, 6))
plt.plot(sorted_Y, empirical_cdf_vals, label='Эмпирическая F(x)', linewidth=2)
plt.plot(sorted_Y, theoretical_cdf_vals, label='Теоретическая F(x) (Wiener)', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title(f'Сравнение эмпирической и теоретической функций распределения\nσ̂ = {sigma_hat:.3f}, Δ = {max_deviation:.3f}')
plt.legend()
plt.grid(True)
plt.show()

# ====================
# 10. Вычисление E[φ(q_t)] по формуле (1.12)
# ====================
def expected_phi(phi: Callable[[float], float], sigma: float, t: float) -> float:
    """
    Приближенное вычисление E[φ(q_t)] через винеровский процесс:
    E[φ(q_t)] ≈ E[φ(|W_t * σ̂|)]
    """
    def integrand(x):
        return norm.pdf(x, scale=sigma * np.sqrt(t)) * phi(np.abs(x * sigma))
    
    integral, error = quad(integrand, -np.inf, np.inf)
    return integral

# Пример: φ(x) = x^2
def phi_example(x: float) -> float:
    return x**2

E_phi_approx = expected_phi(phi_example, sigma_hat, T)
E_phi_empirical = np.mean(phi_example(np.abs(Y_values_at_T)))

print(f"E[φ(q_T)] приближенно (винеровский) = {E_phi_approx:.4f}")
print(f"E[φ(q_T)] эмпирически = {E_phi_empirical:.4f}")

# ====================
# 11. Построение нескольких траекторий (рис.2)
# ====================
plt.figure(figsize=(12, 8))

for i in range(5):  # покажем 5 траекторий
    result = simulate_processes(lambda_, mu, steps, dt)
    plt.plot(result['time'], result['Y'], alpha=0.7, label=f'Траектория {i+1}')

plt.xlabel('Время t')
plt.ylabel('Y_t')
plt.title('Траектории рандомизированного процесса Y_t')
plt.legend()
plt.grid(True)
plt.show()