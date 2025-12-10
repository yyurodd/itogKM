"""
Математическая модель системы массового обслуживания
с методом рандомизации для имитационного моделирования случайных блужданий
Исправленная версия, соответствующая статье
"""

import numpy as np
from scipy.stats import poisson


class QueueModel:
    """
    Класс для моделирования системы массового обслуживания
    с применением метода рандомизации согласно статье
    """
    
    def __init__(self, lambda_param, mu_param, t_max, dt=0.01, seed=None):
        """
        Инициализация модели
        
        Parameters:
        -----------
        lambda_param : float
            Интенсивность процесса поступления заявок A_t
        mu_param : float
            Интенсивность процесса обслуживания B_t
        t_max : float
            Максимальное время моделирования
        dt : float
            Шаг дискретизации времени
        seed : int, optional
            Seed для генератора случайных чисел
        """
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.t_max = t_max
        self.dt = dt
        self.time_points = np.arange(0, t_max + dt, dt)
        self.n_steps = len(self.time_points)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Проверка условия λ = μ
        if abs(lambda_param - mu_param) > 1e-6:
            raise ValueError("Для применения метода рандомизации необходимо λ = μ")
    
    def simulate_trajectory(self):
        """
        Генерация одной траектории всех процессов согласно статье
        
        Returns:
        --------
        dict : словарь с траекториями процессов
            - 'time': массив временных точек
            - 'A': процесс поступления заявок
            - 'B': процесс обслуживания
            - 'D': процесс обслуженных заявок
            - 'q': длина очереди
            - 'U': процесс рандомизации
            - 'R': процесс расклейки очереди
            - 'Y': процесс Y_t = q_t * R_t (не нормированный!)
        """
        # Инициализация массивов
        A = np.zeros(self.n_steps, dtype=int)
        B = np.zeros(self.n_steps, dtype=int)
        D = np.zeros(self.n_steps, dtype=int)
        q = np.zeros(self.n_steps, dtype=int)
        U = np.zeros(self.n_steps, dtype=int)
        R = np.zeros(self.n_steps, dtype=int)
        Y = np.zeros(self.n_steps, dtype=int)
        
        # Для процесса U нужны случайные ξ_j
        # Максимально возможное количество поступлений
        max_expected_A = int(self.lambda_param * self.t_max * 3)  # С запасом
        
        # ξ_j ~ {-1, 1} с вероятностью 1/2
        xi = np.random.choice([-1, 1], size=max_expected_A)
        
        # Моделирование для каждого временного шага
        for i in range(1, self.n_steps):
            dt = self.time_points[i] - self.time_points[i-1]
            
            # 1. Процессы A и B - пуассоновские
            dA = np.random.poisson(self.lambda_param * dt)
            dB = np.random.poisson(self.mu_param * dt)
            
            A[i] = A[i-1] + dA
            B[i] = B[i-1] + dB
            
            # 2. Процесс D с компенсатором: D_t = ∫₀ᵗ I{q_s > 0} dB_s
            if q[i-1] > 0:
                dD = dB
            else:
                dD = 0
            D[i] = D[i-1] + dD
            
            # 3. Длина очереди: q_t = A_t - D_t
            q[i] = A[i] - D[i]
            q[i] = max(0, q[i])  # Очередь не может быть отрицательной
            
            # 4. Процесс U_t = Σ_{j=1}^{A_t} ξ_j
            if A[i] > 0:
                U[i] = np.sum(xi[:A[i]])
            else:
                U[i] = 0
            
            # 5. Процесс R_t - расклейка очереди (дискретная аппроксимация формулы 1.6)
            # R_t = ∫₀ᵗ I(q_{s-} = 0) dU_s - ∫₀ᵗ I(q_{s-} = 1) R_{s-} dD_s
            dU = U[i] - U[i-1]
            dD_step = D[i] - D[i-1]
            
            if q[i-1] == 0:
                dR = dU
            elif q[i-1] == 1:
                dR = -R[i-1] * dD_step
            else:
                dR = 0
            
            R[i] = R[i-1] + dR
            
            # 6. Процесс Y_t = q_t * R_t (формула 1.7)
            Y[i] = q[i] * R[i]
        
        return {
            'time': self.time_points,
            'A': A,
            'B': B,
            'D': D,
            'q': q,
            'U': U,
            'R': R,
            'Y': Y
        }
    
    def simulate_multiple(self, n_trajectories=1000, n_display=20):
        """
        Генерация n траекторий
        
        Parameters:
        -----------
        n_trajectories : int
            Количество траекторий для генерации
        n_display : int
            Количество траекторий для сохранения и отображения
        
        Returns:
        --------
        numpy.ndarray : массив значений Y_t_max для всех траекторий
        list : список траекторий для визуализации
        """
        Y_values_at_T = []
        example_trajectories = []
        
        for i in range(n_trajectories):
            traj = self.simulate_trajectory()
            # Берем значение Y_t в момент t_max (не нормированное!)
            Y_t = traj['Y'][-1]
            Y_values_at_T.append(Y_t)
            
            if i < n_display:
                example_trajectories.append(traj)
        
        return np.array(Y_values_at_T), example_trajectories
    
    def estimate_sigma(self, Y_values_at_T):
        """
        Оценка параметра σ̂ по формуле (1.8) из статьи:
        σ̂ = √(E[Y_t²] / t)
        
        Parameters:
        -----------
        Y_values_at_T : numpy.ndarray
            Массив значений Y_t (не нормированных!) для n траекторий
        
        Returns:
        --------
        float : оценка параметра σ̂
        """
        n = len(Y_values_at_T)
        t = self.t_max
        
        if t <= 0 or n == 0:
            return 0.0
        
        # E[Y_t²] / t
        E_Y2 = np.mean(Y_values_at_T ** 2)
        sigma_hat_squared = E_Y2 / t
        
        return np.sqrt(sigma_hat_squared)