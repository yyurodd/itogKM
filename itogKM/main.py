"""
Графический интерфейс для имитационного моделирования
случайных блужданий методами рандомизации
Использует Streamlit для простого веб-интерфейса
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from model import QueueModel

from utils import (empirical_distribution, compute_theoretical_cdf_for_empirical,
                   max_deviation, compute_sigma_hat)

# Настройка страницы
st.set_page_config(page_title="Имитационное моделирование случайных блужданий", 
                   layout="wide")

# Инициализация session_state по умолчанию
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.run_simulation = False
    st.session_state.trajectories = []
    st.session_state.Y_values = None
    st.session_state.Y_values_at_T = None
    st.session_state.sigma = None
    st.session_state.sigma_hat = None
    st.session_state.max_dev = None
    st.session_state.t_max = None
    st.session_state.empirical_values = None
    st.session_state.empirical_cdf = None
    st.session_state.theoretical_cdf = None
    st.session_state.model_params = None

st.title("Имитационное моделирование случайных блужданий")
st.markdown("---")

# Информационная панель
with st.expander("Информация о модели"):
    st.markdown("""
    **Модель:** Система массового обслуживания с методом рандомизации
    
    **Условие:** λ = μ (необходимо для применения метода)
    
    **Процессы:**
    - A_t: процесс поступления заявок (Пуассоновский)
    - B_t: процесс обслуживания (Пуассоновский)
    - D_t: процесс обслуженных заявок
    - q_t: длина очереди
    - R_t: процесс расклейки очереди
    - Y_t = q_t * R_t: рандомизированный процесс
    
    **Оценка параметра:** σ̂ = √(E[Y_t²] / t)
    """)

# Боковая панель с параметрами
with st.sidebar:
    st.header("Параметры моделирования")
    
    col1, col2 = st.columns(2)
    with col1:
        lambda_param = st.number_input("λ:", min_value=0.1, value=1.0, step=0.1,
                                      help="Интенсивность поступления заявок")
    with col2:
        mu_param = st.number_input("μ:", min_value=0.1, value=1.0, step=0.1,
                                  help="Интенсивность обслуживания")
    
    t_max = st.number_input("Время t:", min_value=0.1, value=10.0, step=0.5)
    n_trajectories = st.number_input("Количество траекторий n:", 
                                     min_value=10, value=1000, step=100)
    n_display = st.number_input("Траекторий для отображения:", 
                                min_value=1, value=20, step=1)
    
    dt = st.slider("Шаг дискретизации:", 0.001, 0.1, 0.01, 0.001,
                   help="Шаг по времени для дискретизации")
    
    seed = st.number_input("Seed:", value=42, 
                          help="Seed для воспроизводимости результатов")
    
    run_button = st.button("🚀 Запустить моделирование", type="primary", use_container_width=True)

# Основная область
if run_button:
    # Проверка условия λ = μ
    if abs(lambda_param - mu_param) > 1e-6:
        st.error("⚠️ Для применения метода рандомизации необходимо λ = μ")
        st.info("Пожалуйста, установите λ = μ и запустите моделирование снова.")
    else:
        with st.spinner('Выполняется моделирование...'):
            try:
                # Создание модели с исправленным классом
                model = QueueModel(lambda_param, mu_param, t_max, dt, seed)
                
                # Прогресс-бар
                progress_bar = st.progress(0)
                
                # Генерация траекторий (по одной для отслеживания прогресса)
                Y_values_at_T = []
                example_trajectories = []
                
                for i in range(n_trajectories):
                    traj = model.simulate_trajectory()
                    Y_values_at_T.append(traj['Y'][-1])
                    
                    if i < n_display:
                        example_trajectories.append(traj)
                    
                    # Обновление прогресса
                    if i % (n_trajectories // 20) == 0:
                        progress_bar.progress(i / n_trajectories)
                
                Y_values_at_T = np.array(Y_values_at_T)
                
                # Два способа оценки σ̂
                sigma_hat1 = model.estimate_sigma(Y_values_at_T)
                sigma_hat2 = compute_sigma_hat(Y_values_at_T, t_max)
                
                # Используем один из них (они должны быть одинаковыми)
                sigma_hat = sigma_hat1
                
                # Построение функций распределения
                empirical_values, empirical_cdf = empirical_distribution(Y_values_at_T)
                theoretical_cdf = compute_theoretical_cdf_for_empirical(
                    empirical_values, sigma_hat, t_max)
                
                # Вычисление максимального отклонения
                max_dev = max_deviation(empirical_values, empirical_cdf, theoretical_cdf)
                
                # Сохранение результатов
                st.session_state.update({
                    'trajectories': example_trajectories,
                    'Y_values_at_T': Y_values_at_T,
                    'sigma_hat': sigma_hat,
                    'max_dev': max_dev,
                    't_max': t_max,
                    'empirical_values': empirical_values,
                    'empirical_cdf': empirical_cdf,
                    'theoretical_cdf': theoretical_cdf,
                    'model_params': {
                        'lambda': lambda_param,
                        'mu': mu_param,
                        't_max': t_max,
                        'n_trajectories': n_trajectories
                    }
                })
                
                progress_bar.progress(1.0)
                st.success("Моделирование завершено!")
                
            except Exception as e:
                st.error(f"Ошибка при моделировании: {str(e)}")

# Отображение результатов
if 'trajectories' in st.session_state:
    st.markdown("---")
    st.header("📊 Результаты моделирования")
    
    # Панель с метриками
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sigma_hat_val = st.session_state.sigma_hat
        st.metric("Оценка σ̂", f"{sigma_hat_val:.6f}" if sigma_hat_val is not None else "—")
    with col2:
        max_dev_val = st.session_state.max_dev
        st.metric("Макс. отклонение D_n", f"{max_dev_val:.6f}" if max_dev_val is not None else "—")
    with col3:
        y_vals = st.session_state.get('Y_values_at_T')
        st.metric("Количество траекторий", f"{len(y_vals)}" if y_vals is not None else "—")
    with col4:
        t_val = st.session_state.t_max
        st.metric("Время t", f"{t_val}" if t_val is not None else "—")
    
    # Информация о параметрах
    with st.expander("Параметры модели"):
        params = st.session_state.model_params
        if params:
            st.write(f"λ = {params['lambda']}, μ = {params['mu']}")
            st.write(f"t = {params['t_max']}, n = {params['n_trajectories']}")
        else:
            st.info("Параметры будут доступны после запуска моделирования.")
    
    # Графики процессов
st.markdown("---")
st.header("📈 Графики процессов")

# Выбор процессов для отображения
processes_to_show = st.multiselect(
    "Выберите процессы для отображения:",
    ['A_t', 'B_t', 'D_t', 'q_t', 'U_t', 'R_t', 'Y_t'],
    default=['q_t', 'Y_t']
)

# Маппинг имен процессов
process_mapping = {
    'A_t': ('A', 'Процесс поступления A_t'),
    'B_t': ('B', 'Процесс обслуживания B_t'),
    'D_t': ('D', 'Процесс обслуженных D_t'),
    'q_t': ('q', 'Длина очереди q_t'),
    'U_t': ('U', 'Процесс рандомизации U_t'),
    'R_t': ('R', 'Расклейка очереди R_t'),
    'Y_t': ('Y', 'Рандомизированный процесс Y_t')
}

if processes_to_show:
    n_cols = 2
    n_rows = (len(processes_to_show) + n_cols - 1) // n_cols
    
    # Создаем фигуру с подходящей высотой
    fig_height = 4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, fig_height))
    # Приводим к плоскому массиву осей
    axes = np.atleast_1d(axes).reshape(-1)
    
    trajectories = st.session_state.trajectories
    
    for idx, process_name in enumerate(processes_to_show):
        ax = axes[idx]
        
        process_key, process_title = process_mapping[process_name]
        
        # Рисуем траектории
        for i, traj in enumerate(trajectories):
            time = traj['time']
            values = traj[process_key]
            ax.plot(time, values, linewidth=0.8, alpha=0.6)
        
        ax.set_xlabel('Время t', fontsize=10)
        ax.set_ylabel(process_name, fontsize=10)
        ax.set_title(process_title, fontsize=12, pad=10)  # pad добавляет отступ
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Скрываем пустые subplots (если их больше, чем выбрано процессов)
    for idx in range(len(processes_to_show), len(axes)):
        axes[idx].set_visible(False)
    
    # Автоматическая регулировка отступов
    plt.tight_layout()
    
    # Дополнительная регулировка если нужно
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Увеличиваем расстояние между графиками
    
    st.pyplot(fig)
    
    # График функций распределения
    st.markdown("---")
    st.header("📊 Функции распределения")
    
    empirical_values = st.session_state.empirical_values
    empirical_cdf = st.session_state.empirical_cdf
    theoretical_cdf = st.session_state.theoretical_cdf
    
    if empirical_values is not None and empirical_cdf is not None and theoretical_cdf is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            
            ax_dist.plot(empirical_values, empirical_cdf, 'b-', 
                        label='Эмпирическая F_n*(y)', linewidth=2, alpha=0.8)
            ax_dist.plot(empirical_values, theoretical_cdf, 'r--', 
                        label=f'Теоретическая F_σW_t(x)', linewidth=2, alpha=0.8)
            
            # Отметка максимального отклонения
            idx_max_dev = np.argmax(np.abs(empirical_cdf - theoretical_cdf))
            ax_dist.plot([empirical_values[idx_max_dev], empirical_values[idx_max_dev]],
                        [empirical_cdf[idx_max_dev], theoretical_cdf[idx_max_dev]],
                        'g-', linewidth=2, alpha=0.5, label=f'D_n = {st.session_state.max_dev:.4f}')
            
            ax_dist.set_xlabel('Значение y')
            ax_dist.set_ylabel('F(y)')
            ax_dist.set_title(f'Сравнение функций распределения (σ̂ = {st.session_state.sigma_hat:.4f})')
            ax_dist.legend()
            ax_dist.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_dist)
        
        with col2:
            st.metric("Макс. отклонение", f"{st.session_state.max_dev:.6f}")
            st.metric("σ̂", f"{st.session_state.sigma_hat:.6f}")
            
            # Статистика по Y_t
            Y_stats = {
                'Среднее': np.mean(st.session_state.Y_values_at_T),
                'Дисперсия': np.var(st.session_state.Y_values_at_T),
                'Мин': np.min(st.session_state.Y_values_at_T),
                'Макс': np.max(st.session_state.Y_values_at_T),
                'E[Y²]': np.mean(st.session_state.Y_values_at_T ** 2),
                'E[Y²]/t': np.mean(st.session_state.Y_values_at_T ** 2) / st.session_state.t_max
            }
            
            st.write("**Статистика Y_t:**")
            for key, value in Y_stats.items():
                st.write(f"{key}: {value:.4f}")
    else:
        st.info("Запустите моделирование, чтобы построить функции распределения.")

else:
    # Начальный экран
    st.info("""
    👈 **Введите параметры в боковой панели и нажмите 'Запустить моделирование'**
    
    **Рекомендуемые параметры для начала:**
    - λ = μ = 1.0
    - t = 10.0
    - n = 1000
    - Шаг дискретизации = 0.01
    """)
    