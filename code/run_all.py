#!/usr/bin/env python3
"""
run_all.py

Главный автозапуск эксперимента.
Запускает обучение моделей, вычисление метрик сложности
и генерацию всех графиков.
"""

import os
import random
import subprocess
import yaml
import numpy as np
import torch


# -----------------------------
# Фиксация seed (воспроизводимость)
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Загрузка конфигурации
# -----------------------------
def load_config():
    """
    Загружает config.yaml из директории code/
    независимо от текущей рабочей директории.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "config.yaml")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise RuntimeError("config.yaml пустой или некорректный")

    return config


# -----------------------------
# Запуск отдельных этапов
# -----------------------------
def run_step(name, command):
    print(f"\n===== {name} =====")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка на этапе: {name}")


# -----------------------------
# Основной pipeline
# -----------------------------
def main():
    print("=== Запуск полного эксперимента ===")

    config = load_config()
    seed = config.get("seed", 42)
    set_seed(seed)

    print(f"Используем seed = {seed}")

    # Пути
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code_dir = os.path.join(root_dir, "code")
    figures_dir = os.path.join(root_dir, "figures")
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    os.chdir(code_dir)

    # 1. Обучение моделей
    run_step(
        "Обучение моделей",
        "python3 train.py"
    )

    # 2. Вычисление accuracy и margin
    run_step(
        "Вычисление accuracy и margin",
        "python3 evaluate.py"
    )

    # 3. Вычисление norm-based сложности (Ψ̂)
    run_step(
        "Вычисление norm-based сложности",
        "python3 complexity.py"
    )

    # 4. Генерация графиков
    run_step(
        "Генерация графиков",
        "python3 plots.py"
    )

    print("\n=== Эксперимент успешно завершён ===")
    print("Все графики сохранены в /figures/")
    print("Результаты сохранены в results/")


if __name__ == "__main__":
    main()
