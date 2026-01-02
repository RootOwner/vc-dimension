"""
utils.py

Вспомогательные функции для обеспечения воспроизводимости,
логирования экспериментов и удобной работы с файловой системой.

Данный модуль не содержит экспериментальной логики и используется
во всех основных скриптах (train.py, evaluate.py, run_all.py).
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


# ---------------------------
# Воспроизводимость
# ---------------------------

def set_seed(seed: int) -> None:
    """
    Фиксирует seed для всех используемых источников случайности.

    Parameters
    ----------
    seed : int
        Значение seed для воспроизводимости экспериментов.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Делаем поведение PyTorch детерминированным
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Работа с директориями
# ---------------------------

def ensure_dir(path: str | Path) -> Path:
    """
    Создаёт директорию, если она не существует.

    Parameters
    ----------
    path : str or Path
        Путь к директории.

    Returns
    -------
    Path
        Объект Path созданной директории.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------
# Логирование
# ---------------------------

def experiment_tag(prefix: str = "exp") -> str:
    """
    Генерирует уникальный идентификатор эксперимента
    на основе текущего времени.

    Parameters
    ----------
    prefix : str
        Префикс имени эксперимента.

    Returns
    -------
    str
        Строковый идентификатор эксперимента.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def log_message(message: str) -> None:
    """
    Унифицированный вывод сообщений в консоль.

    Parameters
    ----------
    message : str
        Сообщение для логирования.
    """
    print(f"[INFO] {message}")
