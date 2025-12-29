"""
Функции для вычисления и визуализации лосс-ландшафта.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import io
from PIL import Image
import torchvision.transforms as transforms


def compute_loss_landscape(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        steps: int = 20,
        range_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляет лосс-ландшафт вокруг текущих параметров модели.

    Args:
        model: Нейронная сеть
        train_loader: DataLoader с тренировочными данными
        criterion: Функция потерь
        device: Устройство для вычислений
        steps: Количество шагов в каждом направлении
        range_scale: Масштаб для исследования

    Returns:
        X, Y, Z: Координаты и значения лосса
    """
    model.eval()

    # Получаем батч данных для оценки
    try:
        data_iter = iter(train_loader)
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)

    # Обрабатываем батч в зависимости от формата
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            inputs, targets = batch
        elif len(batch) == 3:
            targets, inputs = batch[0], batch[1]
        else:
            raise ValueError(f"Неизвестный формат батча: {type(batch)}")
    else:
        inputs = batch
        targets = None

    inputs = inputs.to(device)
    if targets is not None:
        targets = targets.to(device)

    # Сохраняем исходные параметры
    original_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            original_params[name] = param.data.clone()

    # Выбираем два случайных направления
    directions = {}
    for name, param in original_params.items():
        d1 = torch.randn_like(param)
        d2 = torch.randn_like(param)
        d1 = d1 / torch.norm(d1)
        d2 = d2 / torch.norm(d2)
        directions[name] = (d1, d2)

    # Создаем сетку для оценки
    x = np.linspace(-range_scale, range_scale, steps)
    y = np.linspace(-range_scale, range_scale, steps)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    with torch.no_grad():
        for i in range(steps):
            for j in range(steps):
                alpha = X[i, j]
                beta = Y[i, j]

                # Применяем смещение к параметрам
                for name, param in model.named_parameters():
                    if name in original_params:
                        d1, d2 = directions[name]
                        param.data = original_params[name] + alpha * d1 + beta * d2

                # Вычисляем лосс
                outputs = model(inputs)

                # Обработка разных форматов выходов
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                if targets is not None:
                    # Поддерживаем разные задачи
                    if logits.dim() == 3 and targets.dim() == 2:
                        # Для задач типа sequence-to-sequence
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    else:
                        # Для стандартных задач классификации
                        loss = criterion(logits, targets)
                    Z[i, j] = loss.item()
                else:
                    # Если нет таргетов, используем какой-то другой критерий
                    Z[i, j] = logits.mean().item()

    # Восстанавливаем исходные параметры
    for name, param in model.named_parameters():
        if name in original_params:
            param.data = original_params[name]

    return X, Y, Z


def plot_loss_landscape_3d(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
) -> torch.Tensor:
    """Создает 3D визуализацию лосс-ландшафта."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Нормализуем Z для лучшей визуализации
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)

    # Создаем поверхность
    surf = ax.plot_surface(
        X, Y, Z_norm,
        cmap='viridis',
        linewidth=0,
        antialiased=True,
        alpha=0.8,
        rstride=1,
        cstride=1
    )

    # Добавляем контурные линии
    ax.contour(X, Y, Z, 10, offset=Z_norm.min(), cmap='coolwarm', alpha=0.5)

    # Настройки
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Normalized Loss')
    ax.set_title(f'Loss Landscape (Min: {Z.min():.4f}, Max: {Z.max():.4f})')

    # Цветовая шкала
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Сохраняем в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # Преобразуем в тензор
    image = Image.open(buf)
    image_tensor = transforms.ToTensor()(image)

    return image_tensor


def plot_loss_landscape_2d(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
) -> torch.Tensor:
    """Создает 2D тепловую карту лосс-ландшафта."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Нормализуем для лучшей визуализации
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)

    # Контурная карта
    contour = ax.contourf(X, Y, Z_norm, levels=20, cmap='viridis')
    ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)

    # Текущая позиция
    ax.plot(0, 0, 'ro', markersize=10, label='Current position')

    # Настройки
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_title('Loss Landscape Contour')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Цветовая шкала
    plt.colorbar(contour, ax=ax)

    # Сохраняем
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # В тензор
    image = Image.open(buf)
    image_tensor = transforms.ToTensor()(image)

    return image_tensor