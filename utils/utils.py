"""
Вспомогательные утилиты для трейнера.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
import numpy as np


def compute_layer_statistics(model: nn.Module) -> Dict[str, Any]:
    """Вычисляет статистику по слоям модели."""
    stats = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                stats[f'{name}.weight_mean'] = weight.mean().item()
                stats[f'{name}.weight_std'] = weight.std().item()
                stats[f'{name}.weight_sparsity'] = (weight.abs() < 1e-3).float().mean().item()

            if hasattr(module, 'bias') and module.bias is not None:
                bias = module.bias.data
                stats[f'{name}.bias_mean'] = bias.mean().item()
                stats[f'{name}.bias_std'] = bias.std().item()

    return stats


def compute_parameter_count(model: nn.Module) -> Dict[str, int]:
    """Считает количество параметров по типам."""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def check_nan_in_model(model: nn.Module) -> List[str]:
    """Проверяет наличие NaN в параметрах модели."""
    nan_layers = []

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_layers.append(name)

    return nan_layers


def get_learning_rates(optimizer: torch.optim.Optimizer) -> List[float]:
    """Получает текущие learning rates из оптимизатора."""
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    return lrs


def compute_weight_updates(old_state_dict: Dict, new_state_dict: Dict) -> Dict[str, float]:
    """Вычисляет изменения весов между двумя состояниями."""
    updates = {}

    for key in old_state_dict:
        if key in new_state_dict:
            old_weight = old_state_dict[key]
            new_weight = new_state_dict[key]

            # Вычисляем различные метрики изменения
            diff = new_weight - old_weight
            updates[f'{key}_update_l1'] = diff.abs().mean().item()
            updates[f'{key}_update_l2'] = diff.norm().item()
            updates[f'{key}_update_max'] = diff.abs().max().item()

    return updates