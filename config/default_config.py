"""
Конфигурации по умолчанию для разных типов моделей.
"""

from typing import Dict


def create_default_config(model_type: str = 'transformer') -> Dict:
    """Создает конфигурацию по умолчанию для разных типов моделей."""

    configs = {
        'cnn': {
            'num_epochs': 100,
            'log_interval': 10,
            'log_weights_every': 5,
            'log_activations_every': 10,
            'log_loss_landscape_every': 20,
            'loss_landscape_steps': 25,
            'save_checkpoint_every': 50,
            'early_stopping_patience': 20,
            'grad_clip': 5.0,
        },

    return configs.get(model_type, configs['transformer']).copy()