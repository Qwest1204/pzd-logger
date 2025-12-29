"""
Пример использования UniversalTrainer для CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from trainer import UniversalTrainer
from config.default_config import create_default_config


class SimpleCNN(nn.Module):
    """Простая CNN для CIFAR/MNIST."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_cnn_example():
    """Пример обучения CNN."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Фиктивные данные (замените на реальные)
    train_data = torch.randn(5000, 3, 32, 32)
    train_labels = torch.randint(0, 10, (5000,))
    test_data = torch.randn(1000, 3, 32, 32)
    test_labels = torch.randint(0, 10, (1000,))

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Конфигурация
    config = create_default_config('cnn')

    # Создаем модель
    model = SimpleCNN(num_classes=10).to(device)

    # Оптимизатор
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Функция потерь
    criterion = nn.CrossEntropyLoss()

    # Создаем тренер
    trainer = UniversalTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config
    )

    # Запускаем обучение
    trained_model = trainer.train()

    return trained_model, trainer


if __name__ == '__main__':
    train_cnn_example()