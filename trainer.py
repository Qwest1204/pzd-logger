"""
Основной класс UniversalTrainer.
Содержит логику обучения, тестирования и базового логирования.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import datetime
import os
from typing import Dict
import warnings

from utils.loss_visual import compute_loss_landscape, plot_loss_landscape_3d, plot_loss_landscape_2d


class UniversalTrainer:
    """
    Универсальный тренер с полным логированием для любых архитектур нейронных сетей.
    Поддерживает: трансформеры, CNN, RNN, линейные модели и т.д.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            config: Dict
    ):
        """
        Инициализация тренера.

        Args:
            model: Нейронная сеть (любая архитектура)
            train_loader: DataLoader для тренировочных данных
            test_loader: DataLoader для тестовых данных
            optimizer: Оптимизатор
            criterion: Функция потерь
            device: Устройство (CPU/GPU)
            config: Конфигурация обучения
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

        # Создаем директории для сохранения
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"runs/experiment_{current_time}"
        self.checkpoint_dir = f"checkpoints/experiment_{current_time}"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Инициализируем TensorBoard
        self.writer = SummaryWriter(self.log_dir)

        # Состояние обучения
        self.epoch = 0
        self.global_step = 0
        self.best_test_acc = 0.0
        self.train_history = {
            'loss': [], 'accuracy': [], 'grad_norm': []
        }
        self.test_history = {
            'loss': [], 'accuracy': []
        }

        # Внутренние флаги
        self._has_attention = False
        self._data_format_checked = False

    def compute_gradient_norm(self) -> float:
        """Вычисляет норму градиентов модели."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def log_model_weights_and_gradients(self, step: int):
        """Логирует веса и градиенты модели в TensorBoard."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Веса
                self.writer.add_histogram(f'weights/{name}', param.data, step)

                # Градиенты
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, step)

                # Статистика весов
                self.writer.add_scalar(f'weights_stats/{name}_mean', param.data.mean().item(), step)
                self.writer.add_scalar(f'weights_stats/{name}_std', param.data.std().item(), step)

    def log_loss_landscape(self, epoch: int):
        """Логирует лосс-ландшафт в TensorBoard."""
        try:
            print(f"\n[{datetime.datetime.now()}] Computing loss landscape for epoch {epoch}...")

            # Вычисляем с разными масштабами
            scales = [0.1, 0.5, 1.0, 2.0]
            steps = self.config.get('loss_landscape_steps', 30)

            for scale in scales:
                X, Y, Z = compute_loss_landscape(
                    model=self.model,
                    train_loader=self.train_loader,
                    criterion=self.criterion,
                    device=self.device,
                    steps=steps,
                    range_scale=scale
                )

                # 3D визуализация
                landscape_3d = plot_loss_landscape_3d(X, Y, Z)
                self.writer.add_image(f'loss_landscape/3d_scale_{scale}', landscape_3d, epoch)

                # 2D визуализация
                landscape_2d = plot_loss_landscape_2d(X, Y, Z)
                self.writer.add_image(f'loss_landscape/2d_scale_{scale}', landscape_2d, epoch)

                # Статистика
                self.writer.add_scalar(f'loss_landscape_scale_{scale}/min', Z.min(), epoch)
                self.writer.add_scalar(f'loss_landscape_scale_{scale}/max', Z.max(), epoch)
                self.writer.add_scalar(f'loss_landscape_scale_{scale}/mean', Z.mean(), epoch)
                self.writer.add_scalar(f'loss_landscape_scale_{scale}/std', Z.std(), epoch)

                # Градиент поверхности
                if Z.shape[0] > 1 and Z.shape[1] > 1:
                    grad_x, grad_y = np.gradient(Z)
                    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
                    self.writer.add_scalar(f'loss_landscape_scale_{scale}/grad_max', grad_magnitude.max(), epoch)
                    self.writer.add_scalar(f'loss_landscape_scale_{scale}/grad_mean', grad_magnitude.mean(), epoch)

            print(f"[{datetime.datetime.now()}] Loss landscape computed successfully.")

        except Exception as e:
            print(f"[{datetime.datetime.now()}] Error computing loss landscape: {e}")
            warnings.warn(f"Loss landscape computation failed: {e}")

    def log_attention_maps(self, attentions: torch.Tensor, step: int, prefix: str = "train"):
        """Логирует карты внимания (если модель их возвращает)."""
        if attentions is not None:
            self._has_attention = True
            # Поддерживаем разные форматы внимания
            if attentions.dim() == 4:
                # Формат: (batch, heads, seq_len, seq_len)
                for head_idx in range(min(attentions.shape[1], 8)):  # Логируем до 8 голов
                    attention_map = attentions[0, head_idx].detach().cpu()
                    self.writer.add_image(
                        f'{prefix}/attention_head_{head_idx}',
                        attention_map.unsqueeze(0),
                        step
                    )
            elif attentions.dim() == 3:
                # Формат: (batch, seq_len, seq_len)
                attention_map = attentions[0].detach().cpu()
                self.writer.add_image(
                    f'{prefix}/attention',
                    attention_map.unsqueeze(0),
                    step
                )

    def log_activations(self, step: int):
        """Логирует активации модели (через хуки)."""
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()

            return hook

        # Регистрируем хуки для всех слоев
        hooks = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.Sigmoid, nn.Tanh)):
                hook = layer.register_forward_hook(get_activation(name))
                hooks.append(hook)

        # Пропускаем один батч
        try:
            data_iter = iter(self.train_loader)
            batch = next(data_iter)

            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    inputs = batch[0]
                else:
                    inputs = batch
            else:
                inputs = batch

            inputs = inputs.to(self.device)
            _ = self.model(inputs)

            # Логируем активации
            for name, activation in activations.items():
                if activation.numel() > 0:
                    self.writer.add_histogram(f'activations/{name}', activation, step)

                    # Средняя активация
                    self.writer.add_scalar(f'activations_stats/{name}_mean',
                                           activation.mean().item(), step)
                    self.writer.add_scalar(f'activations_stats/{name}_sparsity',
                                           (activation.abs() < 0.01).float().mean().item(), step)

        except Exception as e:
            print(f"Error logging activations: {e}")

        # Удаляем хуки
        for hook in hooks:
            hook.remove()

    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Вычисляет точность для разных типов задач.

        TODO: Реализуйте эту функцию под свою задачу!

        Примеры для разных задач:

        1. Классификация (logits: [batch, classes], targets: [batch]):
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == targets).float().mean() * 100

        2. Sequence-to-sequence (logits: [batch, seq_len, vocab], targets: [batch, seq_len]):
            _, predicted = torch.max(logits, dim=-1)
            accuracy = (predicted == targets).float().mean() * 100

        3. Регрессия (может потребоваться RMSE или другая метрика):
            mse = F.mse_loss(logits, targets)
            accuracy = 0.0  # для регрессии точность не применима

        4. Multi-label классификация:
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == targets).float().mean() * 100
        """
        raise NotImplementedError(
            "Метод compute_accuracy должен быть реализован под вашу задачу.\n"
            "Примеры реализации для разных задач смотрите в docstring метода."
        )

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Сохраняет чекпоинт модели."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_test_acc': self.best_test_acc,
            'train_history': self.train_history,
            'test_history': self.test_history,
            'config': self.config
        }

        # Регулярный чекпоинт
        torch.save(checkpoint, f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt")

        # Лучшая модель
        if is_best:
            torch.save(checkpoint, f"{self.checkpoint_dir}/best_model.pt")
            print(
                f"[{datetime.datetime.now()}] Best model saved at epoch {epoch} with accuracy: {self.best_test_acc:.2f}%")

    def _prepare_batch(self, batch):
        """
        Подготавливает батч данных. Переопределите этот метод под свой формат данных.

        Должен возвращать кортеж (inputs, targets).

        Примеры:
        1. Стандартный формат: batch = (inputs, targets)
        2. Grokking формат: batch = (targets, inputs, ...)
        3. Только inputs: batch = inputs
        """
        if not self._data_format_checked:
            print("\n" + "=" * 60)
            print("ВАЖНО: Проверьте формат данных в вашем DataLoader!")
            print("Батч имеет тип:", type(batch))
            if isinstance(batch, (list, tuple)):
                print(f"Длина батча: {len(batch)}")
                for i, item in enumerate(batch):
                    print(f"  Элемент {i}: тип={type(item)}, форма={item.shape if hasattr(item, 'shape') else 'N/A'}")
            elif hasattr(batch, 'shape'):
                print(f"Форма батча: {batch.shape}")
            print("=" * 60 + "\n")
            self._data_format_checked = True

        # Пример реализации - настройте под свой формат
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                # Предполагаем (inputs, targets) или (targets, inputs)
                try:
                    # Пробуем определить по форме
                    if batch[0].dim() >= batch[1].dim():
                        # Первый элемент вероятнее targets
                        inputs, targets = batch[1], batch[0]
                    else:
                        inputs, targets = batch[0], batch[1]
                except:
                    # По умолчанию берем первые два элемента
                    inputs, targets = batch[0], batch[1]
            else:
                inputs = batch[0]
                targets = None
        else:
            inputs = batch
            targets = None

        return inputs, targets

    def train_epoch(self, epoch: int) -> Dict:
        """Выполняет одну эпоху обучения."""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [Train]')

        for batch_idx, batch in enumerate(train_bar):
            ###################################################################
            # ШАГ 1: ПОДГОТОВКА ДАННЫХ (НАСТРОЙТЕ ПОД СВОЮ ЗАДАЧУ)
            ###################################################################
            inputs, targets = self._prepare_batch(batch)
            inputs = inputs.to(self.device)

            if targets is not None:
                targets = targets.to(self.device)

            ###################################################################
            # ШАГ 2: ПРОХОД ВПЕРЕД (НАСТРОЙТЕ ПОД СВОЮ МОДЕЛЬ)
            ###################################################################
            # Примеры:
            # outputs = self.model(inputs)  # Стандартный вызов
            # outputs, attentions = self.model(inputs)  # Если модель возвращает внимание
            # outputs = self.model(inputs, targets)  # Если нужны таргеты

            # TODO: Вызовите вашу модель с правильными аргументами
            # outputs = self.model(???)

            ###################################################################
            # ШАГ 3: ВЫЧИСЛЕНИЕ ЛОССА (НАСТРОЙТЕ ПОД СВОЮ ЗАДАЧУ)
            ###################################################################
            # Примеры:
            # loss = self.criterion(outputs, targets)  # Стандартный
            # loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))  # Seq2seq
            # loss = self.criterion(outputs, targets.squeeze(1))  # Если targets лишняя размерность

            # TODO: Вычислите loss для вашей задачи
            # loss = ???

            ###################################################################
            # ШАГ 4: ОБРАТНЫЙ ПРОХОД И ОПТИМИЗАЦИЯ
            ###################################################################
            # self.optimizer.zero_grad()
            # loss.backward()

            # # Gradient clipping (опционально)
            # if self.config.get('grad_clip', 0) > 0:
            #     torch.nn.utils.clip_grad_norm_(
            #         self.model.parameters(),
            #         self.config['grad_clip']
            #     )

            # self.optimizer.step()
            ###################################################################

            ###################################################################
            # ШАГ 5: СБОР СТАТИСТИКИ (ОБЩИЙ ДЛЯ ВСЕХ ЗАДАЧ)
            ###################################################################
            # TODO: Обновите статистику
            # train_loss += loss.item()
            # accuracy = self.compute_accuracy(outputs, targets)
            # train_correct += (accuracy / 100) * targets.size(0)
            # train_total += targets.size(0)
            ###################################################################

            # Временные значения для демонстрации (удалите в реальном использовании)
            if batch_idx == 0:
                print("\n" + "=" * 60)
                print("ВАЖНО: Заполните шаги обучения в train_epoch!")
                print("Смотрите комментарии TODO в коде.")
                print("=" * 60 + "\n")

            # Логирование на каждом шаге
            if batch_idx % self.config.get('log_interval', 10) == 0:
                # Норма градиентов
                grad_norm = self.compute_gradient_norm()

                # Временные значения (замените на реальные)
                current_loss = 0.0  # Замените на loss.item()
                current_accuracy = 0.0  # Замените на accuracy

                # Логируем в TensorBoard
                self.writer.add_scalar('train/loss_step', current_loss, self.global_step)
                self.writer.add_scalar('train/gradient_norm', grad_norm, self.global_step)
                self.writer.add_scalar('train/learning_rate',
                                       self.optimizer.param_groups[0]['lr'],
                                       self.global_step)

                # Логируем внимание (если есть)
                if 'attentions' in locals():
                    self.log_attention_maps(attentions, self.global_step, "train")

                # Обновляем прогресс-бар
                train_bar.set_postfix({
                    'loss': current_loss,
                    'acc': current_accuracy,
                    'grad': grad_norm
                })

            self.global_step += 1

        # Вычисляем средние значения за эпоху
        avg_train_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0

        # Сохраняем историю
        self.train_history['loss'].append(avg_train_loss)
        self.train_history['accuracy'].append(train_accuracy)

        return {
            'loss': avg_train_loss,
            'accuracy': train_accuracy
        }

    def test_epoch(self, epoch: int) -> Dict:
        """Выполняет одну эпоху тестирования."""
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            test_bar = tqdm(self.test_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [Test]')

            for batch_idx, batch in enumerate(test_bar):
                ###################################################################
                # ТЕСТИРОВАНИЕ (АНАЛОГИЧНО ТРЕНИРОВКЕ, НО БЕЗ ОПТИМИЗАЦИИ)
                ###################################################################
                # TODO: Реализуйте тестирование аналогично train_epoch
                # но без backward() и optimizer.step()
                ###################################################################

                # Временные значения
                if batch_idx == 0:
                    loss = torch.tensor(0.0)
                    accuracy = 0.0

                # Обновление статистики
                # test_loss += loss.item()
                # test_correct += (accuracy / 100) * targets.size(0)
                # test_total += targets.size(0)

                # Логируем внимание на первом батче
                if batch_idx == 0 and self._has_attention:
                    # Предполагаем, что attentions определены
                    if 'attentions' in locals():
                        self.log_attention_maps(attentions, epoch, "test")

        # Вычисляем средние значения
        avg_test_loss = test_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0

        # Сохраняем историю
        self.test_history['loss'].append(avg_test_loss)
        self.test_history['accuracy'].append(test_accuracy)

        # Обновляем лучшую точность
        if test_accuracy > self.best_test_acc:
            self.best_test_acc = test_accuracy
            self.save_checkpoint(epoch, is_best=True)

        return {
            'loss': avg_test_loss,
            'accuracy': test_accuracy
        }

    def train(self, num_epochs: int = None):
        """Основной цикл обучения."""
        if num_epochs is not None:
            self.config['num_epochs'] = num_epochs

        print(f"[{datetime.datetime.now()}] Starting training...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        print(f"Test samples: {len(self.test_loader.dataset):,}")
        print("-" * 60)

        # Логируем граф модели
        try:
            dummy_batch = next(iter(self.train_loader))
            inputs, _ = self._prepare_batch(dummy_batch)

            if inputs.shape[0] > 1:
                inputs = inputs[:1]  # Берем один пример

            inputs = inputs.to(self.device)
            self.writer.add_graph(self.model, inputs)
        except Exception as e:
            print(f"Could not log model graph: {e}")

        # Основной цикл обучения
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch

            # Тренировка
            train_metrics = self.train_epoch(epoch)

            # Тестирование
            test_metrics = self.test_epoch(epoch)

            # Логирование в TensorBoard
            self.writer.add_scalar('train/loss_epoch', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/accuracy_epoch', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('test/loss_epoch', test_metrics['loss'], epoch)
            self.writer.add_scalar('test/accuracy_epoch', test_metrics['accuracy'], epoch)

            # Разрыв между train и test
            loss_gap = train_metrics['loss'] - test_metrics['loss']
            acc_gap = train_metrics['accuracy'] - test_metrics['accuracy']

            self.writer.add_scalar('metrics/loss_gap', loss_gap, epoch)
            self.writer.add_scalar('metrics/accuracy_gap', acc_gap, epoch)
            self.writer.add_scalar('metrics/best_test_accuracy', self.best_test_acc, epoch)

            # Логирование весов и градиентов
            if epoch % self.config.get('log_weights_every', 10) == 0:
                self.log_model_weights_and_gradients(epoch)

            # Логирование активаций
            if epoch % self.config.get('log_activations_every', 50) == 0:
                self.log_activations(epoch)

            # Логирование лосс-ландшафта
            if self.config.get('log_loss_landscape_every', 50) > 0 and \
                    epoch % self.config.get('log_loss_landscape_every', 50) == 0:
                self.log_loss_landscape(epoch)

            # Сохранение чекпоинта
            if epoch % self.config.get('save_checkpoint_every', 100) == 0:
                self.save_checkpoint(epoch)

            # Вывод статистики
            print(f"\n[{datetime.datetime.now()}] Epoch {epoch + 1}/{self.config['num_epochs']}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%")
            print(f"  Best Test Acc: {self.best_test_acc:.2f}%")
            print(f"  Gap - Loss: {loss_gap:.4f}, Acc: {acc_gap:.2f}%")
            print("-" * 60)

            # Ранняя остановка (опционально)
            if self.config.get('early_stopping_patience', 0) > 0:
                if epoch > self.config['early_stopping_patience']:
                    recent_acc = self.test_history['accuracy'][-self.config['early_stopping_patience']:]
                    if max(recent_acc) <= self.best_test_acc:
                        print(f"\n[{datetime.datetime.now()}] Early stopping triggered!")
                        break

        # Финальное сохранение
        self.save_checkpoint(self.epoch)

        # Закрываем writer
        self.writer.close()

        print(f"\n[{datetime.datetime.now()}] Training completed!")
        print(f"Best test accuracy: {self.best_test_acc:.2f}%")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.log_dir}")
        print("\nTo view results, run:")
        print(f"tensorboard --logdir={self.log_dir}")

        return self.model


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    device: torch.device = None) -> Dict:
    """Загружает чекпоинт и восстанавливает состояние."""

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Загружаем веса модели
    model.load_state_dict(checkpoint['model_state_dict'])

    # Загружаем состояние оптимизатора (если передан)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Best test accuracy: {checkpoint['best_test_acc']:.2f}%")

    return checkpoint