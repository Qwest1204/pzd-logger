# Пиздатый логгер

Универсальный тренер для глубокого обучения с максимальным логированием всех аспектов обучения.

## Особенности

✅ **Универсальность** - работает с любыми архитектурами (Transformers, CNN, RNN, Linear)  
✅ **Полное логирование** - веса, градиенты, активации, внимание  
✅ **Loss Landscape** - 2D/3D визуализация поверхности функции потерь  
✅ **TensorBoard** - все метрики в реальном времени  
✅ **Чекпоинты** - автоматическое сохранение лучших моделей  
✅ **Гибкость** - легко настраивается под ваши задачи  

## Быстрый старт

```python
from trainer.core import UniversalTrainer
from configs.default_configs import create_default_config

# Создайте конфигурацию
config = create_default_config('transformer')

# Инициализируйте тренер
trainer = UniversalTrainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    config=config
)

# Запустите обучение
trained_model = trainer.train()
```

### я делаю это чисто для себя но если вам это приглянётся то поставте звёздочку