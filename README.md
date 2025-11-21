# Система детекции средств индивидуальной защиты (СИЗ)

Проект для обучения модели **YOLOv8-OBB** (Oriented Bounding Box) на детекцию средств индивидуальной защиты с поддержкой **rotated bounding boxes**:

- **Защитная каска** (оранжевая или белая) - класс 0
- **Сигнальный жилет** - класс 1

**Особенности:**
- ✅ **Поддержка rotated bounding boxes** - детекция объектов под наклоном
- ✅ **Оптимизировано для Linux** - до 12 workers, быстрый 'fork' multiprocessing
- ✅ **Высокая скорость обучения** - ~40-60 итераций/сек на Linux

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Извлечение кадров из видео

Если у вас есть видео файлы (AVI, MP4, MOV и т.д.), извлеките из них кадры:

```bash
# Извлечение кадров из videos/ в data/images/
python extract.py

# С дополнительными параметрами
python extract.py --input videos/ --output data/images/ --step 15
```

**Параметры:**
- `--input` или `-i`: Папка с видео файлами (по умолчанию: `videos/`)
- `--output` или `-o`: Папка для сохранения кадров (по умолчанию: `data/images/`)
- `--step` или `-s`: Извлекать каждый N-й кадр (по умолчанию: 15)

### 3. Настройка классов

**Важно**: Создайте файл `classes.txt` в корне проекта с перечислением классов (по одному на строку):

```
helmet
vest
```

Порядок важен: первая строка = класс 0, вторая = класс 1, и т.д.

Файл можно разместить в одном из мест (ищется в таком порядке):
- `classes.txt` (корень проекта)
- `data/classes.txt`
- `config/classes.txt`

### 4. Разметка данных

Разметьте извлеченные кадры вручную, используя инструмент с поддержкой OBB (LabelMe, CVAT, Roboflow).

**Формат аннотаций OBB:**
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Где `(x1,y1), (x2,y2), (x3,y3), (x4,y4)` - 4 точки углов rotated bounding box в нормализованных координатах (0.0 - 1.0).

**Структура до разделения:**
```
data/
├── images/          # Все изображения
├── labels/          # Все аннотации (.txt файлы)
├── classes.txt
└── notes.json
```

**Формат аннотаций OBB**: Каждая строка в файле `.txt` содержит 9 значений: `class_id x1 y1 x2 y2 x3 y3 x4 y4`, где координаты нормализованы (0.0 - 1.0).

### 5. Разделение на train/val

После разметки разделите данные на train/val:

```bash
# Разделение с 20% валидации (по умолчанию)
python data.py

# С другими параметрами
python data.py --val-ratio 0.3    # 30% валидации
python data.py --val-ratio 0.1     # 10% валидации
```

**Структура после разделения:**
```
data/
├── images/
│   ├── train/          # Изображения для обучения
│   └── val/            # Изображения для валидации
├── labels/
│   ├── train/          # Файлы разметки (.txt) для обучения
│   └── val/            # Файлы разметки (.txt) для валидации
├── classes.txt
└── notes.json
```

### 6. Проверка данных

Перед обучением проверьте структуру данных и формат аннотаций:

```bash
# Проверка структуры данных
python check.py

# Проверка формата OBB аннотаций
python check.py --obb

# Проверка конкретной папки
python check.py --obb --labels data/labels/train
```

### 7. Запуск обучения

```bash
python run_full_pipeline.py
```

Этот скрипт автоматически:
- Создаст структуру проекта и конфигурацию
- Проверит структуру данных и разметку
- Обучит модель YOLOv8-OBB (60 эпох)
- Выполнит быстрый тест модели

### 8. Использование обученной модели

После обучения используйте программу `detect.py` для детекции:

```bash
# Детекция на изображении
python detect.py --model models/ppe_detection_obb/weights/best.pt --source image.jpg

# Детекция на видео
python detect.py --model models/ppe_detection_obb/weights/best.pt --source video.mp4

# Пакетная обработка папки
python detect.py --model models/ppe_detection_obb/weights/best.pt --source data/images/val/

# Детекция с камеры в реальном времени
python detect.py --model models/ppe_detection_obb/weights/best.pt --camera

# Настройка порога уверенности
python detect.py --model models/ppe_detection_obb/weights/best.pt --source image.jpg --conf 0.3
```

**Параметры:**
- `--model` или `-m`: Путь к обученной модели OBB (.pt файл)
- `--source` или `-s`: Источник (изображение, видео или папка)
- `--camera` или `-c`: Использовать камеру для детекции в реальном времени
- `--conf`: Порог уверенности (по умолчанию: 0.2)
- `--output` или `-o`: Папка для сохранения результатов (по умолчанию: output/detections)
- `--device`: Устройство ('cpu', '0' для GPU, 'auto' по умолчанию)


## Структура проекта

```
hahaton123/
├── config/
│   └── ppe_data.yaml          # Конфигурация датасета
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/
│   └── ppe_detection_obb/     # Обученные модели
├── output/                     # Результаты детекции
├── logs/                       # Логи выполнения
├── src/                        # Исходный код
│   ├── data/                   # Модули для работы с данными
│   ├── models/                 # Модули для обучения
│   ├── inference/              # Модули для инференса
│   └── utils/                 # Утилиты
├── extract.py                  # Извлечение кадров из видео
├── data.py                     # Разделение датасета на train/val
├── check.py                    # Проверка структуры данных и формата OBB
├── detect.py                   # Детекция с обученной моделью
├── run_full_pipeline.py        # Полный автоматический пайплайн
├── requirements.txt            # Зависимости
├── README.md                   # Документация
└── classes.txt.example         # Пример файла классов
```

## Параметры обучения

- **Эпохи**: 60
- **Размер изображения**: 640x640
- **Размер батча**: 8 (GPU) / 4 (CPU)
- **Ранняя остановка**: после 10 эпох без улучшения
- **Workers**: до 12 (Linux) / до 6 (Windows)

## Требования

- Python 3.8+
- ultralytics (YOLOv8)
- opencv-python
- numpy
- torch

## Инструменты для разметки OBB

⚠️ **Важно**: Для rotated bounding boxes нужны инструменты с поддержкой OBB!

1. **LabelMe** (рекомендуется)
   - Установка: `pip install labelme`
   - Запуск: `labelme`
   - Поддержка rotated bounding boxes
   - Экспорт в формат YOLO-OBB

2. **CVAT** (для команд)
   - https://github.com/openvinotoolkit/cvat
   - Полная поддержка OBB
   - Веб-интерфейс

3. **Roboflow** (онлайн)
   - https://roboflow.com/
   - Поддержка OBB аннотаций
   - Бесплатный аккаунт

## Дополнительная документация

- `classes.txt.example` - Пример файла классов
