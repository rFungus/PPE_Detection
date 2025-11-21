"""
Конфигурация проекта детекции СИЗ.

Централизованное управление всеми настройками.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from dataclasses import dataclass


@dataclass
class ProjectConfig:
    """Основная конфигурация проекта."""
    
    # Пути
    project_root: Path = Path(".").resolve()
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    output_dir: Path = project_root / "output"
    videos_dir: Path = project_root / "videos"
    config_dir: Path = project_root / "config"
    notebooks_dir: Path = project_root / "notebooks"
    src_dir: Path = project_root / "src"
    
    # Данные
    image_extensions: List[str] = None
    video_extensions: List[str] = None
    classes: Dict[int, str] = None
    class_colors: Dict[int, tuple] = None
    
    # Модель OBB (Oriented Bounding Box) для rotated bounding boxes
    model_name: str = "yolov8l-obb.pt"  # Large OBB модель для rotated bounding boxes
    experiment_name: str = "ppe_detection_obb"
    conf_threshold: float = 0.2  # Понижен для очень маленьких объектов
    
    # Обучение (максимально оптимизировано для скорости)
    epochs: int = 60  # Количество эпох для обучения
    img_size: int = 640  # Стандартный размер YOLO для максимальной скорости (~40-60 it/s)
    batch_size: int = 8  # Максимальная утилизация GPU (было 4)
    patience: int = 10  # Быстрая остановка при отсутствии улучшений
    workers: Optional[int] = None  # None = автоопределение (на Linux до 12, на Windows до 6)
    device: str = "auto"
    
    # Разметка
    val_ratio: float = 0.2
    random_seed: int = 42
    prelabel_conf_threshold: float = 0.3
    
    # Извлечение кадров (уменьшен шаг для большего количества кадров)
    frame_extraction_step: int = 15  # Извлекать каждый 15-й кадр (было 30) - увеличивает датасет в 2 раза
    
    def __post_init__(self):
        """Инициализация после создания объекта."""
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        if self.video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
        
        # Загрузка классов из файла classes.txt
        if self.classes is None:
            self.classes = self._load_classes_from_file()
        
        # Генерация цветов для классов (если не заданы)
        if self.class_colors is None:
            self.class_colors = self._generate_class_colors()
        
        # Создание директорий
        self._create_directories()
        
        # Определение устройства
        self._detect_device()
    
    def _load_classes_from_file(self) -> Dict[int, str]:
        """
        Загружает классы из файла classes.txt.
        
        Формат файла: одна строка = один класс
        Пример:
            helmet
            vest
        
        Returns:
            Словарь {id: название_класса}
        """
        # Ищем файл classes.txt в нескольких местах
        possible_paths = [
            self.project_root / "classes.txt",
            self.data_dir / "classes.txt",
            self.config_dir / "classes.txt"
        ]
        
        classes_file = None
        for path in possible_paths:
            if path.exists():
                classes_file = path
                break
        
        if classes_file is None:
            raise FileNotFoundError(
                f"Файл classes.txt не найден! Искали в:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\n\nСоздайте файл classes.txt с перечислением классов (по одному на строку)."
            )
        
        classes = {}
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    class_name = line.strip()
                    if class_name and not class_name.startswith('#'):  # Пропускаем пустые строки и комментарии
                        classes[idx] = class_name
            
            if len(classes) == 0:
                raise ValueError(f"Файл classes.txt пуст или не содержит классов: {classes_file}")
            
            print(f"✓ Классы загружены из {classes_file}: {len(classes)} классов")
            for idx, name in classes.items():
                print(f"  {idx}: {name}")
            
            return classes
            
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки классов из {classes_file}: {e}")
    
    def _generate_class_colors(self) -> Dict[int, tuple]:
        """
        Генерирует цвета для классов автоматически.
        
        Returns:
            Словарь {id: (B, G, R)}
        """
        # Предопределенная палитра цветов (BGR формат для OpenCV)
        color_palette = [
            (0, 165, 255),    # Оранжевый
            (0, 255, 255),    # Желтый
            (255, 0, 0),      # Синий
            (255, 0, 255),   # Пурпурный
            (0, 255, 0),     # Зеленый
            (255, 165, 0),   # Голубой
            (128, 0, 128),   # Фиолетовый
            (255, 192, 203), # Розовый
            (0, 128, 255),   # Коричневый
            (255, 255, 0),   # Циан
        ]
        
        colors = {}
        for idx in range(len(self.classes)):
            colors[idx] = color_palette[idx % len(color_palette)]
        
        return colors
    
    def _create_directories(self):
        """Создает необходимые директории."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.output_dir,
            self.videos_dir,
            self.config_dir,
            self.src_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("Директории созданы/проверены")
    
    def _detect_device(self):
        """Определяет доступное устройство с детальной информацией."""
        print("[GPU CHECK] Начало проверки устройства...")
        try:
            import torch
            print("[GPU CHECK] PyTorch импортирован успешно")
            print(f"[GPU CHECK] Версия PyTorch: {torch.__version__}")
            
            # Детальная проверка CUDA
            print("[GPU CHECK] Проверка доступности CUDA...")
            cuda_available = torch.cuda.is_available()
            print(f"[GPU CHECK] CUDA доступна: {cuda_available}")
            
            if cuda_available:
                print("[GPU CHECK] Получение информации о GPU...")
                gpu_count = torch.cuda.device_count()
                print(f"[GPU CHECK] Найдено GPU: {gpu_count}")
                
                cuda_version = torch.version.cuda
                print(f"[GPU CHECK] CUDA версия: {cuda_version}")
                
                cudnn_version = None
                if torch.backends.cudnn.is_available():
                    cudnn_version = torch.backends.cudnn.version()
                    print(f"[GPU CHECK] cuDNN версия: {cudnn_version}")
                else:
                    print("[GPU CHECK] cuDNN недоступен")
                
                # Получаем информацию о каждой GPU
                print("[GPU CHECK] Сбор детальной информации о GPU...")
                gpu_info = []
                for i in range(gpu_count):
                    print(f"[GPU CHECK] Обработка GPU {i}...")
                    props = torch.cuda.get_device_properties(i)
                    gpu_name = props.name
                    gpu_memory_total = props.total_memory / (1024**3)  # GB
                    gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    gpu_memory_free = gpu_memory_total - gpu_memory_allocated
                    gpu_capability = props.major, props.minor
                    gpu_multiprocessors = props.multi_processor_count
                    
                    gpu_info.append({
                        'id': i,
                        'name': gpu_name,
                        'memory_total_gb': round(gpu_memory_total, 2),
                        'memory_allocated_gb': round(gpu_memory_allocated, 2),
                        'memory_free_gb': round(gpu_memory_free, 2),
                        'capability': f"{gpu_capability[0]}.{gpu_capability[1]}",
                        'multiprocessors': gpu_multiprocessors
                    })
                    print(f"[GPU CHECK] GPU {i} обработан: {gpu_name}")
                
                # Выбираем устройство
                if gpu_count > 1:
                    self.device = "0"  # Используем первую GPU по умолчанию
                    print(f"[GPU CHECK] GPU обнаружено: {gpu_count} устройств")
                    print(f"[GPU CHECK] CUDA версия: {cuda_version}")
                    if cudnn_version:
                        print(f"[GPU CHECK] cuDNN версия: {cudnn_version}")
                    for info in gpu_info:
                        print(f"[GPU CHECK]   GPU {info['id']}: {info['name']}")
                        print(f"[GPU CHECK]     Память: {info['memory_free_gb']:.2f} GB свободно / {info['memory_total_gb']:.2f} GB всего")
                        print(f"[GPU CHECK]     CUDA Capability: {info['capability']}")
                        print(f"[GPU CHECK]     Multiprocessors: {info['multiprocessors']}")
                    print(f"[GPU CHECK] Используется GPU: {self.device}")
                else:
                    self.device = "0"
                    info = gpu_info[0]
                    print(f"[GPU CHECK] GPU обнаружено: {info['name']}")
                    print(f"[GPU CHECK]   Память: {info['memory_free_gb']:.2f} GB свободно / {info['memory_total_gb']:.2f} GB всего")
                    print(f"[GPU CHECK]   CUDA версия: {cuda_version}")
                    if cudnn_version:
                        print(f"[GPU CHECK]   cuDNN версия: {cudnn_version}")
                    print(f"[GPU CHECK]   CUDA Capability: {info['capability']}")
                    print(f"[GPU CHECK]   Multiprocessors: {info['multiprocessors']}")
                    print(f"[GPU CHECK] Используется GPU: {self.device}")
                
                # Проверка текущего устройства
                print("[GPU CHECK] Получение текущего устройства...")
                current_device = torch.cuda.current_device()
                print(f"[GPU CHECK] Текущее устройство: GPU {current_device}")
                
                # Тест производительности GPU (с таймаутом и более безопасный)
                print("[GPU CHECK] Запуск теста производительности GPU...")
                try:
                    import time
                    start_time = time.time()
                    print("[GPU CHECK] Создание тестового тензора...")
                    test_tensor = torch.randn(1000, 1000).cuda()
                    print(f"[GPU CHECK] Тензор создан за {time.time() - start_time:.2f} сек")
                    
                    print("[GPU CHECK] Выполнение матричного умножения...")
                    start_time = time.time()
                    result = test_tensor @ test_tensor
                    print(f"[GPU CHECK] Умножение выполнено за {time.time() - start_time:.2f} сек")
                    
                    print("[GPU CHECK] Синхронизация CUDA...")
                    torch.cuda.synchronize()
                    print("[GPU CHECK] Тест GPU: успешно (GPU работает корректно)")
                    
                    # Очистка памяти
                    del test_tensor, result
                    torch.cuda.empty_cache()
                    print("[GPU CHECK] Память GPU очищена")
                except Exception as e:
                    print(f"[GPU CHECK] WARNING: тест GPU не прошел: {e}")
                    import traceback
                    print(f"[GPU CHECK] Traceback: {traceback.format_exc()}")
                
                print("[GPU CHECK] Проверка GPU завершена успешно")
                
            else:
                self.device = "cpu"
                print("[GPU CHECK] CUDA недоступна, используется CPU")
                print("[GPU CHECK] Для использования GPU установите PyTorch с поддержкой CUDA:")
                print("[GPU CHECK]   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                
        except ImportError as e:
            self.device = "cpu"
            print(f"[GPU CHECK] ERROR: PyTorch не установлен: {e}")
            print("[GPU CHECK] Используется CPU")
            print("[GPU CHECK] Установите: pip install torch torchvision")
        except Exception as e:
            self.device = "cpu"
            print(f"[GPU CHECK] ERROR: Ошибка при определении устройства: {e}")
            import traceback
            print(f"[GPU CHECK] Traceback: {traceback.format_exc()}")
            print("[GPU CHECK] Используется CPU")
        
        print(f"[GPU CHECK] Финальное устройство: {self.device}")
        print("[GPU CHECK] Проверка устройства завершена")
    
    def load_dataset_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Загружает конфигурацию датасета из YAML файла.
        
        Args:
            config_path: Путь к YAML файлу (если None - config/ppe_data.yaml)
            
        Returns:
            Словарь с конфигурацией датасета
        """
        if config_path is None:
            config_path = self.config_dir / "ppe_data.yaml"
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            self.create_dataset_config(config_path)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"Конфигурация загружена: {config_file}")
            print(f"Классов: {config.get('nc', 'N/A')}")
            print(f"Классы: {config.get('names', 'N/A')}")
            
            return config
            
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации: {e}")
            return {}
    
    def create_dataset_config(self, config_path: Optional[str] = None) -> Path:
        """
        Создает стандартную конфигурацию датасета.
        
        Args:
            config_path: Путь для сохранения (если None - config/ppe_data.yaml)
            
        Returns:
            Путь к созданному файлу
        """
        if config_path is None:
            config_path = self.config_dir / "ppe_data.yaml"
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Формируем комментарий с классами
        classes_comment = ", ".join([f"{id}={name}" for id, name in self.classes.items()])
        
        config_content = f"""# Конфигурация датасета для детекции СИЗ
# Классы загружены из classes.txt: {classes_comment}

path: {self.data_dir}  # Корневая папка с данными
train: images/train     # Путь к обучающим изображениям (относительно path)
val: images/val         # Путь к валидационным изображениям (относительно path)

# Количество классов
nc: {len(self.classes)}

# Имена классов
names:
"""
        
        for class_id, class_name in self.classes.items():
            config_content += f"  {class_id}: {class_name}\n"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            print(f"Конфигурация создана: {config_file}")
            print(f"Классы: {list(self.classes.values())}")
            
            return config_file
            
        except Exception as e:
            print(f"Ошибка создания конфигурации: {e}")
            return config_file
    
    def create_classes_file(self, classes_path: Optional[str] = None) -> Path:
        """
        Создает файл классов для LabelImg на основе classes.txt.
        
        Args:
            classes_path: Путь для сохранения (если None - data/predefined_classes.txt)
            
        Returns:
            Путь к созданному файлу
        """
        if classes_path is None:
            classes_path = self.data_dir / "predefined_classes.txt"
        
        classes_file = Path(classes_path)
        classes_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(classes_file, 'w', encoding='utf-8') as f:
                for class_name in self.classes.values():
                    f.write(f"{class_name}\n")
            
            print(f"Файл классов для LabelImg создан: {classes_file}")
            print(f"Классы: {list(self.classes.values())}")
            
            return classes_file
            
        except Exception as e:
            print(f"Ошибка создания файла классов: {e}")
            return classes_file
    
    def get_paths_summary(self) -> Dict[str, Path]:
        """
        Возвращает сводку по путям проекта.
        
        Returns:
            Словарь с основными путями
        """
        return {
            'project_root': self.project_root,
            'data_dir': self.data_dir,
            'models_dir': self.models_dir,
            'output_dir': self.output_dir,
            'videos_dir': self.videos_dir,
            'config_dir': self.config_dir,
            'dataset_config': self.config_dir / "ppe_data.yaml",
            'classes_file': self.data_dir / "predefined_classes.txt"
        }
    
    def validate_paths(self) -> Dict[str, bool]:
        """
        Проверяет существование основных путей.
        
        Returns:
            Словарь {путь: существует}
        """
        paths = self.get_paths_summary()
        validation = {}
        
        print("ПРОВЕРКА ПУТЕЙ ПРОЕКТА")
        print("-" * 40)
        
        for name, path in paths.items():
            exists = path.exists()
            validation[name] = exists
            
            status = "OK" if exists else "MISSING"
            print(f"{status} {name}: {path}")
        
        # Подробная проверка данных
        if self.data_dir.exists():
            img_count = len(list(self.data_dir.rglob("*.jpg"))) + \
                       len(list(self.data_dir.rglob("*.png")))
            label_count = len(list(self.data_dir.rglob("*.txt")))
            
            print(f"\nДанные: {img_count} изображений, {label_count} разметок")
        
        missing_count = sum(1 for exists in validation.values() if not exists)
        if missing_count == 0:
            print("\nВсе пути корректны!")
        else:
            print(f"\nОтсутствует {missing_count} путей")
        
        return validation


# Глобальная конфигурация проекта
config = ProjectConfig()


if __name__ == "__main__":
    # Демонстрация использования
    print("=== КОНФИГУРАЦИЯ ПРОЕКТА ===")
    
    # Проверка путей
    validation = config.validate_paths()
    
    # Создание конфигурации (если нужно)
    if not config.config_dir.exists() or not (config.config_dir / "ppe_data.yaml").exists():
        config.create_dataset_config()
    
    # Создание файла классов
    config.create_classes_file()
    
    # Загрузка конфигурации датасета
    dataset_config = config.load_dataset_config()
    
    print(f"\nКонфигурация готова!")
    print(f"Классов: {dataset_config.get('nc', 0)}")
    print(f"Устройство: {config.device}")
    
    # Пример путей
    paths = config.get_paths_summary()
    print(f"\nОсновные пути:")
    for name, path in paths.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"  {status} {name}: {path}")
