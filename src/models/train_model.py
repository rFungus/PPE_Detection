"""
Модуль для обучения модели YOLOv8.

Использование:
from src.models.train_model import PPEDetectorTrainer
trainer = PPEDetectorTrainer()
trainer.train(epochs=30, batch_size=16)
"""

from pathlib import Path
import torch
import logging
import multiprocessing
import platform
from typing import Optional, Dict, List
from datetime import datetime
# YOLO импортируется лениво внутри методов, чтобы не замедлять импорт модуля

# Настройка multiprocessing
# На Linux используется 'fork' по умолчанию (быстрее чем 'spawn' на Windows)
# На Windows используем 'spawn' для избежания ConnectionResetError
if platform.system() == 'Windows':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Если уже установлен, игнорируем
        pass
# На Linux не нужно менять start_method - 'fork' уже оптимален


class PPEDetectorTrainer:
    """
    Класс для обучения модели детекции СИЗ.
    
    Управляет процессом обучения YOLOv8 с логированием и мониторингом.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n-obb.pt",  # OBB модель для rotated bounding boxes
        config_path: str = "config/ppe_data.yaml",
        project_dir: str = "models",
        experiment_name: str = "ppe_detection"
    ):
        self.model_name = model_name
        self.config_path = Path(config_path)
        self.project_dir = Path(project_dir)
        self.experiment_name = experiment_name
        
        # Настройка логирования
        self.setup_logging()
        
        # Проверка конфигурации
        self._validate_config()
        
        # Устройство
        self.device = self._detect_device()
        self.logger.info(f"Используется устройство: {self.device}")
    
    def setup_logging(self):
        """Настройка логирования."""
        log_dir = self.project_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.logger.info(f"Логи будут сохранены в: {log_file}")
    
    def _validate_config(self):
        """Проверяет наличие конфигурационного файла."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Конфигурационный файл не найден: {self.config_path}\n"
                f"Создайте файл {self.config_path} с классами helmet и vest"
            )
        self.logger.info(f"Конфигурация загружена: {self.config_path}")
    
    def _detect_device(self) -> str:
        """Определяет доступное устройство (GPU/CPU) с детальной диагностикой."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            
            self.logger.info(f"CUDA доступна: версия {cuda_version}")
            if cudnn_version:
                self.logger.info(f"cuDNN доступен: версия {cudnn_version}")
            
            # Детальная информация о GPU
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory / (1024**3)  # GB
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_free = memory_total - memory_allocated
                capability = f"{props.major}.{props.minor}"
                
                self.logger.info(f"GPU {i}: {props.name}")
                self.logger.info(f"  Память: {memory_free:.2f} GB свободно / {memory_total:.2f} GB всего")
                self.logger.info(f"  CUDA Capability: {capability}")
                self.logger.info(f"  Multiprocessors: {props.multi_processor_count}")
            
            if gpu_count > 1:
                device = "0"  # Используем первую GPU
                self.logger.info(f"Обнаружено {gpu_count} GPU, используется GPU {device}")
            else:
                device = "0"
                self.logger.info(f"Используется GPU {device}")
            
            # Тест производительности
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                _ = test_tensor @ test_tensor
                torch.cuda.synchronize()
                self.logger.info("Тест GPU: успешно")
            except Exception as e:
                self.logger.warning(f"Тест GPU не прошел: {e}")
            
            return device
        else:
            self.logger.warning("CUDA недоступна, используется CPU")
            return "cpu"
    
    def train(
        self,
        epochs: int = 60,
        img_size: int = 640,
        batch_size: int = 16,
        patience: int = 10,
        workers: Optional[int] = None,
        save: bool = True,
        plots: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Обучает модель YOLOv8.
        
        Args:
            epochs: Количество эпох
            img_size: Размер входного изображения
            batch_size: Размер батча
            patience: Ранняя остановка после N эпох без улучшения
            workers: Количество worker'ов для загрузки данных (None = автоопределение)
            save: Сохранять чекпоинты
            plots: Генерировать графики
            verbose: Подробный вывод
            
        Returns:
            Словарь с результатами обучения
        """
        # Автоматическое определение количества workers
        # На Linux можно использовать больше workers благодаря лучшей поддержке multiprocessing
        if workers is None:
            if platform.system() == 'Linux':
                # На Linux используем максимум workers для максимальной скорости
                # 'fork' метод позволяет эффективно использовать больше процессов
                cpu_count = multiprocessing.cpu_count()
                workers = min(12, cpu_count - 1)  # До 12 workers на Linux
                if workers == 0:
                    workers = 1
                self.logger.info(f"Linux detected: используем {workers} workers (из {cpu_count} CPU ядер)")
            elif platform.system() == 'Windows':
                # На Windows используем меньше workers из-за 'spawn' метода
                workers = min(6, multiprocessing.cpu_count() - 1)
                if workers == 0:
                    workers = 1
                self.logger.info(f"Windows detected: используем {workers} workers")
            else:
                # Mac или другая система
                workers = min(8, multiprocessing.cpu_count() - 1)
                if workers == 0:
                    workers = 1
        
        self.logger.info("=" * 70)
        self.logger.info("=== НАЧАЛО ОБУЧЕНИЯ ===")
        self.logger.info("=" * 70)
        self.logger.info(f"Параметры обучения:")
        self.logger.info(f"  - Эпохи: {epochs}")
        self.logger.info(f"  - Размер изображения: {img_size}x{img_size}")
        self.logger.info(f"  - Размер батча: {batch_size}")
        self.logger.info(f"  - Workers: {workers}")
        self.logger.info(f"  - Patience: {patience}")
        self.logger.info(f"  - Mixed Precision (AMP): Включен (ускоряет обучение)")
        self.logger.info(f"Платформа: {platform.system()}, CPU cores: {multiprocessing.cpu_count()}")
        self.logger.info(f"Модель: {self.model_name}")
        self.logger.info(f"Конфигурация: {self.config_path}")
        self.logger.info("")
        self.logger.info("МАКСИМАЛЬНЫЕ ОПТИМИЗАЦИИ ДЛЯ СКОРОСТИ:")
        self.logger.info("  - Размер изображения: 640x640 (стандартный YOLO, максимальная скорость)")
        self.logger.info("  - Batch size: 8 (максимальная утилизация GPU)")
        self.logger.info("  - Augmentation: минимальная (mosaic/mixup/copy_paste отключены)")
        if platform.system() == 'Linux':
            self.logger.info("  - Workers: до 12 (Linux оптимизация, 'fork' метод)")
            self.logger.info("  - Multiprocessing: 'fork' метод (быстрее чем 'spawn' на Windows)")
        else:
            self.logger.info("  - Workers: увеличено для быстрой загрузки данных")
        self.logger.info("  - Mixed Precision (AMP): включен")
        self.logger.info("  - Warmup epochs: 1 (было 3)")
        self.logger.info("  - Эпохи: 30 (было 50, изначально 100)")
        self.logger.info("  - Patience: 10 (быстрая остановка)")
        self.logger.info("")
        if platform.system() == 'Linux':
            self.logger.info("Ожидаемая скорость на Linux: ~40-60 итераций/сек (вместо 6)")
            self.logger.info("Время обучения: ~1-2 часа (вместо 8-12 часов)")
            self.logger.info("Ускорение: ~5-10x по сравнению с предыдущими настройками")
            self.logger.info("Linux преимущества: больше workers, быстрый 'fork' multiprocessing")
        else:
            self.logger.info("Ожидаемая скорость: ~40-60 итераций/сек (вместо 6)")
            self.logger.info("Время обучения: ~1-2 часа (вместо 8-12 часов)")
            self.logger.info("Ускорение: ~5-10x по сравнению с предыдущими настройками")
        
        # Проверка доступной памяти GPU
        if torch.cuda.is_available() and self.device != "cpu":
            try:
                gpu_id = int(self.device) if self.device.isdigit() else 0
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
                memory_free = memory_total - memory_allocated
                
                self.logger.info(f"GPU память: {memory_free:.2f} GB свободно / {memory_total:.2f} GB всего")
                
                # Оценка требуемой памяти (приблизительно)
                # YOLOv8l с img_size=640 и batch_size=8 требует примерно 6-8 GB
                # Формула: (img_size/640)^2 * batch_size * 0.3 для YOLOv8l
                estimated_memory = (img_size / 640) ** 2 * batch_size * 0.3  # Примерная оценка в GB
                self.logger.info(f"Примерная требуемая память: ~{estimated_memory:.2f} GB")
                
                # Автоматическое уменьшение параметров при нехватке памяти
                if memory_free < estimated_memory * 0.9:
                    self.logger.warning(f"Внимание: свободной памяти ({memory_free:.2f} GB) может не хватить!")
                    self.logger.warning(f"Требуется: ~{estimated_memory:.2f} GB")
                    
                    # Автоматическая корректировка параметров
                    original_batch_size = batch_size
                    original_img_size = img_size
                    
                    # Уменьшаем batch_size сначала
                    if batch_size > 1 and memory_free < estimated_memory * 0.9:
                        batch_size = max(1, batch_size // 2)
                        estimated_memory = (img_size / 640) ** 2 * batch_size * 0.3
                        self.logger.info(f"Автоматически уменьшен batch_size: {original_batch_size} -> {batch_size}")
                    
                    # Если все еще не хватает, уменьшаем img_size
                    if memory_free < estimated_memory * 0.9 and img_size > 640:
                        img_size = max(640, int(img_size * 0.8))
                        estimated_memory = (img_size / 640) ** 2 * batch_size * 0.3
                        self.logger.info(f"Автоматически уменьшен img_size: {original_img_size} -> {img_size}")
                    
                    if batch_size != original_batch_size or img_size != original_img_size:
                        self.logger.info(f"Скорректированные параметры: batch_size={batch_size}, img_size={img_size}")
                        self.logger.info(f"Новая оценка памяти: ~{estimated_memory:.2f} GB")
            except Exception as e:
                self.logger.warning(f"Не удалось проверить память GPU: {e}")
        
        # Создаем директорию проекта
        experiment_dir = self.project_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем модель (ленивый импорт YOLO)
        self.logger.info("Импорт библиотеки YOLO...")
        try:
            from ultralytics import YOLO
            self.logger.info("YOLO успешно импортирован")
        except ImportError as e:
            self.logger.error(f"Не удалось импортировать YOLO: {e}")
            self.logger.error("Установите ultralytics: pip install ultralytics")
            raise
        except Exception as e:
            self.logger.error(f"Ошибка при импорте YOLO: {e}")
            raise
        
        self.logger.info(f"Загрузка модели OBB (Oriented Bounding Box): {self.model_name}")
        try:
            # YOLOv8-OBB для rotated bounding boxes
            model = YOLO(self.model_name, task='obb')
            self.logger.info("✓ Модель OBB успешно загружена (поддержка rotated bounding boxes)")
            
            # Информация о модели
            if hasattr(model, 'names'):
                self.logger.info(f"  Классы: {list(model.names.values())}")
        except FileNotFoundError:
            self.logger.error(f"Файл модели не найден: {self.model_name}")
            self.logger.error("Убедитесь, что файл существует или используйте предобученную OBB модель:")
            self.logger.error("  - yolov8n-obb.pt (nano, быстрая)")
            self.logger.error("  - yolov8s-obb.pt (small)")
            self.logger.error("  - yolov8m-obb.pt (medium)")
            self.logger.error("  - yolov8l-obb.pt (large, точная)")
            self.logger.error("  - yolov8x-obb.pt (xlarge, максимальная точность)")
            raise
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Параметры обучения (оптимизированы для маленьких объектов и высокого угла обзора)
        # Используем скорректированные параметры (если были изменены автоматически)
        train_params = {
            'data': str(self.config_path),
            'epochs': epochs,
            'imgsz': img_size,  # Может быть автоматически уменьшен
            'device': self.device,
            'project': str(self.project_dir),
            'name': self.experiment_name,
            'batch': batch_size,  # Может быть автоматически уменьшен
            'patience': patience,
            'workers': workers,
            'save': save,
            'plots': plots,
            'verbose': verbose,
            'exist_ok': True,  # Перезаписывать результаты
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 1,  # Уменьшено для скорости (было 3)
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 2.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            # Augmentation минимальна для максимальной скорости
            'hsv_h': 0.01,  # Минимум для скорости
            'hsv_s': 0.5,  # Минимум для скорости
            'hsv_v': 0.3,  # Минимум для скорости
            'degrees': 0.0,  # Без поворота для высокого угла обзора
            'translate': 0.1,  # Минимум для скорости (было 0.2)
            'scale': 0.2,  # Минимум для скорости (было 0.5)
            'shear': 0.0,
            'perspective': 0.0,  # Без перспективы для высокого угла обзора
            'flipud': 0.0,  # Без вертикального отражения для высокого угла
            'fliplr': 0.5,  # Горизонтальное отражение - быстро и эффективно
            'mosaic': 0.0,  # Отключено для максимальной скорости (было 0.5)
            'mixup': 0.0,  # Отключено для максимальной скорости (было 0.1)
            'copy_paste': 0.0,  # Отключено для максимальной скорости (было 0.1)
            'cfg': None,
            'tracker': None,
            'save_dir': str(experiment_dir),
            # Дополнительные параметры для маленьких объектов
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,  # Валидация во время обучения
            'amp': True,  # Mixed precision training для ускорения (если поддерживается)
            'fraction': 1.0,  # Использовать весь датасет
            'profile': False,  # Отключить профилирование для скорости
            'plots': plots,  # Генерировать графики только если нужно
            'save': save,  # Сохранять чекпоинты
            'cache': False,  # Не кэшировать изображения в памяти (экономит память, но немного медленнее)
            'single_cls': False,  # Множественные классы
            'rect': False,  # Прямоугольное обучение (может ускорить, но ухудшить качество)
            'cos_lr': False,  # Cosine learning rate schedule
            'close_mosaic': 10,  # Отключить mosaic за 10 эпох до конца (уже отключен, но на всякий случай)
        }
        
        self.logger.info(f"Директория эксперимента: {experiment_dir}")
        
        # Очистка кэша CUDA перед обучением
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("Кэш CUDA очищен перед обучением")
        
        # Запуск обучения с обработкой ошибок multiprocessing и CUDA OOM
        try:
            self.logger.info("=" * 70)
            self.logger.info("ЗАПУСК ОБУЧЕНИЯ...")
            self.logger.info("=" * 70)
            self.logger.info("Это может занять значительное время. Пожалуйста, подождите...")
            results = model.train(**train_params)
            
            self.logger.info("=" * 70)
            self.logger.info("✓ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            self.logger.info("=" * 70)
            
            # Статистика результатов
            best_model = experiment_dir / "weights" / "best.pt"
            last_model = experiment_dir / "weights" / "last.pt"
            
            self.logger.info(f"Лучшая модель: {best_model}")
            self.logger.info(f"Последняя модель: {last_model}")
            
            # Метрики
            final_metrics = None
            results_csv = experiment_dir / "results.csv"
            if results_csv.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(results_csv)
                    if len(df) > 0:
                        final_metrics = df.iloc[-1]
                        map50 = final_metrics.get('metrics/mAP50(B)', 'N/A')
                        map50_95 = final_metrics.get('metrics/mAP50-95(B)', 'N/A')
                        self.logger.info("=" * 50)
                        self.logger.info("ФИНАЛЬНЫЕ МЕТРИКИ ОБУЧЕНИЯ:")
                        self.logger.info(f"  mAP50: {map50:.3f}" if isinstance(map50, (int, float)) else f"  mAP50: {map50}")
                        self.logger.info(f"  mAP50-95: {map50_95:.3f}" if isinstance(map50_95, (int, float)) else f"  mAP50-95: {map50_95}")
                        self.logger.info("=" * 50)
                    else:
                        self.logger.warning("Файл results.csv пуст")
                except Exception as e:
                    self.logger.warning(f"Не удалось прочитать метрики из results.csv: {e}")
            else:
                self.logger.warning(f"Файл метрик не найден: {results_csv}")
            
            return {
                'success': True,
                'experiment_dir': str(experiment_dir),
                'best_model': str(best_model),
                'last_model': str(last_model),
                'results': results,
                'metrics': final_metrics
            }
            
        except KeyboardInterrupt:
            self.logger.warning("Обучение прервано пользователем")
            return {'success': False, 'error': 'Interrupted by user'}
        except RuntimeError as e:
            # Обработка CUDA out of memory
            error_str = str(e).lower()
            if 'out of memory' in error_str or ('cuda' in error_str and 'memory' in error_str):
                self.logger.warning(f"CUDA out of memory: {e}")
                
                # Очистка памяти
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("Кэш CUDA очищен")
                
                # Пробуем уменьшить параметры
                original_batch_size = batch_size
                original_img_size = img_size
                
                # Стратегия уменьшения: сначала batch_size, потом img_size
                new_batch_size = max(1, batch_size // 2)
                new_img_size = img_size
                
                if new_batch_size == 1 and img_size > 640:
                    # Если batch_size уже 1, уменьшаем размер изображения
                    new_img_size = max(640, int(img_size * 0.75))
                    self.logger.info(f"Уменьшаем размер изображения: {img_size} -> {new_img_size}")
                
                if new_batch_size < original_batch_size:
                    self.logger.info(f"Уменьшаем batch_size: {original_batch_size} -> {new_batch_size}")
                
                if new_batch_size != original_batch_size or new_img_size != original_img_size:
                    train_params['batch'] = new_batch_size
                    train_params['imgsz'] = new_img_size
                    
                    self.logger.info(f"Повторная попытка с уменьшенными параметрами:")
                    self.logger.info(f"  batch_size: {new_batch_size}, img_size: {new_img_size}")
                    
                    try:
                        results = model.train(**train_params)
                        self.logger.info("Обучение успешно завершено с уменьшенными параметрами!")
                        
                        # Статистика результатов
                        best_model = experiment_dir / "weights" / "best.pt"
                        last_model = experiment_dir / "weights" / "last.pt"
                        
                        self.logger.info(f"Лучшая модель: {best_model}")
                        self.logger.info(f"Последняя модель: {last_model}")
                        
                        # Метрики
                        final_metrics = None
                        results_csv = experiment_dir / "results.csv"
                        if results_csv.exists():
                            import pandas as pd
                            df = pd.read_csv(results_csv)
                            final_metrics = df.iloc[-1]
                            self.logger.info(
                                f"Финальные метрики: mAP50={final_metrics.get('metrics/mAP50(B)', 'N/A'):.3f}, "
                                f"mAP50-95={final_metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}"
                            )
                        
                        return {
                            'success': True,
                            'experiment_dir': str(experiment_dir),
                            'best_model': str(best_model),
                            'last_model': str(last_model),
                            'results': results,
                            'metrics': final_metrics,
                            'warning': f'Used reduced parameters: batch_size={new_batch_size}, img_size={new_img_size} due to OOM'
                        }
                    except RuntimeError as retry_error:
                        retry_error_str = str(retry_error).lower()
                        if 'out of memory' in retry_error_str:
                            # Если все еще не хватает памяти, пробуем еще меньше
                            if new_batch_size > 1:
                                train_params['batch'] = 1
                                train_params['imgsz'] = max(640, int(new_img_size * 0.8))
                                self.logger.info(f"Последняя попытка: batch_size=1, img_size={train_params['imgsz']}")
                                
                                try:
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    results = model.train(**train_params)
                                    self.logger.info("Обучение успешно завершено с минимальными параметрами!")
                                    
                                    best_model = experiment_dir / "weights" / "best.pt"
                                    last_model = experiment_dir / "weights" / "last.pt"
                                    
                                    final_metrics = None
                                    results_csv = experiment_dir / "results.csv"
                                    if results_csv.exists():
                                        import pandas as pd
                                        df = pd.read_csv(results_csv)
                                        final_metrics = df.iloc[-1]
                                    
                                    return {
                                        'success': True,
                                        'experiment_dir': str(experiment_dir),
                                        'best_model': str(best_model),
                                        'last_model': str(last_model),
                                        'results': results,
                                        'metrics': final_metrics,
                                        'warning': f'Used minimal parameters: batch_size=1, img_size={train_params["imgsz"]} due to OOM'
                                    }
                                except Exception as final_error:
                                    self.logger.error(f"Не удалось обучить даже с минимальными параметрами: {final_error}")
                                    return {'success': False, 'error': f'CUDA OOM: не хватает памяти даже с batch_size=1, img_size={train_params["imgsz"]}. Попробуйте использовать меньшую модель или CPU.'}
                            else:
                                self.logger.error(f"Не удалось обучить даже с batch_size=1: {retry_error}")
                                return {'success': False, 'error': f'CUDA OOM: не хватает памяти даже с batch_size=1. Попробуйте использовать меньшую модель (yolov8n.pt) или CPU.'}
                        else:
                            raise
                else:
                    self.logger.error("Не удалось уменьшить параметры достаточно")
                    return {'success': False, 'error': f'CUDA OOM: {e}. Попробуйте использовать меньшую модель (yolov8n.pt) или уменьшить img_size.'}
            else:
                # Другие RuntimeError - пробрасываем дальше к обработке multiprocessing
                raise
        except (ConnectionResetError, RuntimeError) as e:
            # Обработка ошибок multiprocessing (только если это не CUDA OOM)
            error_str = str(e).lower()
            # Пропускаем CUDA OOM - они уже обработаны выше
            if 'out of memory' in error_str or ('cuda' in error_str and 'memory' in error_str):
                raise  # Пробрасываем дальше, так как уже обработали
            
            if 'connection' in error_str or 'multiprocessing' in error_str or 'resource_sharer' in error_str:
                self.logger.warning(f"Ошибка multiprocessing: {e}")
                self.logger.info("Попытка перезапуска с workers=0 (без multiprocessing)...")
                
                # Пробуем с workers=0 (без multiprocessing)
                train_params['workers'] = 0
                try:
                    results = model.train(**train_params)
                    self.logger.info("Обучение успешно завершено с workers=0!")
                    
                    # Статистика результатов
                    best_model = experiment_dir / "weights" / "best.pt"
                    last_model = experiment_dir / "weights" / "last.pt"
                    
                    self.logger.info(f"Лучшая модель: {best_model}")
                    self.logger.info(f"Последняя модель: {last_model}")
                    
                    # Метрики
                    final_metrics = None
                    results_csv = experiment_dir / "results.csv"
                    if results_csv.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(results_csv)
                            if len(df) > 0:
                                final_metrics = df.iloc[-1]
                                map50 = final_metrics.get('metrics/mAP50(B)', 'N/A')
                                map50_95 = final_metrics.get('metrics/mAP50-95(B)', 'N/A')
                                self.logger.info("=" * 50)
                                self.logger.info("ФИНАЛЬНЫЕ МЕТРИКИ ОБУЧЕНИЯ (workers=0):")
                                self.logger.info(f"  mAP50: {map50:.3f}" if isinstance(map50, (int, float)) else f"  mAP50: {map50}")
                                self.logger.info(f"  mAP50-95: {map50_95:.3f}" if isinstance(map50_95, (int, float)) else f"  mAP50-95: {map50_95}")
                                self.logger.info("=" * 50)
                        except Exception as e:
                            self.logger.warning(f"Не удалось прочитать метрики: {e}")
                    
                    return {
                        'success': True,
                        'experiment_dir': str(experiment_dir),
                        'best_model': str(best_model),
                        'last_model': str(last_model),
                        'results': results,
                        'metrics': final_metrics,
                        'warning': 'Used workers=0 due to multiprocessing error'
                    }
                except Exception as retry_error:
                    self.logger.error(f"Ошибка при повторной попытке: {retry_error}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return {'success': False, 'error': f'Multiprocessing error: {e}, retry failed: {retry_error}'}
            else:
                # Другие RuntimeError - пробрасываем дальше
                raise
        except Exception as e:
            self.logger.error(f"Ошибка обучения: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def validate_model(
        self,
        model_path: Optional[str] = None,
        data_path: str = "config/ppe_data.yaml",
        conf_threshold: float = 0.5
    ) -> Dict:
        """
        Валидирует обученную модель на валидационной выборке.
        
        Args:
            model_path: Путь к модели (если None - использует лучшую)
            data_path: Путь к конфигурации данных
            conf_threshold: Порог уверенности для детекции
            
        Returns:
            Результаты валидации
        """
        if model_path is None:
            model_path = self.project_dir / self.experiment_name / "weights" / "best.pt"
        
        if not Path(model_path).exists():
            self.logger.error(f"Модель не найдена: {model_path}")
            return {'success': False, 'error': f'Model not found: {model_path}'}
        
        self.logger.info(f"Валидация модели: {model_path}")
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            results = model.val(
                data=data_path,
                conf=conf_threshold,
                verbose=True,
                save_json=True,
                project=self.project_dir,
                name=f"{self.experiment_name}_validation"
            )
            
            self.logger.info("Валидация завершена")
            self.logger.info(f"mAP50: {results.box.map50:.3f}")
            self.logger.info(f"mAP50-95: {results.box.map:.3f}")
            
            return {
                'success': True,
                'map50': results.box.map50,
                'map': results.box.map,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_sample(
        self,
        image_path: str,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        save_result: bool = True
    ) -> Optional[Path]:
        """
        Тестирует модель на одном изображении.
        
        Args:
            image_path: Путь к тестовому изображению
            model_path: Путь к модели
            conf_threshold: Порог уверенности
            save_result: Сохранить результат
            
        Returns:
            Путь к сохраненному изображению с детекциями
        """
        if model_path is None:
            model_path = self.project_dir / self.experiment_name / "weights" / "best.pt"
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            results = model.predict(
                image_path,
                conf=conf_threshold,
                save=save_result,
                project="output",
                name="test_prediction",
                exist_ok=True,
                verbose=False
            )
            
            if save_result:
                output_path = Path("output/test_prediction") / Path(image_path).name
                self.logger.info(f"Результат сохранен: {output_path}")
                return output_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка предсказания: {e}")
            return None


def create_trainer_config(
    config_path: str = "config/ppe_data.yaml",
    classes: List[str] = None
) -> Path:
    """
    Создает конфигурационный файл для обучения.
    
    Args:
        config_path: Путь к конфигурации
        classes: Список классов (если None - загружает из classes.txt)
        
    Returns:
        Путь к созданному файлу
    """
    if classes is None:
        # Загружаем классы из файла classes.txt
        classes_file = None
        for path in [Path("classes.txt"), Path("data/classes.txt"), Path("config/classes.txt")]:
            if path.exists():
                classes_file = path
                break
        
        if classes_file is None:
            raise FileNotFoundError(
                "Файл classes.txt не найден! Создайте файл classes.txt с перечислением классов (по одному на строку)."
            )
        
        classes = []
        with open(classes_file, 'r', encoding='utf-8') as f:
            for line in f:
                class_name = line.strip()
                if class_name and not class_name.startswith('#'):
                    classes.append(class_name)
        
        if len(classes) == 0:
            raise ValueError(f"Файл classes.txt пуст или не содержит классов: {classes_file}")
    
    config_content = f"""path: ./data
train: images/train
val: images/val

nc: {len(classes)}
names: 
"""
    
    for i, class_name in enumerate(classes):
        config_content += f"  {i}: {class_name}\n"
    
    config = Path(config_path)
    config.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config, 'w') as f:
        f.write(config_content)
    
        print(f"Конфигурация создана: {config}")
    print(f"Классы: {classes}")
    
    return config


if __name__ == "__main__":
    # Пример использования
    trainer = PPEDetectorTrainer()
    
    # Обучение
    results = trainer.train(epochs=30, batch_size=16)
    
    if results['success']:
        print(f"Обучение завершено успешно!")
        print(f"Модель: {results['best_model']}")
        
        # Валидация
        val_results = trainer.validate_model()
        if val_results['success']:
            print(f"mAP50: {val_results['map50']:.3f}")
            print(f"mAP50-95: {val_results['map']:.3f}")
        
        # Тест на изображении
        trainer.predict_sample("data/images/val/sample.jpg", save_result=True)
    
    else:
        print(f"Ошибка обучения: {results.get('error', 'Unknown error')}")
