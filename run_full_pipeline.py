"""
Полный автоматический запуск пайплайна для детекции СИЗ (каска + жилет).

Что делает этот скрипт:
- создает структуру проекта и конфигурацию (если их ещё нет);
- проверяет корректность структуры данных и разметки;
- обучает модель YOLOv8 с подобранными параметрами;
- выполняет быстрый тест модели на одном изображении из валидации.

ВАЖНО: Перед запуском pipeline необходимо:
1. Извлечь кадры из видео: python extract.py
2. Разметить кадры вручную (используя LabelImg или другой инструмент)
3. Убедиться, что данные находятся в правильной структуре:
   - data/images/train/ - изображения для обучения
   - data/labels/train/ - разметка для обучения
   - data/images/val/ - изображения для валидации (опционально, можно разделить автоматически)
   - data/labels/val/ - разметка для валидации (опционально)

Запуск (из корня проекта):

    python run_full_pipeline.py
"""

from pathlib import Path
import sys
import os
import logging
import platform
from datetime import datetime


def setup_logging(log_dir: Path = None) -> logging.Logger:
    """
    Настраивает логирование в файл и консоль.
    
    Args:
        log_dir: Директория для логов (если None - logs/ в корне проекта)
        
    Returns:
        Настроенный logger
    """
    if log_dir is None:
        log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Имя файла лога с временной меткой
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Формат логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Настройка root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("ПОЛНЫЙ АВТОМАТИЧЕСКИЙ ПАЙПЛАЙН ДЛЯ ДЕТЕКЦИИ СИЗ")
    logger.info("=" * 70)
    logger.info(f"Логи сохраняются в: {log_file}")
    logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def main() -> None:
    # Настройка логирования
    logger = setup_logging()
    
    # Добавляем корень проекта в PYTHONPATH
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    logger.info(f"Корень проекта: {project_root}")
    
    try:
        logger.info("Начало импорта модулей...")
        # Импорты локальных модулей (после добавления пути)
        logger.info("Импорт config...")
        import time
        start_time = time.time()
        from src.utils.config import config, ProjectConfig
        logger.info(f"Импорт config завершен за {time.time() - start_time:.2f} сек")
        
        logger.info("Импорт data модулей...")
        start_time = time.time()
        
        start_time = time.time()
        logger.info("  Импорт data_utils...")
        from src.data.data_utils import check_data_structure, get_dataset_stats
        logger.info(f"  data_utils импортирован за {time.time() - start_time:.2f} сек")
        
        logger.info("Импорт data модулей завершен")
        
        logger.info("Импорт models модулей...")
        start_time = time.time()
        from src.models.train_model import PPEDetectorTrainer
        logger.info(f"Импорт models модулей завершен за {time.time() - start_time:.2f} сек")
        
        logger.info("Импорт inference модулей...")
        start_time = time.time()
        from src.inference.detect_utils import PPEDetector
        logger.info(f"Импорт inference модулей завершен за {time.time() - start_time:.2f} сек")
        
        logger.info("Все модули успешно импортированы")
        
        # 1. Конфигурация и структура проекта
        logger.info("=" * 70)
        logger.info("ШАГ 1: Настройка конфигурации и структуры проекта")
        logger.info("=" * 70)
        try:
            logger.info("Получение сводки путей...")
            paths_summary = config.get_paths_summary()
            logger.info(f"Пути получены: {len(paths_summary)} элементов")
            
            logger.info("Валидация путей...")
            config.validate_paths()
            logger.info("Валидация путей завершена")
            
            logger.info("Создание конфигурации датасета...")
            config.create_dataset_config()
            logger.info("Конфигурация датасета создана")
            
            logger.info("Создание файла классов...")
            config.create_classes_file()
            logger.info("Файл классов создан")
            
            logger.info("Конфигурация и структура проекта подготовлены")
        except Exception as e:
            logger.error(f"Ошибка при настройке конфигурации: {e}", exc_info=True)
            raise
        
        # 2. Проверка структуры данных и разметки
        logger.info("=" * 70)
        logger.info("ШАГ 2: Проверка структуры данных и разметки")
        logger.info("=" * 70)
        try:
            data_ok = check_data_structure(data_root=str(config.data_dir))
            dataset_stats = get_dataset_stats(data_root=str(config.data_dir))
            total_images = dataset_stats.get("total_images", 0)
            class_distribution = dataset_stats.get("class_distribution", {})
            
            logger.info(f"Всего изображений: {total_images}")
            logger.info(f"  Train: {data_ok.get('train_images', 0)} изображений, {data_ok.get('train_labels', 0)} разметок")
            logger.info(f"  Val: {data_ok.get('val_images', 0)} изображений, {data_ok.get('val_labels', 0)} разметок")
            logger.info(f"Распределение классов (train): {class_distribution}")
            
            if total_images == 0:
                logger.error("Не найдено данных для обучения (нет изображений в data/images/train/ и data/images/val/).")
                logger.error("Загрузите данные (или извлеките кадры из видео) и запустите скрипт снова.")
                return
            
            # Проверяем наличие разметки
            train_images = data_ok.get('train_images', 0)
            train_labels = data_ok.get('train_labels', 0)
            missing_labels = data_ok.get('missing_labels', 0)
            
            if train_images > 0 and train_labels == 0:
                logger.warning(f"Найдено {train_images} изображений, но нет разметки!")
                logger.warning("Запустите автоматическую предразметку или разметьте данные вручную.")
            elif missing_labels > 0:
                logger.warning(f"Найдено {missing_labels} изображений без разметки.")
                logger.info("Можно продолжить обучение с имеющимися данными или доразметить недостающие изображения.")
        except Exception as e:
            logger.error(f"Ошибка при проверке данных: {e}", exc_info=True)
            raise
        
        # 3. Обучение модели
        logger.info("=" * 70)
        logger.info("ШАГ 3: Обучение модели YOLOv8")
        logger.info("=" * 70)
        try:
            logger.info("Инициализация PPEDetectorTrainer...")
            logger.info(f"  model_name: {config.model_name}")
            logger.info(f"  config_path: {config.config_dir / 'ppe_data.yaml'}")
            logger.info(f"  project_dir: {config.models_dir}")
            logger.info(f"  experiment_name: {config.experiment_name}")
            
            trainer = PPEDetectorTrainer(
                model_name=config.model_name,
                config_path=str(config.config_dir / "ppe_data.yaml"),
                project_dir=str(config.models_dir),
                experiment_name=config.experiment_name,
            )
            logger.info("PPEDetectorTrainer инициализирован")
            
            # Автоматический выбор параметров в зависимости от устройства
            if config.device == "cpu":
                epochs = 60
                batch_size = 4
                img_size = 640
            else:
                epochs = config.epochs
                batch_size = config.batch_size
                img_size = config.img_size
            
            logger.info(f"Параметры обучения: epochs={epochs}, img_size={img_size}, batch_size={batch_size}, patience={config.patience}, workers={config.workers}")
            logger.info("Запуск обучения...")
            
            train_results = trainer.train(
                epochs=epochs,
                img_size=img_size,  # Используем адаптированный размер
                batch_size=batch_size,
                patience=config.patience,
                workers=config.workers,
            )
            
            logger.info(f"Результат обучения получен: success={train_results.get('success', False)}")
            
            if not train_results.get("success", False):
                logger.error("Обучение завершилось с ошибкой.")
                logger.error(f"Ошибка: {train_results.get('error')}")
                return
            
            best_model_path = Path(train_results.get("best_model", ""))
            logger.info("Обучение завершено успешно!")
            logger.info(f"Лучшая модель: {best_model_path}")
            logger.info(f"Модель существует: {best_model_path.exists()}")
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
            raise
        
        # 4. Быстрый тест модели на одном изображении
        logger.info("=" * 70)
        logger.info("ШАГ 4: Быстрый тест обученной модели")
        logger.info("=" * 70)
        try:
            # Проверяем, что best_model_path был определен
            if 'best_model_path' not in locals():
                logger.warning("Путь к модели не определен, пропускаю тест инференса.")
            elif not best_model_path.exists():
                logger.warning("Файл лучшей модели не найден, пропускаю тест инференса.")
            else:
                val_images_dir = config.data_dir / "images" / "val"
                val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")) + list(
                    val_images_dir.glob("*.jpeg")
                )
                if not val_images:
                    logger.warning("Нет изображений в data/images/val/ для теста.")
                else:
                    test_img = val_images[0]
                    logger.info(f"Тестовое изображение: {test_img.name}")
                    logger.info("Инициализация детектора...")
                    try:
                        detector = PPEDetector(str(best_model_path))
                        logger.info("Детектор успешно загружен")
                        
                        logger.info("Выполнение детекции...")
                        result_img, detections = detector.detect_image(str(test_img), save_result=True)
                        
                        logger.info("=" * 50)
                        logger.info("РЕЗУЛЬТАТЫ ТЕСТОВОЙ ДЕТЕКЦИИ:")
                        logger.info(f"  Найдено детекций: {len(detections)}")
                        if detections:
                            for i, det in enumerate(detections, 1):
                                logger.info(f"  [{i}] {det['class_name']}: уверенность {det['confidence']:.3f}")
                        else:
                            logger.warning("  Детекции не найдены")
                        logger.info("=" * 50)
                        logger.info("Результат детекции сохранен в папке 'output/detections/'.")
                    except FileNotFoundError as e:
                        logger.error(f"Файл модели не найден: {e}")
                    except Exception as e:
                        logger.error(f"Ошибка при тестовом инференсе: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Ошибка при тестировании модели: {e}", exc_info=True)
            logger.warning("Продолжаем...")
        
        # Финальное сообщение
        logger.info("=" * 70)
        logger.info("ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО")
        logger.info("=" * 70)
        
        if 'best_model_path' in locals() and best_model_path.exists():
            logger.info(f"Обученная модель: {best_model_path}")
            experiment_dir = best_model_path.parent.parent
            logger.info(f"Метрики: {experiment_dir / 'results.csv'}")
            logger.info(f"Графики: {experiment_dir / 'results.png'}")
            logger.info("")
            logger.info("Использование модели:")
            logger.info("  python detect.py --model <путь_к_модели> --source <изображение/видео>")
            logger.info("  python check.py --obb  # Проверка формата аннотаций")
        
    except Exception as e:
        logger.critical(f"Критическая ошибка в пайплайне: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
