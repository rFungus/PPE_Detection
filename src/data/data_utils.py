"""
Утилиты для работы с данными датасета.

Содержит функции проверки структуры, валидации разметки и статистики.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import os


def check_data_structure(
    data_root: str = "data"
) -> Dict[str, int]:
    """
    Проверяет структуру данных и возвращает статистику.
    
    Args:
        data_root: Корневая папка с данными (data/)
        
    Returns:
        Словарь со статистикой по папкам и файлам
    """
    data_path = Path(data_root)
    stats = {
        'train_images': 0, 'train_labels': 0,
        'val_images': 0, 'val_labels': 0,
        'total_images': 0, 'total_labels': 0,
        'missing_labels': 0, 'extra_labels': 0,
        'invalid_labels': 0
    }
    
    # Проверяем наличие основных папок
    required_dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    
    print("ПРОВЕРКА СТРУКТУРЫ ДАННЫХ")
    print("-" * 40)
    
    for dir_path in required_dirs:
        full_path = data_path / dir_path
        if full_path.exists():
            # Подсчет изображений (только для папок images)
            img_count = 0
            if "images" in dir_path:
                img_count = len(list(full_path.glob("*.jpg"))) + \
                           len(list(full_path.glob("*.png"))) + \
                           len(list(full_path.glob("*.jpeg"))) + \
                           len(list(full_path.glob("*.JPG"))) + \
                           len(list(full_path.glob("*.PNG"))) + \
                           len(list(full_path.glob("*.JPEG")))
            
            # Подсчет разметки (только для папок labels)
            label_count = 0
            if "labels" in dir_path:
                label_count = len(list(full_path.glob("*.txt"))) + \
                             len(list(full_path.glob("*.TXT")))
            
            # Устанавливаем значения только для соответствующих типов (не перезаписываем)
            if "train" in dir_path:
                if "images" in dir_path:
                    stats['train_images'] = img_count
                if "labels" in dir_path:
                    stats['train_labels'] = label_count
            else:  # val
                if "images" in dir_path:
                    stats['val_images'] = img_count
                if "labels" in dir_path:
                    stats['val_labels'] = label_count
            
            status = "OK" if (img_count > 0 or label_count > 0) else "EMPTY"
            print(f"{status} {dir_path}: {img_count} изображений, {label_count} разметок")
        else:
            print(f"MISSING {dir_path}")
    
    stats['total_images'] = stats['train_images'] + stats['val_images']
    stats['total_labels'] = stats['train_labels'] + stats['val_labels']
    
    # Проверка соответствия файлов
    print("\nПРОВЕРКА СООТВЕТСТВИЯ ФАЙЛОВ")
    print("-" * 40)
    
    # Train - проверка соответствия файлов (все расширения изображений)
    train_missing_labels = 0
    train_extra_labels = 0
    
    train_images_dir = data_path / "images/train"
    train_labels_dir = data_path / "labels/train"
    
    # Проверяем только если папки существуют
    if train_images_dir.exists() and train_labels_dir.exists():
        # Проверяем все расширения изображений
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        for ext in image_extensions:
            for img_file in train_images_dir.glob(ext):
                label_file = train_labels_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    train_missing_labels += 1
        
        # Проверяем разметки без изображений (проверяем все возможные расширения)
        for label_file in train_labels_dir.glob("*.txt"):
            img_found = False
            for ext in image_extensions:
                img_file = train_images_dir / f"{label_file.stem}{ext[1:]}"  # убираем *
                if img_file.exists():
                    img_found = True
                    break
            if not img_found:
                train_extra_labels += 1
    
    stats['missing_labels'] = train_missing_labels
    stats['extra_labels'] = train_extra_labels
    
    if train_missing_labels > 0:
        print(f"WARNING Train: {train_missing_labels} изображений без разметки")
    if train_extra_labels > 0:
        print(f"WARNING Train: {train_extra_labels} разметок без изображений")
    
    if train_missing_labels == 0 and train_extra_labels == 0:
        print("OK Train: все файлы соответствуют")
    
    # Проверка формата разметки
    print("\nПРОВЕРКА ФОРМАТА РАЗМЕТКИ")
    print("-" * 40)
    
    invalid_labels = 0
    total_annotations = 0
    
    for label_file in (data_path / "labels/train").glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                content = f.readlines()
                total_annotations += len(content)
                
                for line_num, line in enumerate(content, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    # OBB формат: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 значений)
                    # Обычный YOLO формат: class_id x_center y_center width height (5 значений)
                    if len(parts) != 9 and len(parts) != 5:
                        print(f"ERROR {label_file.name}:{line_num} - неверный формат (ожидается 9 значений для OBB или 5 для обычного YOLO, получено {len(parts)})")
                        invalid_labels += 1
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        # Проверка класса (динамическая, не хардкод)
                        coords = [float(x) for x in parts[1:]]
                        if any(coord < 0 or coord > 1 for coord in coords):
                            print(f"WARNING {label_file.name}:{line_num} - координаты вне диапазона [0,1]")
                            invalid_labels += 1
                        
                        # Проверка для OBB формата (9 значений)
                        if len(parts) == 9:
                            # Проверяем, что все 8 координат (4 точки) в диапазоне [0,1]
                            if len(coords) != 8:
                                print(f"ERROR {label_file.name}:{line_num} - OBB формат должен содержать 8 координат (4 точки)")
                                invalid_labels += 1
                        # Проверка для обычного YOLO формата (5 значений)
                        elif len(parts) == 5:
                            # Проверка разумных размеров
                            if len(coords) >= 4:
                                width, height = coords[2], coords[3]
                                if width < 0.01 or height < 0.01 or width > 1 or height > 1:
                                    print(f"WARNING {label_file.name}:{line_num} - нереалистичные размеры box")
                                    invalid_labels += 1
                            
                    except ValueError as e:
                        print(f"ERROR {label_file.name}:{line_num} - ошибка парсинга: {e}")
                        invalid_labels += 1
                        
        except Exception as e:
            print(f"ERROR Ошибка чтения {label_file.name}: {e}")
            invalid_labels += 1
    
    stats['invalid_labels'] = invalid_labels
    stats['total_annotations'] = total_annotations
    
    if invalid_labels == 0:
        print("OK Формат разметки корректен")
    else:
        print(f"WARNING Найдено {invalid_labels} проблем в разметке")
    
    # Общая статистика
    print("\nОБЩАЯ СТАТИСТИКА")
    print("-" * 40)
    print(f"Всего изображений: {stats['total_images']}")
    if stats['total_images'] > 0:
        train_percent = stats['train_images'] / stats['total_images'] * 100
        val_percent = stats['val_images'] / stats['total_images'] * 100
        print(f"  Train: {stats['train_images']} ({train_percent:.1f}%)")
        print(f"  Val: {stats['val_images']} ({val_percent:.1f}%)")
    else:
        print(f"  Train: {stats['train_images']} (N/A - нет данных)")
        print(f"  Val: {stats['val_images']} (N/A - нет данных)")
    
    print(f"Всего аннотаций: {total_annotations}")
    print(f"  Train: {stats['train_labels']} файлов разметки")
    print(f"  Val: {stats['val_labels']} файлов разметки")
    
    if stats['total_images'] >= 100:
        print("OK Достаточно данных для обучения")
    else:
        print("WARNING Рекомендуется больше данных (минимум 100 изображений)")
    
    if stats['missing_labels'] == 0 and stats['extra_labels'] == 0 and invalid_labels == 0:
        print("\nДанные готовы к обучению!")
    else:
        print("\nWARNING Рекомендуется исправить проблемы перед обучением")
    
    return stats


def get_dataset_stats(data_root: str = "data") -> Dict[str, any]:
    """
    Получает подробную статистику датасета.
    
    Returns:
        Расширенная статистика включая классы и размеры
    """
    stats = check_data_structure(data_root)
    
    # Дополнительная статистика по классам
    class_stats = {'helmet': 0, 'vest': 0, 'unknown': 0}
    
    for label_file in Path(f"{data_root}/labels/train").glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        if class_id == 0:
                            class_stats['helmet'] += 1
                        elif class_id == 1:
                            class_stats['vest'] += 1
                        else:
                            class_stats['unknown'] += 1
        except:
            pass
    
    stats['class_distribution'] = class_stats
    stats['balance_score'] = min(class_stats['helmet'], class_stats['vest']) / max(class_stats['helmet'], class_stats['vest']) if max(class_stats['helmet'], class_stats['vest']) > 0 else 0
    
    return stats


def check_obb_labels(labels_dir: Path) -> Tuple[int, int, int]:
    """
    Проверяет формат OBB аннотаций в файлах .txt.
    Ожидаемый формат: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 значений)
    
    Args:
        labels_dir: Директория с файлами разметки
        
    Returns:
        Кортеж (валидных файлов, всего файлов, всего ошибок)
    """
    if not labels_dir.exists():
        print(f"Ошибка: Директория разметки не найдена: {labels_dir}")
        return 0, 0, 0

    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        print(f"В директории {labels_dir} не найдено файлов разметки (.txt).")
        return 0, 0, 0

    print(f"Проверка {len(label_files)} файлов разметки в {labels_dir}...")

    valid_files = 0
    total_errors = 0

    for label_file in label_files:
        file_valid = True
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) != 9:
                        print(f"  ❌ {label_file.name}:{line_num} - Неверное количество значений ({len(parts)} вместо 9).")
                        file_valid = False
                        total_errors += 1
                        continue

                    try:
                        class_id = int(parts[0])
                        coords = [float(p) for p in parts[1:]]

                        if not all(0.0 <= c <= 1.0 for c in coords):
                            print(f"  ⚠️ {label_file.name}:{line_num} - Координаты вне диапазона [0, 1].")
                            file_valid = False
                            total_errors += 1

                    except ValueError as ve:
                        print(f"  ❌ {label_file.name}:{line_num} - Ошибка парсинга чисел: {ve}")
                        file_valid = False
                        total_errors += 1

        except Exception as e:
            print(f"  ❌ Ошибка чтения файла {label_file.name}: {e}")
            file_valid = False
            total_errors += 1
        
        if file_valid:
            valid_files += 1

    if valid_files == len(label_files):
        print(f"\n✅ Все {len(label_files)} файлов разметки имеют корректный OBB формат.")
    else:
        print(f"\n⚠️ Найдено {len(label_files) - valid_files} файлов с ошибками в формате OBB.")
    
    return valid_files, len(label_files), total_errors


if __name__ == "__main__":
    # Пример использования
    print("=== ПРОВЕРКА ДАННЫХ ===")
    data_stats = check_data_structure()
    
    if data_stats['total_images'] > 0:
        detailed_stats = get_dataset_stats()
        print(f"\nРаспределение классов: {detailed_stats['class_distribution']}")
        print(f"Баланс классов: {detailed_stats['balance_score']:.2f}")
        
        if detailed_stats['balance_score'] > 0.5:
            print("OK Классы относительно сбалансированы")
        else:
            print("WARNING Рекомендуется добавить больше примеров меньшего класса")
