"""
Программа для разделения датасета на train/val.

Структура до разделения:
    data/
    ├── images/          # Все изображения
    ├── labels/         # Все аннотации
    ├── classes.txt
    └── notes.json

Структура после разделения:
    data/
    ├── images/
    │   ├── train/      # Изображения для обучения
    │   └── val/        # Изображения для валидации
    ├── labels/
    │   ├── train/      # Аннотации для обучения
    │   └── val/        # Аннотации для валидации
    ├── classes.txt
    └── notes.json

Использование:
    python data.py                          # Разделение с val_ratio=0.2 (20% валидации)
    python data.py --val-ratio 0.3          # 30% валидации
    python data.py --val-ratio 0.1          # 10% валидации
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Tuple, List


def split_dataset(
    images_dir: str = "data/images",
    labels_dir: str = "data/labels",
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Разделяет датасет на обучающую и валидационную выборки.
    
    Args:
        images_dir: Директория с изображениями (data/images)
        labels_dir: Директория с аннотациями (data/labels)
        val_ratio: Доля данных для валидации (0.2 = 20%)
        seed: Seed для воспроизводимости
        
    Returns:
        Кортеж (количество перемещенных изображений, количество перемещенных аннотаций)
    """
    random.seed(seed)
    
    # Пути
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Создаем структуру train/val
    train_images_path = images_path / "train"
    val_images_path = images_path / "val"
    train_labels_path = labels_path / "train"
    val_labels_path = labels_path / "val"
    
    # Создаем папки
    train_images_path.mkdir(parents=True, exist_ok=True)
    val_images_path.mkdir(parents=True, exist_ok=True)
    train_labels_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)
    
    # Проверяем наличие исходных данных
    if not images_path.exists():
        raise FileNotFoundError(f"Директория с изображениями не найдена: {images_dir}")
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Директория с аннотациями не найдена: {labels_dir}")
    
    # Находим все изображения
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
    
    if len(image_files) == 0:
        raise ValueError(f"Не найдено изображений в {images_dir}")
    
    print(f"Найдено изображений: {len(image_files)}")
    
    # Перемешиваем для случайного разделения
    random.shuffle(image_files)
    
    # Вычисляем количество для валидации
    val_count = int(len(image_files) * val_ratio)
    train_count = len(image_files) - val_count
    
    if val_count == 0:
        print("⚠️  Предупреждение: val_ratio слишком мал, все данные будут в train")
        val_count = 0
        train_count = len(image_files)
    
    print(f"Разделение: {train_count} train, {val_count} val")
    
    # Разделяем изображения и соответствующие аннотации
    moved_images = 0
    moved_labels = 0
    
    for i, image_file in enumerate(image_files):
        # Определяем, в train или val
        if i < train_count:
            target_images_dir = train_images_path
            target_labels_dir = train_labels_path
        else:
            target_images_dir = val_images_path
            target_labels_dir = val_labels_path
        
        # Перемещаем изображение
        target_image = target_images_dir / image_file.name
        if image_file.exists():
            shutil.move(str(image_file), str(target_image))
            moved_images += 1
        
        # Перемещаем соответствующую аннотацию
        label_file = labels_path / f"{image_file.stem}.txt"
        if label_file.exists():
            target_label = target_labels_dir / label_file.name
            shutil.move(str(label_file), str(target_label))
            moved_labels += 1
        else:
            print(f"⚠️  Предупреждение: аннотация не найдена для {image_file.name}")
    
    print(f"✅ Разделение завершено!")
    print(f"   Перемещено изображений: {moved_images}")
    print(f"   Перемещено аннотаций: {moved_labels}")
    print(f"   Train: {train_images_path}")
    print(f"   Val: {val_images_path}")
    
    return moved_images, moved_labels


def check_structure(images_dir: str = "data/images", labels_dir: str = "data/labels") -> bool:
    """
    Проверяет, нужно ли разделение данных.
    
    Returns:
        True если нужно разделение (данные в корне images/ и labels/)
        False если уже разделено (есть train/val)
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Проверяем, есть ли train/val папки
    has_train_val = (
        (images_path / "train").exists() or
        (images_path / "val").exists() or
        (labels_path / "train").exists() or
        (labels_path / "val").exists()
    )
    
    # Проверяем, есть ли файлы в корне
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    has_root_files = False
    for ext in image_extensions:
        if list(images_path.glob(f"*{ext}")):
            has_root_files = True
            break
    
    # Если есть train/val, но нет файлов в корне - уже разделено
    if has_train_val and not has_root_files:
        return False
    
    # Если есть файлы в корне - нужно разделение
    if has_root_files:
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Разделение датасета на train/val",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python data.py
  python data.py --val-ratio 0.3
  python data.py --val-ratio 0.1
  python data.py --images data/images --labels data/labels
        """
    )
    
    parser.add_argument(
        "--images", "-i",
        type=str,
        default="data/images",
        help="Папка с изображениями (по умолчанию: data/images)"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default="data/labels",
        help="Папка с аннотациями (по умолчанию: data/labels)"
    )
    parser.add_argument(
        "--val-ratio", "-r",
        type=float,
        default=0.2,
        help="Доля данных для валидации (по умолчанию: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Seed для воспроизводимости (по умолчанию: 42)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Принудительное разделение, даже если train/val уже существуют"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("РАЗДЕЛЕНИЕ ДАТАСЕТА НА TRAIN/VAL")
    print("=" * 70)
    print(f"Папка с изображениями: {args.images}")
    print(f"Папка с аннотациями: {args.labels}")
    print(f"Доля валидации: {args.val_ratio * 100:.1f}%")
    print(f"Seed: {args.seed}")
    print("-" * 70)
    
    # Проверяем структуру
    if not args.force:
        needs_split = check_structure(args.images, args.labels)
        if not needs_split:
            print("   Данные уже разделены на train/val")
            print("   Используйте --force для принудительного разделения")
            return
    
    try:
        moved_images, moved_labels = split_dataset(
            images_dir=args.images,
            labels_dir=args.labels,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        
        print("=" * 70)
        print(f"   Разделение завершено успешно!")
        print(f"   Перемещено изображений: {moved_images}")
        print(f"   Перемещено аннотаций: {moved_labels}")
        print("=" * 70)
        
    except Exception as e:
        print(f"   Ошибка при разделении: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

