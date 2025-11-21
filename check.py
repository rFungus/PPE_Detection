"""
Единый скрипт для проверки структуры данных и формата OBB аннотаций.

Использование:
    python check.py                    # Проверка структуры данных
    python check.py --obb              # Проверка формата OBB аннотаций
    python check.py --obb --labels data/labels/train
"""

import argparse
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_utils import check_data_structure, get_dataset_stats, check_obb_labels
from src.utils.config import ProjectConfig


def main():
    parser = argparse.ArgumentParser(
        description="Проверка структуры данных и формата OBB аннотаций",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python check.py                    # Проверка структуры данных
  python check.py --obb              # Проверка формата OBB аннотаций
  python check.py --obb --labels data/labels/train
        """
    )
    
    parser.add_argument(
        "--obb",
        action="store_true",
        help="Проверить формат OBB аннотаций"
    )
    
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default=None,
        help="Путь к директории с файлами разметки (по умолчанию: data/labels/train)"
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Корневая папка с данными (по умолчанию: data)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 70)
    
    # Проверка структуры данных
    if not args.obb:
        print("\n1. ПРОВЕРКА СТРУКТУРЫ ДАННЫХ")
        print("-" * 70)
        data_stats = check_data_structure(data_root=args.data_root)
        
        if data_stats['total_images'] > 0:
            print("\n2. ДЕТАЛЬНАЯ СТАТИСТИКА")
            print("-" * 70)
            detailed_stats = get_dataset_stats(data_root=args.data_root)
            print(f"Распределение классов: {detailed_stats['class_distribution']}")
            if 'balance_score' in detailed_stats:
                print(f"Баланс классов: {detailed_stats['balance_score']:.2f}")
    
    # Проверка формата OBB
    if args.obb:
        print("\n3. ПРОВЕРКА ФОРМАТА OBB АННОТАЦИЙ")
        print("-" * 70)
        
        if args.labels:
            labels_dir = Path(args.labels)
        else:
            config = ProjectConfig()
            labels_dir = config.data_dir / "labels" / "train"
        
        valid_files, total_files, total_errors = check_obb_labels(labels_dir)
        print(f"\nРезультаты: {valid_files}/{total_files} файлов валидны, {total_errors} ошибок")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

