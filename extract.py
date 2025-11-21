"""
Скрипт для извлечения кадров из AVI видео в датасет.

Использование:
    python extract.py                          # Извлечение из videos/ в data/images/train/
    python extract.py --input videos/          # Указать папку с видео
    python extract.py --output data/images/train/  # Указать папку для кадров
    python extract.py --step 30                # Извлекать каждый 30-й кадр
    python extract.py --input videos/ --output data/images/train/ --step 30
"""

import argparse
import sys
from pathlib import Path
from src.data.extract_frames import auto_extract_frames


def main():
    parser = argparse.ArgumentParser(
        description="Извлечение кадров из AVI видео для создания датасета",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python extract.py
  python extract.py --input videos/ --output data/images/train/
  python extract.py --step 30
  python extract.py --input videos/ --output data/images/train/ --step 15
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='videos',
        help='Папка с видео файлами (по умолчанию: videos/)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/images/train',
        help='Папка для сохранения извлеченных кадров (по умолчанию: data/images/train/)'
    )
    
    parser.add_argument(
        '--step', '-s',
        type=int,
        default=30,
        help='Извлекать каждый N-й кадр (по умолчанию: 30)'
    )
    
    args = parser.parse_args()
    
    # Проверка существования папки с видео
    videos_dir = Path(args.input)
    if not videos_dir.exists():
        print(f"ОШИБКА: Папка '{videos_dir}' не найдена!")
        print(f"Создайте папку '{videos_dir}' и поместите туда видео файлы (AVI, MP4, MOV, и т.д.)")
        sys.exit(1)
    
    # Проверка наличия видео файлов
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_dir.glob(f"*{ext}"))
        video_files.extend(videos_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"ОШИБКА: Не найдено видео файлов в '{videos_dir}'!")
        print(f"Поддерживаемые форматы: {', '.join(video_extensions)}")
        sys.exit(1)
    
    print("=" * 70)
    print("ИЗВЛЕЧЕНИЕ КАДРОВ ИЗ ВИДЕО")
    print("=" * 70)
    print(f"Папка с видео: {videos_dir}")
    print(f"Папка для кадров: {args.output}")
    print(f"Шаг извлечения: каждый {args.step}-й кадр")
    print(f"Найдено видео файлов: {len(video_files)}")
    print()
    
    # Извлечение кадров
    try:
        total_frames = auto_extract_frames(
            videos_dir=str(videos_dir),
            output_dir=str(args.output),
            step=args.step
        )
        
        print()
        print("=" * 70)
        print("ИЗВЛЕЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 70)
        print(f"Всего извлечено кадров: {total_frames}")
        print(f"Кадры сохранены в: {args.output}")
        print()
        print("Следующие шаги:")
        print("1. Разметьте кадры вручную (используйте LabelImg или другой инструмент)")
        print("2. Или используйте автоматическую предразметку (если доступна)")
        print("3. Запустите pipeline для обучения: python run_full_pipeline.py")
        
    except Exception as e:
        print(f"\nОШИБКА при извлечении кадров: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

