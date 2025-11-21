"""
Программа для работы с обученной моделью детекции СИЗ с поддержкой rotated bounding boxes.

Использование:
    python detect.py --model models/ppe_detection_obb/weights/best.pt --source image.jpg
    python detect.py --model models/ppe_detection_obb/weights/best.pt --source video.mp4
    python detect.py --model models/ppe_detection_obb/weights/best.pt --source data/images/val/
    python detect.py --model models/ppe_detection_obb/weights/best.pt --camera
"""

import argparse
import sys
from pathlib import Path
from src.inference.detect_utils import PPEDetector


def main():
    parser = argparse.ArgumentParser(
        description="Детекция СИЗ с поддержкой rotated bounding boxes (OBB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
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
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Путь к обученной модели OBB (.pt файл)"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Источник: путь к изображению, видео или папке с изображениями"
    )
    parser.add_argument(
        "--camera", "-c",
        action="store_true",
        help="Использовать камеру для детекции в реальном времени"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Порог уверенности для детекции (по умолчанию: 0.2)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/detections",
        help="Папка для сохранения результатов (по умолчанию: output/detections)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Устройство: 'cpu', '0' (GPU), 'auto' (по умолчанию: auto)"
    )
    
    args = parser.parse_args()
    
    # Проверка аргументов
    if not args.camera and args.source is None:
        parser.error("Необходимо указать --source или --camera")
    
    if args.camera and args.source is not None:
        parser.error("Используйте либо --source, либо --camera, но не оба одновременно")
    
    # Проверка модели
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Ошибка: Модель не найдена: {model_path}", file=sys.stderr)
        print(f"Убедитесь, что модель обучена и находится по указанному пути.", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 70)
    print("ДЕТЕКЦИЯ СИЗ С ПОДДЕРЖКОЙ ROTATED BOUNDING BOXES (OBB)")
    print("=" * 70)
    print(f"Модель: {model_path}")
    print(f"Порог уверенности: {args.conf}")
    print(f"Устройство: {args.device}")
    print("-" * 70)
    
    try:
        # Инициализация детектора
        detector = PPEDetector(
            model_path=str(model_path),
            conf_threshold=args.conf,
            device=args.device
        )
        
        # Обработка в зависимости от источника
        if args.camera:
            # Детекция с камеры
            print("Запуск детекции с камеры...")
            print("Нажмите 'q' для выхода, 's' для скриншота")
            detector.detect_camera(
                camera_id=0,
                conf_threshold=args.conf
            )
        
        elif args.source:
            source_path = Path(args.source)
            
            if not source_path.exists():
                print(f"❌ Ошибка: Источник не найден: {source_path}", file=sys.stderr)
                sys.exit(1)
            
            if source_path.is_file():
                # Файл (изображение или видео)
                if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Изображение
                    print(f"Обработка изображения: {source_path}")
                    result_img, detections = detector.detect_image(
                        image_path=str(source_path),
                        save_result=True,
                        output_dir=args.output,
                        show_confidence=True
                    )
                    
                    print(f"✅ Детекция завершена!")
                    print(f"Найдено объектов: {len(detections)}")
                    for i, det in enumerate(detections, 1):
                        print(f"  {i}. {det['class_name']}: {det['confidence']:.2f}")
                    print(f"Результат сохранен в: {Path(args.output) / source_path.name}")
                
                elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                    # Видео
                    print(f"Обработка видео: {source_path}")
                    output_video = detector.detect_video(
                        video_path=str(source_path),
                        conf_threshold=args.conf,
                        show_progress=True
                    )
                    print(f"✅ Видео обработано!")
                    print(f"Результат: {output_video}")
                
                else:
                    print(f"❌ Неподдерживаемый формат файла: {source_path.suffix}", file=sys.stderr)
                    sys.exit(1)
            
            elif source_path.is_dir():
                # Папка с изображениями
                print(f"Пакетная обработка папки: {source_path}")
                stats = detector.batch_predict(
                    image_folder=str(source_path),
                    output_folder=args.output,
                    conf_threshold=args.conf,
                    save_results=True
                )
                print(f"✅ Пакетная обработка завершена!")
                print(f"Обработано: {stats['processed']} изображений")
                print(f"Найдено детекций: {stats['detections']}")
                print(f"Ошибок: {stats['errors']}")
            
            else:
                print(f"❌ Неизвестный тип источника: {source_path}", file=sys.stderr)
                sys.exit(1)
        
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

