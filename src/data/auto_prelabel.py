"""
Модуль для автоматической предразметки изображений.

Использование:
from src.data.auto_prelabel import auto_prelabel
auto_prelabel(images_dir="data/images/train", labels_dir="data/labels/train")
"""

from pathlib import Path
from typing import Optional
import cv2
# YOLO импортируется лениво внутри функции, чтобы не замедлять импорт модуля


def auto_prelabel(
    images_dir: str = "data/images/train",
    labels_dir: str = "data/labels/train",
    conf_threshold: float = 0.3
) -> dict:
    """
    Автоматически предразмечает изображения с использованием предобученной модели.
    
    Args:
        images_dir: Папка с изображениями для разметки
        labels_dir: Папка для сохранения файлов разметки
        conf_threshold: Минимальная уверенность детекции
        
    Returns:
        Словарь со статистикой: {'processed': int, 'annotations': int, 'errors': int}
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    # Поиск изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(ext))
        image_files.extend(images_path.glob(ext.upper()))
    
    if len(image_files) == 0:
        print(f"Не найдено изображений в {images_dir}")
        print(f"Поддерживаемые форматы: {', '.join(image_extensions)}")
        return {'processed': 0, 'annotations': 0, 'errors': 0}
    
    print(f"Найдено изображений: {len(image_files)}")
    print(f"Порог уверенности: {conf_threshold}")
    
    # Загрузка модели (ленивый импорт YOLO)
    try:
        print("Импорт YOLO...")
        from ultralytics import YOLO
        print("Загрузка предобученной модели YOLOv8n...")
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Установите: pip install ultralytics")
        return {'processed': 0, 'annotations': 0, 'errors': 1}
    
    # COCO класс 'person' = 0
    person_class_id = 0
    
    stats = {'processed': 0, 'annotations': 0, 'errors': 0}
    
    for i, image_file in enumerate(image_files, 1):
        try:
            # Загрузка изображения
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"[{i}/{len(image_files)}] Не удалось загрузить: {image_file.name}")
                stats['errors'] += 1
                continue
            
            # Детекция
            results = model(image, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
            
            if boxes is None or len(boxes) == 0:
                print(f"  [{i}/{len(image_files)}] {image_file.name} - нет детекций")
                stats['processed'] += 1
                continue
            
            # Фильтрация людей
            person_boxes = []
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                if cls_id == person_class_id:
                    person_boxes.append(box)
            
            if len(person_boxes) == 0:
                print(f"  [{i}/{len(image_files)}] {image_file.name} - людей не найдено")
                stats['processed'] += 1
                continue
            
            # Создание разметки
            label_file = labels_path / f"{image_file.stem}.txt"
            with open(label_file, 'w') as f:
                img_height, img_width = image.shape[:2]
                
                for box in person_boxes:
                    # Координаты
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Нормализация для YOLO
                    center_x = (x1 + x2) / 2 / img_width
                    center_y = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Класс 0 (helmet) - для корректировки в LabelImg
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            stats['processed'] += 1
            stats['annotations'] += len(person_boxes)
            
            if i % 10 == 0 or i == len(image_files):
                progress = i / len(image_files) * 100
                print(f"  Прогресс: {progress:.1f}% ({i}/{len(image_files)})")
        
        except Exception as e:
            print(f"[{i}/{len(image_files)}] Ошибка обработки {image_file.name}: {e}")
            stats['errors'] += 1
    
    # Финальная статистика
    print("\n" + "=" * 60)
    print("АВТОМАТИЧЕСКАЯ ПРЕДРАЗМЕТКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print(f"Обработано изображений: {stats['processed']}/{len(image_files)}")
    print(f"Создано аннотаций: {stats['annotations']}")
    print(f"Ошибок: {stats['errors']}")
    print(f"Файлы разметки: {labels_dir}/")
    
    print("\nСЛЕДУЮЩИЕ ШАГИ:")
    print("1. Откройте LabelImg: labelImg")
    print("2. File → Open Dir → data/images/train/")
    print("3. Выберите формат YOLO (внизу окна)")
    print("4. View → Auto Save mode")
    print("5. Корректируйте:")
    print("   - 1 = helmet (каска) - оставьте")
    print("   - 2 = vest (жилет) - поменяйте класс")
    print("   - Delete = удалите лишний box")
    print("   - W = добавьте недостающий объект")
    print("   - D = следующее изображение")
    print("6. После корректировки: python run_training.py")
    
    return stats


if __name__ == "__main__":
    # Автоматический запуск с параметрами по умолчанию
    stats = auto_prelabel()
    print(f"\nГотово! Статистика: {stats}")
