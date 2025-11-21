"""
Модуль для извлечения кадров из видео.

Использование:
from src.data.extract_frames import auto_extract_frames
auto_extract_frames(videos_dir="videos", output_dir="data/images/train")
"""

import cv2
from pathlib import Path
from typing import List, Tuple
import os


def extract_frames_from_video(
    video_path: Path, 
    output_dir: Path, 
    step: int = 30
) -> int:
    """
    Извлекает кадры из видео с заданным шагом.
    
    Args:
        video_path: Путь к видео файлу
        output_dir: Директория для сохранения кадров
        step: Извлекать каждый N-й кадр
        
    Returns:
        Количество сохраненных кадров
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    video_name = video_path.stem
    
    print(f"  Обработка: {video_path.name}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % step == 0:
            frame_filename = output_dir / f"{video_name}_frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"  Сохранено кадров: {saved_count} (из {frame_count} всего)")
    return saved_count


def auto_extract_frames(
    videos_dir: str = "videos", 
    output_dir: str = "data/images/train", 
    step: int = 30
) -> int:
    """
    Автоматически извлекает кадры из всех видео в указанной папке.
    
    Args:
        videos_dir: Папка с видео файлами
        output_dir: Папка для сохранения кадров
        step: Извлекать каждый N-й кадр
        
    Returns:
        Общее количество сохраненных кадров
    """
    videos_path = Path(videos_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not videos_path.exists():
        print(f"Папка '{videos_dir}' не найдена!")
        print(f"Создайте папку '{videos_dir}' и поместите туда видео файлы.")
        return 0
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_files: List[Path] = []
    for ext in video_extensions:
        video_files.extend(videos_path.glob(f"*{ext}"))
        video_files.extend(videos_path.glob(f"*{ext.upper()}"))
    
    if len(video_files) == 0:
        print(f"Не найдено видео файлов в '{videos_dir}'!")
        print(f"Поддерживаемые форматы: {', '.join(video_extensions)}")
        return 0
    
    print(f"Найдено видео: {len(video_files)}")
    total_saved = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] {video_file.name}")
        saved = extract_frames_from_video(video_file, output_path, step)
        total_saved += saved
    
    print(f"\nИзвлечено {total_saved} кадров в {output_dir}/")
    return total_saved


if __name__ == "__main__":
    # Пример использования
    total_frames = auto_extract_frames()
    print(f"Готово! Извлечено {total_frames} кадров.")
