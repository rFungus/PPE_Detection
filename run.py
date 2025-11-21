"""
GUI приложение для работы с обученной моделью детекции СИЗ.

Функции:
- Выбор и обработка фото
- Выбор и обработка видео
- Трансляция с камеры (с выбором доступных устройств)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import threading
from typing import Optional, List
import sys

# Импорт детектора
from src.inference.detect_utils import PPEDetector


class PPEDetectionApp:
    """Главное окно приложения для детекции СИЗ."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Детекция СИЗ - Приложение")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Переменные
        self.detector: Optional[PPEDetector] = None
        self.model_path: Optional[Path] = None
        self.camera_thread: Optional[threading.Thread] = None
        self.camera_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Создание папки для демонстрационных фотографий
        self.stock_pic_dir = Path("stock_pic")
        self.stock_pic_dir.mkdir(exist_ok=True)
        
        # Создание интерфейса
        try:
            self._create_widgets()
        except Exception as e:
            print(f"Ошибка при создании интерфейса: {e}")
            import traceback
            traceback.print_exc()
            # Показываем базовое окно с ошибкой
            error_label = tk.Label(
                self.root,
                text=f"Ошибка создания интерфейса:\n{e}",
                fg="red",
                font=("Arial", 10)
            )
            error_label.pack(pady=50)
            return
        
        # Автопоиск модели (не блокируем запуск приложения)
        try:
            self._find_model()
        except Exception as e:
            print(f"Ошибка при поиске модели: {e}")
            self.model_label.config(
                text=f"Ошибка при поиске модели: {e}",
                fg="red"
            )
    
    def _create_widgets(self):
        """Создает элементы интерфейса."""
        # Заголовок
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Детекция СИЗ",
            font=("TkDefaultFont", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Основной контейнер
        main_frame = tk.Frame(self.root, padx=30, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Информация о модели
        model_frame = tk.LabelFrame(main_frame, text="Модель", font=("TkDefaultFont", 10, "bold"), padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.model_label = tk.Label(
            model_frame,
            text="Модель не загружена",
            font=("TkDefaultFont", 9),
            fg="gray",
            wraplength=500,
            justify=tk.LEFT
        )
        self.model_label.pack(anchor=tk.W)
        
        model_btn_frame = tk.Frame(model_frame)
        model_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.model_btn = tk.Button(
            model_btn_frame,
            text="Выбрать модель",
            command=self._select_model,
            bg="#3498db",
            fg="white",
            font=("TkDefaultFont", 10),
            padx=15,
            pady=5
        )
        self.model_btn.pack(side=tk.LEFT)
        
        # Кнопки действий
        actions_frame = tk.LabelFrame(main_frame, text="Действия", font=("TkDefaultFont", 10, "bold"), padx=10, pady=10)
        actions_frame.pack(fill=tk.BOTH, expand=True)
        
        # Кнопка выбора фото
        self.photo_btn = tk.Button(
            actions_frame,
            text="Выбрать фото",
            command=self._select_photo,
            bg="#27ae60",
            fg="white",
            font=("TkDefaultFont", 11, "bold"),
            padx=20,
            pady=15,
            width=25
        )
        self.photo_btn.pack(pady=10)
        
        # Кнопка выбора видео
        self.video_btn = tk.Button(
            actions_frame,
            text="Выбрать видео",
            command=self._select_video,
            bg="#e74c3c",
            fg="white",
            font=("TkDefaultFont", 11, "bold"),
            padx=20,
            pady=15,
            width=25
        )
        self.video_btn.pack(pady=10)
        
        # Кнопка трансляции
        self.stream_btn = tk.Button(
            actions_frame,
            text="Трансляция с камеры",
            command=self._start_stream,
            bg="#9b59b6",
            fg="white",
            font=("TkDefaultFont", 11, "bold"),
            padx=20,
            pady=15,
            width=25
        )
        self.stream_btn.pack(pady=10)
        
        # Мини-кнопка Alarm_t
        alarm_frame = tk.Frame(actions_frame)
        alarm_frame.pack(pady=5)
        
        self.alarm_btn = tk.Button(
            alarm_frame,
            text="Alarm_t",
            command=self._show_alarm,
            bg="#e67e22",
            fg="white",
            font=("TkDefaultFont", 9, "bold"),
            padx=10,
            pady=5
        )
        self.alarm_btn.pack()
        
        # Статус
        self.status_label = tk.Label(
            main_frame,
            text="Готов к работе",
            font=("TkDefaultFont", 9),
            fg="green"
        )
        self.status_label.pack(pady=(10, 0))
    
    def _find_model(self):
        """Автоматически ищет модель в стандартных местах."""
        try:
            possible_paths = [
                Path("models/ppe_detection_obb/weights/best.pt"),
                Path("models/ppe_detection_obb/weights/last.pt"),
            ]
            
            # Также ищем в подпапках models
            models_dir = Path("models")
            if models_dir.exists():
                for exp_dir in models_dir.iterdir():
                    if exp_dir.is_dir():
                        best_path = exp_dir / "weights" / "best.pt"
                        if best_path.exists():
                            possible_paths.append(best_path)
            
            for model_path in possible_paths:
                if model_path.exists():
                    self.model_path = model_path
                    self._load_model()
                    return
            
            # Если модель не найдена
            self.model_label.config(
                text="Модель не найдена. Пожалуйста, выберите модель вручную.",
                fg="red"
            )
        except Exception as e:
            print(f"Ошибка при поиске модели: {e}")
            self.model_label.config(
                text=f"Ошибка при поиске модели: {e}",
                fg="red"
            )
    
    def _select_model(self):
        """Позволяет пользователю выбрать модель вручную."""
        file_path = filedialog.askopenfilename(
            title="Выберите модель",
            filetypes=[("PyTorch модели", "*.pt"), ("Все файлы", "*.*")]
        )
        
        if file_path:
            self.model_path = Path(file_path)
            self._load_model()
    
    def _load_model(self):
        """Загружает модель детектора."""
        if not self.model_path or not self.model_path.exists():
            try:
                messagebox.showerror("Ошибка", "Файл модели не найден!")
            except:
                print("Ошибка: Файл модели не найден!")
            return
        
        try:
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Загрузка модели...", fg="orange")
            self.root.update()
            
            self.detector = PPEDetector(
                model_path=str(self.model_path),
                conf_threshold=0.2,
                device="auto"
            )
            
            if hasattr(self, 'model_label'):
                self.model_label.config(
                    text=f"Модель: {self.model_path.name}\nПуть: {self.model_path}",
                    fg="green"
                )
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Модель загружена успешно", fg="green")
            
        except Exception as e:
            error_msg = f"Не удалось загрузить модель:\n{e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            try:
                messagebox.showerror("Ошибка", error_msg)
            except:
                pass
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Ошибка загрузки модели", fg="red")
            self.detector = None
    
    def _select_photo(self):
        """Обработка выбора фото."""
        if not self._check_detector():
            return
        
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp"),
                ("Все файлы", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.status_label.config(text="Обработка изображения...", fg="orange")
            self.root.update()
            
            # Обработка в отдельном потоке, чтобы не блокировать GUI
            def process_image():
                try:
                    result_img, detections = self.detector.detect_image(
                        image_path=file_path,
                        save_result=True,
                        output_dir="output/detections",
                        show_confidence=True
                    )
                    
                    # Показываем результат
                    self.root.after(0, lambda: self._show_image_result(result_img, detections, file_path))
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка обработки:\n{e}"))
                    self.root.after(0, lambda: self.status_label.config(text="Ошибка обработки", fg="red"))
            
            thread = threading.Thread(target=process_image, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать изображение:\n{e}")
            self.status_label.config(text="Ошибка", fg="red")
    
    def _show_image_result(self, result_img, detections, original_path):
        """Показывает результат обработки изображения."""
        self.status_label.config(
            text=f"Обработано! Найдено объектов: {len(detections)}",
            fg="green"
        )
        
        # Показываем изображение
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        # Создаем новое окно для результата
        result_window = tk.Toplevel(self.root)
        result_window.title(f"Результат: {Path(original_path).name}")
        result_window.geometry("800x600")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(result_img)
        ax.set_title(f"Детекция СИЗ\nНайдено объектов: {len(detections)}", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Информация о детекциях
        if detections:
            info_text = "\n".join([
                f"{i+1}. {det['class_name']}: {det['confidence']:.2f}"
                for i, det in enumerate(detections)
            ])
            info_label = tk.Label(
                result_window,
                text=f"Детекции:\n{info_text}",
                font=("TkDefaultFont", 9),
                justify=tk.LEFT,
                padx=10,
                pady=10
            )
            info_label.pack()
        
        # Кнопка закрытия
        close_btn = tk.Button(
            result_window,
            text="Закрыть",
            command=result_window.destroy,
            bg="#3498db",
            fg="white",
            font=("TkDefaultFont", 9),
            padx=20,
            pady=5
        )
        close_btn.pack(pady=10)
    
    def _select_video(self):
        """Обработка выбора видео."""
        if not self._check_detector():
            return
        
        file_path = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[
                ("Видео", "*.mp4 *.avi *.mov *.mkv *.flv"),
                ("Все файлы", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.status_label.config(text="Обработка видео...", fg="orange")
            self.root.update()
            
            # Обработка в отдельном потоке
            def process_video():
                try:
                    output_path = self.detector.detect_video(
                        video_path=file_path,
                        conf_threshold=0.2,
                        show_progress=True
                    )
                    
                    self.root.after(0, lambda: self._show_video_result(output_path))
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка обработки видео:\n{e}"))
                    self.root.after(0, lambda: self.status_label.config(text="Ошибка обработки", fg="red"))
            
            thread = threading.Thread(target=process_video, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать видео:\n{e}")
            self.status_label.config(text="Ошибка", fg="red")
    
    def _show_video_result(self, output_path):
        """Показывает результат обработки видео."""
        self.status_label.config(text="Видео обработано!", fg="green")
        messagebox.showinfo(
            "Готово",
            f"Видео обработано успешно!\n\nРезультат сохранен:\n{output_path}"
        )
    
    def _get_available_cameras(self) -> List[int]:
        """Определяет доступные камеры."""
        available = []
        # Проверяем первые 10 индексов камер
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
    
    def _start_stream(self):
        """Запускает трансляцию с камеры."""
        if not self._check_detector():
            return
        
        # Получаем доступные камеры
        cameras = self._get_available_cameras()
        
        if not cameras:
            messagebox.showwarning(
                "Камеры не найдены",
                "Не найдено доступных камер.\nПроверьте подключение камеры."
            )
            return
        
        # Если только одна камера, используем её
        if len(cameras) == 1:
            self._run_camera_stream(cameras[0])
        else:
            # Показываем диалог выбора камеры
            self._show_camera_selection_dialog(cameras)
    
    def _show_camera_selection_dialog(self, cameras: List[int]):
        """Показывает диалог выбора камеры."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Выбор камеры")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Центрируем окно
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (300 // 2)
        dialog.geometry(f"400x300+{x}+{y}")
        
        # Заголовок
        title_label = tk.Label(
            dialog,
            text="Выберите камеру",
            font=("TkDefaultFont", 12, "bold"),
            pady=20
        )
        title_label.pack()
        
        # Список камер
        camera_frame = tk.Frame(dialog, padx=20, pady=10)
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        selected_camera = tk.IntVar(value=cameras[0])
        
        for cam_id in cameras:
            # Пытаемся получить информацию о камере
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                # Пробуем получить разрешение
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                camera_info = f"Камера {cam_id} ({width}x{height})"
            else:
                camera_info = f"Камера {cam_id}"
            
            radio = tk.Radiobutton(
                camera_frame,
                text=camera_info,
                variable=selected_camera,
                value=cam_id,
                font=("TkDefaultFont", 9),
                anchor=tk.W
            )
            radio.pack(fill=tk.X, pady=5)
        
        # Кнопки
        btn_frame = tk.Frame(dialog, pady=20)
        btn_frame.pack()
        
        def start():
            dialog.destroy()
            self._run_camera_stream(selected_camera.get())
        
        start_btn = tk.Button(
            btn_frame,
            text="Запустить",
            command=start,
            bg="#27ae60",
            fg="white",
            font=("TkDefaultFont", 9, "bold"),
            padx=20,
            pady=5
        )
        start_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(
            btn_frame,
            text="Отмена",
            command=dialog.destroy,
            bg="#95a5a6",
            fg="white",
            font=("TkDefaultFont", 9),
            padx=20,
            pady=5
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def _run_camera_stream(self, camera_id: int):
        """Запускает поток трансляции с камеры."""
        if self.camera_running:
            messagebox.showwarning("Внимание", "Трансляция уже запущена!")
            return
        
        self.camera_running = True
        self.status_label.config(text=f"Трансляция с камеры {camera_id}...", fg="orange")
        
        # Запускаем в отдельном потоке
        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            args=(camera_id,),
            daemon=True
        )
        self.camera_thread.start()
    
    def _camera_loop(self, camera_id: int):
        """Основной цикл обработки камеры."""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Не удалось открыть камеру {camera_id}"))
                self.root.after(0, lambda: self.status_label.config(text="Ошибка открытия камеры", fg="red"))
                self.camera_running = False
                return
            
            # Настройка камеры
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            window_name = f"Детекция СИЗ - Камера {camera_id}"
            frame_count = 0
            
            while self.camera_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Детекция каждые N кадров (для производительности)
                if frame_count % 2 == 0:  # Обрабатываем каждый 2-й кадр
                    try:
                        # Детекция OBB
                        results = self.detector.model.predict(
                            frame,
                            conf=self.detector.conf_threshold,
                            verbose=False,
                            device=self.detector.device,
                            imgsz=640,  # Меньший размер для скорости
                            iou=0.4,
                            max_det=500,
                        )
                        
                        # Рисуем детекции
                        obbs = results[0].obbs
                        if obbs is not None and len(obbs) > 0:
                            from src.inference.detect_utils import CLASS_NAMES, CLASS_COLORS
                            
                            for obb in obbs:
                                points = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                                class_id = int(obb.cls[0].cpu().numpy())
                                confidence = float(obb.conf[0].cpu().numpy())
                                
                                if confidence >= self.detector.conf_threshold:
                                    color = CLASS_COLORS.get(class_id, (255, 255, 255))
                                    
                                    # Рисуем rotated bounding box
                                    pts = points.reshape((-1, 1, 2))
                                    cv2.polylines(frame, [pts], True, color, 2)
                                    
                                    # Текст
                                    x_coords = points[:, 0]
                                    y_coords = points[:, 1]
                                    x1_text = int(x_coords.min())
                                    y1_text = int(y_coords.min())
                                    
                                    label = f"{CLASS_NAMES.get(class_id, 'unknown')}"
                                    if confidence < 0.9:
                                        label += f": {confidence:.2f}"
                                    
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    cv2.rectangle(
                                        frame, (x1_text, y1_text - label_size[1] - 10),
                                        (x1_text + label_size[0], y1_text), color, -1
                                    )
                                    cv2.putText(
                                        frame, label, (x1_text, y1_text - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                                    )
                        
                        # Информация на кадре
                        cv2.putText(
                            frame, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                        )
                        cv2.putText(
                            frame, "Press 'q' to quit", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                        )
                        
                    except Exception as e:
                        print(f"Ошибка детекции: {e}")
                
                # Показываем кадр
                cv2.imshow(window_name, frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                frame_count += 1
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка трансляции:\n{e}"))
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.camera_running = False
            self.root.after(0, lambda: self.status_label.config(text="Трансляция остановлена", fg="gray"))
    
    def _check_detector(self) -> bool:
        """Проверяет, загружена ли модель."""
        if self.detector is None:
            messagebox.showwarning(
                "Модель не загружена",
                "Пожалуйста, сначала загрузите модель."
            )
            return False
        return True
    
    def _get_stock_images(self) -> List[Path]:
        """Получает список изображений из папки stock_pic."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        if not self.stock_pic_dir.exists():
            return images
        
        for ext in image_extensions:
            images.extend(self.stock_pic_dir.glob(f"*{ext}"))
            images.extend(self.stock_pic_dir.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def _show_alarm(self):
        """Показывает окно предупреждения о нарушении СИЗ с фотографией из stock_pic."""
        # Получаем список изображений из stock_pic
        stock_images = self._get_stock_images()
        
        if not stock_images:
            messagebox.showwarning(
                "Нет фотографий",
                f"В папке {self.stock_pic_dir} нет демонстрационных фотографий.\n"
                f"Пожалуйста, загрузите фотографии в папку stock_pic."
            )
            return
        
        # Берем первое изображение (или можно сделать выбор)
        selected_image_path = stock_images[0]
        
        # Если несколько изображений, можно показать выбор
        if len(stock_images) > 1:
            # Показываем диалог выбора изображения
            dialog = tk.Toplevel(self.root)
            dialog.title("Выбор фотографии")
            dialog.geometry("400x300")
            dialog.resizable(False, False)
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Центрируем окно
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
            y = (dialog.winfo_screenheight() // 2) - (300 // 2)
            dialog.geometry(f"400x300+{x}+{y}")
            
            # Заголовок
            title_label = tk.Label(
                dialog,
                text="Выберите фотографию",
                font=("TkDefaultFont", 11, "bold"),
                pady=15
            )
            title_label.pack()
            
            # Список изображений
            list_frame = tk.Frame(dialog, padx=20, pady=10)
            list_frame.pack(fill=tk.BOTH, expand=True)
            
            selected_path = tk.StringVar(value=str(selected_image_path))
            
            for img_path in stock_images:
                radio = tk.Radiobutton(
                    list_frame,
                    text=img_path.name,
                    variable=selected_path,
                    value=str(img_path),
                    font=("TkDefaultFont", 9),
                    anchor=tk.W,
                    wraplength=350
                )
                radio.pack(fill=tk.X, pady=3)
            
            # Кнопки
            btn_frame = tk.Frame(dialog, pady=15)
            btn_frame.pack()
            
            def confirm():
                selected_image_path = Path(selected_path.get())
                dialog.destroy()
                self._display_alarm_window(selected_image_path)
            
            confirm_btn = tk.Button(
                btn_frame,
                text="Выбрать",
                command=confirm,
                bg="#27ae60",
                fg="white",
                font=("TkDefaultFont", 9, "bold"),
                padx=15,
                pady=5
            )
            confirm_btn.pack(side=tk.LEFT, padx=5)
            
            cancel_btn = tk.Button(
                btn_frame,
                text="Отмена",
                command=dialog.destroy,
                bg="#95a5a6",
                fg="white",
                font=("TkDefaultFont", 9),
                padx=15,
                pady=5
            )
            cancel_btn.pack(side=tk.LEFT, padx=5)
        else:
            # Если только одно изображение, сразу показываем окно
            self._display_alarm_window(selected_image_path)
    
    def _display_alarm_window(self, image_path: Path):
        """Отображает окно предупреждения с выбранной фотографией."""
        # Загружаем изображение
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{image_path}")
                return
            
            # Конвертируем BGR в RGB для matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки изображения:\n{e}")
            return
        
        # Создаем окно предупреждения
        alarm_window = tk.Toplevel(self.root)
        alarm_window.title("⚠️ Нарушение соблюдения СИЗ")
        alarm_window.geometry("700x600")
        alarm_window.resizable(False, False)
        alarm_window.transient(self.root)
        
        # Центрируем окно
        alarm_window.update_idletasks()
        x = (alarm_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (alarm_window.winfo_screenheight() // 2) - (600 // 2)
        alarm_window.geometry(f"700x600+{x}+{y}")
        
        # Заголовок предупреждения
        warning_frame = tk.Frame(alarm_window, bg="#e74c3c", height=80)
        warning_frame.pack(fill=tk.X)
        warning_frame.pack_propagate(False)
        
        warning_label = tk.Label(
            warning_frame,
            text="НАРУШЕНИЕ СОБЛЮДЕНИЯ СИЗ",
            font=("TkDefaultFont", 16, "bold"),
            bg="#e74c3c",
            fg="white"
        )
        warning_label.pack(pady=25)
        
        # Изображение
        image_frame = tk.Frame(alarm_window, padx=20, pady=20)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image_rgb)
        ax.set_title("Фотография нарушения", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Информация о файле
        info_label = tk.Label(
            image_frame,
            text=f"Фото: {image_path.name}",
            font=("TkDefaultFont", 8),
            fg="gray"
        )
        info_label.pack(pady=(5, 0))
        
        # Кнопка закрытия
        btn_frame = tk.Frame(alarm_window, pady=15)
        btn_frame.pack()
        
        close_btn = tk.Button(
            btn_frame,
            text="Закрыть",
            command=alarm_window.destroy,
            bg="#95a5a6",
            fg="white",
            font=("TkDefaultFont", 9),
            padx=20,
            pady=5
        )
        close_btn.pack()


def main():
    """Главная функция приложения."""
    try:
        root = tk.Tk()
        app = PPEDetectionApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print(f"Критическая ошибка: {e}")
        traceback.print_exc()
        # Показываем ошибку в окне, если tkinter работает
        try:
            import tkinter.messagebox as mb
            mb.showerror("Критическая ошибка", f"Программа завершилась с ошибкой:\n{e}\n\nПроверьте консоль для деталей.")
        except:
            pass


if __name__ == "__main__":
    main()

