import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageProcessingApp(tk.Tk):
    """Главный класс приложения для обработки изображений.

    Реализует функционал:
    - Загрузка изображений (JPG, PNG)
    - Захват с веб-камеры
    - Отображение цветовых каналов
    - Операции по варианту: маска по красному, повышение резкости,
    рисование прямоугольника
    """

    def __init__(self):
        """Инициализация главного окна приложения."""
        super().__init__()
        self.title("OZPR - работа с изображением")
        self.geometry("1200x800")
        self.configure(bg="#f0f0f0")

        try:
            self.iconbitmap("icon.ico")
        except BaseException:
            pass

        self.original_image = None
        self.current_image = None
        self.photo_image = None
        self.camera_available = self.check_camera()
        self.create_widgets()
        self.update_camera_status()

    def check_camera(self):
        """Проверяет доступность веб-камеры."""
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            return False
        cap.release()
        return True

    def create_widgets(self):
        """Создает элементы интерфейса."""
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.LabelFrame(
            main_frame, text="Управление", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 15))
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)
        ttk.Button(
            btn_frame,
            text="Загрузить изображение",
            command=self.load_image,
            width=20
        ).pack(pady=5)

        self.camera_btn = ttk.Button(
            btn_frame,
            text="Сделать снимок",
            command=self.capture_from_camera,
            width=20,
            state=tk.NORMAL if self.camera_available else tk.DISABLED
        )
        self.camera_btn.pack(pady=5)

        ttk.Button(
            btn_frame,
            text="Сбросить изменения",
            command=self.reset_image,
            width=20
        ).pack(pady=5)

        self.camera_status = ttk.Label(
            btn_frame,
            text="Камера: проверка...",
            foreground="green" if self.camera_available else "red"
        )
        self.camera_status.pack(pady=5)

        channel_frame = ttk.LabelFrame(
            control_frame, text="Цветовые каналы", padding=10)
        channel_frame.pack(side=tk.LEFT, padx=10)

        self.channel_var = tk.StringVar(value="Все")
        channels = ["Все", "Красный", "Зеленый", "Синий"]

        for channel in channels:
            ttk.Radiobutton(
                channel_frame,
                text=channel,
                variable=self.channel_var,
                value=channel,
                command=self.apply_channel
            ).pack(side=tk.LEFT, padx=5)

        operation_frame = ttk.LabelFrame(
            control_frame, text="Операции обработки", padding=10)
        operation_frame.pack(side=tk.LEFT, padx=10)

        self.operation_var = tk.StringVar()
        operations = [
            "1. Маска по красному",
            "2. Повышение резкости",
            "3. Рисование прямоугольника"
        ]

        op_combo = ttk.Combobox(
            operation_frame,
            textvariable=self.operation_var,
            values=operations,
            state="readonly",
            width=25
        )
        op_combo.pack(side=tk.LEFT, padx=5)
        op_combo.bind("<<ComboboxSelected>>", self.show_operation_params)

        self.apply_btn = ttk.Button(
            operation_frame,
            text="Применить",
            command=self.apply_operation,
            state=tk.DISABLED
        )
        self.apply_btn.pack(side=tk.LEFT, padx=5)
        self.param_frame = ttk.Frame(control_frame)
        self.param_frame.pack(side=tk.LEFT, padx=10)

        self.threshold_frame = ttk.Frame(self.param_frame)
        ttk.Label(self.threshold_frame,
                  text="Порог (0-255):").pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value="100")
        ttk.Entry(
            self.threshold_frame,
            textvariable=self.threshold_var,
            width=5).pack(
            side=tk.LEFT,
            padx=5)

        self.rect_frame = ttk.Frame(self.param_frame)
        coords = ["x1", "y1", "x2", "y2"]
        self.coord_vars = []

        for coord in coords:
            frame = ttk.Frame(self.rect_frame)
            frame.pack(side=tk.LEFT, padx=2)
            ttk.Label(frame, text=f"{coord}:").pack(side=tk.LEFT)
            var = tk.StringVar(value="0")
            ttk.Entry(frame, textvariable=var, width=5).pack(side=tk.LEFT)
            self.coord_vars.append(var)

        img_frame = ttk.LabelFrame(main_frame, text="Изображение", padding=10)
        img_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(img_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_camera_status(self):
        """Обновляет статус камеры в интерфейсе."""
        status = "Доступна" if self.camera_available else "Недоступна"
        color = "green" if self.camera_available else "red"
        self.camera_status.config(text=f"Камера: {status}", foreground=color)

    def show_operation_params(self, event=None):
        """Показывает параметры для выбранной операции."""
        # Скрываем все параметры
        for widget in self.param_frame.winfo_children():
            widget.pack_forget()

        operation = self.operation_var.get()

        if operation == "1. Маска по красному":
            self.threshold_frame.pack(side=tk.LEFT)
        elif operation == "3. Рисование прямоугольника":
            self.rect_frame.pack(side=tk.LEFT)

        self.apply_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Выбрана операция: {operation}")

    def load_image(self):
        """Загружает изображение из файла"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        try:
            pil_image = Image.open(file_path)

            if pil_image.mode == 'RGBA':
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[3])
                pil_image = background
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = self.original_image.copy()
            self.show_image()
            self.channel_var.set("Все")
            self.status_var.set(f"Изображение загружено: {file_path}")

        except Exception as e:
            messagebox.showerror(
                "Ошибка загрузки",
                f"Не удалось загрузить изображение:\n{str(e)}"
            )

    def capture_from_camera(self):
        """Захватывает изображение с веб-камеры."""
        if not self.camera_available:
            messagebox.showerror(
                "Ошибка камеры",
                "Веб-камера недоступна. Решения:\n"
                "1. Проверьте подключение камеры\n"
                "2. Предоставьте права приложению\n"
                "3. Перезагрузите устройство"
            )
            return

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror(
                "Ошибка", "Не удалось получить изображение с камеры")
            return
        self.original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_image = self.original_image.copy()
        self.show_image()
        self.channel_var.set("Все")
        self.status_var.set("Изображение с камеры захвачено")

    def show_image(self):
        """Отображает текущее изображение в интерфейсе."""
        if self.current_image is None:
            return

        img = self.current_image.copy()
        h, w = img.shape[:2]
        max_size = 800

        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_pil = Image.fromarray(img)
        self.photo_image = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.photo_image)

    def apply_channel(self):
        """Применяет выбранный цветовой канал."""
        if self.current_image is None:
            return

        channel = self.channel_var.get()

        if channel == "Все":
            self.current_image = self.original_image.copy()
        else:
            b, g, r = cv2.split(self.original_image)
            zeros = np.zeros_like(b)

            if channel == "Красный":
                self.current_image = cv2.merge([zeros, zeros, r])
            elif channel == "Зеленый":
                self.current_image = cv2.merge([zeros, g, zeros])
            elif channel == "Синий":
                self.current_image = cv2.merge([b, zeros, zeros])

        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        self.show_image()
        self.status_var.set(f"Применен канал: {channel}")

    def reset_image(self):
        """Сбрасывает все изменения к оригиналу."""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.channel_var.set("Все")
            self.show_image()
            self.status_var.set("Изображение сброшено к оригиналу")

    def apply_operation(self):
        """Применяет выбранную операцию обработки."""
        if self.original_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return

        operation = self.operation_var.get()

        if operation == "1. Маска по красному":
            try:
                threshold = int(self.threshold_var.get())
                if not 0 <= threshold <= 255:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Ошибка", "Порог должен быть целым числом от 0 до 255")
                return

            self.apply_red_mask(threshold)
            self.status_var.set(
                f"Применена маска по красному (порог: {threshold})")

        elif operation == "2. Повышение резкости":
            self.sharpen_image()
            self.status_var.set("Применено повышение резкости")

        elif operation == "3. Рисование прямоугольника":
            try:
                coords = [int(var.get()) for var in self.coord_vars]
                x1, y1, x2, y2 = coords

                # Проверка координат
                h, w = self.original_image.shape[:2]
                if (any(
                        coord < 0 for coord in coords)
                        or x1 > w or y1 > h or x2 > w or y2 > h):
                    raise ValueError

            except ValueError:
                messagebox.showerror(
                    "Ошибка", "Некорректные координаты прямоугольника")
                return

            self.draw_rectangle(x1, y1, x2, y2)
            self.status_var.set(
                f"Нарисован прямоугольник: ({x1},{y1})-({x2},{y2})")

        self.show_image()

    def apply_red_mask(self, threshold):
        """Маска по красному"""
        rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(rgb)

        mask = (r > threshold).astype(np.uint8) * 255
        self.current_image = cv2.merge([mask, np.zeros_like(g), np.zeros_like(b)])

    def sharpen_image(self):
        """Повышает резкость изображения с помощью фильтра."""
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        self.current_image = cv2.filter2D(self.original_image, -1, kernel)

    def draw_rectangle(self, x1, y1, x2, y2):
        """Рисует прямоугольник на изображении.

        Args:
            x1, y1: Координаты верхнего левого угла
            x2, y2: Координаты нижнего правого угла
        """
        img = self.original_image.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      (0, 0, 255), 2)
        self.current_image = img


if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()
