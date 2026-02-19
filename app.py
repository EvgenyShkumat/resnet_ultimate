import sys
import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QMessageBox, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt

# ==========================================
# 1. ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ МОДЕЛИ
# (Должно точно совпадать с тем, что было при обучении)
# ==========================================
class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetEmotion, self).__init__()
        # Веса здесь можно не грузить (weights=None), так как мы будем грузить свои
        self.net = models.resnet34(weights=None)
        
        # Переделываем первый слой под 1 канал (Grayscale)
        original_weights = self.net.conv1.weight.data.mean(dim=1, keepdim=True)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.conv1.weight.data = original_weights 
        
        # Переделываем выходной слой
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)

# Классы эмоций (FER-2013 стандарт)
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ==========================================
# 2. ОСНОВНОЕ ПРИЛОЖЕНИЕ PYQT5
# ==========================================
class EmotionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система распознавания эмоций (Участник: Ivanov)")
        self.setGeometry(100, 100, 1000, 700)
        
        # Переменные состояния
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image = None # Здесь хранится загруженное изображение (PIL или CV2)
        self.cap = None # Для веб-камеры
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Интерфейс
        self.init_ui()
        
        # Трансформации (такие же, как при валидации/тесте)
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.log("Приложение запущено. Пожалуйста, выберите модель нейронной сети.")

    def init_ui(self):
        """Создание элементов интерфейса согласно ТЗ"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        # --- БЛОК 1: Выбор модели (ТЗ: элемент окна... выбрать модель) ---
        model_group = QGroupBox("Конфигурация модели")
        model_layout = QHBoxLayout()
        
        self.lbl_model_path = QLabel("Модель не выбрана")
        self.lbl_model_path.setStyleSheet("color: red;")
        btn_load_model = QPushButton("Выбрать файл модели (.pth)")
        btn_load_model.clicked.connect(self.load_model_dialog)
        
        model_layout.addWidget(btn_load_model)
        model_layout.addWidget(self.lbl_model_path)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # --- БЛОК 2: Отображение изображения / Камеры ---
        img_group = QGroupBox("Просмотр")
        img_layout = QVBoxLayout()
        self.image_label = QLabel("Изображение не загружено")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")
        self.image_label.setMinimumHeight(400)
        img_layout.addWidget(self.image_label)
        img_group.setLayout(img_layout)
        main_layout.addWidget(img_group)

        # --- БЛОК 3: Управление (ТЗ: сделать фото, выбрать изображение, запуск) ---
        controls_layout = QHBoxLayout()
        
        self.btn_open_file = QPushButton("Выбрать изображение (Файл)")
        self.btn_open_file.clicked.connect(self.open_image_file)
        
        self.btn_webcam_start = QPushButton("Включить веб-камеру")
        self.btn_webcam_start.clicked.connect(self.toggle_webcam)
        
        self.btn_webcam_capture = QPushButton("Сделать фото")
        self.btn_webcam_capture.setEnabled(False)
        self.btn_webcam_capture.clicked.connect(self.capture_photo)
        
        self.btn_predict = QPushButton("ЗАПУСК РАСПОЗНАВАНИЯ") # ТЗ: кнопка «запуск распознавания»
        self.btn_predict.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.btn_predict.clicked.connect(self.run_recognition)

        controls_layout.addWidget(self.btn_open_file)
        controls_layout.addWidget(self.btn_webcam_start)
        controls_layout.addWidget(self.btn_webcam_capture)
        controls_layout.addWidget(self.btn_predict)
        main_layout.addLayout(controls_layout)

        # --- БЛОК 4: Информация (ТЗ: дополнительная информация) ---
        info_group = QGroupBox("Журнал и Результаты")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        central_widget.setLayout(main_layout)

    # ==========================================
    # ЛОГИКА РАБОТЫ
    # ==========================================

    def log(self, message):
        """Вывод сообщений в текстовое поле"""
        self.info_text.append(f"> {message}")

    def load_model_dialog(self):
        """Загрузка весов модели"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл модели", "", "PyTorch Model (*.pth);;All Files (*)", options=options)
        
        if file_path:
            try:
                self.log(f"Загрузка модели из: {file_path}...")
                
                # Инициализация архитектуры
                self.model = ResNetEmotion(num_classes=len(EMOTION_CLASSES))
                
                # Загрузка весов (map_location='cpu' на случай, если учили на GPU, а запускаем на CPU)
                state_dict = torch.load(file_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                self.model.to(self.device)
                self.model.eval() # Режим предсказания
                
                self.lbl_model_path.setText(os.path.basename(file_path))
                self.lbl_model_path.setStyleSheet("color: green; font-weight: bold;")
                self.log("Модель успешно загружена и готова к работе.")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{str(e)}")
                self.log(f"Ошибка загрузки модели: {e}")

    def open_image_file(self):
        """Выбор изображения с диска"""
        self.stop_webcam() # Если камера работает, выключаем
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                QMessageBox.warning(self, "Ошибка", "Не удалось прочитать изображение.")
                return
            
            # Конвертация BGR -> RGB для отображения в Qt
            self.display_image(self.current_image)
            self.log(f"Загружено изображение: {os.path.basename(file_path)}")

    def toggle_webcam(self):
        """Включение/выключение стрима с камеры"""
        if self.timer.isActive():
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Ошибка", "Не удалось подключиться к веб-камере.")
            return
        
        self.timer.start(30) # 30 мс обновление (около 30 FPS)
        self.btn_webcam_start.setText("Остановить камеру")
        self.btn_webcam_capture.setEnabled(True)
        self.log("Веб-камера включена.")

    def stop_webcam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.btn_webcam_start.setText("Включить веб-камеру")
        self.btn_webcam_capture.setEnabled(False)

    def update_frame(self):
        """Чтение кадра с камеры и показ в интерфейсе"""
        ret, frame = self.cap.read()
        if ret:
            self.current_image = frame
            self.display_image(frame)

    def capture_photo(self):
        """Заморозка текущего кадра (сделать фото)"""
        if self.current_image is not None:
            self.stop_webcam()
            self.log("Фотография сделана.")
            # current_image уже содержит последний кадр

    def display_image(self, img_bgr):
        """Отображение OpenCV (BGR) картинки в QLabel"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Масштабирование под размер лейбла
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def run_recognition(self):
        """Запуск инференса (обработки)"""
        # Проверки
        if self.model is None:
            QMessageBox.warning(self, "Внимание", "Сначала выберите обученную модель (.pth)!")
            return
        
        if self.current_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение или сделайте фото!")
            return

        self.log("Начинаю обработку...")
        
        try:
            # 1. Подготовка изображения (Preprocessing)
            # OpenCV (BGR) -> PIL (RGB) -> Grayscale (внутри transforms)
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Применяем те же трансформации, что и при обучении
            input_tensor = self.transform(pil_img).unsqueeze(0) # Добавляем Batch dimension [1, 1, 224, 224]
            input_tensor = input_tensor.to(self.device)
            
            # 2. Прогон через модель
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)
                
            predicted_class = EMOTION_CLASSES[predicted_idx.item()]
            conf_val = confidence.item() * 100
            
            # 3. Вывод результата
            result_msg = f"РЕЗУЛЬТАТ: {predicted_class} ({conf_val:.2f}%)"
            self.log(result_msg)
            
            # Можно вывести прямо на картинку
            QMessageBox.information(self, "Результат распознавания", result_msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при распознавании:\n{e}")
            self.log(f"Error: {e}")

# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())