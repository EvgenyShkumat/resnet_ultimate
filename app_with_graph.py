import sys
import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QMessageBox, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt

# ==========================================
# 1. МОДЕЛЬ
# ==========================================
class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetEmotion, self).__init__()
        self.net = models.resnet34(weights=None)
        original_weights = self.net.conv1.weight.data.mean(dim=1, keepdim=True)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.conv1.weight.data = original_weights 
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ==========================================
# 2. КЛАСС ДЛЯ ГРАФИКА (MATPLOTLIB WIDGET)
# ==========================================
class ProbabilityCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(ProbabilityCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.ax.set_title("Вероятности классов")
        self.bars = None

    def plot(self, probs):
        self.ax.clear()
        # Цвета для красоты
        colors = ['red', 'green', 'purple', 'orange', 'gray', 'blue', 'cyan']
        bars = self.ax.bar(EMOTION_CLASSES, probs, color=colors)
        
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Уверенность (%)')
        self.ax.set_title("Распределение вероятностей")
        
        # Добавляем цифры над столбиками
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
            
        self.draw()

# ==========================================
# 3. ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ==========================================
class EmotionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система распознавания эмоций (Участник: Ivanov)")
        self.setGeometry(100, 100, 1200, 700) # Чуть шире для графика
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image = None 
        self.cap = None 
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()
        
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.log("Приложение готово. Загрузите модель.")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный горизонтальный слой: Слева управление/камера, Справа график/лог
        main_h_layout = QHBoxLayout()
        
        # --- ЛЕВАЯ КОЛОНКА ---
        left_layout = QVBoxLayout()
        
        # 1. Выбор модели
        model_group = QGroupBox("1. Конфигурация")
        model_layout = QVBoxLayout()
        self.lbl_model_path = QLabel("Модель не загружена")
        self.lbl_model_path.setStyleSheet("color: red;")
        btn_load = QPushButton("Загрузить веса (.pth)")
        btn_load.clicked.connect(self.load_model_dialog)
        model_layout.addWidget(btn_load)
        model_layout.addWidget(self.lbl_model_path)
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # 2. Камера/Изображение
        img_group = QGroupBox("2. Изображение")
        img_l = QVBoxLayout()
        self.image_label = QLabel("Нет изображения")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; background: #eee;")
        self.image_label.setMinimumSize(400, 300)
        img_l.addWidget(self.image_label)
        img_group.setLayout(img_l)
        left_layout.addWidget(img_group)
        
        # 3. Кнопки управления
        ctrl_layout = QHBoxLayout()
        btn_file = QPushButton("Файл")
        btn_file.clicked.connect(self.open_image_file)
        btn_cam = QPushButton("Веб-камера")
        btn_cam.clicked.connect(self.toggle_webcam)
        self.btn_snap = QPushButton("Снять фото")
        self.btn_snap.setEnabled(False)
        self.btn_snap.clicked.connect(self.capture_photo)
        
        ctrl_layout.addWidget(btn_file)
        ctrl_layout.addWidget(btn_cam)
        ctrl_layout.addWidget(self.btn_snap)
        left_layout.addLayout(ctrl_layout)
        
        # 4. Кнопка пуск
        self.btn_run = QPushButton("ЗАПУСК РАСПОЗНАВАНИЯ")
        self.btn_run.setStyleSheet("background: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.run_recognition)
        left_layout.addWidget(self.btn_run)

        main_h_layout.addLayout(left_layout, stretch=1)

        # --- ПРАВАЯ КОЛОНКА ---
        right_layout = QVBoxLayout()
        
        # 5. График
        graph_group = QGroupBox("3. Анализ вероятностей")
        graph_l = QVBoxLayout()
        self.canvas = ProbabilityCanvas(self, width=5, height=4)
        graph_l.addWidget(self.canvas)
        graph_group.setLayout(graph_l)
        right_layout.addWidget(graph_group, stretch=2)
        
        # 6. Лог
        log_group = QGroupBox("Журнал событий")
        log_l = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        log_l.addWidget(self.info_text)
        log_group.setLayout(log_l)
        right_layout.addWidget(log_group, stretch=1)
        
        main_h_layout.addLayout(right_layout, stretch=1)
        
        central_widget.setLayout(main_h_layout)

    # --- ЛОГИКА ---
    def log(self, msg):
        self.info_text.append(f"> {msg}")

    def load_model_dialog(self):
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать модель", "", "Model (*.pth);;All (*)", options=options)
        if path:
            try:
                self.model = ResNetEmotion(len(EMOTION_CLASSES))
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.lbl_model_path.setText(os.path.basename(path))
                self.lbl_model_path.setStyleSheet("color: green; font-weight: bold")
                self.log("Модель загружена!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))

    def open_image_file(self):
        self.stop_webcam()
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать фото", "", "Image (*.png *.jpg *.jpeg);;All (*)")
        if path:
            self.current_image = cv2.imread(path)
            self.display_image(self.current_image)
            self.log("Фото загружено из файла")

    def toggle_webcam(self):
        if self.timer.isActive(): self.stop_webcam()
        else: self.start_webcam()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.btn_snap.setEnabled(True)
        self.log("Камера включена")

    def stop_webcam(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        self.btn_snap.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_image = frame
            self.display_image(frame)

    def capture_photo(self):
        if self.current_image is not None:
            self.stop_webcam()
            self.log("Снимок сделан")

    def display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def run_recognition(self):
        if not self.model or self.current_image is None:
            QMessageBox.warning(self, "Ошибка", "Нет модели или фото")
            return
        
        try:
            # Препроцессинг
            rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Инференс
            with torch.no_grad():
                out = self.model(tensor)
                probs = torch.softmax(out, 1)[0] * 100 # переводим в проценты
                
            # Результаты
            conf, idx = torch.max(probs, 0)
            res_cls = EMOTION_CLASSES[idx.item()]
            
            self.log(f"Результат: {res_cls} ({conf.item():.1f}%)")
            
            # !! ОБНОВЛЕНИЕ ГРАФИКА !!
            probs_np = probs.cpu().numpy()
            self.canvas.plot(probs_np)
            
        except Exception as e:
            self.log(f"Ошибка: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())