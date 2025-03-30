import sys
import cv2
import numpy as np
import time
import os
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QProgressBar, QTextEdit, QMessageBox, QListWidget, 
                           QInputDialog, QDialog, QSlider, QSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

class VideoRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание номеров")
        self.setGeometry(100, 100, 1600, 800)
        
        # Инициализация переменных
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.frame_count = 0
        self.total_frames = 0
        self.processing = False
        self.number_plate_detection_and_reading = None
        self.unique_numbers = set()  # Множество для хранения уникальных номеров
        self.recognized_numbers_data = []  # Список для хранения данных о распознанных номерах
        
        # Параметры распознавания (начальные значения - самые лояльные)
        self.confidence_threshold = 0.05  # Минимальный порог уверенности (0.0 - 1.0)
        self.min_plate_width = 30  # Минимальная ширина номера в пикселях
        self.min_plate_height = 10  # Минимальная высота номера в пикселях
        
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # Основной горизонтальный layout
        
        # Создание левой панели (видео и управление)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Создание верхней панели с кнопками
        top_panel = QHBoxLayout()
        
        # Кнопка выбора файла
        self.select_file_btn = QPushButton("Выбрать видео")
        self.select_file_btn.clicked.connect(self.select_video)
        top_panel.addWidget(self.select_file_btn)
        
        # Кнопка выбора камеры
        self.select_camera_btn = QPushButton("Выбрать камеру")
        self.select_camera_btn.clicked.connect(self.select_camera)
        top_panel.addWidget(self.select_camera_btn)
        
        # Кнопка старт/пауза
        self.start_btn = QPushButton("Старт")
        self.start_btn.clicked.connect(self.toggle_processing)
        self.start_btn.setEnabled(False)
        top_panel.addWidget(self.start_btn)
        
        # Кнопка стоп
        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        top_panel.addWidget(self.stop_btn)
        
        # Кнопка настроек
        self.settings_btn = QPushButton("Настройки")
        self.settings_btn.clicked.connect(self.show_settings)
        top_panel.addWidget(self.settings_btn)
        
        # Кнопка отчета
        self.report_btn = QPushButton("Отчет")
        self.report_btn.clicked.connect(self.generate_report)
        top_panel.addWidget(self.report_btn)
        
        left_layout.addLayout(top_panel)
        
        # Создание области отображения видео
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        left_layout.addWidget(self.video_label)
        
        # Создание прогресс-бара
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        # Создание области вывода логов
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        left_layout.addWidget(self.log_text)
        
        # Добавляем левую панель в основной layout
        main_layout.addWidget(left_panel)
        
        # Создание правой панели (список уникальных номеров)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Заголовок для списка номеров
        numbers_label = QLabel("Уникальные номера:")
        numbers_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(numbers_label)
        
        # Список уникальных номеров
        self.numbers_list = QListWidget()
        self.numbers_list.setMinimumWidth(300)
        right_layout.addWidget(self.numbers_list)
        
        # Добавляем правую панель в основной layout
        main_layout.addWidget(right_panel)
        
        # Создание таймера для обновления видео
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Инициализация пайплайна
        self.log("Инициализация системы распознавания...")
        self.number_plate_detection_and_reading = pipeline(
            "number_plate_detection_and_reading",
            image_loader="opencv"
        )
        self.log("Система распознавания инициализирована")

    def log(self, message):
        """Добавление сообщения в лог"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def select_camera(self):
        """Выбор камеры для захвата видеопотока"""
        # Получаем список доступных камер
        available_cameras = []
        for i in range(10):  # Проверяем первые 10 индексов
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        
        if not available_cameras:
            QMessageBox.warning(self, "Предупреждение", "Камеры не найдены")
            return
            
        # Если найдена только одна камера, используем её
        if len(available_cameras) == 1:
            camera_index = available_cameras[0]
        else:
            # Если найдено несколько камер, показываем диалог выбора
            camera_index, ok = QInputDialog.getInt(
                self, 
                "Выбор камеры", 
                "Введите номер камеры (0-9):", 
                value=0, 
                min=0, 
                max=9, 
                step=1
            )
            if not ok or camera_index not in available_cameras:
                return
        
        # Открываем выбранную камеру
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Ошибка", "Не удалось открыть камеру")
            return
            
        # Получаем параметры камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Устанавливаем разрешение
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Активируем кнопки
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        self.log(f"Выбрана камера {camera_index}")
        
        # Показываем первый кадр
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

    def show_settings(self):
        """Отображение окна настроек"""
        # Создаем диалоговое окно
        dialog = QDialog(self)
        dialog.setWindowTitle("Настройки распознавания")
        dialog.setModal(True)
        
        # Создаем layout
        layout = QVBoxLayout(dialog)
        
        # Слайдер для порога уверенности
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Порог уверенности:")
        confidence_slider = QSlider(Qt.Horizontal)
        confidence_slider.setMinimum(0)
        confidence_slider.setMaximum(100)
        confidence_slider.setValue(int(self.confidence_threshold * 100))
        confidence_value = QLabel(f"{self.confidence_threshold:.2f}")
        confidence_slider.valueChanged.connect(
            lambda v: confidence_value.setText(f"{v/100:.2f}")
        )
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(confidence_slider)
        confidence_layout.addWidget(confidence_value)
        layout.addLayout(confidence_layout)
        
        # Поля для минимальных размеров
        size_layout = QHBoxLayout()
        
        # Минимальная ширина
        width_layout = QVBoxLayout()
        width_label = QLabel("Мин. ширина (пикс):")
        width_spin = QSpinBox()
        width_spin.setRange(30, 500)
        width_spin.setValue(self.min_plate_width)
        width_layout.addWidget(width_label)
        width_layout.addWidget(width_spin)
        
        # Минимальная высота
        height_layout = QVBoxLayout()
        height_label = QLabel("Мин. высота (пикс):")
        height_spin = QSpinBox()
        height_spin.setRange(10, 200)
        height_spin.setValue(self.min_plate_height)
        height_layout.addWidget(height_label)
        height_layout.addWidget(height_spin)
        
        size_layout.addLayout(width_layout)
        size_layout.addLayout(height_layout)
        layout.addLayout(size_layout)
        
        # Кнопки
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Отмена")
        
        def on_ok():
            self.confidence_threshold = confidence_slider.value() / 100
            self.min_plate_width = width_spin.value()
            self.min_plate_height = height_spin.value()
            dialog.accept()
            
        ok_button.clicked.connect(on_ok)
        cancel_button.clicked.connect(dialog.reject)
        
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)
        
        dialog.exec_()

    def generate_report(self):
        """Генерация отчета"""
        if not self.recognized_numbers_data:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для отчета")
            return
            
        # Создаем директорию для отчета, если её нет
        report_dir = "reports"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
        # Генерируем имя файла отчета с текущей датой и временем
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"report_{timestamp}.txt")
        
        # Создаем отчет
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Отчет по распознаванию номеров\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Дата и время создания: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего уникальных номеров: {len(self.unique_numbers)}\n\n")
            
            f.write("Список распознанных номеров:\n")
            f.write("-" * 50 + "\n")
            
            # Сортируем данные по времени
            sorted_data = sorted(self.recognized_numbers_data, key=lambda x: x['timestamp'])
            
            for data in sorted_data:
                f.write(f"Номер: {data['number']}\n")
                f.write(f"Время: {data['timestamp']}\n")
                f.write(f"Уверенность: {data['confidence']:.2f}\n")
                f.write("-" * 50 + "\n")
        
        # Сохраняем скриншот текущего кадра
        if self.current_frame is not None:
            screenshot_path = os.path.join(report_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, self.current_frame)
        
        self.log(f"Отчет сохранен в файл: {report_path}")
        QMessageBox.information(self, "Успех", f"Отчет успешно создан:\n{report_path}")

    def select_video(self):
        """Выбор видеофайла"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл",
            "",
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
        )
        
        if file_name:
            self.video_path = file_name
            self.log(f"Выбран файл: {file_name}")
            
            # Открываем видео
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Ошибка", "Не удалось открыть видеофайл")
                return
                
            # Получаем общее количество кадров
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar.setMaximum(self.total_frames)
            
            # Активируем кнопки
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            
            # Показываем первый кадр
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)

    def toggle_processing(self):
        """Переключение режима обработки"""
        if not self.processing:
            self.processing = True
            self.start_btn.setText("Пауза")
            self.timer.start(30)  # 30ms = ~33 FPS
            self.log("Начата обработка видео")
        else:
            self.processing = False
            self.start_btn.setText("Старт")
            self.timer.stop()
            self.log("Обработка приостановлена")

    def stop_processing(self):
        """Остановка обработки"""
        self.processing = False
        self.timer.stop()
        self.start_btn.setText("Старт")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.frame_count = 0
        self.progress_bar.setValue(0)
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.log("Обработка остановлена")

    def add_unique_number(self, number):
        """Добавление уникального номера в список"""
        if number not in self.unique_numbers:
            self.unique_numbers.add(number)
            self.numbers_list.addItem(number)

    def update_frame(self):
        """Обновление кадра"""
        if not self.processing or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_processing()
            self.log("Обработка завершена")
            return
            
        self.current_frame = frame
        self.frame_count += 1
        self.progress_bar.setValue(self.frame_count)
        
        # Обработка кадра
        try:
            # Сохраняем кадр во временный файл
            temp_path = f"temp_frame_{self.frame_count}.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Обрабатываем кадр
            result = self.number_plate_detection_and_reading([temp_path])
            (images, images_bboxs, 
             images_points, images_zones, region_ids, 
             region_names, count_lines, 
             confidences, texts) = unzip(result)
            
            # Удаляем временный файл
            os.remove(temp_path)
            
            # Обработка результатов
            for i, (text_list, conf_list) in enumerate(zip(texts[0], confidences[0])):
                if text_list:
                    text = ''.join(text_list)
                    conf = np.mean(conf_list) if conf_list else 0.0
                    
                    # Проверяем размер номера
                    if len(images_bboxs[0]) > i:
                        bbox = images_bboxs[0][i]
                        width = int(bbox[2] - bbox[0])
                        height = int(bbox[3] - bbox[1])
                        
                        # Пропускаем номера меньше минимального размера
                        if width < self.min_plate_width or height < self.min_plate_height:
                            continue
                    
                    # Проверяем порог уверенности
                    if conf < self.confidence_threshold:
                        continue
                    
                    if is_valid_russian_plate(text):
                        # Форматируем номер в кириллице для лога
                        formatted_text_cyrillic = format_plate_number(text)
                        # Форматируем номер в латинице для отображения на видео
                        formatted_text_latin = format_plate_number_latin(text)
                        
                        self.log(f"Найден номер: {formatted_text_cyrillic} (уверенность: {conf:.2f})")
                        self.add_unique_number(formatted_text_cyrillic)
                        
                        # Сохраняем данные о распознанном номере
                        self.recognized_numbers_data.append({
                            'number': formatted_text_cyrillic,
                            'confidence': conf,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Рисуем рамку вокруг номера
                        if len(images_bboxs[0]) > i:
                            bbox = images_bboxs[0][i]
                            cv2.rectangle(frame, 
                                        (int(bbox[0]), int(bbox[1])), 
                                        (int(bbox[2]), int(bbox[3])), 
                                        (0, 255, 0), 2)
                            cv2.putText(frame, formatted_text_latin, 
                                      (int(bbox[0]), int(bbox[1]-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.9, (0, 255, 0), 2)
        
        except Exception as e:
            self.log(f"Ошибка при обработке кадра: {str(e)}")
            # Удаляем временный файл в случае ошибки
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Конвертация кадра для отображения
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Масштабирование изображения под размер label
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        self.stop_processing()
        if self.cap:
            self.cap.release()
        event.accept()

def is_valid_russian_plate(text):
    """Проверяет соответствие номера формату российских номеров"""
    # Разрешенные буквы (кириллица и латиница)
    allowed_letters_cyrillic = 'АВЕКМНОРСТУХ'
    allowed_letters_latin = 'ABEKMHOPCTYX'
    
    # Словарь соответствия латинских букв кириллическим
    latin_to_cyrillic = {
        'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
        'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
        'U': 'У', 'X': 'Х', 'Y': 'У'
    }
    
    # Удаляем пробелы и приводим к верхнему регистру
    text = text.replace(' ', '').upper()
    
    # Проверяем длину (8 или 9 символов)
    if len(text) not in [8, 9]:
        return False
    
    # Заменяем латинские буквы на кириллические
    text_cyrillic = ''
    for char in text:
        if char in latin_to_cyrillic:
            text_cyrillic += latin_to_cyrillic[char]
        else:
            text_cyrillic += char
    
    # Проверяем формат: буква-цифры-буквы-цифры
    pattern = f'^[{allowed_letters_cyrillic}][0-9]{{3}}[{allowed_letters_cyrillic}]{{2}}[0-9]{{2,3}}$'
    if not re.match(pattern, text_cyrillic):
        return False
    
    # Проверяем код региона (не должен быть 00)
    region = text_cyrillic[-2:] if len(text_cyrillic) == 8 else text_cyrillic[-3:]
    if region == '00':
        return False
    
    return True

def format_plate_number(text):
    """Форматирует номер в стандартный вид"""
    # Словарь соответствия латинских букв кириллическим
    latin_to_cyrillic = {
        'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
        'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
        'U': 'У', 'X': 'Х', 'Y': 'У'
    }
    
    # Удаляем пробелы и приводим к верхнему регистру
    text = text.replace(' ', '').upper()
    
    # Заменяем латинские буквы на кириллические
    text_cyrillic = ''
    for char in text:
        if char in latin_to_cyrillic:
            text_cyrillic += latin_to_cyrillic[char]
        else:
            text_cyrillic += char
    
    # Добавляем пробелы в нужных местах
    if len(text_cyrillic) == 8:
        return f"{text_cyrillic[0]} {text_cyrillic[1:4]} {text_cyrillic[4:6]} {text_cyrillic[6:]}"
    else:  # 9 символов
        return f"{text_cyrillic[0]} {text_cyrillic[1:4]} {text_cyrillic[4:6]} {text_cyrillic[6:]}"

def format_plate_number_latin(text):
    """Форматирует номер в латинские символы"""
    # Словарь соответствия кириллических букв латинским
    cyrillic_to_latin = {
        'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M',
        'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T',
        'У': 'Y', 'Х': 'X'
    }
    
    # Удаляем пробелы и приводим к верхнему регистру
    text = text.replace(' ', '').upper()
    
    # Заменяем кириллические буквы на латинские
    text_latin = ''
    for char in text:
        if char in cyrillic_to_latin:
            text_latin += cyrillic_to_latin[char]
        else:
            text_latin += char
    
    # Добавляем пробелы в нужных местах
    if len(text_latin) == 8:
        return f"{text_latin[0]} {text_latin[1:4]} {text_latin[4:6]} {text_latin[6:]}"
    else:  # 9 символов
        return f"{text_latin[0]} {text_latin[1:4]} {text_latin[4:6]} {text_latin[6:]}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Показываем информационное сообщение при запуске
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setWindowTitle("Информация")
    msg.setText("Добро пожаловать!")
    msg.setInformativeText(
        "При работе с приложением могут возникать небольшие задержки при анализе видео.\n\n"
        "Рекомендации для оптимальной работы:\n"
        "• Используйте видео невысокого разрешения\n"
        "• Закройте другие ресурсоемкие приложения\n"
        "• При зависании используйте кнопку 'Стоп'\n\n"
        "Приятной работы!"
    )
    msg.setStandardButtons(QMessageBox.Ok)
    msg.setDefaultButton(QMessageBox.Ok)
    
    if msg.exec_() == QMessageBox.Ok:
        window = VideoRecognitionApp()
        window.show()
        sys.exit(app.exec_())
