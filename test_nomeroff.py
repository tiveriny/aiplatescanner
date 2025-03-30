import os
import cv2
import sys
from pathlib import Path
import time

# Добавляем путь к текущей директории в PYTHONPATH
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

try:
    print("Импортируем необходимые библиотеки...")
    from nomeroff_net import pipeline
    from nomeroff_net.tools import unzip
except Exception as e:
    print(f"Ошибка при импорте nomeroff_net: {e}")
    print("Пожалуйста, убедитесь, что все зависимости установлены:")
    print("pip install -r requirements.txt")
    sys.exit(1)

print("Начинаем инициализацию...")

try:
    # Инициализация пайплайна с указанием локального пути к моделям
    print("Создаем пайплайн...")
    models_dir = current_dir / "models"
    print(f"Путь к моделям: {models_dir}")
    
    start_time = time.time()
    print("Загружаем модели (это может занять несколько минут)...")
    # Используем оптимизированный режим для быстрой работы
    number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading_runtime", 
                                                image_loader="opencv",
                                                models_dir=str(models_dir),
                                                region="ru")  # Указываем только русский регион
    print(f"Модели загружены за {time.time() - start_time:.2f} секунд")

    # Полный путь к тестовому изображению
    test_image = r"C:\Users\tim\Downloads\nomeroff-net-master\nomeroff-net-master\2ex.png"
    print(f"Путь к изображению: {test_image}")
    print(f"Файл существует: {os.path.exists(test_image)}")

    # Проверяем, можем ли мы открыть изображение
    img = cv2.imread(test_image)
    if img is None:
        print("Ошибка: Не удалось открыть изображение!")
        sys.exit(1)
    print("Изображение успешно открыто")

    print("Начинаем распознавание...")
    start_time = time.time()
    # Распознавание номеров
    (images, images_bboxs, images_points, images_zones, region_ids, 
     region_names, count_lines, confidences, texts) = unzip(number_plate_detection_and_reading([test_image]))
    print(f"Распознавание завершено за {time.time() - start_time:.2f} секунд")

    # Вывод результатов
    print("\nРезультаты распознавания:")
    print(f"\nИзображение: {os.path.basename(test_image)}")
    print(f"Распознанные номера: {texts[0]}")
    print(f"Уверенность: {confidences[0]}")

except Exception as e:
    print(f"Произошла ошибка: {e}")
    print("\nВозможные решения:")
    print("1. Убедитесь, что датасет распакован в директорию models")
    print("2. Проверьте, что все зависимости установлены")
    print("3. Проверьте наличие и права доступа к файлу изображения")
    sys.exit(1) 