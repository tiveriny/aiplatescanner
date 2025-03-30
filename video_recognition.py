import cv2
import numpy as np
import time
import os
import re
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

def is_valid_russian_plate(text):
    """Проверяет соответствие номера формату российских номеров"""
    # Разрешенные буквы (кириллица и латиница)
    allowed_letters_cyrillic = 'АВЕКМНОРСТУХ'
    allowed_letters_latin = 'ABEKMHOPCTYX'
    
    # Словарь соответствия латинских букв кириллическим
    latin_to_cyrillic = {
        'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
        'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
        'U': 'У', 'X': 'Х', 'Y': 'У'  # Добавлена буква Y
    }
    
    # Удаляем пробелы и приводим к верхнему регистру
    text = text.replace(' ', '').upper()
    print(f"Проверка номера: {text}")
    
    # Проверяем длину (8 или 9 символов)
    if len(text) not in [8, 9]:
        print(f"Неверная длина: {len(text)}")
        return False
    
    # Заменяем латинские буквы на кириллические
    text_cyrillic = ''
    for char in text:
        if char in latin_to_cyrillic:
            text_cyrillic += latin_to_cyrillic[char]
        else:
            text_cyrillic += char
    
    print(f"Преобразованный номер: {text_cyrillic}")
    
    # Проверяем формат: буква-цифры-буквы-цифры
    pattern = f'^[{allowed_letters_cyrillic}][0-9]{{3}}[{allowed_letters_cyrillic}]{{2}}[0-9]{{2,3}}$'
    if not re.match(pattern, text_cyrillic):
        print(f"Не соответствует паттерну: {pattern}")
        return False
    
    # Проверяем код региона (не должен быть 00)
    region = text_cyrillic[-2:] if len(text_cyrillic) == 8 else text_cyrillic[-3:]
    if region == '00':
        print("Недопустимый код региона: 00")
        return False
    
    print("Номер валидный")
    return True

def format_plate_number(text):
    """Форматирует номер в стандартный вид"""
    # Словарь соответствия латинских букв кириллическим
    latin_to_cyrillic = {
        'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
        'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
        'U': 'У', 'X': 'Х', 'Y': 'У'  # Добавлена буква Y
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

def extract_frames(video_path, output_dir):
    """Извлекает кадры из видео и сохраняет их в директорию"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео файл {video_path}")
        return 0
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Извлечено кадров: {frame_count}")
    
    cap.release()
    return frame_count

def process_frame(frame_path, number_plate_detection_and_reading):
    """Обрабатывает один кадр и возвращает результаты"""
    try:
        result = number_plate_detection_and_reading([frame_path])
        (images, images_bboxs, 
         images_points, images_zones, region_ids, 
         region_names, count_lines, 
         confidences, texts) = unzip(result)
        
        return {
            'texts': texts[0],
            'confidences': confidences[0],
            'bboxs': images_bboxs[0],
            'success': True
        }
    except Exception as e:
        print(f"Ошибка при обработке кадра {frame_path}: {str(e)}")
        return {'success': False, 'error': str(e)}

def process_video(video_path):
    # Проверяем существование файла
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл {video_path} не найден!")
        return
    
    # Создаем директории для кадров и результатов
    frames_dir = "frames"
    results_dir = "results"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Инициализация пайплайна
    print("Инициализация системы распознавания...")
    number_plate_detection_and_reading = pipeline(
        "number_plate_detection_and_reading",
        image_loader="opencv"
    )
    
    # Извлекаем кадры из видео
    print("Извлечение кадров из видео...")
    total_frames = extract_frames(video_path, frames_dir)
    print(f"Всего извлечено кадров: {total_frames}")
    
    # Обрабатываем каждый кадр
    processed_count = 0
    start_time = time.time()
    
    print("\nНачинаем обработку кадров...")
    for frame_num in range(total_frames):
        frame_path = os.path.join(frames_dir, f"frame_{frame_num:06d}.jpg")
        if not os.path.exists(frame_path):
            continue
            
        print(f"\nОбработка кадра {frame_num + 1}/{total_frames}")
        
        # Обрабатываем кадр
        result = process_frame(frame_path, number_plate_detection_and_reading)
        
        if result['success']:
            # Загружаем кадр для визуализации
            frame = cv2.imread(frame_path)
            
            # Выводим результаты
            print(f"Найдено номеров: {len(result['texts'])}")
            valid_plates = 0
            for i, (text_list, conf_list) in enumerate(zip(result['texts'], result['confidences'])):
                if text_list:  # Если номер распознан
                    # Преобразуем список символов в строку
                    text = ''.join(text_list)
                    # Вычисляем среднюю уверенность
                    conf = np.mean(conf_list) if conf_list else 0.0
                    
                    print(f"\nПроверка номера {i+1}:")
                    print(f"Исходный текст: {text}")
                    print(f"Уверенность: {conf:.2f}")
                    
                    # Проверяем формат номера
                    if is_valid_russian_plate(text):
                        valid_plates += 1
                        formatted_text = format_plate_number(text)
                        print(f"Номер {i+1}: {formatted_text} (уверенность: {conf:.2f}) [Валидный]")
                        
                        # Рисуем рамку вокруг номера
                        if len(result['bboxs']) > i:
                            bbox = result['bboxs'][i]
                            cv2.rectangle(frame, 
                                        (int(bbox[0]), int(bbox[1])), 
                                        (int(bbox[2]), int(bbox[3])), 
                                        (0, 255, 0), 2)
                            cv2.putText(frame, formatted_text, 
                                      (int(bbox[0]), int(bbox[1]-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.9, (0, 255, 0), 2)
                    else:
                        print(f"Номер {i+1}: {text} (уверенность: {conf:.2f}) [Не соответствует формату]")
            
            if valid_plates > 0:
                # Сохраняем обработанный кадр только если найдены валидные номера
                output_path = os.path.join(results_dir, f"processed_{frame_num:06d}.jpg")
                cv2.imwrite(output_path, frame)
                print(f"Сохранен обработанный кадр: {output_path}")
                processed_count += 1
        
        # Выводим информацию о прогрессе
        elapsed_time = time.time() - start_time
        current_fps = (frame_num + 1) / elapsed_time
        progress = ((frame_num + 1) / total_frames) * 100
        print(f"Прогресс: {progress:.1f}% | Обработано кадров: {frame_num + 1}/{total_frames} | FPS: {current_fps:.2f}")
    
    total_time = time.time() - start_time
    print(f"\nОбработка завершена:")
    print(f"Всего кадров: {total_frames}")
    print(f"Успешно обработано кадров: {processed_count}")
    print(f"Общее время: {total_time:.2f} секунд")
    print(f"Средний FPS: {total_frames/total_time:.2f}")
    print(f"\nОбработанные кадры сохранены в директории: {results_dir}")
    
    # Спрашиваем пользователя, хочет ли он удалить временные файлы
    response = input("\nХотите удалить временные файлы кадров? (y/n): ")
    if response.lower() == 'y':
        import shutil
        shutil.rmtree(frames_dir)
        print("Временные файлы удалены")

if __name__ == "__main__":
    # Пример использования
    video_path = "test.mp4"  # Укажите путь к вашему видео
    process_video(video_path) 