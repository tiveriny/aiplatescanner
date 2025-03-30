import PyInstaller.__main__
import os

# Получаем текущую директорию
current_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к основному файлу приложения
main_script = os.path.join(current_dir, 'video_recognition_gui.py')

# Создаем директорию для сборки, если её нет
dist_dir = os.path.join(current_dir, 'dist')
if not os.path.exists(dist_dir):
    os.makedirs(dist_dir)

# Параметры для сборки
PyInstaller.__main__.run([
    main_script,
    '--name=NumberRecognition',
    '--onefile',
    '--windowed',
    '--icon=icon.ico',  # Если у вас есть иконка
    '--add-data=README.md;.',  # Добавляем README в сборку
    '--clean',
    '--noconfirm',
    f'--distpath={dist_dir}',
    '--hidden-import=nomeroff_net',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    '--hidden-import=PyQt5',
]) 