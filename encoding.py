import os

# Путь к папке с файлами
input_folder = "books"
# Путь к папке для сохранения преобразованных файлов (опционально)
output_folder = "books_utf8"

# Создаем папку для сохранения преобразованных файлов, если её нет
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Проходим по всем файлам в папке
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    
    # Проверяем, что это файл (а не папка)
    if os.path.isfile(file_path):
        # Открываем файл в кодировке windows-1251 и читаем его содержимое
        try:
            with open(file_path, "r", encoding="windows-1251") as infile:
                content = infile.read()
        except UnicodeDecodeError:
            print(f"Файл {filename} не в кодировке windows-1251. Пропускаем.")
            continue
        
        # Сохраняем преобразованный файл в кодировке utf-8
        output_file_path = os.path.join(output_folder, filename)
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            outfile.write(content)
        
        print(f"Файл {filename} успешно преобразован и сохранен в {output_folder}.")