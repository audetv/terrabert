import os

# Путь к папке с текстовыми файлами
input_folder = "books_utf8"
# input_folder = "latin"
# Путь к выходному файлу
output_file = "corpus.txt"

# Открываем выходной файл для записи
with open(output_file, "w", encoding="utf-8") as outfile:
    # Проходим по всем файлам в папке
    for filename in os.listdir(input_folder):
        # Полный путь к файлу
        file_path = os.path.join(input_folder, filename)
        # Проверяем, что это файл (а не папка)
        if os.path.isfile(file_path):
            # Открываем файл и читаем его содержимое
            with open(file_path, "r", encoding="utf-8") as infile:
                # Записываем содержимое в выходной файл
                outfile.write(infile.read())
                # Добавляем перенос строки между файлами (опционально)
                outfile.write("\n")