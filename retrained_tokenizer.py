from transformers import AutoTokenizer

# 1. Загрузка предобученного токенизатора
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# 2. Подготовка корпуса текстов
def read_corpus(file_path):
    """Генератор, который читает тексты из файла."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()

# Путь к корпусу текстов
corpus_file = "corpus.txt"

# 3. Дообучение токенизатора
new_tokenizer = tokenizer.train_new_from_iterator(
    read_corpus(corpus_file),  # Итератор по текстам
    vocab_size=300000,  # Размер нового словаря
    min_frequency=1,  # Минимальная частота для добавления токена
    show_progress=True  # Отображение прогресса
)

# 4. Сохранение дообученного токенизатора
new_tokenizer.save_pretrained("my_retrained_tokenizer")
print("Дообученный токенизатор сохранен в папку: my_retrained_tokenizer")