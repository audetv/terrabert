from collections import Counter

# Чтение корпуса текста
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Разделение текста на слова
words = text.split()

print(len(words))

# Подсчет уникальных слов
unique_words = set(words)
print(f"Количество уникальных слов: {len(unique_words)}")
vocab_size = len(unique_words)

# Ограничение размера словаря (например, не более 100 000)
vocab_size = min(vocab_size, 1000000)

print(f"Размер словаря: {vocab_size}")
