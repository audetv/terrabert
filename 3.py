from tokenizers import Tokenizer

# Загрузка токенизатора
tokenizer = Tokenizer.from_file("my_tokenizer.json")

# Пример токенизации
text = "Это пример текста на кириллице."
encoded = tokenizer.encode(text)
print(encoded.tokens)