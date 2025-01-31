from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Создание токенизатора
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# Обучение токенизатора на вашем корпусе
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Сохранение токенизатора
tokenizer.save("my_tokenizer.json")

# Создание файла vocab.txt
vocab = tokenizer.get_vocab()  # Получаем словарь (токен -> id)
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # Сортируем по id

# Сохраняем в vocab.txt
with open("vocab.txt", "w", encoding="utf-8") as f:
    for token, _ in sorted_vocab:
        f.write(token + "\n")