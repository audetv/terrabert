from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tokenizer = Tokenizer.from_file("my_tokenizer.json")

# Чтение корпуса текста
with open("corpus.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()  # Читаем все строки файла


import torch

def prepare_sequences(texts, tokenizer, max_seq_length=256):
    inputs = tokenizer.encode_batch(texts)
    sequences = []
    for input in inputs:
        tokens = input.ids
        # Разбиение на части по max_seq_length
        for i in range(0, len(tokens), max_seq_length):
            sequence = tokens[i:i + max_seq_length]
            if len(sequence) < max_seq_length:
                # Добавление паддинга
                sequence += [tokenizer.token_to_id("[PAD]")] * (max_seq_length - len(sequence))
            sequences.append(sequence)
    return torch.tensor(sequences)

# Подготовка последовательностей
max_seq_length = 256
sequences = prepare_sequences(texts, tokenizer, max_seq_length=max_seq_length)

# 3. Создание DataLoader
# Для удобства работы с данными создадим DataLoader, который будет формировать батчи для обучения:

from torch.utils.data import DataLoader, TensorDataset

# Создание Dataset
dataset = TensorDataset(sequences)

# Создание DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 4. Создание модели
# Теперь создадим модель, аналогичную BERT, но с увеличенным max_seq_length.
#  Мы будем использовать архитектуру трансформера из библиотеки transformers.

from transformers import BertConfig, BertForMaskedLM

# Параметры модели
vocab_size = tokenizer.get_vocab_size()
d_model = 768  # Размерность эмбеддингов
nhead = 12     # Количество голов в multi-head attention
num_layers = 12  # Количество слоев трансформера
dim_feedforward = 3072  # Размерность скрытого слоя в feedforward сети

# Создание конфигурации BERT
config = BertConfig(
    vocab_size=vocab_size,
    max_position_embeddings=max_seq_length,
    hidden_size=d_model,
    num_attention_heads=nhead,
    num_hidden_layers=num_layers,
    intermediate_size=dim_feedforward
)

# Создание модели
model = BertForMaskedLM(config)

# 5. Обучение модели
# Теперь настроим обучение модели. Мы будем использовать задачу Masked Language Modeling (MLM), как в BERT.

# 5.1. Создание масок
# Для задачи MLM нам нужно замаскировать часть токенов в каждой последовательности:

def create_masked_inputs(sequences, tokenizer, mask_prob=0.15):
    inputs = sequences.clone()
    labels = sequences.clone()
    mask_token_id = tokenizer.token_to_id("[MASK]")
    pad_token_id = tokenizer.token_to_id("[PAD]")

    # Создание масок
    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            if labels[i, j] != pad_token_id and torch.rand(1) < mask_prob:
                labels[i, j] = inputs[i, j]  # Сохраняем оригинальный токен в labels
                if torch.rand(1) < 0.8:
                    inputs[i, j] = mask_token_id  # Заменяем на [MASK]
                elif torch.rand(1) < 0.5:
                    inputs[i, j] = torch.randint(0, tokenizer.get_vocab_size(), (1,))  # Заменяем на случайный токен

    return inputs, labels

# 5.2. Цикл обучения
# Теперь напишем цикл обучения:

from tqdm import tqdm
import torch.optim as optim
import logging

# Настройка логирования
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Оптимизатор
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Функция потерь
criterion = torch.nn.CrossEntropyLoss()

# Цикл обучения
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Используем tqdm для отображения прогресса
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch[0]  # Получаем последовательности из DataLoader

        # Создание масок
        masked_inputs, labels = create_masked_inputs(input_ids, tokenizer)

        # Прямой проход
        outputs = model(input_ids=masked_inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Обновляем описание прогресс-бара
        progress_bar.set_postfix({"Loss": loss.item()})

    # Закрываем прогресс-бар для текущей эпохи
    progress_bar.close()

    # Выводим среднюю потерю за эпоху
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# 6. Сохранение модели
# После обучения сохраним модель и токенизатор:

# Сохранение модели
model.save_pretrained("my_bert_model")
tokenizer.save("my_bert_tokenizer.json")