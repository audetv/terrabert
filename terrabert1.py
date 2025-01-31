import os
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM, BertConfig
import torch
from torch.utils.data import DataLoader, Dataset

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),  # Логи в файл
        logging.StreamHandler()  # Логи в консоль
    ]
)

# 1. Чтение файлов и создание корпуса
def create_corpus(input_folder, output_file):
    """
    Читает все текстовые файлы из папки и объединяет их в один корпус.
    """
    logging.info("Создание корпуса текстов...")
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
    logging.info(f"Корпус сохранен в файл: {output_file}")

# 2. Создание токенизатора и его дообучение на корпусе
def create_and_train_tokenizer(corpus_file, vocab_size=30000):
    """
    Создает и дообучает токенизатор на корпусе текстов.
    """
    logging.info("Создание и дообучение токенизатора...")

    # Загрузка предобученного токенизатора
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Подготовка корпуса текстов
    def read_corpus(file_path):
        """Генератор, который читает тексты из файла."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()

    # Дообучение токенизатора
    new_tokenizer = tokenizer.train_new_from_iterator(
        read_corpus(corpus_file),  # Итератор по текстам
        vocab_size=vocab_size,  # Размер нового словаря
        min_frequency=2,  # Минимальная частота для добавления токена
        show_progress=True  # Отображение прогресса
    )

    # Сохранение дообученного токенизатора
    new_tokenizer.save_pretrained("my_retrained_tokenizer")
    logging.info("Дообученный токенизатор сохранен в папку: my_retrained_tokenizer")

    # Создание файла vocab.txt
    vocab = new_tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # Сортировка по id
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")
    logging.info("Словарь сохранен в файл: vocab.txt")

    return new_tokenizer

# 3. Подготовка данных для обучения модели
class TextDataset(Dataset):
    """
    Класс для создания Dataset из текстов.
    """
    def __init__(self, texts, tokenizer, max_seq_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}

# 4. Обучение модели
def train_model(tokenizer, corpus_file, num_epochs=3, batch_size=8, max_seq_length=512):
    """
    Обучает модель BERT на корпусе текстов.
    """
    logging.info("Подготовка данных для обучения модели...")
    # Чтение корпуса
    with open(corpus_file, "r", encoding="utf-8") as f:
        texts = f.readlines()

    # Создание Dataset и DataLoader
    dataset = TextDataset(texts, tokenizer, max_seq_length=max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Создание модели BERT
    logging.info("Создание модели BERT...")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_seq_length,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        intermediate_size=3072
    )
    model = BertForMaskedLM(config)

    # Оптимизатор и функция потерь
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Цикл обучения
    logging.info("Начало обучения модели...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Используем tqdm для отображения прогресса
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # Прямой проход
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        # Закрываем прогресс-бар для текущей эпохи
        progress_bar.close()

        # Логирование средней потери за эпоху
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Сохранение модели
    model.save_pretrained("my_bert_model")
    logging.info("Модель сохранена в папку: my_bert_model")

# 5. Основной процесс
if __name__ == "__main__":
    # Папка с текстовыми файлами
    input_folder = "dotu"
    # Файл для сохранения корпуса
    corpus_file = "corpus.txt"

    # 1. Создание корпуса
    create_corpus(input_folder, corpus_file)

    # 2. Создание и дообучение токенизатора
    tokenizer = create_and_train_tokenizer(corpus_file)

    # 3. Обучение модели
    train_model(tokenizer, corpus_file, num_epochs=3, batch_size=8, max_seq_length=512)

    logging.info("Модель и токенизатор успешно обучены и сохранены.")