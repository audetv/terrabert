import pytest
from tokenizers import Tokenizer  # Используем библиотеку tokenizers

def test_tokenizer():
    # Загрузка токенизатора
    tokenizer = Tokenizer.from_file("my_bert_tokenizer.json")

    # Пример текста
    text = "Это пример текста для тестирования."

    # Токенизация
    tokens = tokenizer.encode(text).tokens
    ids = tokenizer.encode(text).ids

    # Проверка, что токенизация прошла успешно
    assert len(tokens) > 0, "Токенизатор не вернул токены."
    assert len(ids) > 0, "Токенизатор не вернул идентификаторы."

    # Проверка, что токены содержат ожидаемые подстроки
    assert "Предисловие" in tokens, "Токенизатор не обработал текст корректно."

import torch
from transformers import BertForMaskedLM

# def test_model():
#     # Загрузка модели
#     model = BertForMaskedLM.from_pretrained("my_bert_model")

#     # Пример текста
#     text = "Это пример текста для тестирования модели."

#     # Токенизация
#     tokenizer = BertTokenizer.from_file("my_bert_tokenizer.json")
#     inputs = tokenizer.encode(text, return_tensors="pt")

#     # Получение эмбеддингов
#     with torch.no_grad():
#         outputs = model(input_ids=inputs)
#         embeddings = outputs.last_hidden_state

#     # Проверка, что эмбеддинги имеют правильную форму
#     assert embeddings.shape == (1, len(inputs[0]), 768), "Неверная форма эмбеддингов."