# Импорт библиотек
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import resample

# ========== НАСТРОЙКА УСТРОЙСТВА ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# ========== ЗАГРУЗКА И БАЛАНСИРОВКА ДАННЫХ ==========
df = pd.read_csv(r'C:\Users\Igor\Downloads\train.csv (1).csv')
df['label'] = df['rate'] - 1  # Преобразуем 1-5 в 0-4

print("\nРаспределение классов до балансировки:")
print(df['rate'].value_counts().sort_index())

# Находим размер самого большого класса
class_counts = df['rate'].value_counts()
max_count = class_counts.max()

# Балансировка данных
dfs = []
for rating in range(1, 6):
    df_class = df[df['rate'] == rating]
    # Для классов с количеством примеров меньше максимального делаем upsampling
    if len(df_class) < max_count:
        dfs.append(resample(df_class,
                          replace=True,  # Разрешаем повторение примеров
                          n_samples=max_count,  # Доводим до размера максимального класса
                          random_state=42))
    else:
        dfs.append(df_class)  # Для максимального класса оставляем как есть

df_balanced = pd.concat(dfs).sample(frac=1, random_state=42)  # Перемешиваем

print("\nПосле балансировки:")
print(df_balanced['rate'].value_counts().sort_index())

# Остальной код остается без изменений...
# ========== ЗАГРУЗКА РУССКОЯЗЫЧНОЙ МОДЕЛИ ==========
model_name = 'DeepPavlov/rubert-base-cased'  # Лучшая модель для русского языка
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5).to(device)

# ========== ПОДГОТОВКА ДАТАСЕТА ==========
class RussianDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Разделение данных
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_balanced['text'].tolist(),
    df_balanced['label'].values,
    test_size=0.15,
    random_state=42,
    stratify=df_balanced['label']
)

train_dataset = RussianDataset(train_texts, train_labels, tokenizer)
val_dataset = RussianDataset(val_texts, val_labels, tokenizer)

# ========== ОБУЧЕНИЕ МОДЕЛИ ==========
training_args = TrainingArguments(
    output_dir=r"D:\models\модели 2025\rubert_rating",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
        'f1': f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average='weighted')
    }
)

print("\nНачало обучения модели...")
trainer.train()
model.save_pretrained(r"D:\models\модели 2025\best_rubert_rate_model")
print("Обучение завершено и модель сохранена!")

# ========== ПРЕДСКАЗАНИЕ НА ТЕСТОВЫХ ДАННЫХ ==========
print("\nЗагрузка тестовых данных...")
test_df = pd.read_csv(r'C:\Users\Igor\Downloads\test.csv')
test_texts = test_df['text'].fillna('').astype(str).tolist()

print("Прогнозирование оценок...")
predictions = []
for i in tqdm(range(0, len(test_texts), 32), desc="Обработка"):
    batch = test_texts[i:i+32]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy() + 1
        predictions.extend(preds)

# Сохранение результатов
result_df = pd.DataFrame({'index': test_df.index, 'rate': predictions})
output_path = r"D:\models\модели 2025\rubert_predictions.csv"
result_df.to_csv(output_path, index=False)
print(f"\nРезультаты сохранены в: {output_path}")
print("\nПример первых 5 предсказаний:")
print(result_df.head())