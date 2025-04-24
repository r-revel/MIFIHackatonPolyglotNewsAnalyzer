import requests
import pandas as pd
from pathlib import Path

# Конфигурация
file_path = r"C:\Users\Igor\Downloads\датасет текстов\2.csv"
output_path = file_path

# Базовый prompt
base_prompt = '''
Нужно определить принадлежность текста к одной или нескольким тематикам из заданного списка (Спорт, Юмор, Реклама, Соцсети, Политика, Личная жизнь) 
в соответствии с представленными ниже примерами.
Пример 1:
Пиняев, во, "Какой пеняев хороший, а! И Воробьев! Второй матч, второй гол! Супер, а! Уф!" Спорт: 1.0
Пример 2:
Недавно мы с женой ходили на концерт группы «32nd to Mars» и «Джарда Лета». Билеты на этот концерт я подарил жене на день рождения. Впервые в жизни угадал с подарком. Личная жизнь: 1.0
Пример 3:
комик , который прожил и осознал эту жизнь – Стас Старовойтов  Ищи ответы на все вопросы сегодня в НОВОМ СЕЗОНЕ « стендап » в 23:00 на ТНТ"	Юмор: 1.0
Пример 4:
КАКАЯ КАРТА УКАЖЕТ НА МАСИКА?  РАСПРОДАЖА КУРСА «ОРАКУЛ ЛЕНОРМАН. БАЗОВЫЙ КУРС» В РАССРОЧКУ	Реклама: 1.0
Пример 5:
Приглашаю всей в свое ВК сообщество, будем делиться опытом и травить байки!	Соцсети: 1.0
Пример 6:
Сегодня запланирована встреча председателя госдумы и представителей регионов	Политика: 1.0
Если текст относится к разным категориям, то пиши через черточку: Соцсети: 0.5/Политика: 0.5.
Если текст не принадлежит ни к одной из указанных категорий, пиши: Нет категории
При этом суммарная степень уверенности не должна быть больше 1.0. 
Теперь определи принадлежность к категории у этого текста:
'''

# Настройки Ollama
url = "http://192.168.17.1:11434/api/generate"
model_name = "qwen2.5:32b"

# Загрузка файла
df = pd.read_csv(file_path)

# Проверка наличия колонки для ответов
if 'response' not in df.columns:
    df['response'] = ""

# Подсчет обработанных
processed_since_last_save = 0

# Обработка строк
for i, row in df.iterrows():
    # Пропуск уже обработанных строк
    if isinstance(df.at[i, 'response'], str) and df.at[i, 'response'].strip():
        print(f"⏭️ Строка {i + 1} уже обработана, пропускаем.")
        continue

    # Конкатенация всех столбцов (кроме response) в одну строку
    text_parts = []
    for col in df.columns:
        if col != 'response' and pd.notna(row[col]):
            text_parts.append(str(row[col]))
    text = ' '.join(text_parts).strip()

    if not text:
        print(f"⚠️ Строка {i + 1} пустая, пропускаем.")
        continue

    full_prompt = f"{base_prompt}\n\n{text}"
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False
    }

    print(f"🧠 Генерация по строке {i + 1}...")

    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            generated = response.json()["response"]
            df.at[i, 'response'] = generated
            print(f"✅ Ответ записан в строку {i + 1}")
            print(df.at[i, 'response'])
            processed_since_last_save += 1

            # Сохраняем каждые 10 строк
            if processed_since_last_save >= 10:
                df.to_csv(output_path, index=False)
                print("💾 Промежуточное сохранение.")
                processed_since_last_save = 0

        else:
            print(f"❌ Ошибка запроса ({response.status_code}): {response.text}")
            # При ошибке запроса сохраняем прогресс
            df.to_csv(output_path, index=False)
            print("💾 Экстренное сохранение при ошибке запроса.")
    except Exception as e:
        print(f"❗ Ошибка строки {i + 1}: {e}")
        # При любой другой ошибке сохраняем прогресс
        df.to_csv(output_path, index=False)
        print("💾 Экстренное сохранение при ошибке обработки.")

# Финальное сохранение
df.to_csv(output_path, index=False)
print(f"\n✅ Финальное сохранение. Результаты сохранены в: {output_path}")