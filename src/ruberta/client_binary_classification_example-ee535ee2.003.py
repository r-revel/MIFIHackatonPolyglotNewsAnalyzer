import requests
from pprint import pprint

# Адрес сервера
API_URL = "http://localhost:8000/predict"

# Список текстов для проверки
sample_texts = [
    "Sample 1",  # true
    "Sample 2" # false
]


def send_request(text):
    """Отправляет текст на сервер и возвращает результат."""
    response = requests.post(API_URL, json={"text": text})
    return response.json()


if __name__ == "__main__":
    print("Отправка текстов на сервер...\n")

    for text in sample_texts:
        result = send_request(text)
        print(f"Текст: {text}")
        print(f"Результат: {result['label']} (уверенность: {result['confidence']:.4f})")
        print("-" * 50)