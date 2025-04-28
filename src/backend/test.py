import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# Фикстура для подмены classifier
@pytest.fixture(autouse=True)
def mock_classifier(monkeypatch):
    def fake_classifier(text):
        if "bad" in text:
            return [[{"label": "negative", "score": 0.95}]]
        else:
            return [[{"label": "positive", "score": 0.9}]]

    monkeypatch.setattr("main.classifier", fake_classifier)


# Фикстуры для тестовых данных
@pytest.fixture
def positive_text():
    return "Good text"


@pytest.fixture
def negative_text():
    return "This is a bad example"


@pytest.fixture
def empty_text():
    return ""


# Вспомогательная функция для проверки ответов
def assert_response(response, status_code, expected_text=None, expected_label=None):
    assert response.status_code == status_code
    if status_code == 200:
        json_data = response.json()
        assert json_data["text"] == expected_text
        if expected_label:
            assert json_data["labels"][0]["label"] == expected_label


# Тесты
def test_get_form():
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.headers["content-type"]


@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("Good text", "positive"),
        ("Another great example", "positive"),
        ("This is a bad example", "negative"),
    ],
)
def test_predict_json(text, expected_label):
    response = client.post("/predict", json={"text": text})
    assert_response(response, 200, expected_text=text, expected_label=expected_label)


def test_predict_form_positive(positive_text):
    response = client.post("/predict/form", data={"text": positive_text})
    assert_response(response, 200, expected_text=positive_text, expected_label="positive")


def test_predict_form_empty_text(empty_text):
    response = client.post("/predict/form", data={"text": empty_text})
    assert response.status_code == 400
    assert response.json()["detail"] == "Text cannot be empty."


def test_predict_json_missing_text():
    response = client.post("/predict", json={"wrong_field": "oops"})
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_invalid_content_type():
    response = client.post("/predict", content="Just some raw text", headers={"Content-Type": "text/plain"})
    assert response.status_code == 422
