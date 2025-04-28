from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os
import uvicorn
from pydantic import BaseModel
import dotenv

# Загрузка переменных окружения из .env файла
dotenv.load_dotenv()
model_path = os.getenv("MODEL_PATH")

app = FastAPI(docs_url=None)

# Настройка шаблонизатора Jinja2 для работы с HTML-страницами
templates = Jinja2Templates(directory="templates")


class TextRequest(BaseModel):
    text: str


# Получение главной страницы
@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    """
    Обработчик GET-запроса для отображения главной страницы с формой.

    Args:
        request (Request): Объект запроса FastAPI.

    Returns:
        TemplateResponse: HTML-страница с формой для ввода текста.
    """
    return templates.TemplateResponse(request, "form.html", {"labels": None})


# Обработка запроса формы
@app.post("/predict/form", response_class=JSONResponse)
def predict_form(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = classifier(text)
    labels = [label for label in result[0] if label['score'] >= 0.5]
    return {"text": text, "labels": labels}


# Обработка запросов с json
@app.post("/predict", response_class=JSONResponse)
def predict_json(request: TextRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = classifier(text)
    labels = [label for label in result[0] if label['score'] >= 0.5]
    return {"text": text, "labels": labels}


# Инициализация модели и классификатора
model = None
tokenizer = None
classifier = None

if __name__ == "__main__":

    # Загрузка модели и токенизатора
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

    # Запуск сервера FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
