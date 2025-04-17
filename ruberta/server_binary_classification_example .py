from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import uvicorn

app = FastAPI()

# Указываем абсолютный путь к модели
model_path = os.path.abspath("./best_distilbert_model")

# Загрузка модели
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    result = classifier(request.text)
    label = "Проблемы" if result[0]['label'] == 'LABEL_1' else 'OK'
    return {
        "text": request.text,
        "label": label,
        "confidence": result[0]['score']
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)