# MIFI Hackaton Polyglot News Analyzer

## 
Камерзан И.Н. - лид, предобработка данных<br>
Колесников В.В. - обучение моделей ML<br>
Ревель Р. С. - разметка данных<br>
Люльчак Е. С. - фронтенд (чат-бот TG)<br>
Сырников А. С. - деплой модели<br>

Система, которая определяет вероятность принадлежности текста к одной или нескольким тематикам из заданного списка. 

## Project Organization
------------

```
MIFI Hackaton Polyglot News Analyzer
├── LICENSE     
├── README.md                  
├── Makefile                     # Makefile with commands like `make data` or `make train`                   
├── configs                      # Config files (models and training hyperparameters)
│   └── model1.yaml              
│
├── data                         
│   ├── external                 # Data from third party sources.
│   ├── interim                  # Intermediate data that has been transformed.
│   ├── processed                # The final, canonical data sets for modeling.
│   └── raw                      # The original, immutable data dump.
│
├── docs                         # Project documentation.
│
├── models                       # Trained and serialized models.
│
├── notebooks                    # Jupyter notebooks.
│
├── references                   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                  # Generated graphics and figures to be used in reporting.
│
├── requirements.txt             # The requirements file for reproducing the analysis environment.
└── src                          # Source code for use in this project.
    ├── __init__.py              # Makes src a Python module.
    │
    ├── data                     # Data engineering scripts.
    │   ├── build_features.py    
    │   ├── cleaning.py          
    │   ├── ingestion.py         
    │   ├── labeling.py          
    │   ├── splitting.py         
    │   └── validation.py        
    │
    ├── models                   # ML model engineering (a folder for each model).
    │   └── model1      
    │       ├── dataloader.py    
    │       ├── hyperparameters_tuning.py 
    │       ├── model.py         
    │       ├── predict.py       
    │       ├── preprocessing.py 
    │       └── train.py         
    │
    └── visualization        # Scripts to create exploratory and results oriented visualizations.
        ├── evaluation.py        
        └── exploration.py       
```

# Модель для классификации текстов по темам (Multi-label)

Модель на основе `cointegrated/rubert-tiny2` для классификации русскоязычных текстов по 7 темам:
- Спорт
- Личная жизнь
- Юмор
- Соцсети
- Политика
- Реклама
- Нет категории

## Основные характеристики

### Метрики качества (на валидации)
| Метрика          | Значение |
|------------------|----------|
| Micro F1         | 0.696    |
| Macro F1         | 0.665    |
| Macro ROC-AUC    | 0.890    |
| Subset Accuracy  | 0.480    |

### Оптимальные гиперпараметры
| Параметр       | Значение   |
|---------------|------------|
| Learning Rate | 2.43e-5    |
| Batch Size    | 4          |
| Epochs        | 6          |
| Weight Decay  | 0.0246     |

## Использование

### 1. Установка зависимостей
```bash
pip install transformers torch scikit-learn
```

2. Загрузка модели
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "models/cointegrated/rubert-tiny2/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```
3. Предсказание

```python
def predict(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).detach().numpy()[0]
    
    return {label: (prob > threshold) for label, prob in zip(model.config.id2label.values(), probs)}
```

4. Оптимальные пороги для классов
Для лучшего качества используйте индивидуальные пороги:

Тема	Порог
Спорт	0.347
Личная жизнь	0.264
Юмор	0.538
Соцсети	0.192
Политика	0.553
Реклама	0.575
Нет категории	0.269
Ограничения
Модель лучше всего работает с короткими текстами (до 128 токенов)

Качество ниже для классов с малым количеством примеров ("Личная жизнь", "Нет категории")

Для длинных текстов рекомендуется разбивать на предложения

5. Лицензия
Модель доступна по лицензии MIT. Использование коммерческое разрешено с указанием авторства.

--------
<p><small>Project based on the <a target="_blank" href="https://github.com/Chim-SO/cookiecutter-mlops/">cookiecutter MLOps project template</a>
that is originally based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
#cookiecuttermlops #cookiecutterdatascience</small></p>


--------
<p><small>Project based on the <a target="_blank" href="https://github.com/Chim-SO/cookiecutter-mlops/">cookiecutter MLOps project template</a>
that is originally based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
#cookiecuttermlops #cookiecutterdatascience</small></p>
