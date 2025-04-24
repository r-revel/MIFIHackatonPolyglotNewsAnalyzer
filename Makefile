.PHONY: clean data lint requirements \
        env kernel jupyter split train evaluate \
        sync_data_to_s3 sync_data_from_s3 \
        test_environment help

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# BUCKET = {{ cookiecutter.s3_bucket }} 
# PROFILE = {{ cookiecutter.aws_profile }}
PROJECT_NAME = MIFI_Hackaton_Polyglot_News_Analyzer
PYTHON_INTERPRETER = python3  

# Имя conda-окружения и файл-манифест
ENV_NAME = textcls
ENV_FILE = environment.yml

#################################################################################
# DEPENDENCY CHECK                                                              #
#################################################################################

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# BASE COMMANDS                                                                 #
#################################################################################

## Установка зависимостей через pip (используется реже, см. env)
requirements: test_environment              ## pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Создание/обновление conda-окружения из environment.yml
env:                                        ## conda env create / update
ifeq (True,$(HAS_CONDA))
	@echo ">>> Creating / updating conda environment '$(ENV_NAME)'"
	@conda env create -f $(ENV_FILE) -n $(ENV_NAME) || \
	  conda env update -f $(ENV_FILE) -n $(ENV_NAME) --prune
else
	@echo "❌ Conda не найден. Установите Anaconda / Miniconda."
endif

## Регистрация ядра Jupyter для окружения
kernel: env                                 ## ipykernel install
	@conda run -n $(ENV_NAME) python -m ipykernel install \
		--user --name $(ENV_NAME) \
		--display-name "TextCls ($(ENV_NAME))"

## Запуск Jupyter Lab внутри окружения
jupyter:                                    ## conda run jupyter lab
	@echo ">>> Starting Jupyter Lab ..."
	@conda run -n $(ENV_NAME) jupyter lab --no-browser --ip=0.0.0.0 --port=8888

#################################################################################
# DATA PIPELINE                                                                 #
#################################################################################

## Предобработка (Excel → CSV) и сплит на train/test
split: env                                  ## python src/data/splitting.py
	conda run -n $(ENV_NAME) python src/data/splitting.py

## Построение фичей / очистка (пример)
data: env                                   ## python src/data/make_dataset.py
	conda run -n $(ENV_NAME) python src/data/make_dataset.py data/raw data/processed

#################################################################################
# MODEL TRAINING & EVAL                                                         #
#################################################################################

## Обучение модели (ruRoberta-large-sentiment)
train: env split                            ## python src/models/model1/train.py
	conda run -n $(ENV_NAME) python src/models/model1/train.py

## Быстрая оценка сохранённой модели
evaluate: env                               ## python src/models/model1/evaluate.py
	conda run -n $(ENV_NAME) python src/models/model1/evaluate.py

#################################################################################
# QUALITY TOOLS                                                                 #
#################################################################################

## Очистка *.pyc, __pycache__
clean:                                      ## удалить *.pyc / __pycache__
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## flake8 + isort + black (только проверка)
lint: env                                   ## запуск flake8
	conda run -n $(ENV_NAME) flake8 src

## Проверка, что окружение & Python работают
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# SELF-DOCUMENTING HELP                                                         #
#################################################################################

.DEFAULT_GOAL := help

help:                                       ## показать список правил
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"; echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'