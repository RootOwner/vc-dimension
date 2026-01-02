# ==============================
# Project: Norm-Based Complexity Measures
# ==============================

# ---- Variables ----
PYTHON      := python3
VENV        := venv
VENV_BIN    := $(VENV)/bin
PIP         := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

CODE_DIR    := code
MAIN_TEX    := main.tex
PDF         := main.pdf

LATEX_PKGS  := \
	texlive-latex-extra \
	texlive-latex-recommended \
	texlive-fonts-recommended \
	texlive-fonts-extra \
	texlive-lang-cyrillic \
	texlive-lang-european \
	texlive-science \
	texlive-bibtex-extra \
	texlive-publishers \
	biber

# ---- Default target ----
.DEFAULT_GOAL := all

# ==============================
# Help
# ==============================
help:
	@echo "Доступные цели:"
	@echo "  setup      : Создание виртуального окружения и установка зависимостей"
	@echo "  run        : Запуск полного эксперимента (обучение + оценка + графики)"
	@echo "  train      : Только обучение моделей"
	@echo "  evaluate   : Только оценка моделей"
	@echo "  plots      : Только генерация графиков"
	@echo "  pdf        : Сборка LaTeX-документа (main.pdf) (Установка зависимостей при необходимости)"
	@echo "  clean      : Удаление вспомогательных файлов LaTeX"
	@echo "  clean-all  : Удаление всех сгенерированных файлов"
	@echo "  help       : Показать это сообщение справки"
	@echo ""
	@echo "Пример рабочего процесса:"
	@echo "  make setup"
	@echo "  make run"
	@echo "  make pdf"
	@echo ""
	@echo "Системные требования (Ubuntu):"
	@echo "  sudo apt update && sudo apt install python3 python3-pip python3-venv make"
	@echo "--------------------------------------------------------------------------------"

# ==============================
# Setup
# ==============================
setup: $(VENV)/.installed

$(VENV)/.installed:
	@echo "[SETUP] Создание виртуального окружения"
	$(PYTHON) -m venv $(VENV)
	@echo "[SETUP] Обновление pip"
	$(PIP) install --upgrade pip
	@echo "[SETUP] Установка Python-зависимостей"
	$(PIP) install -r requirements.txt
	@touch $(VENV)/.installed
	@echo "[SETUP] Готово"

# ==============================
# Experiment
# ==============================
run: setup
	@echo "[RUN] Полный запуск эксперимента"
	$(PYTHON_VENV) code/run_all.py

train: setup
	@echo "[TRAIN] Обучение моделей"
	$(PYTHON_VENV) code/train.py

evaluate: setup
	@echo "[EVAL] Оценка моделей"
	$(PYTHON_VENV) code/evaluate.py

plots: setup
	@echo "[PLOTS] Генерация графиков"
	$(PYTHON_VENV) && code/plots.py

# ==============================
# LaTeX
# ==============================
pdf: latex-deps
	@echo "[LATEX] Компиляция LaTeX-документа"
	pdflatex $(MAIN_TEX)
	biber main
	pdflatex $(MAIN_TEX)
	pdflatex $(MAIN_TEX)
	@echo "[LATEX] Готово: $(PDF)"

latex-deps:
	@echo "[LATEX] Проверка LaTeX-зависимостей"
	@which pdflatex >/dev/null || sudo apt update && sudo apt install -y $(LATEX_PKGS)
	@which biber    >/dev/null || sudo apt update && sudo apt install -y biber


# ==============================
# Clean
# ==============================
clean:
	@echo "[CLEAN] Удаление LaTeX-временных файлов"
	rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *.toc *.lof *.lot

clean-all: clean
	@echo "[CLEAN-ALL] Удаление результатов и LaTeX-временных файлов"
	rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *.toc *.lof *.lot
	rm -rf results/*
	rm -rf figures/*
