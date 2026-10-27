# postech-fiap-tech-challenge-2

Sistema preditivo de propensão de compra a partir do comportamento de navegação
(dataset [Online Shoppers Purchasing Intention](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)).

## Instalação

Pré-requisitos: Python 3.11–3.13 e [Poetry](https://python-poetry.org/) 2.x.

```bash
pipx install poetry        # caso ainda não tenha o Poetry
poetry install             # instala dependências de produção e de desenvolvimento
cp .env.example .env       # ajuste os valores se necessário
```

## Desenvolvimento

```bash
poetry run ruff check .          # lint
poetry run ruff format --check . # formatação
poetry run pytest --cov          # testes com cobertura
```
