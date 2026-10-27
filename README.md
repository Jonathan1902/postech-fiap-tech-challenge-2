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

## Dados (DVC)

O dataset bruto é versionado pelo DVC: o Git guarda só o ponteiro
`data/raw/online_shoppers_intention.csv.dvc` (hash md5); o arquivo fica no cache do DVC e
no remote.

**Remote atual: local** (`../dvc-storage`, pasta irmã do repositório). A migração para um
remote em nuvem (DagsHub) está planejada.

```bash
poetry run dvc pull              # baixa o dataset do remote
```

Sem acesso ao remote local (ex.: clone em outra máquina), recrie-o a partir do Kaggle:

```bash
# 1. Baixe online_shoppers_intention.csv do Kaggle e salve em data/raw/
# 2. Confira se o hash bate com o versionado (md5 em data/raw/*.dvc):
md5sum data/raw/online_shoppers_intention.csv
# 3. Crie o remote local e envie os dados:
mkdir -p ../dvc-storage && poetry run dvc push
```

## Pipeline

```
preprocess ──► train ──► register
```

| Estágio | O que faz | Saídas |
|---|---|---|
| `preprocess` | Remove duplicatas, converte tipos e faz split estratificado | `data/processed/{train,test}.csv` |
| `train` | Escolhe o melhor candidato por validação cruzada no treino, avalia no teste e loga cada run no MLflow | `models/model.pkl`, `reports/metrics.json`, `reports/best_run.json` |
| `register` | Registra o run escolhido no Model Registry e aplica o alias `champion` | `reports/registry.json` |

Parâmetros (seed, split, features, hiperparâmetros) ficam em `params.yaml`.

```bash
poetry run dvc repro             # roda só os estágios cujas dependências mudaram
poetry run dvc metrics show      # métricas do melhor modelo
poetry run dvc params diff       # parâmetros alterados desde o último commit
```

Enquanto o servidor MLflow (docker-compose) não existe, use um backend local no `.env`:
`MLFLOW_TRACKING_URI=sqlite:///mlflow.db`, e visualize com
`poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

**Ordem para rastreabilidade:** commite o código **antes** de rodar `dvc repro`, para que a
tag de commit gravada em cada run do MLflow aponte para o código que gerou o modelo. Depois,
commite `dvc.lock` e `reports/`. Cada run também recebe a tag `data_md5`, que liga o run à
versão do dataset no DVC.

## Desenvolvimento

```bash
poetry run ruff check .          # lint
poetry run ruff format --check . # formatação
poetry run pytest --cov          # testes com cobertura
```
