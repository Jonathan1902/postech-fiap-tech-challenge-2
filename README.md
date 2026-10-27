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

Sem o servidor MLflow do docker-compose, use um backend local no `.env`:
`MLFLOW_TRACKING_URI=sqlite:///mlflow.db`, e visualize com
`poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

**Ordem para rastreabilidade:** commite o código **antes** de rodar `dvc repro`, para que a
tag de commit gravada em cada run do MLflow aponte para o código que gerou o modelo. Depois,
commite `dvc.lock` e `reports/`. Cada run também recebe a tag `data_md5`, que liga o run à
versão do dataset no DVC.

## Docker

`Dockerfile` multi-stage (Poetry 2.5.1 só no build; runtime `python:3.13-slim`, a mesma versão do
ambiente local, usuário não-root) com os alvos `mlflow`, `trainer` (inclui `dvc` e `git`) e `api`.
O `docker-compose.yml` orquestra:

```bash
docker compose up -d --wait mlflow   # UI e Registry em http://localhost:5000
docker compose run --rm trainer      # dvc repro apontando para http://mlflow:5000
docker compose --profile api up -d   # API de inferência (implementada na etapa 4)
docker compose down                  # para os serviços (acrescente -v para apagar o volume do MLflow)
```

**Fluxo oficial:** o servidor do compose é a fonte dos `run_id` gravados em `reports/`. Os
relatórios commitados devem vir de uma execução no Docker; o SQLite local serve só para
experimentar. Num servidor MLflow novo (ou após `down -v`) o `dvc.lock` já está em dia e o
`dvc repro` não faria nada, então force a reexecução e commite o resultado:

```bash
docker compose run --rm trainer dvc repro --force
git add dvc.lock reports/ && git commit -m "chore: reexecuta pipeline no Docker"
```

- O `trainer` monta o repositório em `/app`: dados do DVC, `models/` e `reports/` ficam no host.
  O remote local (`../dvc-storage`) também é montado, então `dvc pull`/`dvc push` funcionam no
  container: `docker compose run --rm trainer dvc pull`.
- O `trainer` roda com `init: true`, para que o `dvc` não seja o PID 1. Se um container for
  morto no meio do pipeline e o DVC acusar lock ativo, apague `.dvc/tmp/rwlock`.
- O servidor MLflow usa SQLite + artefatos no volume `mlflow-data` e `--serve-artifacts`, então
  os clientes enviam artefatos por HTTP, sem montar o volume.
- `MLFLOW_TRACKING_URI` é sobrescrita no compose para `http://mlflow:5000`; o `.env` do host
  continua valendo para execuções fora do Docker.
- O servidor roda com `--workers 1` para caber em VMs pequenas do Docker Desktop. O treino dos
  três candidatos precisa de ~2 GB livres: se o container for morto (exit 137), aumente a memória
  em *Settings → Resources*.

## Desenvolvimento

```bash
poetry run ruff check .          # lint
poetry run ruff format --check . # formatação
poetry run pytest --cov          # testes com cobertura
```
