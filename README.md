# Shopper Intent: propensão de compra em e-commerce

Sistema preditivo que estima a **probabilidade de uma sessão de navegação terminar em compra**,
a partir do dataset [Online Shoppers Purchasing Intention](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
(12.330 sessões, 17 features, alvo `Revenue`).

O foco é a **engenharia de Machine Learning**: pipeline reprodutível com DVC, experimentos e
Model Registry no MLflow, execução em Docker, dependências com Poetry e código limpo com
testes. Tech Challenge da Fase 2 da Pós-Tech FIAP (Machine Learning Engineering).

## Arquitetura

```
                 params.yaml (seed, split, features, hiperparâmetros)
                        │
 data/raw/*.csv ──► [preprocess] ──► data/processed/ ──► [train] ──► [register]
   (DVC)            dedup + split       (DVC)            CV + teste    alias champion
                    estratificado                          │              │
                                                           ▼              ▼
                                         MLflow server: runs, métricas, Model Registry
                                                                          │
                                         API FastAPI  ◄── models:/shopper-intent-model@champion
                                         POST /predict
```

| Serviço (docker-compose) | Função |
|---|---|
| `mlflow` | Tracking e Model Registry (SQLite + artefatos em volume), em http://localhost:5000 |
| `trainer` | Roda `dvc repro`: preprocess → train → register |
| `api` | Serve o modelo `champion`, em http://localhost:8000/docs |

## Resultados

Três candidatos, todos dentro do mesmo `Pipeline` sklearn (pré-processamento + estimador). A
escolha usa **validação cruzada estratificada (5 folds) no treino**; o teste fica reservado
para a avaliação final.

| Modelo | CV PR-AUC | Teste PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| **HistGradientBoosting** (champion) | **0,742** | **0,741** | 0,934 | 0,551 | 0,809 |
| RandomForest | 0,726 | 0,722 | 0,929 | 0,554 | 0,819 |
| LogisticRegression (baseline) | 0,646 | 0,663 | 0,910 | 0,508 | 0,796 |

Matriz de confusão do champion no teste (limiar 0,5): 309 compradores encontrados e 73
perdidos; 252 falsos positivos entre 2.059 não compradores.

Principais decisões (detalhes e gráficos em [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)):

- **Métrica PR-AUC:** só 15,6% das sessões compram; um modelo que sempre responde "não
  compra" teria 84% de acurácia.
- **`PageValues`** é a feature dominante: sem ela, o PR-AUC de validação cai de 0,74 para 0,38.
  Foi mantida, mas em produção é preciso confirmar que ela existe no momento da previsão.
- **Limiar 0,5** com `class_weight="balanced"` favorece o recall (81%). É configurável em
  `params.yaml` e a API usa o limiar logado no run do modelo promovido.

## Início rápido (Docker)

Pré-requisitos: Docker com Compose v2 e o dataset disponível (veja [Dados](#dados-dvc)).

```bash
docker compose up -d --wait mlflow                  # 1. servidor MLflow
docker compose run --rm trainer dvc repro --force   # 2. pipeline completo (~3 min)
docker compose --profile api up -d --wait api       # 3. API com o modelo champion
curl localhost:8000/health                          # {"status":"ok","model_version":"..."}
```

Para parar: `docker compose --profile api down` (acrescente `-v` para apagar o volume do MLflow).

## Instalação local

Pré-requisitos: Python 3.11–3.13 e [Poetry](https://python-poetry.org/) 2.x.

```bash
pipx install poetry        # caso ainda não tenha o Poetry
poetry install             # dependências de produção, pipeline (dvc) e desenvolvimento
cp .env.example .env       # ajuste os valores se necessário
```

| Grupo do Poetry | Pacotes | Onde é instalado |
|---|---|---|
| main | scikit-learn, pandas, mlflow, fastapi, uvicorn, pydantic-settings, pyyaml, joblib | todas as imagens |
| pipeline | dvc | imagem `trainer` |
| dev | pytest, pytest-cov, ruff, jupyter, matplotlib, seaborn | só localmente |

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

| Estágio | O que faz | Saídas |
|---|---|---|
| `preprocess` | Remove duplicatas, converte tipos e faz split estratificado | `data/processed/{train,test}.csv` |
| `train` | Escolhe o melhor candidato por validação cruzada no treino, avalia no teste e loga cada run no MLflow | `models/model.pkl`, `reports/metrics.json`, `reports/best_run.json` |
| `register` | Registra o run escolhido no Model Registry e aplica o alias `champion` | `reports/registry.json` |

Parâmetros (seed, split, features, hiperparâmetros) ficam em `params.yaml`; cada estágio
declara só os parâmetros e arquivos que usa, então o DVC reexecuta apenas o necessário.

```bash
poetry run dvc repro             # roda só os estágios cujas dependências mudaram
poetry run dvc metrics show      # métricas do melhor modelo
poetry run dvc params diff       # parâmetros alterados desde o último commit
```

Sem o servidor MLflow do docker-compose, use um backend local no `.env`:
`MLFLOW_TRACKING_URI=sqlite:///mlflow.db`, e visualize com
`poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## MLflow

Cada candidato gera um run no experimento `shopper-intent` com:

- **Parâmetros:** hiperparâmetros do modelo, `seed`, `cv_folds` e `threshold`.
- **Métricas:** `cv_average_precision` (critério de escolha) e, no teste, `average_precision`,
  `roc_auc`, `f1`, `precision` e `recall`.
- **Artefatos:** o `Pipeline` completo (com assinatura e exemplo de entrada) e
  `confusion_matrix.json`.
- **Tags:** `mlflow.source.git.commit` (código) e `data_md5` (versão do dataset no DVC).

O estágio `register` registra o run vencedor como uma nova versão de `shopper-intent-model` e
move o alias `champion` para ela. A API sempre carrega a versão apontada pelo alias.

**Ordem para rastreabilidade:** commite o código **antes** de rodar `dvc repro`, para que a
tag de commit gravada em cada run aponte para o código que gerou o modelo. Depois, commite
`dvc.lock` e `reports/`.

## API de inferência

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "Administrative": 3, "Administrative_Duration": 87.83, "Informational": 0,
  "Informational_Duration": 0, "ProductRelated": 27, "ProductRelated_Duration": 798.33,
  "BounceRates": 0, "ExitRates": 0.0126, "PageValues": 22.92, "SpecialDay": 0.8,
  "Month": "Feb", "OperatingSystems": 2, "Browser": 2, "Region": 3, "TrafficType": 1,
  "VisitorType": "Returning_Visitor", "Weekend": false
}'
# {"purchase_probability":0.909,"will_purchase":true,"threshold":0.5,"model_version":"3"}
```

- O corpo usa os nomes originais das colunas e é validado pelo Pydantic: campo ausente, campo
  extra, mês ou tipo de visitante inválido, ou taxa fora de [0, 1] retornam **422**.
- `GET /health` retorna **503** enquanto não houver modelo com o alias no Registry (por
  exemplo, antes do primeiro `dvc repro`).
- Documentação interativa (Swagger) em http://localhost:8000/docs.
- Fora do Docker: `poetry run uvicorn shopper_intent.api.main:app --reload`.

## Docker

`Dockerfile` multi-stage (Poetry 2.5.1 só no build; runtime `python:3.13-slim`, a mesma versão do
ambiente local, usuário não-root) com os alvos `mlflow`, `trainer` (inclui `dvc` e `git`) e `api`.

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

## Estrutura

```
src/shopper_intent/
├── config.py               # Settings (.env) e load_params (params.yaml)
├── data/                   # loader, preprocess (limpeza e split), versioning (md5 do DVC)
├── features/               # FeatureBuilder: ColumnTransformer (scaler + one-hot)
├── models/                 # factory, validation (CV), evaluator, trainer (MLflow), registry
├── pipelines/              # entrypoints dos estágios do dvc.yaml
└── api/                    # FastAPI: schemas (Pydantic), model_loader, main
tests/                      # pytest (inclui MLflow isolado em SQLite temporário)
notebooks/01_eda.ipynb      # análise exploratória e justificativa das decisões
dvc.yaml · params.yaml · Dockerfile · docker-compose.yml · pyproject.toml · poetry.lock
```

## Desenvolvimento

```bash
poetry run ruff check .          # lint (inclui o notebook)
poetry run ruff format --check . # formatação
poetry run pytest --cov          # testes com cobertura
```
