"""Registro e promoção do melhor modelo no MLflow Model Registry."""

import mlflow
from mlflow import MlflowClient

from shopper_intent.models.trainer import MODEL_ARTIFACT_NAME


class ModelRegistry:
    """Escolhe o melhor run, registra o modelo e aplica o alias de produção."""

    def __init__(self, tracking_uri: str, model_name: str, alias: str) -> None:
        """Guarda a URI do MLflow, o nome do modelo registrado e o alias."""
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.alias = alias
        mlflow.set_tracking_uri(tracking_uri)

    def best_run_id(self, experiment_name: str, metric: str) -> str:
        """Devolve o ID do run com maior valor da métrica informada."""
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
            output_format="list",
        )
        if not runs:
            raise LookupError(f"Nenhum run encontrado no experimento '{experiment_name}'.")
        return runs[0].info.run_id

    def register(self, run_id: str) -> str:
        """Registra o modelo do run no Registry e devolve a versão criada."""
        model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
        return str(mlflow.register_model(model_uri, self.model_name).version)

    def promote(self, version: str) -> None:
        """Aponta o alias configurado para a versão informada."""
        MlflowClient().set_registered_model_alias(self.model_name, self.alias, version)
