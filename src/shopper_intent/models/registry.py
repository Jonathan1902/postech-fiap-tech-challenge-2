"""Registro e promoção do melhor modelo no MLflow Model Registry."""

import mlflow
from mlflow import MlflowClient

from shopper_intent.models.trainer import MODEL_ARTIFACT_NAME


class ModelRegistry:
    """Registra o modelo de um run e aplica o alias de produção."""

    def __init__(self, tracking_uri: str, model_name: str, alias: str) -> None:
        """Guarda a URI do MLflow, o nome do modelo registrado e o alias."""
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.alias = alias

    def register(self, run_id: str) -> str:
        """Registra o modelo do run no Registry e devolve a versão criada."""
        mlflow.set_tracking_uri(self.tracking_uri)
        model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
        return str(mlflow.register_model(model_uri, self.model_name).version)

    def promote(self, version: str) -> None:
        """Aponta o alias configurado para a versão informada."""
        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.set_registered_model_alias(self.model_name, self.alias, version)
