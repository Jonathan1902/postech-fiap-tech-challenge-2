"""Registro e promoção do melhor modelo no MLflow Model Registry."""

from dataclasses import dataclass, field

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from shopper_intent.models.trainer import MODEL_ARTIFACT_NAME


@dataclass(frozen=True)
class ModelRegistry:
    """Registra o modelo de um run e move o alias de produção quando ele é melhor.

    Usa a tracking URI global do MLflow, configurada pelo entrypoint.
    """

    model_name: str
    alias: str
    metric: str
    client: MlflowClient = field(default_factory=MlflowClient)

    def register(self, run_id: str) -> str:
        """Registra o modelo do run e devolve a versão; reaproveita a versão se já existir."""
        existing = self.client.search_model_versions(
            f"name='{self.model_name}' and run_id='{run_id}'"
        )
        if existing:
            return str(existing[0].version)
        model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
        return str(mlflow.register_model(model_uri, self.model_name).version)

    def promote(self, version: str) -> bool:
        """Aponta o alias para a versão se ela não for pior que a atual; diz se promoveu."""
        champion = self._champion_score()
        if champion is not None and self._score(version) < champion:
            return False
        self.client.set_registered_model_alias(self.model_name, self.alias, version)
        return True

    def _champion_score(self) -> float | None:
        """Métrica da versão que hoje tem o alias, ou None se ninguém o tiver."""
        try:
            champion = self.client.get_model_version_by_alias(self.model_name, self.alias)
        except MlflowException:
            return None
        return self._score(champion.version)

    def _score(self, version: str) -> float:
        """Lê a métrica de comparação no run que gerou a versão."""
        run_id = self.client.get_model_version(self.model_name, version).run_id
        return self.client.get_run(run_id).data.metrics[self.metric]
