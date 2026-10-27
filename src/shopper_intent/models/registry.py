"""Registro e promoção do melhor modelo no MLflow Model Registry."""


class ModelRegistry:
    """Escolhe o melhor run, registra o modelo e aplica o alias de produção."""

    def __init__(self, tracking_uri: str, model_name: str, alias: str) -> None:
        """Guarda a URI do MLflow, o nome do modelo registrado e o alias."""
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.alias = alias

    def best_run_id(self, experiment_name: str, metric: str) -> str:
        """Devolve o ID do run com maior valor da métrica informada."""
        raise NotImplementedError

    def register(self, run_id: str) -> str:
        """Registra o modelo do run no Registry e devolve a versão criada."""
        raise NotImplementedError

    def promote(self, version: str) -> None:
        """Aponta o alias configurado para a versão informada."""
        raise NotImplementedError
