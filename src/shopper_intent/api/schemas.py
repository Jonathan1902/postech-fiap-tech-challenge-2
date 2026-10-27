"""Contratos de entrada e saída da API, validados pelo Pydantic."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# Grafia do dataset original ("June" por extenso). Jan e Apr não aparecem no treino:
# são aceitos e o OneHotEncoder (handle_unknown="ignore") os trata como categoria nova.
Month = Literal["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VisitorType = Literal["Returning_Visitor", "New_Visitor", "Other"]


class SessionFeatures(BaseModel):
    """Comportamento de navegação de uma sessão (as 17 features do dataset).

    Os campos são declarados na mesma ordem das colunas usadas no treino e aceitam o
    nome original da coluna (alias) no JSON.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    administrative: int = Field(alias="Administrative", ge=0)
    administrative_duration: float = Field(alias="Administrative_Duration", ge=0)
    informational: int = Field(alias="Informational", ge=0)
    informational_duration: float = Field(alias="Informational_Duration", ge=0)
    product_related: int = Field(alias="ProductRelated", ge=0)
    product_related_duration: float = Field(alias="ProductRelated_Duration", ge=0)
    bounce_rates: float = Field(alias="BounceRates", ge=0, le=1)
    exit_rates: float = Field(alias="ExitRates", ge=0, le=1)
    page_values: float = Field(alias="PageValues", ge=0)
    special_day: float = Field(alias="SpecialDay", ge=0, le=1)
    month: Month = Field(alias="Month")
    operating_systems: int = Field(alias="OperatingSystems", ge=1)
    browser: int = Field(alias="Browser", ge=1)
    region: int = Field(alias="Region", ge=1)
    traffic_type: int = Field(alias="TrafficType", ge=1)
    visitor_type: VisitorType = Field(alias="VisitorType")
    weekend: bool = Field(alias="Weekend")

    def to_frame(self) -> pd.DataFrame:
        """Converte a sessão em um DataFrame de uma linha, no formato visto no treino."""
        row = self.model_dump(by_alias=True)
        row["Weekend"] = int(row["Weekend"])
        return pd.DataFrame([row])


class PredictionResponse(BaseModel):
    """Probabilidade de compra e a decisão tomada com o limiar do modelo."""

    purchase_probability: float
    will_purchase: bool
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    """Estado da API e versão do modelo carregado."""

    status: Literal["ok", "unavailable"]
    model_version: str | None
