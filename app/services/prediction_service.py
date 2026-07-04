from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.model_loader import ModelLoader
from api.schemas.prediction import (
    FeatureOverride,
    PredictionResponse,
)




class PredictionService:
    """Business logic for predictions."""

    def __init__(self, db: AsyncSession, model_loader: ModelLoader):
        self.db = db
        self.loader = model_loader

    async def predict_single(
        self,
        order_id: str,
        feature_override: FeatureOverride | None = None,
    ) -> PredictionResponse:
        """Predict delay for a single order."""

        # ─────────────────────────────────────────────────────────────
        # 1) Get features (from DB or override)
        # ─────────────────────────────────────────────────────────────
        if feature_override:
            features_df = self._build_features_from_override(order_id, feature_override)
        else:
            features_df = await self._fetch_features_from_db(order_id)

        if features_df is None or features_df.empty:
            raise ValueError(f"Order {order_id} not found in feature table")

        # ─────────────────────────────────────────────────────────────
        # 2) Prepare input (drop non-feature columns)
        # ─────────────────────────────────────────────────────────────
        X = self._prepare_model_input(features_df)

        # ─────────────────────────────────────────────────────────────
        # 3) Predict
        # ─────────────────────────────────────────────────────────────
        proba = self.loader.predict(X)[0]  # returns array, take first element
        if isinstance(proba, (list, tuple)):
            proba = proba[1]  # binary classifier: [prob_class_0, prob_class_1]

        # ─────────────────────────────────────────────────────────────
        # 4) Build response
        # ─────────────────────────────────────────────────────────────
        metadata = self.loader.metadata
        return PredictionResponse.build(
            order_id=order_id,
            probability=float(proba),
            model_name=metadata.name,
            model_version=metadata.version,
        )

    async def _fetch_features_from_db(self, order_id: str) -> pd.DataFrame | None:
        """Fetch pre-computed features from PostgreSQL."""
        query = text(
            """
            SELECT * FROM processed.features
            WHERE order_id = :order_id
            LIMIT 1
            """
        )
        result = await self.db.execute(query, {"order_id": order_id})
        row = result.fetchone()

        if not row:
            return None

        # Convert to DataFrame
        df = pd.DataFrame([row._mapping])
        logger.debug(f"Fetched features for {order_id}: {df.shape}")
        return df

    def _build_features_from_override(
        self, order_id: str, override: FeatureOverride
    ) -> pd.DataFrame:
        """Build feature DataFrame from manual override."""
        data = override.model_dump()
        data["order_id"] = order_id
        return pd.DataFrame([data])

    def _prepare_model_input(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Drop metadata columns, keep only model inputs."""
        drop_cols = [
            "order_id",
            "target_late_delivery",
            "target",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "avg_delivery_delay_days",  # leakage
            "max_delivery_delay_days",  # leakage
            "feature_version",
            "created_at",
            "customer_city",  # high cardinality, may not be in model
            "seller_city",
        ]

        X = features_df.drop(columns=drop_cols, errors="ignore")
        logger.debug(f"Model input shape: {X.shape}, columns: {list(X.columns)}")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction (bonus)
# ─────────────────────────────────────────────────────────────────────────────

class BatchPredictionService(PredictionService):
    """Extended service for batch predictions."""

    async def predict_batch(
        self, orders: list[tuple[str, FeatureOverride | None]]
    ) -> tuple[list[PredictionResponse], list[dict[str, str]]]:
        """
        Predict multiple orders in one call.
        Returns (successful_predictions, errors).
        """
        predictions = []
        errors = []

        for order_id, feature_override in orders:
            try:
                pred = await self.predict_single(order_id, feature_override)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {order_id}: {e}")
                errors.append({"order_id": order_id, "error": str(e)})

        return predictions, errors
