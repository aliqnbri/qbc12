"""
Central import point for all SQLModel table classes.
Importing this module registers all tables in SQLModel.metadata.
"""

# Raw schema tables
from app.models.raw import (
    OrderRaw,
    OrderItemRaw,
    OrderReviewRaw,
    CustomerRaw,
    ProductRaw,
    SellerRaw,
    OrderPaymentRaw,
    MarketingQualifiedLeadRaw,
    ClosedDealRaw,
    ProductCategoryNameTranslationRaw,
)

# Processed schema tables
from app.models.processed import (
    FeatureStore,
)

# Predictions schema tables
from app.models.predictions import (
    PredictionLog,
)

__all__ = [
    # Raw
    "OrderRaw",
    "OrderItemRaw",
    "OrderReviewRaw",
    "CustomerRaw",
    "ProductRaw",
    "SellerRaw",
    "OrderPaymentRaw",
    "MarketingQualifiedLeadRaw",
    "ClosedDealRaw",
    "ProductCategoryNameTranslationRaw",
    # Processed
    "FeatureStore",
    # Predictions
    "PredictionLog",
]
