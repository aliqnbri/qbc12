# File: app/pipelines/transformation/schema_registry.py
"""
Schema registry with column specifications, validators, and Polars type mappings.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional
import polars as pl


@dataclass
class ColumnSpec:
    """Column specification for schema validation and transformation."""
    
    dtype: pl.DataType
    nullable: bool = True
    primary_key: bool = False
    validator: Optional[Callable[[Any], bool]] = None
    default: Optional[Any] = None
    description: str = ""


# Validators
def validate_order_status(value: str) -> bool:
    """Validate order status values."""
    valid_statuses = {
        "delivered", "shipped", "canceled", "unavailable",
        "invoiced", "processing", "created", "approved"
    }
    return value.lower() in valid_statuses


def validate_review_score(value: int) -> bool:
    """Validate review score range."""
    return 1 <= value <= 5


def validate_br_state(value: str) -> bool:
    """Validate Brazilian state codes."""
    valid_states = {
        "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
        "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
        "RS", "RO", "RR", "SC", "SP", "SE", "TO"
    }
    return value.upper() in valid_states


def validate_payment_type(value: str) -> bool:
    """Validate payment type values."""
    valid_types = {"credit_card", "boleto", "voucher", "debit_card", "not_defined"}
    return value.lower() in valid_types


def validate_lead_origin(value: str) -> bool:
    """Validate marketing lead origin values."""
    valid_origins = {"organic_search", "paid_search", "social", "direct", "other"}
    return value.lower() in valid_origins


# Table schemas
TABLE_SCHEMAS: dict[str, dict[str, ColumnSpec]] = {
    "orders": {
        "order_id": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "customer_id": ColumnSpec(pl.Utf8, nullable=False),
        "order_status": ColumnSpec(pl.Utf8, nullable=True, validator=validate_order_status),
        "order_purchase_timestamp": ColumnSpec(pl.Datetime, nullable=False),
        "order_approved_at": ColumnSpec(pl.Datetime, nullable=True),
        "order_delivered_carrier_date": ColumnSpec(pl.Datetime, nullable=True),
        "order_delivered_customer_date": ColumnSpec(pl.Datetime, nullable=True),
        "order_estimated_delivery_date": ColumnSpec(pl.Datetime, nullable=False),
    },
    "order_items": {
        "order_id": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "order_item_id": ColumnSpec(pl.Int32, nullable=False, primary_key=True),
        "product_id": ColumnSpec(pl.Utf8, nullable=False),
        "seller_id": ColumnSpec(pl.Utf8, nullable=False),
        "shipping_limit_date": ColumnSpec(pl.Datetime, nullable=False),
        "price": ColumnSpec(pl.Float64, nullable=False),
        "freight_value": ColumnSpec(pl.Float64, nullable=False),
    },
    "order_reviews": {
        "review_id": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "order_id": ColumnSpec(pl.Utf8, nullable=False),
        "review_score": ColumnSpec(pl.Int8, nullable=True, validator=validate_review_score),
        "review_comment_title": ColumnSpec(pl.Utf8, nullable=True),
        "review_comment_message": ColumnSpec(pl.Utf8, nullable=True),
        "review_creation_date": ColumnSpec(pl.Datetime, nullable=False),
        "review_answer_timestamp": ColumnSpec(pl.Datetime, nullable=True),
    },
    "customers": {
        "customer_id": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "customer_unique_id": ColumnSpec(pl.Utf8, nullable=False),
        "customer_zip_code_prefix": ColumnSpec(pl.Utf8, nullable=False),  # Fixed from Int32
        "customer_city": ColumnSpec(pl.Utf8, nullable=False),
        "customer_state": ColumnSpec(pl.Utf8, nullable=False, validator=validate_br_state),
    },
    "products": {
        "product_id": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "product_category_name": ColumnSpec(pl.Utf8, nullable=True),
        "product_name_length": ColumnSpec(pl.Int32, nullable=True),  
        "product_description_length": ColumnSpec(pl.Int32, nullable=True),
        "product_photos_qty": ColumnSpec(pl.Int32, nullable=True),
        "product_weight_g": ColumnSpec(pl.Int32, nullable=True),
        "product_length_cm": ColumnSpec(pl.Int32, nullable=True),
        "product_height_cm": ColumnSpec(pl.Int32, nullable=True),
        "product_width_cm": ColumnSpec(pl.Int32, nullable=True),
    },
    "sellers": {
        "seller_id": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "seller_zip_code_prefix": ColumnSpec(pl.Utf8, nullable=False),  # Fixed from Int32
        "seller_city": ColumnSpec(pl.Utf8, nullable=False),
        "seller_state": ColumnSpec(pl.Utf8, nullable=False, validator=validate_br_state),
    },
    "order_payments": {
        "order_id": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "payment_sequential": ColumnSpec(pl.Int32, nullable=False, primary_key=True),
        "payment_type": ColumnSpec(pl.Utf8, nullable=False, validator=validate_payment_type),
        "payment_installments": ColumnSpec(pl.Int32, nullable=False),
        "payment_value": ColumnSpec(pl.Float64, nullable=False),
    },
    "marketing_qualified_leads": {
        "mql_id": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "first_contact_date": ColumnSpec(pl.Date, nullable=False),
        "landing_page_id": ColumnSpec(pl.Utf8, nullable=False),
        "origin": ColumnSpec(pl.Utf8, nullable=True, validator=validate_lead_origin),
    },
    "closed_deals": {
        "mql_id": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "seller_id": ColumnSpec(pl.Utf8, nullable=False),
        "sdr_id": ColumnSpec(pl.Utf8, nullable=True),
        "sr_id": ColumnSpec(pl.Utf8, nullable=True),
        "won_date": ColumnSpec(pl.Datetime, nullable=False),
        "business_segment": ColumnSpec(pl.Utf8, nullable=True),
        "lead_type": ColumnSpec(pl.Utf8, nullable=True),
        "lead_behaviour_profile": ColumnSpec(pl.Utf8, nullable=True),
        "has_company": ColumnSpec(pl.Boolean, nullable=True),
        "has_gtin": ColumnSpec(pl.Boolean, nullable=True),
        "average_stock": ColumnSpec(pl.Utf8, nullable=True),
        "business_type": ColumnSpec(pl.Utf8, nullable=True),
        "declared_product_catalog_size": ColumnSpec(pl.Float64, nullable=True),
        "declared_monthly_revenue": ColumnSpec(pl.Float64, nullable=True),
    },
    "product_category_name_translation": {
        "product_category_name": ColumnSpec(pl.Utf8, nullable=False, primary_key=True),
        "product_category_name_english": ColumnSpec(pl.Utf8, nullable=False),
    },
}


def get_table_schema(table_name: str) -> dict[str, ColumnSpec]:
    """Get schema specification for a table."""
    if table_name not in TABLE_SCHEMAS:
        raise ValueError(f"Schema not found for table: {table_name}")
    return TABLE_SCHEMAS[table_name]


def get_polars_schema(table_name: str) -> dict[str, pl.DataType]:
    """Get Polars dtype mapping for a table."""
    schema = get_table_schema(table_name)
    return {col: spec.dtype for col, spec in schema.items()}


def validate_row(
    table_name: str,
    row: dict[str, Any],
    raise_on_error: bool = False
) -> tuple[bool, list[str]]:
    """
    Validate a single row against table schema.
    
    Returns:
        (is_valid, list_of_errors)
    """
    schema = get_table_schema(table_name)
    errors = []
    
    for col_name, spec in schema.items():
        value = row.get(col_name)
        
        # Check nullable
        if value is None and not spec.nullable:
            errors.append(f"Column {col_name} cannot be null")
            continue
        
        # Run validator
        if value is not None and spec.validator:
            try:
                if not spec.validator(value):
                    errors.append(f"Validation failed for {col_name}={value}")
            except Exception as e:
                errors.append(f"Validator error for {col_name}: {e}")
    
    is_valid = len(errors) == 0
    
    if raise_on_error and not is_valid:
        raise ValueError(f"Row validation failed: {errors}")
    
    return is_valid, errors
