"""Table definitions with dependency resolution and auto-discovery.

Defines the mapping between CSV files and PostgreSQL tables using
normalized names derived from the Olist naming convention.
"""

from __future__ import annotations
from app.utils.utils import normalize_dataset_name
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar



@dataclass(frozen=True)
class TableDefinition:
    """Metadata for a single CSV-to-table mapping.
    
    Attributes:
        table: PostgreSQL table name (normalized from CSV filename)
        schema: PostgreSQL schema name (default: 'raw')
        primary_keys: Column names forming the primary key
        depends_on: Table names that must be loaded first (FK dependencies)
        csv_pattern: Optional explicit CSV filename pattern
    """
    
    table: str
    schema: str
    primary_keys: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    csv_pattern: str | None = None
    
    @property
    def full_table_name(self) -> str:
        """Qualified table name: schema.table"""
        return f"{self.schema}.{self.table}"
    
    def matches_csv(self, csv_path: Path) -> bool:
        """Check if this definition matches a CSV file.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            True if the normalized stem matches this table name
        """
        if self.csv_pattern:
            return csv_path.name == self.csv_pattern
        
        # Default: match by normalized name
        normalized = normalize_dataset_name(csv_path.stem)
        return normalized == self.table


class TableRegistry:
    """Registry of all tables with dependency-aware ordering.
    
    Tables are defined in topological order respecting foreign key constraints.
    The registry uses normalized table names derived from CSV filenames.
    """
    
    # Ordered by FK dependencies
    DEFINITIONS: ClassVar[tuple[TableDefinition, ...]] = (
        # Independent tables (no dependencies)
        TableDefinition(
            table="customers",
            schema="raw",
            primary_keys=("customer_id",),
        ),
        TableDefinition(
            table="sellers",
            schema="raw",
            primary_keys=("seller_id",),
        ),
        TableDefinition(
            table="products",
            schema="raw",
            primary_keys=("product_id",),
        ),
        TableDefinition(
            table="product_category_name_translation",
            schema="raw",
            primary_keys=("product_category_name",),
        ),
        TableDefinition(
            table="marketing_qualified_leads",
            schema="raw",
            primary_keys=("mql_id",),
        ),
        
        # Dependent tables (orders → items/payments/reviews)
        TableDefinition(
            table="orders",
            schema="raw",
            primary_keys=("order_id",),
            depends_on=("customers",),
        ),
        TableDefinition(
            table="order_items",
            schema="raw",
            primary_keys=("order_id", "order_item_id"),
            depends_on=("orders", "products", "sellers"),
        ),
        TableDefinition(
            table="order_payments",
            schema="raw",
            primary_keys=("order_id", "payment_sequential"),
            depends_on=("orders",),
        ),
        TableDefinition(
            table="order_reviews",
            schema="raw",
            primary_keys=("review_id",),
            depends_on=("orders",),
        ),
        TableDefinition(
            table="closed_deals",
            schema="raw",
            primary_keys=("mql_id",),
            depends_on=("marketing_qualified_leads",),
        ),
    )
    
    @classmethod
    def get_definition_for_csv(cls, csv_path: Path) -> TableDefinition | None:
        """Find table definition matching a CSV file.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Matching TableDefinition or None if no match found
        """
        for defn in cls.DEFINITIONS:
            if defn.matches_csv(csv_path):
                return defn
        return None
    
    @classmethod
    def get_by_table(cls, table: str) -> TableDefinition | None:
        """Lookup table definition by table name.
        
        Args:
            table: Table name (normalized)
        
        Returns:
            TableDefinition or None
        """
        return next(
            (defn for defn in cls.DEFINITIONS if defn.table == table),
            None,
        )
    
    @classmethod
    def get_load_order(cls) -> list[TableDefinition]:
        """Get tables in dependency-respecting load order.
        
        Returns:
            List of TableDefinitions in the order they should be loaded
        """
        return list(cls.DEFINITIONS)
